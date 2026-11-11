"""The incremental architecture options: Derf/DyT, XSA, document masking,
Block AttnRes, mHC -- and that the L-tier configs keep the 2x2 fair."""

import glob
import os

import pytest
import torch

from src.model import Config, ModelConfig, Transformer, count_params
from src.model.attention import PointwiseNorm, document_mask, exclude_self
from src.model.residual import AttnRes, sinkhorn
from tests.conftest import make_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = [
    dict(norm=norm, residual=residual, xsa=xsa, doc_mask=True, eos_id=0)
    for norm in ("rmsnorm", "derf", "dyt")
    for residual in ("standard", "attnres", "mhc")
    for xsa in (False, True)
]


@pytest.fixture(params=ARCH, ids=lambda a: f"{a['norm']}-{a['residual']}-xsa{int(a['xsa'])}")
def arch_model(request, attention_kind):
    torch.manual_seed(0)
    return Transformer(make_config(attention=attention_kind, ffn="moe", n_layers=4,
                                   n_dense_layers=1, attnres_block_layers=2, **request.param))


def test_every_param_gets_grad(arch_model):
    x = torch.randint(1, arch_model.cfg.vocab_size, (2, arch_model.cfg.seq_len))
    x[:, 5] = 0  # a document boundary
    logits, aux, z = arch_model(x)
    (logits.float().sum() + aux + z).backward()
    dead = [n for n, p in arch_model.named_parameters() if p.grad is None or torch.all(p.grad == 0)]
    assert not dead, f"parameters with no/zero gradient: {dead}"


def test_future_tokens_dont_leak(arch_model):
    arch_model.eval()
    x = torch.randint(1, arch_model.cfg.vocab_size, (1, arch_model.cfg.seq_len))
    t = arch_model.cfg.seq_len // 2
    x2 = x.clone()
    x2[0, t:] = x2[0, t:] % (arch_model.cfg.vocab_size - 1) + 1
    with torch.no_grad():
        a, b = arch_model(x)[0], arch_model(x2)[0]
    assert torch.allclose(a[0, :t], b[0, :t], atol=1e-5)


@pytest.mark.parametrize("use_flex", [False, True])
def test_document_mask_isolates_documents(attention_kind, use_flex):
    torch.manual_seed(0)
    model = Transformer(make_config(attention=attention_kind, doc_mask=True, eos_id=0, xsa=True)).eval()
    model.use_flex = use_flex
    x = torch.randint(1, 64, (1, 16))
    x[0, 6] = 0  # doc A = positions 0..6 (EOS included), doc B = 7..15
    x2 = x.clone()
    x2[0, :6] = x2[0, :6] % 63 + 1  # rewrite doc A entirely
    with torch.no_grad():
        a, b = model(x)[0], model(x2)[0]
    assert torch.allclose(a[0, 7:], b[0, 7:], atol=1e-5), "doc B saw doc A"
    assert not torch.allclose(a[0, :6], b[0, :6])


def test_flex_and_dense_masks_agree(attention_kind):
    torch.manual_seed(0)
    model = Transformer(make_config(attention=attention_kind, doc_mask=True, eos_id=0)).eval()
    x = torch.randint(1, 64, (2, 16))
    x[0, 3] = x[1, 9] = x[1, 12] = 0
    with torch.no_grad():
        model.use_flex = False
        dense = model(x)[0]
        model.use_flex = True
        flex = model(x)[0]
    assert torch.allclose(dense, flex, atol=1e-5)


def test_document_mask_is_none_without_eos():
    assert document_mask(torch.ones(2, 8, dtype=torch.long), 0) is None


def test_xsa_output_is_orthogonal_to_self_value():
    o, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8)
    z = exclude_self(o, v)
    assert torch.allclose((z * v).sum(-1), torch.zeros(2, 3, 5), atol=1e-5)


@pytest.mark.parametrize("kind, fn", [("derf", lambda x: torch.erf(0.5 * x)), ("dyt", lambda x: torch.tanh(0.5 * x))])
def test_pointwise_norm_init(kind, fn):
    x = torch.randn(4, 16) * 5
    assert torch.allclose(PointwiseNorm(16, kind, 0.5)(x), fn(x))


def test_attnres_starts_as_uniform_average():
    sources = [torch.randn(2, 3, 8) for _ in range(4)]
    assert torch.allclose(AttnRes(8)(sources), torch.stack(sources).mean(0), atol=1e-6)


def test_attnres_matches_stacked_reference():
    """The stack-free forward equals softmax_n(q . RMSNorm(s_n)) . s_n, values and grads,
    with mixed fp32/bf16 sources like the autocast path (fp32 embedding, bf16 block sums)."""
    torch.manual_seed(0)
    res = AttnRes(16)
    with torch.no_grad():
        res.query.normal_()
    sources = [torch.randn(2, 5, 16, requires_grad=True)] + [
        (3 * torch.randn(2, 5, 16)).bfloat16().requires_grad_() for _ in range(3)]
    values = torch.stack([s.float() for s in sources])
    keys = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + res.eps)
    ref = torch.einsum("nbs,nbsd->bsd", torch.einsum("d,nbsd->nbs", res.query, keys).softmax(0), values)
    out = res(sources)
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, atol=0.05, rtol=0.02)  # bf16 gemv logits
    g_out = torch.autograd.grad(out.sum(), [res.query, *sources])
    g_ref = torch.autograd.grad(ref.sum(), [res.query, *sources])
    for a, b in zip(g_out, g_ref):
        assert torch.allclose(a.float(), b.float(), atol=0.1, rtol=0.05)


def test_sinkhorn_is_doubly_stochastic():
    m = sinkhorn(torch.randn(10, 4, 4), 20)  # 20 iterations: approximate for very skewed logits
    assert (m >= 0).all()
    assert torch.allclose(m.sum(-1), torch.ones(10, 4), atol=1e-3)
    assert torch.allclose(m.sum(-2), torch.ones(10, 4), atol=1e-5)


def test_mhc_starts_close_to_standard_residual():
    torch.manual_seed(0)
    std = Transformer(make_config())
    torch.manual_seed(0)
    mhc = Transformer(make_config(residual="mhc"))
    mhc.load_state_dict(std.state_dict(), strict=False)
    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        a, b = std(x)[0], mhc(x)[0]
    assert (a - b).abs().max() < 0.05 * a.abs().max()


def test_activation_checkpointing_matches_plain_backward():
    torch.manual_seed(0)
    model = Transformer(make_config(ffn="moe", residual="attnres", norm="derf", attnres_block_layers=1))
    x = torch.randint(0, 64, (2, 16))
    grads = []
    for ckpt in (False, True):
        model.zero_grad()
        model.gradient_checkpointing = ckpt
        logits, aux, z = model(x)
        (logits.sum() + aux + z).backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    for g0, g1 in zip(*grads):
        assert torch.allclose(g0, g1, atol=1e-5)


def test_yaml_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("model:\n  dim: 32\n  n_heads: 4\n  xsa: true\ntrain:\n  source_weights: {a: 2}\n")
    (tmp_path / "cell.yaml").write_text("base: base.yaml\nrun_name: cell\nmodel:\n  attention: differential\n")
    cfg = Config.from_yaml(str(tmp_path / "cell.yaml"), ["model.xsa=false", "train.source_weights.b=3"])
    assert (cfg.run_name, cfg.model.dim, cfg.model.attention, cfg.model.xsa) == ("cell", 32, "differential", False)
    assert cfg.train.source_weights == {"a": 2, "b": 3}


def l_tier_counts():
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "configs", "l_*.yaml"))):
        if path.endswith("l_base.yaml"):
            continue
        cfg = Config.from_yaml(path)
        with torch.device("meta"):
            out[cfg.run_name] = (cfg, count_params(Transformer(cfg.model)))
    return out


def test_l_tier_cells_are_parameter_matched_and_fit_one_48gb_gpu():
    counts = l_tier_counts()
    assert {"l_dense", "l_diff", "l_moe", "l_diffmoe", "l_latentmoe", "l_diff_latentmoe"} <= set(counts)
    ref = counts["l_dense"][1]["active"]
    for name, (cfg, c) in counts.items():
        assert abs(c["active"] - ref) / ref < 0.002, f"{name} active params drift"
        # fp32 weights + grads + AdamW moments must leave room for activations on 48 GB
        assert c["total"] * 16 / 2**30 < 32, f"{name} train state too large for one 48 GB GPU"
        assert cfg.train.global_batch_tokens % (cfg.train.batch_size * cfg.model.seq_len * 4) == 0
    moe_total = counts["l_moe"][1]["total"]
    assert abs(counts["l_latentmoe"][1]["total"] - moe_total) / moe_total < 0.002
