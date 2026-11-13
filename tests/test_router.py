"""Guards against silent expert collapse: the aux loss should push a skewed
router toward uniform usage, and after a few optimizer steps on random data
every expert should receive at least one token."""

import pytest
import torch

from src.model.config import ModelConfig
from src.model.moe import MoE


def test_aux_loss_penalizes_imbalance():
    cfg = ModelConfig(dim=16, n_heads=4, n_experts=4, top_k=1, expert_inter_dim=16, vocab_size=32)
    moe = MoE(cfg)

    # skew the gate weight so expert 0 always wins
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0] += 10.0

    x = torch.randn(1, 8, 16)
    _, aux_skewed, _ = moe(x)

    with torch.no_grad():
        moe.gate.weight.zero_()  # uniform logits

    _, aux_uniform, _ = moe(x)

    assert aux_skewed.item() > aux_uniform.item(), "aux loss should be higher for a skewed router"


def test_all_experts_used_after_training_steps():
    torch.manual_seed(0)
    cfg = ModelConfig(dim=16, n_heads=4, n_experts=4, top_k=2, expert_inter_dim=16, vocab_size=32)
    moe = MoE(cfg)
    opt = torch.optim.Adam(moe.parameters(), lr=1e-2)

    for _ in range(50):
        x = torch.randn(4, 8, 16)
        out, aux, z = moe(x)
        loss = out.pow(2).mean() + 0.05 * aux + 0.001 * z
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert (moe.last_counts > 0).all(), f"dead experts after training: {moe.last_counts.tolist()}"


def _loop_dispatch(experts, x, idx, weights):
    """The original per-expert loop, kept as the reference."""
    out = torch.zeros(x.size(0), experts[0].w2.out_features, dtype=x.dtype)
    for e in range(len(experts)):
        tok, slot = torch.where(idx == e)
        out[tok] += experts[e](x[tok]) * weights[tok, slot, None]
    return out


@pytest.mark.parametrize("top_k", [2, 4])
def test_sorted_dispatch_matches_per_expert_loop(top_k):
    from src.model.moe import SwiGLU, dispatch

    torch.manual_seed(0)
    n_experts, T, D = 6, 40, 16
    experts = torch.nn.ModuleList([SwiGLU(D, 24) for _ in range(n_experts)])
    x = torch.randn(T, D, requires_grad=True)
    # expert 3 receives no tokens: it must still run (on zero rows) and get a grad
    idx = torch.stack([torch.randperm(n_experts - 1)[:top_k] for _ in range(T)])
    idx[idx >= 3] += 1
    weights = torch.rand(T, top_k, requires_grad=True)

    results = []
    for fn in (_loop_dispatch, lambda *a: dispatch(*a)[0]):
        experts.zero_grad(set_to_none=True)
        x.grad = weights.grad = None
        out = fn(experts, x, idx, weights)
        out.pow(2).sum().backward()
        results.append([out.detach(), x.grad.clone(), weights.grad.clone(),
                        *[p.grad.clone() for p in experts.parameters()]])
    for ref, new in zip(*results):
        assert torch.allclose(ref, new, atol=1e-6, rtol=1e-5)
    counts = dispatch(experts, x, idx, weights)[1]
    assert counts.tolist() == torch.bincount(idx.flatten(), minlength=n_experts).tolist() and counts[3] == 0
