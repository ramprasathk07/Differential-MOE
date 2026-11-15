"""Gradient clipping with a separate group for scalar parameters, and spike attribution.

In l_diffmoe one Derf shift (a single scalar) carried up to 66% of the squared gradient norm and
spiked to 1e13; a global clip then divided every other weight's gradient by the same factor."""

import copy
import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.train import clip_gradients, grad_norm_by_param
from tests.conftest import make_config

SPIKY = "blocks.0.attn_norm.shift"


def _model_with_grads(spike: float):
    from src.model import Transformer

    torch.manual_seed(0)
    model = Transformer(make_config(norm="derf", residual="attnres"))
    logits, aux, z = model(_batch())
    (logits.float().pow(2).mean() * 1e-3).backward()  # small, so the weight group stays under the clip
    dict(model.named_parameters())[SPIKY].grad += spike
    return model


def _batch():
    return torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(1))


def _weights(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.numel() > 1}


def test_scalar_spike_no_longer_shrinks_the_other_weights():
    model = _model_with_grads(1e6)
    before = _weights(model)
    w_norm = math.sqrt(sum(g.pow(2).sum().item() for g in before.values()))
    assert w_norm < 1.0
    total, norms, top = clip_gradients(model, 1.0, scalars_apart=True, spike_at=10.0)
    assert total == pytest.approx(math.hypot(*norms))
    assert norms[0] == pytest.approx(w_norm, rel=1e-5) and norms[1] > 1e5
    assert top[0][0] == SPIKY
    for n, g in _weights(model).items():
        assert torch.equal(g, before[n]), f"{n} was rescaled"
    scalars = [p.grad for p in model.parameters() if p.numel() == 1]
    assert math.sqrt(sum(g.pow(2).sum().item() for g in scalars)) == pytest.approx(1.0, rel=1e-4)

    model = _model_with_grads(1e6)  # one global group: the scalar's spike shrinks everything ~1e6x
    total, norms, top = clip_gradients(model, 1.0, scalars_apart=False)
    assert len(norms) == 1 and top is None
    for n, g in _weights(model).items():
        assert torch.allclose(g, before[n] / total, rtol=1e-3, atol=0)


def test_spike_attribution_only_above_threshold_or_nonfinite():
    assert clip_gradients(_model_with_grads(0.0), 1.0, True, spike_at=1e9)[2] is None
    assert clip_gradients(_model_with_grads(0.0), 1.0, True, spike_at=None)[2] is None
    top = clip_gradients(_model_with_grads(float("inf")), 1.0, True, spike_at=1e9)[2]
    assert top[0][0] == SPIKY and math.isinf(top[0][1])


def _fsdp_worker(rank, store_path, out_path):
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group("gloo", init_method=f"file:///{store_path}", rank=rank, world_size=2)
    try:
        ref = _model_with_grads(0.0)
        model = copy.deepcopy(ref)
        model.zero_grad(set_to_none=True)
        for block in model.blocks:
            fully_shard(block)
        fully_shard(model)
        logits, aux, z = model(_batch())  # same batch on both ranks: averaged grads == ref grads
        (logits.float().pow(2).mean() * 1e-3).backward()
        dict(model.named_parameters())[SPIKY].grad.to_local().add_(5e5)  # (1,) shard: rank 0 holds it
        top = grad_norm_by_param(model, top=50)
        total, norms, _ = clip_gradients(model, 1.0, scalars_apart=True)
        if rank == 0:
            torch.save({"top": top, "total": total, "norms": norms}, out_path)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="needs gloo")
def test_attribution_and_clipping_under_fsdp_match_unsharded(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")  # gloo ranks on CPU; FSDP2 would pick cuda:0 if visible
    out = tmp_path / "r.pt"
    mp.spawn(_fsdp_worker, args=(str(tmp_path / "store").replace("\\", "/"), str(out)), nprocs=2)
    r = torch.load(out)
    ref = _model_with_grads(0.0)
    dict(ref.named_parameters())[SPIKY].grad += 5e5
    want = {n: p.grad.norm().item() for n, p in ref.named_parameters()}
    for n, v in r["top"]:
        assert v == pytest.approx(want[n], rel=1e-3, abs=1e-6), n
    assert r["top"][0][0] == SPIKY
    total, norms, _ = clip_gradients(ref, 1.0, scalars_apart=True)
    assert r["total"] == pytest.approx(total, rel=1e-4)
    assert r["norms"] == pytest.approx(norms, rel=1e-4)
