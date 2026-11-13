"""FSDP2 gradient accumulation with hold_gathered: same weights after the optimizer
steps as FSDP's default resharding, with far fewer all-gathers. Two CPU ranks on gloo."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = ["model.n_layers=4", "model.n_dense_layers=1", "model.dim=64", "model.n_heads=4", "model.seq_len=64",
        "model.inter_dim=128", "model.vocab_size=256", "model.expert_inter_dim=64"]
STEPS, ACCUM = 2, 3


def _train(hold: bool):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from src.model import Config, Transformer
    from src.train import hold_gathered

    cfg = Config.from_yaml(os.path.join(ROOT, "configs", "l_diffmoe.yaml"), TINY).model
    torch.manual_seed(0)
    model = Transformer(cfg)
    model.gradient_checkpointing = True
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in model.blocks:
        fully_shard(block, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(1 + dist.get_rank())

    # Count FSDP2's per-group all-gathers (private module: path may move in newer torch).
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

    gathers = 0
    real = pg.foreach_all_gather

    def counting(*a, **k):
        nonlocal gathers
        gathers += 1
        return real(*a, **k)

    pg.foreach_all_gather = counting
    try:
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            for micro in range(ACCUM):
                last = micro == ACCUM - 1
                model.set_requires_gradient_sync(last)
                if hold:
                    hold_gathered(model, hold=not last)
                x = torch.randint(1, cfg.vocab_size, (2, cfg.seq_len), generator=gen)
                x[:, ::17] = 0
                logits, aux, z = model(x)
                loss = F.cross_entropy(logits.float().view(-1, cfg.vocab_size), x.view(-1)) + 0.01 * aux
                (loss / ACCUM).backward()
            opt.step()
    finally:
        pg.foreach_all_gather = real
    sharded = all(type(p).__name__ == "DTensor" for p in model.parameters())
    return {n: p.full_tensor().float() for n, p in model.named_parameters()}, gathers, sharded


def _worker(rank, store_path, out_path):
    dist.init_process_group("gloo", init_method=f"file:///{store_path}", rank=rank, world_size=2)
    try:
        base, base_gathers, _ = _train(hold=False)
        held, held_gathers, sharded = _train(hold=True)
        if rank == 0:
            diff = max((base[n] - held[n]).abs().max().item() for n in base)
            torch.save({"diff": diff, "base": base_gathers, "held": held_gathers, "sharded": sharded}, out_path)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="needs gloo")
def test_hold_gathered_matches_default_with_fewer_gathers(tmp_path):
    out = tmp_path / "result.pt"
    mp.spawn(_worker, args=(str(tmp_path / "store").replace("\\", "/"), str(out)), nprocs=2)
    r = torch.load(out)
    assert r["diff"] == 0.0, f"weights differ by {r['diff']}"
    assert r["sharded"], "weights must be resharded before the optimizer step"
    assert r["held"] * 2 < r["base"], f"all-gathers: held {r['held']} vs default {r['base']}"
