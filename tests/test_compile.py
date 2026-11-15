"""compile_blocks under activation checkpointing: same loss and grads as eager.
Uses the aot_eager backend (same Dynamo graphs and AOT autograd, no C++/Triton codegen),
so it runs on any CPU box."""

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import Config, Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = ["model.n_layers=4", "model.n_dense_layers=1", "model.dim=64", "model.n_heads=4", "model.seq_len=64",
        "model.inter_dim=128", "model.vocab_size=256", "model.expert_inter_dim=64"]


def test_compiled_blocks_match_eager_under_checkpointing(monkeypatch):
    orig = nn.Module.compile
    monkeypatch.setattr(nn.Module, "compile", lambda self, **kw: orig(self, backend="aot_eager", **kw))
    torch._dynamo.reset()
    cfg = Config.from_yaml(os.path.join(ROOT, "configs", "l_diffmoe.yaml"), TINY).model
    torch.manual_seed(0)
    eager = Transformer(cfg)
    compiled = copy.deepcopy(eager)
    compiled.compile_blocks()
    assert compiled.final_res._compiled_call_impl is not None
    assert all(b.ffn_res._compiled_call_impl is not None for b in compiled.blocks)
    gen = torch.Generator().manual_seed(1)
    try:
        for _ in range(3):  # later iterations hit cached graphs, as in training
            x = torch.randint(1, cfg.vocab_size, (2, cfg.seq_len), generator=gen)
            x[:, ::17] = 0
            losses = []
            for m in (eager, compiled):
                m.gradient_checkpointing = True
                m.zero_grad(set_to_none=True)
                logits, aux, _ = m(x)
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), x.view(-1)) + 0.01 * aux
                loss.backward()
                losses.append(loss.item())
            assert abs(losses[0] - losses[1]) < 1e-5
            for (n, a), b in zip(eager.named_parameters(), compiled.parameters()):
                assert torch.allclose(a.grad, b.grad, atol=1e-5, rtol=1e-4), n
    finally:
        torch._dynamo.reset()
