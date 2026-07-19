"""Catches the v0 bug class: @torch.inference_mode() on forward, and KV-cache
writes that silently detach gradients from wk/wv."""

import torch


def test_every_param_gets_grad(tiny_model):
    x = torch.randint(0, tiny_model.cfg.vocab_size, (2, tiny_model.cfg.seq_len))
    logits, aux, z = tiny_model(x)
    loss = logits.float().sum() + aux + z
    loss.backward()

    dead = [name for name, p in tiny_model.named_parameters() if p.grad is None or torch.all(p.grad == 0)]
    assert not dead, f"parameters with no/zero gradient: {dead}"
