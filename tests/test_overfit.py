"""Classic sanity check: a model that can't drive loss to ~0 on one repeated
batch has a wiring bug, not a hyperparameter problem."""

import torch
import torch.nn.functional as F

from src.model import ModelConfig, Transformer


def test_overfit_single_batch():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, dim=32, n_layers=2, n_heads=4, seq_len=16,
                       attention="standard", ffn="dense", inter_dim=64)
    model = Transformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    x = torch.randint(0, cfg.vocab_size, (4, cfg.seq_len))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.seq_len))

    losses = []
    for _ in range(300):
        logits, aux, z = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1)) + aux + z
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < 0.5, f"failed to overfit a single batch, final loss {losses[-1]:.4f}"
