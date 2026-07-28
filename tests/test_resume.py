"""Checkpoint + resume must reproduce the exact same training trajectory --
critical on Kaggle where 12h session limits force multi-session training."""

import torch
import torch.nn.functional as F

from src.model import ModelConfig, Transformer


def make_model_and_opt(seed):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=32, dim=32, n_layers=2, n_heads=4, seq_len=16,
                       attention="standard", ffn="dense", inter_dim=64)
    model = Transformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt, cfg


def run_steps(model, opt, cfg, gen, n_steps):
    losses = []
    for _ in range(n_steps):
        x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len), generator=gen)
        y = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len), generator=gen)
        logits, aux, z = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1)) + aux + z
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_resume_matches_uninterrupted():
    # uninterrupted: 10 steps straight
    model_a, opt_a, cfg = make_model_and_opt(seed=1)
    gen_a = torch.Generator().manual_seed(123)
    losses_a = run_steps(model_a, opt_a, cfg, gen_a, 10)

    # interrupted: 5 steps, checkpoint, reload, 5 more steps
    model_b, opt_b, _ = make_model_and_opt(seed=1)
    gen_b = torch.Generator().manual_seed(123)
    losses_b_part1 = run_steps(model_b, opt_b, cfg, gen_b, 5)

    ckpt = {
        "model": model_b.state_dict(),
        "optimizer": opt_b.state_dict(),
        "rng_state": gen_b.get_state(),
    }

    model_c, opt_c, _ = make_model_and_opt(seed=1)
    model_c.load_state_dict(ckpt["model"])
    opt_c.load_state_dict(ckpt["optimizer"])
    gen_c = torch.Generator().manual_seed(123)
    gen_c.set_state(ckpt["rng_state"])
    losses_b_part2 = run_steps(model_c, opt_c, cfg, gen_c, 5)

    losses_b = losses_b_part1 + losses_b_part2
    for i, (la, lb) in enumerate(zip(losses_a, losses_b)):
        assert abs(la - lb) < 1e-4, f"resume diverged at step {i}: {la} vs {lb}"
