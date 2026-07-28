"""Guards against silent expert collapse: the aux loss should push a skewed
router toward uniform usage, and after a few optimizer steps on random data
every expert should receive at least one token."""

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
