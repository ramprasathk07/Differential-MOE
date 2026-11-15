import torch

from src.model import ModelConfig, Transformer
from src.model.moe import (
    QuantileBalancedSigmoidGate,
    StableLatentMoE,
    situ_glu_activation,
)


def stable_config(**overrides):
    values = dict(
        vocab_size=64,
        dim=16,
        n_layers=2,
        n_heads=4,
        seq_len=8,
        attention="standard",
        ffn="stable_latent_moe",
        inter_dim=32,
        n_dense_layers=0,
        latent_dim=8,
        n_experts=4,
        top_k=2,
        expert_inter_dim=8,
        n_shared_experts=2,
        shared_inter_dim=4,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        quantile_balance=True,
        aux_loss_coef=0.0,
        router_z_coef=0.0,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_situ_glu_product_is_bounded():
    gate = torch.tensor([-1e6, -10.0, 0.0, 10.0, 1e6])
    up = torch.tensor([1e6, -1e6, 0.0, 1e6, -1e6])
    out = situ_glu_activation(gate, up, beta=4.0, linear_beta=25.0)
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 100.0


def test_sigmoid_router_uses_bias_only_for_selection():
    torch.manual_seed(0)
    gate = QuantileBalancedSigmoidGate(stable_config(quantile_balance=False))
    x = torch.randn(12, 16)
    weights_before, idx_before = gate(x)
    with torch.no_grad():
        gate.expert_bias[3] = 10.0
    weights_after, idx_after = gate(x)

    assert torch.allclose(weights_before.sum(dim=-1), torch.ones(12), atol=1e-6)
    assert torch.allclose(weights_after.sum(dim=-1), torch.ones(12), atol=1e-6)
    assert (idx_after == 3).any(dim=-1).all()
    # Bias changes dispatch, but mixture weights remain normalized raw sigmoid scores.
    logits = torch.nn.functional.linear(x.float(), gate.weight).sigmoid()
    expected = logits.gather(1, idx_after)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    assert torch.allclose(weights_after, expected)
    assert not torch.equal(idx_before, idx_after)


def test_quantile_balancing_is_delayed_and_centered():
    torch.manual_seed(1)
    gate = QuantileBalancedSigmoidGate(stable_config())
    gate.train()
    x = torch.randn(32, 16)
    initial_bias = gate.expert_bias.clone()
    gate(x)

    # The batch that produced the margins must not change its own routing bias.
    assert torch.equal(gate.expert_bias, initial_bias)
    updated = gate.update_quantile_bias()
    assert updated is not None
    assert abs(updated.mean().item()) < 1e-6
    assert not torch.equal(updated, initial_bias)
    assert gate.update_quantile_bias() is None  # pending batch was consumed once


def test_quantile_balancing_reduces_route_imbalance():
    torch.manual_seed(5)
    gate = QuantileBalancedSigmoidGate(stable_config(top_k=1))
    gate.train()
    x = torch.randn(128, 16)
    _, before_idx = gate(x)
    before = torch.bincount(before_idx.flatten(), minlength=gate.n_experts)
    gate.update_quantile_bias()
    _, after_idx = gate(x)
    after = torch.bincount(after_idx.flatten(), minlength=gate.n_experts)

    before_range = before.max() - before.min()
    after_range = after.max() - after.min()
    assert after_range < before_range


def test_stable_latent_moe_paths_receive_gradients():
    torch.manual_seed(2)
    moe = StableLatentMoE(stable_config())
    x = torch.randn(4, 8, 16)
    out, aux, z = moe(x)
    loss = out.square().mean() + aux + z
    loss.backward()

    assert out.shape == x.shape
    assert aux.item() == 0.0 and z.item() == 0.0
    required = {
        "gate.weight",
        "shared_experts.w1.weight",
        "routed_expert_down_proj.weight",
        "routed_expert_norm.weight",
        "routed_expert_up_proj.weight",
    }
    grads = {name for name, param in moe.named_parameters() if param.grad is not None}
    assert required <= grads
    assert moe.last_counts.sum().item() == x.shape[0] * x.shape[1] * moe.gate.top_k


def test_transformer_commits_qb_for_every_latent_layer():
    torch.manual_seed(3)
    model = Transformer(stable_config())
    tokens = torch.randint(0, model.cfg.vocab_size, (2, model.cfg.seq_len))
    model(tokens)
    updates = model.update_quantile_balancing()
    assert set(updates) == {0, 1}


def test_quantile_bias_is_checkpointed():
    torch.manual_seed(4)
    first = Transformer(stable_config())
    tokens = torch.randint(0, first.cfg.vocab_size, (2, first.cfg.seq_len))
    first(tokens)
    first.update_quantile_balancing()

    second = Transformer(stable_config())
    second.load_state_dict(first.state_dict())
    for block_a, block_b in zip(first.blocks, second.blocks):
        assert torch.equal(block_a.ffn.gate.expert_bias, block_b.ffn.gate.expert_bias)
