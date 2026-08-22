"""docs/plan.md's central claim: standard vs differential attention cost the
same params (4*dim^2), and dense vs MoE FFN cost the same *active* params.
These are what make the 2x2 ablation a fair comparison."""

from src.model import ModelConfig, Transformer, count_params


def test_attention_parity():
    dense_std = ModelConfig(dim=384, n_heads=6, attention="standard", ffn="dense",
                             inter_dim=1024, n_layers=8, vocab_size=4096)
    dense_diff = ModelConfig(dim=384, n_heads=6, attention="differential", ffn="dense",
                              inter_dim=1024, n_layers=8, vocab_size=4096)
    a = count_params(Transformer(dense_std))["active"]
    b = count_params(Transformer(dense_diff))["active"]
    assert abs(a - b) / a < 0.01, f"attention parity broken: {a} vs {b}"


def test_ffn_active_parity():
    std_dense = ModelConfig(dim=384, n_heads=6, attention="standard", ffn="dense",
                             inter_dim=1024, n_layers=8, vocab_size=4096)
    std_moe = ModelConfig(dim=384, n_heads=6, attention="standard", ffn="moe",
                           inter_dim=1024, n_dense_layers=2, n_experts=8, top_k=2,
                           expert_inter_dim=512, n_layers=8, vocab_size=4096)
    a = count_params(Transformer(std_dense))["active"]
    b = count_params(Transformer(std_moe))["active"]
    assert abs(a - b) / a < 0.02, f"FFN active-param parity broken: {a} vs {b}"


def test_moe_raw_exceeds_active():
    cfg = ModelConfig(dim=384, n_heads=6, attention="standard", ffn="moe",
                       inter_dim=1024, n_dense_layers=2, n_experts=8, top_k=2,
                       expert_inter_dim=512, n_layers=8, vocab_size=4096)
    counts = count_params(Transformer(cfg))
    assert counts["total"] > counts["active"], "MoE raw params should exceed active params"


def test_stable_latent_moe_matches_trained_moe_budgets():
    # Scale-equivalent miniature of the tier-S 768/384 geometry. Keeping all
    # width ratios and expert counts identical verifies the same accounting
    # without allocating two ~379M-parameter models in the unit test suite.
    common = dict(
        vocab_size=128,
        dim=48,
        n_layers=4,
        n_heads=6,
        seq_len=16,
        attention="standard",
        inter_dim=192,
        n_dense_layers=1,
    )
    baseline = ModelConfig(
        **common,
        ffn="moe",
        n_experts=6,
        top_k=2,
        expert_inter_dim=96,
    )
    latent = ModelConfig(
        **common,
        ffn="stable_latent_moe",
        latent_dim=24,
        n_experts=16,
        top_k=4,
        expert_inter_dim=64,
        n_shared_experts=2,
        shared_inter_dim=24,
        aux_loss_coef=0.0,
        router_z_coef=0.0,
    )
    base_counts = count_params(Transformer(baseline))
    latent_counts = count_params(Transformer(latent))
    expected_fixed_overhead = (common["n_layers"] - common["n_dense_layers"]) * (
        (latent.n_experts - baseline.n_experts) * common["dim"] + latent.latent_dim
    )
    for key in ("total", "active"):
        # Expert/shared/projection budgets match exactly. The only intended
        # delta is the larger router plus one latent RMSNorm per routed layer.
        assert latent_counts[key] - base_counts[key] == expected_fixed_overhead
