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
