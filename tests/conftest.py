import pytest

from src.model import ModelConfig, Transformer


def make_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=64,
        dim=32,
        n_layers=2,
        n_heads=4,
        seq_len=16,
        tie_embeddings=True,
        attention="standard",
        ffn="dense",
        inter_dim=64,
        n_dense_layers=0,
        n_experts=4,
        top_k=2,
        expert_inter_dim=32,
        n_shared_experts=0,
        aux_loss_coef=0.01,
        router_z_coef=0.001,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture(params=["standard", "differential"])
def attention_kind(request):
    return request.param


@pytest.fixture(params=["dense", "moe"])
def ffn_kind(request):
    return request.param


@pytest.fixture
def tiny_model(attention_kind, ffn_kind):
    cfg = make_config(attention=attention_kind, ffn=ffn_kind)
    return Transformer(cfg)
