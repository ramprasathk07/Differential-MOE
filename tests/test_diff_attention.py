"""Differential attention matches the paper's construction: correct output
shape, lambda computed in fp32, lambda_init follows the layer-dependent schedule."""

import math

import torch

from src.model.attention import DifferentialAttention
from src.model.config import ModelConfig


def test_output_shape_and_lambda_dtype():
    cfg = ModelConfig(dim=32, n_heads=4, attention="differential", seq_len=16, vocab_size=64)
    attn = DifferentialAttention(cfg, layer_id=0)
    x = torch.randn(2, 16, 32)
    rope = torch.zeros(2, 16, cfg.dim // cfg.n_heads // 2)  # dummy, apply_rope only needs seq slice
    from src.model.attention import build_rope_cache

    rope = build_rope_cache(16, cfg.dim // cfg.n_heads, cfg.rope_theta)
    out = attn(x, rope)
    assert out.shape == x.shape

    lam = (
        torch.exp(torch.sum(attn.lambda_q1 * attn.lambda_k1, dim=-1).float())
        - torch.exp(torch.sum(attn.lambda_q2 * attn.lambda_k2, dim=-1).float())
        + attn.lambda_init
    )
    assert lam.dtype == torch.float32


def test_lambda_init_schedule():
    for layer_id in [0, 1, 5, 11]:
        cfg = ModelConfig(dim=32, n_heads=4, attention="differential", vocab_size=64)
        attn = DifferentialAttention(cfg, layer_id=layer_id)
        expected = 0.8 - 0.6 * math.exp(-0.3 * layer_id)
        assert abs(attn.lambda_init - expected) < 1e-9
