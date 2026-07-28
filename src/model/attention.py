"""Standard and Differential attention with identical parameter cost (4 * dim^2).

Differential attention follows "Differential Transformer" (Ye et al., 2024):
half the head count, each head twice as wide, attn = softmax(Q1K1) - lambda * softmax(Q2K2).
Implemented as two scaled_dot_product_attention calls sharing V, so the memory-efficient
kernel is used on GPU.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def build_rope_cache(seq_len: int, head_dim: int, theta: float) -> torch.Tensor:
    """Returns (seq_len, head_dim // 2) complex-free cos/sin cache stacked as (2, S, D/2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.stack([freqs.cos(), freqs.sin()])  # (2, S, D/2)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """x: (B, S, H, D). rope: (2, S_max, D/2). Rotate-half convention, fp32 math."""
    seq_len = x.size(1)
    cos = rope[0, :seq_len].unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
    sin = rope[1, :seq_len].unsqueeze(0).unsqueeze(2)
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).type_as(x) * self.weight


class StandardAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim)
        q = apply_rope(q, rope).transpose(1, 2)
        k = apply_rope(k, rope).transpose(1, 2)
        v = v.transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, S, D)
        return self.wo(o)


class DifferentialAttention(nn.Module):
    """Param-parity variant: n_heads // 2 heads, each with two (Q, K) pairs of width
    head_dim and a V of width 2 * head_dim. Total projection cost 4 * dim^2, same as
    StandardAttention with n_heads heads."""

    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.n_heads = cfg.n_heads // 2
        self.head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

        # lambda reparameterization (paper eq. 2); layer_id is 0-indexed, paper is 1-indexed
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_id)
        self.lambda_q1 = nn.Parameter(torch.randn(self.n_heads, self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.n_heads, self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.n_heads, self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.n_heads, self.head_dim) * 0.1)

        # headwise sub-layer norm, weight shared across heads (as in official impl)
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)

    def current_lambda(self) -> torch.Tensor:
        """Per-head lambda, in fp32. Pure function of current parameters -- no
        forward pass needed, so this is cheap enough to log every training step."""
        return (
            torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
            - torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
            + self.lambda_init
        )

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, hd = self.n_heads, self.head_dim

        q = self.wq(x).view(B, S, H, 2 * hd)
        k = self.wk(x).view(B, S, H, 2 * hd)
        v = self.wv(x).view(B, S, H, 2 * hd)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        q1 = apply_rope(q1, rope).transpose(1, 2)
        q2 = apply_rope(q2, rope).transpose(1, 2)
        k1 = apply_rope(k1, rope).transpose(1, 2)
        k2 = apply_rope(k2, rope).transpose(1, 2)
        v = v.transpose(1, 2)  # (B, H, S, 2*hd)

        a1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True)
        a2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True)

        lam = self.current_lambda().view(1, H, 1, 1).type_as(a1)

        o = a1 - lam * a2
        o = self.subln(o) * (1.0 - self.lambda_init)
        o = o.transpose(1, 2).reshape(B, S, D)
        return self.wo(o)


def make_attention(cfg: ModelConfig, layer_id: int) -> nn.Module:
    if cfg.attention == "differential":
        return DifferentialAttention(cfg, layer_id)
    return StandardAttention(cfg, layer_id)
