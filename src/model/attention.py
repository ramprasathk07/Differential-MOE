"""Standard and Differential attention with identical parameter cost (4 * dim^2).

Differential attention follows "Differential Transformer" (Ye et al., 2024):
half the head count, each head twice as wide, attn = softmax(Q1K1) - lambda * softmax(Q2K2).

Optional, off by default:
  * Exclusive Self Attention (XSA, Zhai 2026): remove each head's output component
    along the token's own value vector before the output projection.
  * Document masking: `mask` is None (plain causal), a dense bool tensor (SDPA,
    works everywhere incl. Windows without Triton), or a FlexAttention BlockMask
    (CUDA + Triton; skips fully masked tiles).
"""

import importlib.util
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

HAS_TRITON = importlib.util.find_spec("triton") is not None
_flex_compiled = None


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


class PointwiseNorm(nn.Module):
    """Normalization-free replacement for a norm layer.

    DyT  (Zhu et al. 2025): weight * tanh(alpha * x) + bias
    Derf (Chen et al. 2025): weight * erf(alpha * x + shift) + bias
    alpha and shift are learnable scalars, weight/bias per channel.
    """

    def __init__(self, dim: int, kind: str, alpha_init: float):
        super().__init__()
        self.kind = kind
        # shape (1,) not (): FSDP2 shards along dim 0
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.shift = nn.Parameter(torch.zeros(1)) if kind == "derf" else None
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.tanh(self.alpha * x) if self.kind == "dyt" else torch.erf(self.alpha * x + self.shift)
        return y * self.weight + self.bias


def make_norm(cfg: ModelConfig, dim: int) -> nn.Module:
    if cfg.norm == "rmsnorm":
        return RMSNorm(dim, cfg.norm_eps)
    return PointwiseNorm(dim, cfg.norm, cfg.norm_alpha_init)


def exclude_self(o: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """XSA: drop the component of each head output along its own value vector."""
    vn = F.normalize(v, dim=-1)
    return o - (o * vn).sum(-1, keepdim=True) * vn


def attend(q, k, v, mask=None) -> torch.Tensor:
    """q, k: (B, H, S, d); v: (B, H, S, dv). Causal, plus document mask if given."""
    if mask is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if isinstance(mask, torch.Tensor):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    from torch.nn.attention.flex_attention import flex_attention

    if torch.compiler.is_compiling():  # already inside a compiled attention module
        return flex_attention(q, k, v, block_mask=mask)
    global _flex_compiled
    if _flex_compiled is None:
        # Uncompiled flex materialises the full score matrix; on CPU (tests) that is fine.
        _flex_compiled = torch.compile(flex_attention, dynamic=False) if q.is_cuda else flex_attention
    return _flex_compiled(q, k, v, block_mask=mask)


class StandardAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.xsa = cfg.xsa
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, rope: torch.Tensor, mask=None) -> torch.Tensor:
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim)
        q = apply_rope(q, rope).transpose(1, 2)
        k = apply_rope(k, rope).transpose(1, 2)
        v = v.transpose(1, 2)
        o = attend(q, k, v, mask)
        if self.xsa:
            o = exclude_self(o, v)
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
        self.xsa = cfg.xsa
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
        q1, k1, q2, k2 = (p.float() for p in (self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2))
        return torch.exp(torch.sum(q1 * k1, dim=-1)) - torch.exp(torch.sum(q2 * k2, dim=-1)) + self.lambda_init

    def forward(self, x: torch.Tensor, rope: torch.Tensor, mask=None) -> torch.Tensor:
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

        a1 = attend(q1, k1, v, mask)
        a2 = attend(q2, k2, v, mask)

        lam = self.current_lambda().view(1, H, 1, 1).type_as(a1)

        o = a1 - lam * a2
        if self.xsa:
            o = exclude_self(o, v)
        o = self.subln(o) * (1.0 - self.lambda_init)
        o = o.transpose(1, 2).reshape(B, S, D)
        return self.wo(o)


def make_attention(cfg: ModelConfig, layer_id: int) -> nn.Module:
    if cfg.attention == "differential":
        return DifferentialAttention(cfg, layer_id)
    return StandardAttention(cfg, layer_id)


def document_mask(tokens: torch.Tensor, eos_id: int, use_flex: Optional[bool] = None):
    """Causal + same-document mask from EOS positions; EOS ends its own document.
    Returns None when the batch has no EOS (plain causal keeps the flash path)."""
    eos = tokens == eos_id
    if not bool(eos.any()):
        return None
    doc = eos.long().cumsum(1) - eos.long()
    B, S = tokens.shape
    if use_flex is None:
        use_flex = tokens.is_cuda and HAS_TRITON
    if use_flex:
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (doc[b, q_idx] == doc[b, kv_idx])

        return create_block_mask(mask_mod, B, None, S, S, device=tokens.device)
    causal = torch.ones(S, S, dtype=torch.bool, device=tokens.device).tril()
    return ((doc[:, :, None] == doc[:, None, :]) & causal)[:, None]
