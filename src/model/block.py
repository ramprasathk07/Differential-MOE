"""Pre-norm transformer block. Plain residuals, no cache, no fused tricks."""

from typing import Tuple

import torch
import torch.nn as nn

from .attention import RMSNorm, make_attention
from .config import ModelConfig
from .moe import MoE, StableLatentMoE, SwiGLU


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.attn = make_attention(cfg, layer_id)
        self.is_moe = cfg.ffn in ("moe", "stable_latent_moe") and layer_id >= cfg.n_dense_layers
        if self.is_moe:
            self.ffn = StableLatentMoE(cfg) if cfg.ffn == "stable_latent_moe" else MoE(cfg)
        else:
            self.ffn = SwiGLU(cfg.dim, cfg.inter_dim)
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(
        self, x: torch.Tensor, rope: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x), rope)
        if self.is_moe:
            y, aux, z = self.ffn(self.ffn_norm(x))
        else:
            y = self.ffn(self.ffn_norm(x))
            aux = z = torch.zeros((), device=x.device, dtype=torch.float32)
        x = x + y
        return x, aux, z
