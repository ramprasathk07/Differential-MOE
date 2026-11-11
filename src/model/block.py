"""Pre-norm transformer block with a selectable residual stream.

The residual state passed into a block depends on cfg.residual:
  standard  x                    (B, S, D)
  attnres   (sources, partial)   Block AttnRes: completed block reps + running block sum
  mhc       streams              (B, S, n, D)
Everything goes through Block.forward so FSDP's per-block hooks always fire. A block
returns only tensors it computed: under AttnRes it returns the new partial sum and the
Transformer owns the sources list. Re-returning earlier blocks' tensors would hang every
block's FSDP backward hooks on them and free gathered weights while still in use.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .attention import make_attention, make_norm
from .config import ModelConfig
from .moe import MoE, StableLatentMoE, SwiGLU
from .residual import MHC, AttnRes


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.attn = make_attention(cfg, layer_id)
        self.is_moe = cfg.ffn in ("moe", "stable_latent_moe") and layer_id >= cfg.n_dense_layers
        if self.is_moe:
            self.ffn = StableLatentMoE(cfg) if cfg.ffn == "stable_latent_moe" else MoE(cfg)
        else:
            self.ffn = SwiGLU(cfg.dim, cfg.inter_dim)
        self.attn_norm = make_norm(cfg, cfg.dim)
        self.ffn_norm = make_norm(cfg, cfg.dim)

        self.residual = cfg.residual
        if self.residual == "attnres":
            # Layer 0's attention only ever sees the embedding: a softmax over one source.
            self.attn_res = AttnRes(cfg.dim, cfg.norm_eps) if layer_id > 0 else None
            self.ffn_res = AttnRes(cfg.dim, cfg.norm_eps)
            self.ends_block = (layer_id + 1) % cfg.attnres_block_layers == 0
        elif self.residual == "mhc":
            self.attn_mhc = MHC(cfg.dim, cfg.mhc_streams, cfg.mhc_sinkhorn_iters, cfg.norm_eps)
            self.ffn_mhc = MHC(cfg.dim, cfg.mhc_streams, cfg.mhc_sinkhorn_iters, cfg.norm_eps)

    def _attn(self, h, rope, mask):
        return self.attn(self.attn_norm(h), rope, mask)

    def _ffn(self, h):
        if self.is_moe:
            return self.ffn(self.ffn_norm(h))
        zero = torch.zeros((), device=h.device, dtype=torch.float32)
        return self.ffn(self.ffn_norm(h)), zero, zero

    def forward(self, state, rope: torch.Tensor, mask=None) -> Tuple[object, torch.Tensor, torch.Tensor]:
        if self.residual == "attnres":
            sources, partial = state
            h = sources[0] if self.attn_res is None else self.attn_res(
                sources if partial is None else sources + [partial])
            out = self._attn(h, rope, mask)
            partial = out if partial is None else partial + out
            y, aux, z = self._ffn(self.ffn_res(sources + [partial]))
            return partial + y, aux, z
        if self.residual == "mhc":
            h, (post, res) = self.attn_mhc(state)
            state = MHC.write(state, self._attn(h, rope, mask), post, res)
            h, (post, res) = self.ffn_mhc(state)
            y, aux, z = self._ffn(h)
            return MHC.write(state, y, post, res), aux, z
        x = state + self._attn(state, rope, mask)
        y, aux, z = self._ffn(x)
        return x + y, aux, z
