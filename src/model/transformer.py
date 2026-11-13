"""Decoder-only LM: RoPE, pre-norm blocks, optional MoE FFNs, optional
AttnRes / mHC residual streams, optional document masking.

forward() returns (logits, aux_loss, z_loss); the training loop owns the CE loss.
With cfg.doc_mask the model derives document boundaries from cfg.eos_id itself,
so every caller (trainer, eval scripts, generate) masks identically.
No KV cache anywhere in the training path — generation recomputes the full prefix,
which is fine at this scale.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import build_rope_cache, document_mask, make_norm
from .block import Block
from .config import ModelConfig
from .residual import AttnRes


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = make_norm(cfg, cfg.dim)
        self.final_res = AttnRes(cfg.dim, cfg.norm_eps) if cfg.residual == "attnres" else None
        self.gradient_checkpointing = False  # set by the trainer
        self.ckpt_skip_last = 0  # with checkpointing: the last N blocks keep activations
        # None: FlexAttention on CUDA when Triton exists, else dense SDPA mask
        self.use_flex = None if cfg.flex_attention else False
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

        head_dim = cfg.dim // cfg.n_heads
        self.register_buffer(
            "rope", build_rope_cache(cfg.seq_len, head_dim, cfg.rope_theta), persistent=False
        )
        self.apply(self._init_weights)
        # residual-output projections get the GPT-2 style 1/sqrt(2L) shrink
        for name, p in self.named_parameters():
            if (
                name.endswith("wo.weight")
                or name.endswith("w2.weight")
                or name.endswith("routed_expert_up_proj.weight")
            ):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert tokens.size(1) <= self.cfg.seq_len, "sequence longer than rope cache"
        cfg = self.cfg
        x = self.embed(tokens)
        mask = document_mask(tokens, cfg.eos_id, self.use_flex) if cfg.doc_mask else None
        if cfg.residual == "attnres":
            state = ([x], None)
        elif cfg.residual == "mhc":
            state = x.unsqueeze(2).expand(-1, -1, cfg.mhc_streams, -1).contiguous()
        else:
            state = x
        aux_total = torch.zeros((), device=tokens.device, dtype=torch.float32)
        z_total = torch.zeros((), device=tokens.device, dtype=torch.float32)
        recompute = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        n_recompute = len(self.blocks) - self.ckpt_skip_last
        for i, block in enumerate(self.blocks):
            if recompute and i < n_recompute:
                out, aux, z = checkpoint(block, state, self.rope, mask, use_reentrant=False)
            else:
                out, aux, z = block(state, self.rope, mask)
            if cfg.residual == "attnres":  # out is the block's new partial sum
                sources = state[0]
                state = (sources + [out], None) if block.ends_block else (sources, out)
            else:
                state = out
            aux_total = aux_total + aux
            z_total = z_total + z
        if cfg.residual == "attnres":
            sources, partial = state
            x = self.final_res(sources if partial is None else sources + [partial])
        elif cfg.residual == "mhc":
            x = state.mean(2)
        else:
            x = state
        x = self.norm(x)
        logits = self.head(x)
        return logits, aux_total, z_total

    def compile_blocks(self):
        """torch.compile the per-layer compute; the MoE dispatch loop stays eager."""
        import torch._dynamo
        # All compiled modules share nn.Module._call_impl's cache. With the LRU order, the
        # checkpoint recompute of a dense FFN can hit the experts' newer dynamic graph instead
        # of the static one its forward used -> CheckpointError (pytorch#166926, torchtitan#1991).
        torch._C._dynamo.eval_frame._set_lru_cache(False)
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        for block in self.blocks:
            block.attn.compile()
            block.attn_norm.compile()
            block.ffn_norm.compile()
            if block.is_moe:
                for expert in block.ffn.experts:
                    expert.compile(dynamic=True)
            else:
                block.ffn.compile()

    @torch.no_grad()
    def update_quantile_balancing(self) -> dict:
        """Commit delayed router-bias updates after one accumulated train step."""
        updated = {}
        for i, block in enumerate(self.blocks):
            if block.is_moe:
                bias = block.ffn.update_quantile_bias()
                if bias is not None:
                    updated[i] = bias
        return updated

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 0.9,
        eos_id: int = -1,
    ) -> torch.Tensor:
        """Full-prefix recompute per step (no cache). tokens: (B, S)."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = tokens[:, -self.cfg.seq_len :]
            logits, _, _ = self(ctx)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cum = sorted_probs.cumsum(dim=-1)
                sorted_probs[cum - sorted_probs > top_p] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_tok = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            tokens = torch.cat([tokens, next_tok], dim=1)
            if eos_id >= 0 and (next_tok == eos_id).all():
                break
        return tokens


def count_params(model: Transformer) -> dict:
    """Total / active / non-embedding parameter counts. Active = params used per token
    (routed experts beyond top_k excluded)."""
    cfg = model.cfg
    total = sum(p.numel() for p in model.parameters())
    embed = model.embed.weight.numel()
    if not cfg.tie_embeddings:
        embed += model.head.weight.numel()

    inactive = 0
    for block in model.blocks:
        if block.is_moe:
            per_expert = sum(p.numel() for p in block.ffn.experts[0].parameters())
            inactive += per_expert * (cfg.n_experts - cfg.top_k)
    return {
        "total": total,
        "active": total - inactive,
        "non_embed_total": total - embed,
        "non_embed_active": total - inactive - embed,
    }
