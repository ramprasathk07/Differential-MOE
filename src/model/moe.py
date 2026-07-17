"""Mixture-of-Experts FFN: softmax top-k gate with Switch-style load-balance loss
and router z-loss, dropless per-expert dispatch."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """Returns per-token expert weights/indices plus aux losses.

    aux (load balance, Switch/GShard): n_experts * sum_e f_e * P_e where
      f_e = fraction of routed (token, slot) assignments to expert e,
      P_e = mean softmax probability of expert e.
    Equals 1.0 under a perfectly uniform router.
    z-loss: mean(logsumexp(logits)^2), keeps logits from drifting.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.top_k = cfg.top_k
        self.n_experts = cfg.n_experts
        self.weight = nn.Parameter(torch.randn(cfg.n_experts, cfg.dim) * 0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # router math in fp32 regardless of autocast
        logits = F.linear(x.float(), self.weight)
        probs = logits.softmax(dim=-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)
        weights = top_probs / top_probs.sum(dim=-1, keepdim=True)

        counts = torch.zeros_like(probs).scatter_add_(
            1, top_idx, torch.ones_like(top_probs)
        )
        f = counts.mean(dim=0) / self.top_k  # sums to 1
        p = probs.mean(dim=0)
        aux = self.n_experts * torch.sum(f * p)
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return weights, top_idx, aux, z


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.dim = cfg.dim
        self.n_experts = cfg.n_experts
        self.gate = Gate(cfg)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.dim, cfg.expert_inter_dim) for _ in range(cfg.n_experts)]
        )
        self.shared = (
            SwiGLU(cfg.dim, cfg.n_shared_experts * cfg.shared_inter_dim)
            if cfg.n_shared_experts > 0
            else None
        )
        # for logging: expert assignment counts of the latest forward
        self.last_counts: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        xf = x.reshape(-1, D)
        weights, idx, aux, z = self.gate(xf)
        weights = weights.type_as(xf)

        out = torch.zeros_like(xf)
        for e in range(self.n_experts):
            tok, slot = torch.where(idx == e)
            if tok.numel() == 0:
                continue
            out[tok] += self.experts[e](xf[tok]) * weights[tok, slot, None]
        if self.shared is not None:
            out = out + self.shared(xf)

        self.last_counts = torch.bincount(idx.flatten(), minlength=self.n_experts).detach()
        return out.view(B, S, D), aux, z
