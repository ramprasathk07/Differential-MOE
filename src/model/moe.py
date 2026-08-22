"""Routed feed-forwards used by the project.

``MoE`` is the original softmax/top-k baseline. ``StableLatentMoE`` implements
the Kimi K3 design at training time: full-width shared experts, compact routed
experts, RMSNorm before the routed up-projection, bounded SiTU-GLU activations,
sigmoid routing, and next-step Quantile Balancing without an auxiliary loss.
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .attention import RMSNorm
from .config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def situ_glu_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float,
) -> torch.Tensor:
    """Bounded SiTU-GLU product from Kimi K3.

    Each coordinate is bounded in magnitude by ``beta * linear_beta`` while
    matching SwiGLU to first order near the origin.
    """

    bounded_gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    bounded_up = linear_beta * torch.tanh(up / linear_beta)
    return bounded_gate * bounded_up


class SiTUGLU(nn.Module):
    def __init__(self, dim: int, inter_dim: int, beta: float, linear_beta: float):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = situ_glu_activation(
            self.w1(x), self.w3(x), self.beta, self.linear_beta
        )
        return self.w2(hidden)


class RoutedMoE(nn.Module):
    """Common interface used by routing diagnostics and parameter accounting."""

    last_counts: Optional[torch.Tensor]

    def update_quantile_bias(self) -> Optional[torch.Tensor]:
        return None


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


class MoE(RoutedMoE):
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


class QuantileBalancedSigmoidGate(nn.Module):
    """Sigmoid top-k router with exact, delayed Quantile Balancing.

    During optimizer step ``t`` each micro-batch is routed with the currently
    frozen expert bias. We retain detached score margins from every accumulated
    micro-batch. ``update_quantile_bias`` pools those margins across DDP ranks
    and replaces the bias once, after the optimizer update, so it only affects
    step ``t + 1``. This is the causal update in Kimi K3 Eq. 14.

    The paper uses histograms for million-token global batches. This project has
    a much smaller batch, so an exact kth-value is both affordable and avoids a
    histogram range/bin hyperparameter.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.top_k = cfg.top_k
        self.n_experts = cfg.n_experts
        self.quantile_balance = cfg.quantile_balance
        self.weight = nn.Parameter(torch.randn(cfg.n_experts, cfg.dim) * 0.02)
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts, dtype=torch.float32))
        self._pending_margins: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(x.float(), self.weight)
        scores = logits.sigmoid()
        biased_scores = scores + self.expert_bias

        if self.training and self.quantile_balance:
            candidates, candidate_idx = biased_scores.topk(
                self.top_k + 1, dim=-1, sorted=True
            )
            top_idx = candidate_idx[:, : self.top_k]
            cutoff = candidates[:, self.top_k]
            # Raw score minus the biased token cutoff, exactly as Eq. 14.
            self._pending_margins.append((scores - cutoff[:, None]).detach())
        else:
            top_idx = biased_scores.topk(self.top_k, dim=-1, sorted=False).indices

        top_scores = scores.gather(1, top_idx)
        weights = top_scores / top_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        return weights, top_idx

    @staticmethod
    def _gather_rows(local: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return local

        world_size = dist.get_world_size()
        local_size = torch.tensor([local.size(0)], device=local.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        row_counts = [int(size.item()) for size in sizes]
        max_rows = max(row_counts)

        if local.size(0) < max_rows:
            padding = local.new_zeros(max_rows - local.size(0), local.size(1))
            local = torch.cat((local, padding), dim=0)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        return torch.cat([rows[:count] for rows, count in zip(gathered, row_counts)], dim=0)

    @torch.no_grad()
    def update_quantile_bias(self) -> Optional[torch.Tensor]:
        if not self.quantile_balance or not self._pending_margins:
            return None

        local = torch.cat(self._pending_margins, dim=0).float()
        self._pending_margins.clear()
        margins = self._gather_rows(local)
        n_tokens = margins.size(0)
        if n_tokens == 0:
            return None

        # q target routes/expert -> the (q+1)-th largest margin is the cutoff.
        target_load = round(n_tokens * self.top_k / self.n_experts)
        target_load = min(max(target_load, 1), n_tokens - 1)
        kth_smallest = n_tokens - target_load
        threshold = margins.kthvalue(kth_smallest, dim=0).values
        next_bias = -threshold
        next_bias -= next_bias.mean()  # a common offset cannot change top-k selection
        self.expert_bias.copy_(next_bias)
        return self.expert_bias.detach().clone()


class StableLatentMoE(RoutedMoE):
    """Kimi-style Stable LatentMoE feed-forward.

    The router observes the full-width token. Routed computation happens after
    one shared down-projection; selected outputs are mixed in latent space,
    normalized, and projected back. Shared experts always process the original
    full-width token and are added to the routed branch.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.latent_dim is not None  # guaranteed by ModelConfig validation
        self.dim = cfg.dim
        self.latent_dim = cfg.latent_dim
        self.n_experts = cfg.n_experts
        self.gate = QuantileBalancedSigmoidGate(cfg)
        self.experts = nn.ModuleList(
            [
                SiTUGLU(
                    cfg.latent_dim,
                    cfg.expert_inter_dim,
                    cfg.situ_beta,
                    cfg.situ_linear_beta,
                )
                for _ in range(cfg.n_experts)
            ]
        )
        # A widened SiTU-GLU is algebraically identical to summing independent
        # shared experts and matches the official Kimi implementation.
        self.shared_experts = SiTUGLU(
            cfg.dim,
            cfg.n_shared_experts * cfg.shared_inter_dim,
            cfg.situ_beta,
            cfg.situ_linear_beta,
        )
        self.routed_expert_down_proj = nn.Linear(cfg.dim, cfg.latent_dim, bias=False)
        self.routed_expert_norm = RMSNorm(cfg.latent_dim, cfg.norm_eps)
        self.routed_expert_up_proj = nn.Linear(cfg.latent_dim, cfg.dim, bias=False)
        self.last_counts: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        full = x.reshape(-1, D)
        weights, idx = self.gate(full)
        latent = self.routed_expert_down_proj(full)
        weights = weights.type_as(latent)

        routed = torch.zeros_like(latent)
        for e in range(self.n_experts):
            tok, slot = torch.where(idx == e)
            if tok.numel() == 0:
                continue
            routed[tok] += self.experts[e](latent[tok]) * weights[tok, slot, None]

        routed = self.routed_expert_up_proj(self.routed_expert_norm(routed))
        out = self.shared_experts(full) + routed
        self.last_counts = torch.bincount(idx.flatten(), minlength=self.n_experts).detach()
        zero = torch.zeros((), device=x.device, dtype=torch.float32)
        return out.view(B, S, D), zero, zero

    def update_quantile_bias(self) -> Optional[torch.Tensor]:
        return self.gate.update_quantile_bias()
