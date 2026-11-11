"""Residual-stream alternatives to `x = x + f(x)`.

AttnRes (Kimi Team, "Attention Residuals", 2026), Block variant: before every
sublayer, mix the completed block representations (token embedding first) and
the current block's partial sum with a softmax over depth. One learned
pseudo-query per sublayer; keys are RMS-normalised sources. The query starts
at zero, so training begins from a uniform average over sources.

mHC (DeepSeek-AI, "Manifold-Constrained Hyper-Connections", 2025): n parallel
residual streams. Per sublayer, a token-dependent read vector H_pre (sigmoid),
write vector H_post (2 * sigmoid) and stream-mixing matrix H_res projected
onto the doubly stochastic matrices by Sinkhorn-Knopp:
    X <- H_res X + H_post^T f(H_pre X)
Initialised so all streams stay equal and the update reduces to x + f(x).
"""

import functools
import math
from typing import List, Tuple

import torch
import torch.nn as nn

from .attention import RMSNorm


class AttnRes(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.query = nn.Parameter(torch.zeros(dim))
        self.last_weights = None  # (n_sources,) mean depth weights, for logging

    def forward(self, sources: List[torch.Tensor]) -> torch.Tensor:
        # Parameter-free key norm: a learned key scale would be redundant with the query.
        # q . (s / rms(s)) == (q . s) / rms(s): one gemv and one fp32-accumulated norm per
        # source, so no stacked or fp32 copy of the sources is made or saved for backward.
        dtype = functools.reduce(torch.promote_types, (s.dtype for s in sources))
        logits = torch.stack([
            (s @ self.query.to(s.dtype)).float() * torch.rsqrt(
                torch.linalg.vector_norm(s, dim=-1, dtype=torch.float32).square() / s.size(-1) + self.eps)
            for s in sources])
        weights = logits.softmax(0)  # (N, B, S)
        self.last_weights = weights.detach().mean(dim=(1, 2))
        w = weights.to(dtype).unsqueeze(-1)
        out = sources[0].to(dtype) * w[0]
        for s, wi in zip(sources[1:], w[1:]):
            out = torch.addcmul(out, s, wi)
        return out


def sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Project exp(logits) (..., n, n) onto doubly stochastic matrices."""
    m = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iters):
        m = m / m.sum(-1, keepdim=True)
        m = m / m.sum(-2, keepdim=True)
    return m


class MHC(nn.Module):
    def __init__(self, dim: int, n_streams: int, iters: int, eps: float = 1e-6):
        super().__init__()
        n = n_streams
        self.n, self.iters = n, iters
        self.norm = RMSNorm(n * dim, eps)
        self.proj = nn.Linear(n * dim, 2 * n + n * n, bias=False)  # [pre | post | res] dynamic parts
        self.alpha = nn.Parameter(torch.full((3,), 0.01))  # small gates on the dynamic parts
        self.bias_pre = nn.Parameter(torch.full((n,), -math.log(n - 1)))  # sigmoid -> 1/n: read the mean
        self.bias_post = nn.Parameter(torch.zeros(n))  # 2 * sigmoid(0) = 1: write to every stream
        # Flattened so weight decay (dim >= 2 only) leaves it alone; diag 4 -> ~0.95 identity.
        self.bias_res = nn.Parameter((4.0 * torch.eye(n)).flatten())
        self.last_res_diag = None

    def forward(self, streams: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """streams: (B, S, n, D). Returns the sublayer input and the write coefficients."""
        B, S, n, D = streams.shape
        h = self.proj(self.norm(streams.reshape(B, S, n * D))).float()
        a = self.alpha.float()
        pre = torch.sigmoid(a[0] * h[..., :n] + self.bias_pre.float())
        post = 2.0 * torch.sigmoid(a[1] * h[..., n:2 * n] + self.bias_post.float())
        res_logits = a[2] * h[..., 2 * n:] + self.bias_res.float()
        res = sinkhorn(res_logits.view(B, S, n, n), self.iters)
        self.last_res_diag = res.detach().diagonal(dim1=-2, dim2=-1).mean()
        x_in = torch.einsum("bsn,bsnd->bsd", pre.type_as(streams), streams)
        return x_in, (post, res)

    @staticmethod
    def write(streams: torch.Tensor, out: torch.Tensor, post: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        mixed = torch.einsum("bsij,bsjd->bsid", res.type_as(streams), streams)
        return mixed + post.type_as(streams)[..., None] * out[:, :, None, :]
