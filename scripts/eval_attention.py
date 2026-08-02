"""Does differential attention actually produce sharper attention?

Everything else in this project measures the *effect* (loss goes down, and it goes
down more at later positions). This measures the *cause* directly: it recomputes
the attention matrices themselves and asks whether the differential variant really
concentrates attention the way the mechanism claims.

The model uses F.scaled_dot_product_attention, which never materialises the weights,
so this script monkeypatches both attention classes with equivalent forwards that
also stash the matrix. The model code is untouched; the arithmetic is identical.

Metrics, all computed on the *effective* attention a token actually receives:

  entropy / effective support
      Differential attention's rows are signed and sum to (1 - lambda), so Shannon
      entropy is undefined on them. We normalise |A| per row instead, which is
      well defined for both variants and still answers "how spread out is it?".
      effective_support = exp(entropy): roughly how many positions are really used.

  top-k mass
      Fraction of total |attention| captured by the k strongest positions. A sharper
      mechanism should need fewer positions to account for its mass.

  attention distance
      Mean |i - j| weighted by |A|. Says whether a variant looks further back --
      the natural companion to the position-resolved NLL result.

  negative mass (differential only)
      Fraction of attention mass that ends up NEGATIVE. This is the capability a
      single softmax mathematically cannot have, so it is the cleanest evidence
      that the subtraction is doing something rather than decorating.

    python scripts/eval_attention.py
"""

import gc
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import Transformer                                  # noqa: E402
from src.model.attention import (                                  # noqa: E402
    StandardAttention, DifferentialAttention, apply_rope,
)
from src.data import load_tokens                                   # noqa: E402
from scripts.eval_all import (                                     # noqa: E402
    infer_config, best_checkpoint, window_offsets,
    DATA_DIR, CKPT_ROOT, SEQ_LEN, DEVICE, RUNS,
)

OUT_JSON = os.path.join(ROOT, "docs/blog/runs_export/eval_attention.json")
N_WINDOWS = 128        # attention matrices are B*H*S*S -- keep this modest
BATCH = 4
MIN_POS = 64           # ignore early rows: with <64 tokens of context "spread out"
#                        is not meaningful and the entropy ceiling is tiny
TOPK = (1, 8, 32)
CAPTURE = {}           # module id -> effective attention (B, H, S, S), fp32 on GPU


def _causal_mask(S, device):
    return torch.ones(S, S, dtype=torch.bool, device=device).tril()


def std_forward(self, x, rope):
    B, S, D = x.shape
    q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
    k = self.wk(x).view(B, S, self.n_heads, self.head_dim)
    v = self.wv(x).view(B, S, self.n_heads, self.head_dim)
    q = apply_rope(q, rope).transpose(1, 2)
    k = apply_rope(k, rope).transpose(1, 2)
    v = v.transpose(1, 2)

    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
    scores = scores.masked_fill(~_causal_mask(S, x.device), float("-inf"))
    attn = scores.softmax(-1)
    CAPTURE[id(self)] = attn.detach()

    o = (attn.type_as(v) @ v).transpose(1, 2).reshape(B, S, D)
    return self.wo(o)


def diff_forward(self, x, rope):
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
    v = v.transpose(1, 2)

    mask = ~_causal_mask(S, x.device)
    s1 = (q1.float() @ k1.float().transpose(-1, -2)) / math.sqrt(hd)
    s2 = (q2.float() @ k2.float().transpose(-1, -2)) / math.sqrt(hd)
    a1 = s1.masked_fill(mask, float("-inf")).softmax(-1)
    a2 = s2.masked_fill(mask, float("-inf")).softmax(-1)
    lam = self.current_lambda().view(1, H, 1, 1)
    attn = a1 - lam * a2                       # the EFFECTIVE attention; may be < 0
    CAPTURE[id(self)] = attn.detach()

    o = attn.type_as(v) @ v
    o = self.subln(o) * (1.0 - self.lambda_init)
    o = o.transpose(1, 2).reshape(B, S, D)
    return self.wo(o)


def attention_stats(attn, differential):
    """attn: (B, H, S, S) fp32, rows causal. Returns per-row aggregates over
    positions >= MIN_POS, averaged over batch/heads/positions."""
    B, H, S, _ = attn.shape
    a = attn[:, :, MIN_POS:, :]                        # (B, H, P, S)
    absa = a.abs()
    denom = absa.sum(-1, keepdim=True).clamp_min(1e-9)
    p = absa / denom                                   # normalised magnitude profile

    ent = -(p * (p + 1e-12).log()).sum(-1)             # (B, H, P)
    out = {"entropy": ent.mean().item(),
           "effective_support": ent.mean().exp().item()}

    for k in TOPK:
        topk = p.topk(min(k, p.shape[-1]), dim=-1).values.sum(-1)
        out[f"top{k}_mass"] = topk.mean().item()

    idx = torch.arange(S, device=attn.device).view(1, 1, 1, S)
    row = torch.arange(MIN_POS, S, device=attn.device).view(1, 1, -1, 1)
    dist = (row - idx).clamp_min(0).float()
    out["attn_distance"] = (p * dist).sum(-1).mean().item()

    if differential:
        neg = a.clamp_max(0).abs().sum(-1)
        tot = absa.sum(-1).clamp_min(1e-9)
        out["negative_mass_frac"] = (neg / tot).mean().item()
        out["row_sum"] = a.sum(-1).mean().item()
    else:
        out["negative_mass_frac"] = 0.0
        out["row_sum"] = a.sum(-1).mean().item()
    return out


def main():
    StandardAttention.forward = std_forward
    DifferentialAttention.forward = diff_forward

    test = load_tokens(os.path.join(DATA_DIR, "test.bin"))
    offsets = window_offsets(0, len(test), N_WINDOWS)
    print(f"{len(offsets)} windows, batch {BATCH}, rows from position {MIN_POS} on\n")

    results = {}
    for run in RUNS:
        ckpt_path = best_checkpoint(os.path.join(CKPT_ROOT, run))
        if ckpt_path is None:
            print(f"skip {run}: no checkpoint")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
        sd = ckpt["model"]
        cfg = infer_config(sd)
        model = Transformer(cfg).to(DEVICE)
        model.load_state_dict(sd, strict=False)
        del ckpt, sd
        gc.collect()
        model.eval()
        differential = cfg.attention == "differential"

        attn_mods = [b.attn for b in model.blocks]
        per_layer = [[] for _ in attn_mods]

        with torch.no_grad():
            for b in range(0, len(offsets), BATCH):
                chunk = offsets[b:b + BATCH]
                x = torch.stack([torch.from_numpy(test[i:i + SEQ_LEN].astype("int64"))
                                 for i in chunk]).to(DEVICE)
                CAPTURE.clear()
                model(x)
                for li, m in enumerate(attn_mods):
                    per_layer[li].append(attention_stats(CAPTURE[id(m)], differential))
                CAPTURE.clear()

        keys = per_layer[0][0].keys()
        layers = [{k: float(np.mean([d[k] for d in L])) for k in keys} for L in per_layer]
        overall = {k: float(np.mean([l[k] for l in layers])) for k in keys}
        results[run] = {"attention": cfg.attention, "n_heads_effective": attn_mods[0].n_heads,
                        "per_layer": layers, "overall": overall}
        print(f"{run:10s} ({cfg.attention:12s}) entropy {overall['entropy']:.3f} "
              f"| eff.support {overall['effective_support']:6.1f} "
              f"| top8 {overall['top8_mass']:.3f} | dist {overall['attn_distance']:6.1f} "
              f"| neg mass {overall['negative_mass_frac']:.4f}")

        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", OUT_JSON)


if __name__ == "__main__":
    main()
