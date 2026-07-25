"""Put an error bar on the head-to-head test-set differences.

Every model is scored on byte-identical test windows, so the difference between
two models can be measured *per window* and bootstrapped as a paired statistic --
far tighter than comparing two independent means, and the correct test for "is
this gap bigger than the test sample's own noise?"

What this measures: sampling error from a finite test set. What it does NOT
measure: seed-to-seed variance from retraining, which is a separate and here
larger source of doubt. Reporting both matters -- a tight CI on a single-seed
comparison is precision, not reliability.

    python scripts/eval_uncertainty.py
"""

import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import Transformer                       # noqa: E402
from src.data import load_tokens                        # noqa: E402
from scripts.eval_all import (                          # noqa: E402
    infer_config, best_checkpoint, window_offsets,
    DATA_DIR, CKPT_ROOT, SEQ_LEN, DEVICE, BATCH, N_GLOBAL_WINDOWS, RUNS,
)

N_BOOT = 20000
RNG = np.random.default_rng(0)


@torch.no_grad()
def per_window_nll(model, data, offsets):
    """Mean NLL for each individual window -- the paired unit of analysis."""
    out = np.empty(len(offsets), dtype=np.float64)
    for b in range(0, len(offsets), BATCH):
        chunk = offsets[b:b + BATCH]
        x = torch.stack([torch.from_numpy(data[i:i + SEQ_LEN].astype("int64")) for i in chunk]).to(DEVICE)
        y = torch.stack([torch.from_numpy(data[i + 1:i + SEQ_LEN + 1].astype("int64")) for i in chunk]).to(DEVICE)
        logits, _, _ = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1),
                               reduction="none").view(y.shape)
        out[b:b + len(chunk)] = loss.mean(dim=1).double().cpu().numpy()
    return out


def paired_ci(a, b, n_boot=N_BOOT):
    """Bootstrap the mean paired difference (b - a) over windows."""
    d = b - a
    n = len(d)
    idx = RNG.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi, d.std(ddof=1) / np.sqrt(n)


def main():
    test = load_tokens(os.path.join(DATA_DIR, "test.bin"))
    offsets = window_offsets(0, len(test), N_GLOBAL_WINDOWS)
    print(f"{len(offsets)} paired windows ({len(offsets)*SEQ_LEN/1e6:.2f}M tokens) on {DEVICE}\n")

    nlls = {}
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
        nlls[run] = per_window_nll(model, test, offsets)
        print(f"{run:10s} step {ckpt_path.split('step')[-1][:4]}  "
              f"mean NLL {nlls[run].mean():.4f}  per-window sd {nlls[run].std(ddof=1):.4f}")
        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 74)
    print("PAIRED DIFFERENCES  (negative = second model better), 95% bootstrap CI")
    print("=" * 74)
    pairs = [("s_dense", "s_moe"), ("s_dense", "s_diffmoe"), ("s_moe", "s_diffmoe"),
             ("s_dense", "s_diff"), ("s_diff", "s_diffmoe")]
    results = {}
    for a, b in pairs:
        if a not in nlls or b not in nlls:
            continue
        mean, lo, hi, se = paired_ci(nlls[a], nlls[b])
        wins = (nlls[b] < nlls[a]).mean()
        sig = "yes" if (lo < 0) == (hi < 0) else "NO (CI spans 0)"
        print(f"  {b} - {a}")
        print(f"      delta = {mean:+.4f} nats   95% CI [{lo:+.4f}, {hi:+.4f}]   SE {se:.4f}")
        print(f"      wins on {wins*100:.1f}% of the {len(nlls[a])} windows   "
              f"excludes zero: {sig}")
        results[f"{b}_minus_{a}"] = {"delta": mean, "ci_low": lo, "ci_high": hi,
                                     "se": se, "window_win_rate": wins}

    npz = os.path.join(ROOT, "docs/blog/runs_export/per_window_nll.npz")
    np.savez_compressed(npz, **nlls)
    print("\nwrote", npz, "(per-window NLLs; re-analysis needs no GPU)")

    out = os.path.join(ROOT, "docs/blog/runs_export/eval_uncertainty.json")
    with open(out, "w") as f:
        json.dump({"n_windows": len(offsets), "n_boot": N_BOOT,
                   "mean_nll": {k: float(v.mean()) for k, v in nlls.items()},
                   "paired": results}, f, indent=2)
    print("\nwrote", out)
    print("\nNOTE: these intervals cover test-set sampling error only. Seed-to-seed\n"
          "variance from retraining is not measured here and is the larger unknown.")


if __name__ == "__main__":
    main()
