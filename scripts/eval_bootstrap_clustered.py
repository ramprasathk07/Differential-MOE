"""Re-do the paired confidence intervals without assuming windows are exchangeable.

eval_uncertainty.py bootstraps by resampling the 3,200 test windows independently.
That treats every window as an independent draw, which they are not: windows from
the same BabyLM domain are correlated, and the six domains differ enormously in
difficulty (CHILDES 1.9 nats, Gutenberg 3.8). Ignoring that structure makes the
interval too narrow.

Three estimators, from least to most conservative:

  iid        resample windows independently          (what eval_uncertainty.py does)
  stratified resample within each domain, keeping the domain mix fixed
             -> answers "how much would this move on another sample of THIS corpus?"
  clustered  resample the six DOMAINS with replacement, then windows inside them
             -> answers "how much would this move on a corpus of different domains?"

The clustered interval has only six clusters, so it is deliberately crude and wide.
It is the honest interval to quote if the claim is meant to generalise beyond the
particular six domains BabyLM happens to ship.

Runs on CPU from the cached per-window NLLs -- no GPU, no model loading.

    python scripts/eval_bootstrap_clustered.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.eval_all import window_offsets, DATA_DIR, SEQ_LEN, N_GLOBAL_WINDOWS  # noqa: E402

NPZ = os.path.join(ROOT, "docs/blog/runs_export/per_window_nll.npz")
OUT_JSON = os.path.join(ROOT, "docs/blog/runs_export/eval_bootstrap_clustered.json")
N_BOOT = 20000
RNG = np.random.default_rng(0)
NAMES = {"s_dense": "Dense", "s_diff": "Diff-Dense",
         "s_moe": "MoE", "s_diffmoe": "Diff-MoE"}
PAIRS = [("s_dense", "s_diff"), ("s_dense", "s_moe"), ("s_dense", "s_diffmoe"),
         ("s_moe", "s_diffmoe"), ("s_diff", "s_diffmoe")]


def window_domains():
    """Domain label for each of the evaluation windows, reconstructed from the
    same offsets eval_all.py used."""
    import numpy as _np
    test_len = os.path.getsize(os.path.join(DATA_DIR, "test.bin")) // 4   # uint32
    offsets = window_offsets(0, test_len, N_GLOBAL_WINDOWS)
    with open(os.path.join(DATA_DIR, "domain_offsets.json")) as f:
        spans = json.load(f)["test"]
    labels = _np.empty(len(offsets), dtype=object)
    for name, (lo, hi) in spans.items():
        for i, o in enumerate(offsets):
            if lo <= o < hi:
                labels[i] = name
    return labels, offsets


def ci(boots):
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def boot_iid(d, n_boot=N_BOOT):
    idx = RNG.integers(0, len(d), size=(n_boot, len(d)))
    return d[idx].mean(axis=1)


def boot_stratified(d, groups, n_boot=N_BOOT):
    """Resample within each domain; the domain mix stays exactly as observed."""
    out = np.empty(n_boot)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    weights = np.array([len(m) for m in members], dtype=float)
    weights /= weights.sum()
    for b in range(n_boot):
        total = 0.0
        for w, m in zip(weights, members):
            pick = RNG.integers(0, len(m), size=len(m))
            total += w * d[m[pick]].mean()
        out[b] = total
    return out


def boot_clustered(d, groups, n_boot=N_BOOT):
    """Resample the domains themselves, then windows inside each drawn domain."""
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    k = len(uniq)
    out = np.empty(n_boot)
    for b in range(n_boot):
        drawn = RNG.integers(0, k, size=k)
        vals = []
        for j in drawn:
            m = members[j]
            pick = RNG.integers(0, len(m), size=len(m))
            vals.append(d[m[pick]].mean())
        out[b] = float(np.mean(vals))
    return out


def main():
    z = np.load(NPZ)
    nll = {k: z[k] for k in z.files}
    groups, _ = window_domains()
    print(f"{len(groups)} windows across {len(np.unique(groups))} domains")
    for g in np.unique(groups):
        print(f"  {g:16s} {int((groups == g).sum()):5d} windows")
    print()

    results = {}
    print(f"{'comparison':26s} {'delta':>9} {'iid 95% CI':>22} {'stratified':>22} {'clustered (6 domains)':>26}")
    for a, b in PAIRS:
        if a not in nll or b not in nll:
            continue
        d = nll[b] - nll[a]
        lo_i, hi_i = ci(boot_iid(d))
        lo_s, hi_s = ci(boot_stratified(d, groups))
        lo_c, hi_c = ci(boot_clustered(d, groups))
        name = f"{NAMES[b]} - {NAMES[a]}"
        sig_c = "excludes 0" if (lo_c < 0) == (hi_c < 0) else "SPANS 0"
        print(f"{name:26s} {d.mean():+9.4f} "
              f"[{lo_i:+7.4f},{hi_i:+7.4f}] [{lo_s:+7.4f},{hi_s:+7.4f}] "
              f"[{lo_c:+7.4f},{hi_c:+7.4f}]  {sig_c}")
        results[f"{b}_minus_{a}"] = {
            "delta": float(d.mean()),
            "iid": [lo_i, hi_i], "stratified": [lo_s, hi_s], "clustered": [lo_c, hi_c],
            "clustered_excludes_zero": (lo_c < 0) == (hi_c < 0),
            "width_ratio_clustered_over_iid": float((hi_c - lo_c) / (hi_i - lo_i)),
        }

    with open(OUT_JSON, "w") as f:
        json.dump({"n_boot": N_BOOT, "n_windows": int(len(groups)),
                   "n_domains": int(len(np.unique(groups))), "pairs": results}, f, indent=2)
    print("\nwrote", OUT_JSON)
    print("\nThe clustered interval uses only 6 clusters and is deliberately crude.\n"
          "Quote it when the claim is meant to hold beyond these particular domains.")


if __name__ == "__main__":
    main()
