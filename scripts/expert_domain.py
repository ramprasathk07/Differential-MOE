"""Expert-domain specialization analysis on the trained s_diffmoe checkpoint.

Question: with 6 experts and BabyLM's 6 domains, did the router learn to send
each domain to its own experts -- or did the load-balance loss (which pinned the
marginal routing entropy at ~1.0) flatten any specialization?

Method: for each test-split domain, run ~131k tokens through the model and record,
per MoE layer, what fraction of that domain's tokens each expert received. Expert
index k is only comparable *within* a layer (permutation-free across layers), so
we never sum experts across layers -- we score each layer's domain x expert matrix
for structure and surface the most-specialized one. Per-domain NLL falls out of
the same pass.

Writes docs/blog/runs_export/s_diffmoe/expert_domain.json and two figures.

    python scripts/expert_domain.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.model import Transformer                # noqa: E402
from src.data import load_tokens                 # noqa: E402
from src.eval import evaluate                    # noqa: E402
import eval_local                                # noqa: E402  (reuse infer_config + paths)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 512
BATCH = 16
WINDOWS_PER_DOMAIN = 256      # ~131k tokens/domain -> stable routing + per-domain NLL
CKPT = eval_local.CKPT
DATA_DIR = eval_local.DATA_DIR
OUT = os.path.join(ROOT, "docs/blog/runs_export/s_diffmoe")
ASSETS = os.path.join(ROOT, "docs/blog/assets")

# palette (shared with make_figures_wandb.py)
SURFACE, INK, SECOND, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BLUE = "#e1e0d9", "#c3c2b7", "#2a78d6"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 200, "figure.dpi": 200,
})

DOMAIN_LABEL = {
    "bnc_spoken": "BNC spoken", "childes": "CHILDES", "gutenberg": "Gutenberg",
    "open_subtitles": "Subtitles", "simple_wiki": "Simple Wiki", "switchboard": "Switchboard",
}


def strided_range(data, start, end, batch, seq, n_windows, device):
    lo, hi = start, end - seq - 1
    total = max(1, (hi - lo) // seq)
    n = min(n_windows, total)
    offs = [lo + int(k * total / n) * seq for k in range(n)]
    for b in range(0, n, batch):
        chunk = offs[b:b + batch]
        x = torch.stack([torch.from_numpy(data[i:i + seq].astype("int64")) for i in chunk]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + seq + 1].astype("int64")) for i in chunk]).to(device)
        yield x, y


def tv_from_uniform(frac):
    n = len(frac)
    return 0.5 * sum(abs(f - 1.0 / n) for f in frac)


def main():
    ckpt = torch.load(CKPT, map_location="cpu", mmap=True)
    sd = ckpt["model"]
    cfg = eval_local.infer_config(sd)
    model = Transformer(cfg).to(DEVICE)
    model.load_state_dict(sd, strict=False)
    del ckpt, sd
    model.eval()
    print("loaded", cfg.n_experts, "experts,", cfg.n_layers, "layers on", DEVICE)

    test = load_tokens(os.path.join(DATA_DIR, "test.bin"))
    domains = json.load(open(os.path.join(DATA_DIR, "domain_offsets.json")))["test"]

    per_domain = {}
    for name, (s, e) in domains.items():
        batches = strided_range(test, s, e, BATCH, SEQ, WINDOWS_PER_DOMAIN, DEVICE)
        res = evaluate(model, batches, None)
        counts = {int(L): [int(c) for c in res.expert_counts[L]] for L in res.expert_counts}
        per_domain[name] = {"nll": res.nll, "ppl": res.perplexity,
                            "top1": res.top1_acc, "counts": counts}
        print(f"  {name:16s} nll {res.nll:.3f}  ppl {res.perplexity:6.2f}  top1 {res.top1_acc:.3f}")

    dom_order = list(domains.keys())
    moe_layers = sorted(per_domain[dom_order[0]]["counts"].keys())
    n_exp = cfg.n_experts

    # domain x expert routing fraction, per layer; score each layer's structure
    layer_matrix, layer_score = {}, {}
    for L in moe_layers:
        M = []
        for d in dom_order:
            c = per_domain[d]["counts"][L]
            tot = sum(c) or 1
            M.append([ci / tot for ci in c])
        layer_matrix[L] = M
        layer_score[L] = sum(tv_from_uniform(row) for row in M) / len(M)

    best_L = max(layer_score, key=layer_score.get)
    print("\nper-layer specialization (mean TV from uniform):")
    for L in moe_layers:
        print(f"  L{L}: {layer_score[L]:.3f}" + ("   <- most structured" if L == best_L else ""))

    json.dump({"per_domain_nll": {d: per_domain[d]["nll"] for d in dom_order},
               "per_domain_ppl": {d: per_domain[d]["ppl"] for d in dom_order},
               "layer_specialization_tv": layer_score,
               "best_layer": best_L,
               "best_layer_matrix": {d: layer_matrix[best_L][i] for i, d in enumerate(dom_order)}},
              open(os.path.join(OUT, "expert_domain.json"), "w"), indent=2)

    # ---- Fig 7: per-domain NLL --------------------------------------------
    order = sorted(dom_order, key=lambda d: per_domain[d]["nll"])
    vals = [per_domain[d]["nll"] for d in order]
    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    fig.subplots_adjust(top=0.80, left=0.16, right=0.97, bottom=0.14)
    ax.grid(True, axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(length=0, labelsize=10, colors=MUTED)
    ypos = range(len(order))
    ax.barh(list(ypos), vals, color=BLUE, height=0.62, zorder=3)
    for yp, v in zip(ypos, vals):
        ax.text(v + 0.03, yp, f"{v:.2f}", va="center", ha="left", color=INK, fontsize=9.5)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([DOMAIN_LABEL[d] for d in order], color=SECOND)
    ax.set_xlabel("test NLL (nats) — lower = easier to model")
    ax.set_xlim(0, max(vals) * 1.16)
    ax.text(0, 1.16, "The six domains are wildly different in difficulty",
            transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, va="bottom")
    ax.text(0, 1.035, "Same checkpoint, ~131k held-out tokens per domain",
            transform=ax.transAxes, fontsize=10.2, color=SECOND, va="bottom")
    fig.savefig(os.path.join(ASSETS, "fig7_domain_nll.png"), bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print("wrote fig7_domain_nll.png")

    # ---- Fig 8: domain x expert routing heatmap (most-structured layer) ----
    M = layer_matrix[best_L]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    fig.subplots_adjust(top=0.78, left=0.16, right=0.99, bottom=0.10)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(0.34, max(max(r) for r in M)), aspect="auto")
    ax.set_xticks(range(n_exp))
    ax.set_xticklabels([f"E{i}" for i in range(n_exp)], color=SECOND)
    ax.set_yticks(range(len(dom_order)))
    ax.set_yticklabels([DOMAIN_LABEL[d] for d in dom_order], color=SECOND)
    ax.tick_params(length=0, labelsize=10)
    for i in range(len(dom_order)):
        for j in range(n_exp):
            v = M[i][j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v > 0.22 else INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("fraction of domain's tokens", color=SECOND, fontsize=9.5)
    cb.ax.tick_params(length=0, colors=MUTED, labelsize=9)
    verdict = ("clear specialization" if layer_score[best_L] > 0.15
               else "weak — load balance flattened it" if layer_score[best_L] < 0.06
               else "mild specialization")
    ax.text(0, 1.20, f"Domain → expert routing, layer {best_L} (most structured)",
            transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, va="bottom")
    ax.text(0, 1.06,
            f"Rows sum to 1 · 1/{n_exp}={1/n_exp:.2f} would be no preference · "
            f"structure score {layer_score[best_L]:.3f} → {verdict}",
            transform=ax.transAxes, fontsize=10, color=SECOND, va="bottom")
    fig.savefig(os.path.join(ASSETS, "fig8_expert_domain.png"), bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print("wrote fig8_expert_domain.png")
    print("done")


if __name__ == "__main__":
    main()
