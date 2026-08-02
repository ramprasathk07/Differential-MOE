"""Render the Part 2 (scale-up tier-S) blog figures from the run's own logs.

Unlike make_figures.py (Part 1), this does NOT hardcode numbers transcribed from
Weights & Biases. It reads the keyless files each training/eval run already
writes, so the charts regenerate exactly and stay honest:

    runs_export/<run>/metrics.csv           (src/train.py  -- per-step curves)
    runs_export/<run>/report.json           (src/train.py  -- params, best_val)
    runs_export/<run>/final_test_eval.json   (src/eval.py   -- per-layer diag)

Drop those three files per run under docs/blog/runs_export/ (or pass --root),
then:

    python docs/blog/make_figures_part2.py

Figures are written to docs/blog/assets/ as fig_s0*.png. Any figure whose data
is absent (e.g. no MoE run supplied) is skipped with a note rather than crashing.
"""

import argparse
import csv
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---- palette (identical to Part 1's research-instrument identity) ----
INK = "#121a1f"
INK_SOFT = "#48565e"
INK_FAINT = "#8a949b"
GRID = "#e9edef"
TEAL = "#0b8f86"
INDIGO = "#5b62d6"
AMBER = "#c2740a"
ROSE = "#b5397f"
BG = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": INK_FAINT,
    "axes.labelcolor": INK_SOFT,
    "xtick.color": INK_FAINT,
    "ytick.color": INK_FAINT,
    "axes.linewidth": 0.8,
})

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)

# The four cells of the 2x2, in a fixed draw order. label + colour keyed on the
# (attention, ffn) pair from report.json, so run_name spelling never matters.
CELLS = {
    ("differential", "moe"):   ("differential + MoE",   INDIGO, 2.6),
    ("standard", "dense"):     ("standard + dense",     TEAL,   2.2),
    ("differential", "dense"): ("differential + dense", AMBER,  2.2),
    ("standard", "moe"):       ("standard + MoE",       ROSE,   2.2),
}


def style(ax):
    ax.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def lambda_init(layer, n_layers):
    """The per-layer differential-attention init schedule, from the paper /
    Part 1: lam_init(l) = 0.8 - 0.6 * exp(-0.3 * l)."""
    return 0.8 - 0.6 * math.exp(-0.3 * layer)


def load_runs(root):
    """Return a list of run dicts sorted into the CELLS draw order. Each has:
    key, label, colour, lw, metrics(list of row dicts), report, eval."""
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(run_dir):
            continue
        rep_path = os.path.join(run_dir, "report.json")
        if not os.path.exists(rep_path):
            print("skip (no report.json):", run_dir)
            continue
        report = json.load(open(rep_path))
        attn = report.get("model", {}).get("attention")
        ffn = report.get("model", {}).get("ffn")
        key = (attn, ffn)
        label, colour, lw = CELLS.get(key, (report.get("run_name", os.path.basename(run_dir)), INK_SOFT, 2.0))

        metrics = []
        mpath = os.path.join(run_dir, "metrics.csv")
        if os.path.exists(mpath):
            with open(mpath, newline="") as f:
                for row in csv.DictReader(f):
                    metrics.append({k: (float(v) if v not in ("", None) else None)
                                    for k, v in row.items()})

        evl = None
        epath = os.path.join(run_dir, "final_test_eval.json")
        if os.path.exists(epath):
            evl = json.load(open(epath))

        runs.append(dict(key=key, label=label, colour=colour, lw=lw,
                         metrics=metrics, report=report, eval=evl,
                         name=report.get("run_name", os.path.basename(run_dir))))
    # draw in the canonical 2x2 order, unknown cells last
    order = list(CELLS)
    runs.sort(key=lambda r: order.index(r["key"]) if r["key"] in order else 99)
    if not runs:
        raise SystemExit(f"no runs found under {root} -- drop each run's "
                         f"metrics.csv + report.json (+ final_test_eval.json) there")
    print("runs:", ", ".join(f"{r['name']}({r['label']})" for r in runs))
    return runs


def series(run, ycol, xcol="step"):
    xs, ys = [], []
    for row in run["metrics"]:
        y = row.get(ycol)
        x = row.get(xcol)
        if y is not None and x is not None:
            xs.append(x); ys.append(y)
    return xs, ys


# ---------------------------------------------------------------- figures ----
def fig_train_loss(runs):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    style(ax)
    drew = False
    for r in runs:
        xs, ys = series(r, "train_loss")
        if not xs:
            continue
        ax.plot(xs, ys, color=r["colour"], lw=r["lw"], label=r["label"])
        ax.scatter([xs[-1]], [ys[-1]], color=r["colour"], s=24, zorder=5)
        drew = True
    if not drew:
        plt.close(fig); print("skip fig_s01 (no train_loss)"); return
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("train loss (nats)")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    save(fig, "fig_s01_train_loss.png")


def fig_val_nll(runs):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    style(ax)
    drew = False
    for r in runs:
        xs, ys = series(r, "val_nll")
        if not xs:
            continue
        ax.plot(xs, ys, color=r["colour"], lw=r["lw"], marker="o", ms=4,
                markeredgecolor=BG, markeredgewidth=0.9, label=r["label"])
        ymin = min(ys); xmin = xs[ys.index(ymin)]
        ax.scatter([xmin], [ymin], color=r["colour"], s=44, zorder=6,
                   edgecolor=BG, linewidth=1.3)
        drew = True
    if not drew:
        plt.close(fig); print("skip fig_s02 (no val_nll)"); return
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("val NLL (nats)")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    save(fig, "fig_s02_val_nll.png")


def fig_aux_losses(runs):
    # prefer the thesis run (differential+MoE); else any MoE run
    moe = [r for r in runs if r["key"][1] == "moe"]
    moe.sort(key=lambda r: 0 if r["key"][0] == "differential" else 1)
    for r in moe:
        ax_xs, aux = series(r, "train_aux_loss")
        zx, z = series(r, "train_router_z_loss")
        if not aux or not z:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 3.3))
        style(ax)
        ax.plot(ax_xs, aux, color=TEAL, lw=2.2, label="load-balance aux (left)")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("aux loss  (Σ over MoE layers)", color=TEAL)
        ax.tick_params(axis="y", colors=TEAL)
        ax.spines["left"].set_color(TEAL)
        n_moe = _n_moe_layers(r)
        if n_moe:
            ax.axhline(float(n_moe), color=TEAL, lw=1, ls=(0, (2, 3)), alpha=0.6)
            ax.text(0.99, 0.04, f"{n_moe}.0 = perfectly balanced (1.0 / layer)",
                    transform=ax.transAxes, ha="right", color=TEAL, fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(zx, z, color=AMBER, lw=2.2, label="router z (right)")
        ax2.set_ylabel("router z-loss", color=AMBER)
        ax2.tick_params(axis="y", colors=AMBER, length=0)
        ax2.spines["right"].set_color(AMBER)
        ax2.spines["top"].set_visible(False)
        lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
        ax.legend(lines, [l.get_label() for l in lines], frameon=False,
                  fontsize=9.5, loc="upper center")
        save(fig, "fig_s03_aux_losses.png")
        return
    print("skip fig_s03 (no MoE aux/z series)")


def _n_moe_layers(run):
    ev = run.get("eval") or {}
    if ev.get("expert_entropy"):
        return len(ev["expert_entropy"])
    m = run["report"].get("model", {})
    if m.get("ffn") == "moe" and m.get("n_layers"):
        # n_dense_layers isn't in report.json; entropy dict is the reliable count
        return None
    return None


def fig_lambda_depth(runs):
    diff = [r for r in runs if r["key"][0] == "differential" and r.get("eval")
            and r["eval"].get("lambda_means")]
    if not diff:
        print("skip fig_s04 (no differential run with lambda_means in eval)"); return
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    style(ax)
    n_layers = max(r["report"]["model"]["n_layers"] for r in diff)
    layers = list(range(n_layers))
    ax.plot(layers, [lambda_init(l, n_layers) for l in layers], color=INK_FAINT,
            lw=1.8, ls="--", label="initialization schedule")
    for r in diff:
        lm = r["eval"]["lambda_means"]
        idx = sorted(int(k) for k in lm)
        ax.plot(idx, [lm[str(i)] if str(i) in lm else lm[i] for i in idx],
                color=r["colour"], lw=2.4, marker="o", ms=6,
                markeredgecolor=BG, markeredgewidth=1.3,
                label=f"learned — {r['label']}")
    ax.set_xlabel("layer index")
    ax.set_ylabel(u"λ")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(max(1, n_layers // 8)))
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    save(fig, "fig_s04_lambda_depth.png")


def fig_expert_entropy(runs):
    moe = [r for r in runs if r["key"][1] == "moe" and r.get("eval")
           and r["eval"].get("expert_entropy")]
    moe.sort(key=lambda r: 0 if r["key"][0] == "differential" else 1)
    if not moe:
        print("skip fig_s05 (no MoE run with expert_entropy in eval)"); return
    r = moe[0]
    ent = r["eval"]["expert_entropy"]
    items = sorted(((int(k), v) for k, v in ent.items()))
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    style(ax)
    labels = [f"L{l}" for l, _ in items]
    vals = [v for _, v in items]
    bars = ax.bar(labels, vals, color=r["colour"], width=0.62, zorder=3, alpha=0.9)
    ax.axhline(1.0, color=INK_FAINT, lw=1, ls=(0, (2, 3)))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.5, color=INK_SOFT)
    lo = min(0.9, min(vals) - 0.02)
    ax.set_ylim(lo, 1.01)
    ax.set_ylabel("norm. entropy")
    ax.set_title(f"{r['label']}", fontsize=10, color=INK_SOFT, loc="left")
    ax.text(0.012, 0.04, "1.0 = perfectly uniform", transform=ax.transAxes,
            fontsize=9, color=INK_FAINT)
    save(fig, "fig_s05_expert_entropy.png")


def fig_final_bpb(runs):
    """Bar of best/eval bits-per-byte across the 2x2 -- the tokenizer-agnostic
    money chart. Uses test bits-per-byte from eval when present, else best_val."""
    data = []
    for r in runs:
        bpb = None
        if r.get("eval") and r["eval"].get("test_bits_per_byte") is not None:
            bpb = r["eval"]["test_bits_per_byte"]
        elif r["report"].get("best_val", {}).get("bits_per_byte") is not None:
            bpb = r["report"]["best_val"]["bits_per_byte"]
        if bpb is not None:
            data.append((r["label"], bpb, r["colour"]))
    if not data:
        print("skip fig_s06 (no bits-per-byte anywhere)"); return
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    style(ax)
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    cols = [d[2] for d in data]
    bars = ax.bar(range(len(vals)), vals, color=cols, width=0.6, zorder=3, alpha=0.92)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9.5, color=INK_SOFT)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("test bits-per-byte  (lower = better)")
    save(fig, "fig_s06_bits_per_byte.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "runs_export"),
                    help="dir with one subdir per run, each holding metrics.csv "
                         "+ report.json (+ final_test_eval.json)")
    args = ap.parse_args()
    runs = load_runs(args.root)
    fig_train_loss(runs)
    fig_val_nll(runs)
    fig_aux_losses(runs)
    fig_lambda_depth(runs)
    fig_expert_entropy(runs)
    fig_final_bpb(runs)
    print("done")


if __name__ == "__main__":
    main()
