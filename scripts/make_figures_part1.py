"""Render the Part 1 figures: dense vs MoE vs Diff-MoE, side by side.

Reads the three runs' logged histories (no transcription, no smoothing):

  model_checkpoints/checkpoints/s_dense/metrics.csv     local, 2900 steps
  model_checkpoints/checkpoints/s_moe/metrics.csv       local, 2900 steps
  docs/blog/runs_export/s_diffmoe/wandb_history.csv     W&B, crashed at 2480

plus docs/blog/runs_export/eval_comparison.json (written by scripts/eval_all.py)
for the held-out test figures. PNGs land in the repo-root assets/, which the
README embeds. Any figure whose data is missing is skipped with
a printed note rather than failing.

Design follows the data-viz method: categorical hues assigned in fixed slot order
(blue/orange/aqua -- validated all-pairs, light and dark), one measure per axis
and never a dual axis, 2px lines, >=8px endpoint markers with a surface ring,
hairline gridlines, direct labels in text ink beside a colored mark (never text
in the series color), and a legend whenever two or more series share an axis.

    python scripts/make_figures_part1.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

# ---- validated palette (light mode; same instance as make_figures_wandb.py) --
SURFACE = "#fcfcfb"
INK = "#0b0b0b"       # primary text
SECOND = "#52514e"    # secondary text
MUTED = "#898781"     # axis labels / source
GRID = "#e1e0d9"      # hairline gridline
AXIS = "#c3c2b7"      # baseline / spine
BLUE = "#2a78d6"      # categorical slot 1
ORANGE = "#eb6834"    # categorical slot 2
AQUA = "#1baf7a"      # categorical slot 3
YELLOW = "#eda100"    # categorical slot 4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 11,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": SECOND,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 1.0,
    "savefig.dpi": 200,
    "figure.dpi": 200,
})

HERE = os.path.dirname(os.path.abspath(__file__))   # scripts/
ROOT = os.path.dirname(HERE)                        # repo root
OUT = os.path.join(ROOT, "assets")                  # published: README embeds these
CK = os.path.join(ROOT, "model_checkpoints/checkpoints")
EXPORT = os.path.join(ROOT, "docs/blog/runs_export")  # local artifacts, not published
os.makedirs(OUT, exist_ok=True)

TOK_PER_STEP = 65536          # batch 2 x accum 32 x 2 GPUs x seq 512
SOURCE = "source: logged training histories · BabyLM strict · Kaggle 2×T4"

# entity -> color, fixed. A run keeps its hue in every figure, and the slot order
# here is the order the palette was validated in (adjacent pairlist -- the right
# one for lines and grouped bars). A run with no data is skipped, not recoloured.
RUNS = {
    "s_dense":   {"label": "Dense",      "color": BLUE},
    "s_moe":     {"label": "MoE",        "color": ORANGE},
    "s_diffmoe": {"label": "Diff-MoE",   "color": AQUA},
    "s_diff":    {"label": "Diff-Dense", "color": YELLOW},
}


def base(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


def titles(ax, title, subtitle):
    ax.set_title(title, color=INK, fontsize=14, fontweight="600", loc="left", pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=SECOND, fontsize=10.5, va="bottom")


def endlabel(ax, x, y, text, color, dx=28, dy=0, ha="left"):
    """Direct label in text ink, with the series-colored endpoint mark carrying
    identity -- text never wears the series color. `ha="right"` places the label
    back along the line, for a series that ends short of the others."""
    ax.plot([x], [y], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=5, clip_on=False)
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                color=INK, fontsize=10.5, fontweight="600", va="center", ha=ha,
                clip_on=False)


def source(fig, text=SOURCE):
    fig.text(0.008, 0.012, text, color=MUTED, fontsize=8.5, ha="left")


# ---- load the three histories ----------------------------------------------
def load_curves():
    """Local metrics.csv where the run dir came back from Kaggle; the W&B export
    otherwise (s_diffmoe crashed before its metrics.csv was saved). A run with
    neither is simply absent from every figure."""
    curves = {}
    for run in RUNS:
        local = os.path.join(CK, run, "metrics.csv")
        wandb = os.path.join(EXPORT, run, "wandb_history.csv")
        if os.path.exists(local):
            df = pd.read_csv(local)
            cols = ("step", "train_loss", "val_nll", "tok_per_sec")
        elif os.path.exists(wandb):
            df = pd.read_csv(wandb)
            cols = ("_step", "train/loss", "val/nll", "train/tok_per_sec")
        else:
            print(f"no history for {run} (looked in {local}, {wandb})")
            continue
        s, tl, vn, tk = cols
        curves[run] = {
            "train": df[[s, tl]].dropna().rename(columns={s: "step", tl: "loss"}),
            "val": df[[s, vn]].dropna().rename(columns={s: "step", vn: "nll"}),
            "tok_s": df[tk].dropna().iloc[1:].median(),
        }
    return curves


C = load_curves()
print("loaded runs:", list(C))
for r, c in C.items():
    print(f"  {r:10s} train {len(c['train'])} pts, val {len(c['val'])} pts, "
          f"{c['tok_s']:.0f} tok/s")


# ---- fig 1: validation NLL vs step -----------------------------------------
def fig_val_nll():
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    base(ax)
    for run, meta in RUNS.items():
        if run not in C:
            continue
        v = C[run]["val"]
        ax.plot(v.step, v.nll, "-", lw=2, color=meta["color"], label=meta["label"], zorder=3)
        ax.plot(v.step, v.nll, "o", ms=5, color=meta["color"], mec=SURFACE, mew=1.5, zorder=4)
    dy = {"s_dense": 7, "s_moe": -2, "s_diffmoe": -18}   # Diff-MoE ends early; clear the MoE curve
    for run, meta in RUNS.items():
        if run not in C:
            continue
        v = C[run]["val"]
        endlabel(ax, v.step.iloc[-1], v.nll.iloc[-1],
                 f"{meta['label']}  {v.nll.iloc[-1]:.3f}", meta["color"], dx=16,
                 dy=dy.get(run, 0))
    titles(ax, "Validation NLL: the three runs, step for step",
           "Lower is better. Measured every 250 steps on the identical held-out slice.")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation NLL (nats/token)")
    ax.set_xlim(0, 3560)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.legend(frameon=False, loc="upper right", labelcolor=SECOND, fontsize=10.5)
    source(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(OUT, "fig_p1_val_nll.png"))
    plt.close(fig)
    print("wrote fig_p1_val_nll.png")


# ---- fig 2: head-to-head delta against a named baseline ---------------------
def fig_delta(baseline="s_moe", others=("s_diffmoe",), fname="fig_p1_delta.png",
              title="What differential attention adds to a mixture of experts",
              subtitle=("Validation NLL minus the MoE run at the same step. "
                        "Below zero = differential attention is ahead.")):
    if baseline not in C:
        print(f"skip delta: no {baseline} baseline")
        return
    b = C[baseline]["val"].set_index("step")["nll"]
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    base(ax)
    ax.axhline(0, color=AXIS, lw=1.4, zorder=2)
    ax.text(60, 0.003, f"{RUNS[baseline]['label']} baseline", color=MUTED,
            fontsize=9.5, va="bottom")
    for run in others:
        if run not in C:
            continue
        v = C[run]["val"].set_index("step")["nll"]
        common = v.index.intersection(b.index)
        d = (v.loc[common] - b.loc[common])
        ax.plot(common, d.values, "-", lw=2, color=RUNS[run]["color"],
                label=f"{RUNS[run]['label']} − {RUNS[baseline]['label']}", zorder=3)
        ax.plot(common, d.values, "o", ms=5, color=RUNS[run]["color"],
                mec=SURFACE, mew=1.5, zorder=4)
        endlabel(ax, common[-1], d.values[-1], f"{d.values[-1]:+.3f}",
                 RUNS[run]["color"], dx=16, dy=0)
    titles(ax, title, subtitle)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(f"Δ validation NLL vs {RUNS[baseline]['label']} (nats)")
    ax.set_xlim(0, 3050)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.legend(frameon=False, loc="upper right", labelcolor=SECOND, fontsize=10.5)
    source(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("wrote", fname)


# ---- fig 3: the same curves against GPU-hours, not steps --------------------
def fig_walltime():
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    base(ax)
    for run, meta in RUNS.items():
        if run not in C:
            continue
        v = C[run]["val"]
        hours = v.step * TOK_PER_STEP / C[run]["tok_s"] / 3600
        ax.plot(hours, v.nll, "-", lw=2, color=meta["color"],
                label=f"{meta['label']}  ({C[run]['tok_s']:,.0f} tok/s)", zorder=3)
        ax.plot(hours, v.nll, "o", ms=5, color=meta["color"], mec=SURFACE, mew=1.5, zorder=4)
    # the three runs end within ~3 h of each other -> stagger the labels vertically
    dy = {"s_dense": 10, "s_moe": -14, "s_diffmoe": 10}
    for run, meta in RUNS.items():
        if run not in C:
            continue
        v = C[run]["val"]
        hours = v.step * TOK_PER_STEP / C[run]["tok_s"] / 3600
        endlabel(ax, hours.iloc[-1], v.nll.iloc[-1], meta["label"],
                 meta["color"], dx=14, dy=dy.get(run, 0))
    titles(ax, "The same three runs, priced in GPU-hours",
           "Step-for-step is not what a free-tier budget buys. This is quality per hour of 2×T4.")
    ax.set_xlabel("GPU-hours (2×T4, from each run's measured throughput)")
    ax.set_ylabel("validation NLL (nats/token)")
    ax.set_xlim(0, 17.5)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.legend(frameon=False, loc="upper right", labelcolor=SECOND, fontsize=10.5)
    source(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(OUT, "fig_p1_walltime.png"))
    plt.close(fig)
    print("wrote fig_p1_walltime.png")


# ---- fig 4: training loss ---------------------------------------------------
def fig_train_loss():
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    base(ax)
    for run, meta in RUNS.items():
        if run not in C:
            continue
        t = C[run]["train"]
        ax.plot(t.step, t.loss, "-", lw=2, color=meta["color"], label=meta["label"], zorder=3)
    # Diff-MoE stops ~420 steps short, so its label would land on top of the other
    # two -- place it back along its own line instead of out to the right.
    place = {"s_dense": (16, 8, "left"), "s_moe": (16, -8, "left"),
             "s_diffmoe": (-12, 16, "right")}
    for run, meta in RUNS.items():
        if run not in C:
            continue
        t = C[run]["train"]
        dx, dy, ha = place.get(run, (16, 0, "left"))
        endlabel(ax, t.step.iloc[-1], t.loss.iloc[-1],
                 f"{meta['label']}  {t.loss.iloc[-1]:.3f}", meta["color"],
                 dx=dx, dy=dy, ha=ha)
    titles(ax, "Training cross-entropy",
           "Every logged step, unsmoothed. All three fall the way they should; the gaps are small and real.")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training loss (nats/token)")
    ax.set_xlim(0, 3560)
    ax.set_ylim(2.4, 5.0)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.legend(frameon=False, loc="upper right", labelcolor=SECOND, fontsize=10.5)
    source(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(OUT, "fig_p1_train_loss.png"))
    plt.close(fig)
    print("wrote fig_p1_train_loss.png")


# ---- fig 5: per-domain test NLL (needs eval_comparison.json) ----------------
def fig_domain():
    p = os.path.join(EXPORT, "eval_comparison.json")
    if not os.path.exists(p):
        print("skip per-domain: no eval_comparison.json (run scripts/eval_all.py)")
        return
    with open(p) as f:
        comp = json.load(f)
    runs = [r for r in RUNS if r in comp and comp[r].get("per_domain")]
    if not runs:
        print("skip per-domain: no per_domain data")
        return
    domains = list(comp[runs[0]]["per_domain"])
    order = sorted(domains, key=lambda d: comp[runs[0]]["per_domain"][d]["nll"])
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    base(ax)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=GRID, linewidth=1, zorder=0)
    n = len(runs)
    h = 0.78 / n
    ypos = np.arange(len(order))
    for k, run in enumerate(runs):
        vals = [comp[run]["per_domain"][d]["nll"] for d in order]
        off = (k - (n - 1) / 2) * h
        ax.barh(ypos + off, vals, height=h * 0.88, color=RUNS[run]["color"],
                label=RUNS[run]["label"], zorder=3)   # 2px surface gap via 0.88 factor
        for y, v in zip(ypos + off, vals):
            ax.text(v + 0.03, y, f"{v:.2f}", va="center", color=SECOND, fontsize=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order, color=SECOND)
    ax.invert_yaxis()
    titles(ax, "The six BabyLM domains are not equally hard",
           "Held-out test NLL per domain, identical windows for all three models. Lower is better.")
    ax.set_xlabel("test NLL (nats/token)")
    # domains are sorted easiest-first and the y-axis is inverted, so the top-right
    # of the plot is empty -- the only place a legend does not sit on a bar
    ax.set_xlim(0, 4.35)
    ax.legend(frameon=False, loc="upper right", labelcolor=SECOND, fontsize=10.5)
    source(fig, "source: scripts/eval_all.py · held-out BabyLM test split · RTX 3060, fp32")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(OUT, "fig_p1_domain_nll.png"))
    plt.close(fig)
    print("wrote fig_p1_domain_nll.png")


if __name__ == "__main__":
    fig_val_nll()
    fig_delta()                       # Part 1's pairing: Diff-MoE vs MoE
    fig_delta(baseline="s_dense", others=("s_moe", "s_diffmoe"),
              fname="fig_p1_delta_vs_dense.png",
              title="How much each mechanism buys over a plain dense transformer",
              subtitle=("Validation NLL minus the dense run at the same step, "
                        "at identical active-parameter cost. Below zero = better than dense."))
    fig_walltime()
    fig_train_loss()
    fig_domain()
