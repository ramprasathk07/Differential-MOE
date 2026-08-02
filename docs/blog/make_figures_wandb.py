"""Render report-grade blog figures from the real W&B history of s_diffmoe.

Reads docs/blog/runs_export/s_diffmoe/wandb_history.csv (pulled by
scripts/pull_wandb.py) and writes PNGs to docs/blog/assets/. Every number is the
logged value -- no transcription, no smoothing.

Design follows the data-viz method: one measure per axis (no dual-axis -- the
router losses are two small multiples, not a twin scale), honest full-range axes,
a validated palette, 2px lines, >=8px markers with a surface ring, hairline
gridlines, and titles/subtitles in text ink (never the series color).

    python docs/blog/make_figures_wandb.py
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

# ---- validated palette (light mode) ---------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"       # primary text
SECOND = "#52514e"    # secondary text
MUTED = "#898781"     # axis labels / source
GRID = "#e1e0d9"      # hairline gridline
AXIS = "#c3c2b7"      # baseline / spine
BLUE = "#2a78d6"      # series 1
ORANGE = "#eb6834"    # series 2
BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("blues", BLUE_STEPS)

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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(os.path.join(HERE, "runs_export/s_diffmoe/wandb_history.csv"))

N_LAYERS = 14
MOE_LAYERS = list(range(2, 14))
SOURCE = "source: W&B run  s_diffmoe  ·  BabyLM strict  ·  Kaggle 2×T4"


def base(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(length=0, labelsize=10, colors=MUTED)


def titles(ax, title, subtitle):
    ax.text(0, 1.16, title, transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, va="bottom")
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=10.2,
            color=SECOND, va="bottom")


def save(fig, name, source=True):
    if source:
        fig.text(0.995, 0.008, SOURCE, ha="right", va="bottom",
                 fontsize=7.6, color=MUTED)
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print("wrote", p)


def col(name):
    s = df[["_step", name]].dropna().sort_values("_step")
    return s["_step"].to_numpy(), s[name].to_numpy()


def lambda_init(layer):
    return 0.8 - 0.6 * math.exp(-0.3 * layer)


# ---- Fig 1 — training loss ------------------------------------------------
x, y = col("train/loss")
fig, ax = plt.subplots(figsize=(7.8, 3.7))
fig.subplots_adjust(top=0.80, left=0.10, right=0.97, bottom=0.16)
base(ax)
ax.plot(x, y, color=BLUE, lw=2.0, solid_capstyle="round", zorder=3)
ax.scatter([x[-1]], [y[-1]], s=64, color=BLUE, edgecolor=SURFACE, linewidth=1.8, zorder=5)
ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(-4, 10),
            textcoords="offset points", color=INK, fontsize=10, ha="right", fontweight="bold")
ax.set_xlabel("optimizer step")
ax.set_ylabel("training loss (nats)")
ax.set_ylim(2, 10)
titles(ax, "Training cross-entropy falls cleanly",
       "Differential-MoE · 209M active params · steps 20–2480 (session crash)")
save(fig, "fig1_train_loss.png")

# ---- Fig 2 — validation NLL ----------------------------------------------
vx, vy = col("val/nll")
bi = int(np.argmin(vy))
fig, ax = plt.subplots(figsize=(7.8, 3.7))
fig.subplots_adjust(top=0.80, left=0.10, right=0.97, bottom=0.16)
base(ax)
ax.plot(vx, vy, color=BLUE, lw=2.0, marker="o", ms=7.5, mfc=BLUE,
        markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
ax.scatter([vx[bi]], [vy[bi]], s=120, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=6)
ax.annotate(f"best  {vy[bi]:.3f}  @ step {int(vx[bi])}",
            (vx[bi], vy[bi]), xytext=(-12, 14), textcoords="offset points",
            color=INK, fontsize=10, fontweight="bold", ha="right")
ax.set_xlabel("optimizer step")
ax.set_ylabel("validation NLL (nats)")
titles(ax, "Validation is still falling — no overfit reached",
       "Fixed dev slice, every 250 steps · 4.41 → 3.613, monotone to the crash")
save(fig, "fig2_val_nll.png")

# ---- Fig 3 — router regularizers as TWO small multiples (no dual axis) ----
ax_x, aux = col("train/aux_loss")
zx, z = col("train/router_z_loss")
n_moe = len(MOE_LAYERS)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.7))
fig.subplots_adjust(top=0.78, left=0.075, right=0.975, bottom=0.16, wspace=0.28)
for ax in (a1, a2):
    base(ax)
    ax.set_xlabel("optimizer step")
# left: load-balance aux
a1.plot(ax_x, aux, color=BLUE, lw=2.0, solid_capstyle="round", zorder=3)
a1.axhline(float(n_moe), color=AXIS, lw=1.2, zorder=1)
a1.text(0.96, 0.9, f"{n_moe}.0 = uniform floor", transform=a1.transAxes,
        ha="right", va="top", color=SECOND, fontsize=9)
a1.set_ylabel("load-balance aux  (Σ 12 MoE layers)")
a1.set_xlim(0, ax_x[-1] * 1.02)
a1.set_title("Load balance holds", loc="left", fontsize=11.5, fontweight="bold",
             color=INK, pad=8)
# right: router z
a2.plot(zx, z, color=ORANGE, lw=2.0, solid_capstyle="round", zorder=3)
a2.set_ylabel("router z-loss")
a2.set_title("Router logits tamed", loc="left", fontsize=11.5, fontweight="bold",
             color=INK, pad=8)
fig.text(0.075, 0.98, "Router regularizers do their jobs from the first ~100 steps",
         fontsize=13, fontweight="bold", color=INK, va="top")
save(fig, "fig3_router.png")

# ---- Fig 4 — learned lambda by depth (final val) -------------------------
lam = [col(f"val/lambda_mean_L{i}")[1][-1] for i in range(N_LAYERS)]
init = [lambda_init(i) for i in range(N_LAYERS)]
fig, ax = plt.subplots(figsize=(7.8, 3.7))
fig.subplots_adjust(top=0.80, left=0.09, right=0.97, bottom=0.16)
base(ax)
ax.plot(range(N_LAYERS), init, color=MUTED, lw=1.8, ls=(0, (5, 3)), zorder=2,
        label="initialization schedule")
ax.plot(range(N_LAYERS), lam, color=BLUE, lw=2.0, marker="o", ms=7.5, mfc=BLUE,
        markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3, label="learned (final)")
ax.set_xlabel("layer index")
ax.set_ylabel(u"differential-attention  λ")
ax.set_ylim(0, 1)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.legend(frameon=False, fontsize=10, loc="lower right", labelcolor=SECOND)
titles(ax, "λ moved off its initialization at every layer",
       "Deeper layers subtract harder; layer 0 learns to subtract less")
save(fig, "fig4_lambda_depth.png")

# ---- Fig 5 — expert routing entropy, HONEST 0..1 axis --------------------
ent = [col(f"val/expert_entropy_L{i}")[1][-1] for i in MOE_LAYERS]
fig, ax = plt.subplots(figsize=(7.8, 3.7))
fig.subplots_adjust(top=0.80, left=0.09, right=0.97, bottom=0.16)
base(ax)
labels = [f"L{i}" for i in MOE_LAYERS]
bars = ax.bar(labels, ent, color=BLUE, width=0.66, zorder=3)
ax.axhline(1.0, color=AXIS, lw=1.2, zorder=2)
ax.text(len(labels) - 0.5, 1.0, "1.0 = uniform", va="bottom", ha="right",
        color=SECOND, fontsize=9)
ax.set_ylim(0, 1.08)
ax.set_ylabel("normalized routing entropy")
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
titles(ax, "No expert collapse anywhere",
       "Every one of the 12 MoE layers routes at 0.996–1.000 of the uniform ceiling")
save(fig, "fig5_expert_entropy.png")

# ---- Fig 6 — lambda evolution over training (sequential blue by depth) ----
fig, ax = plt.subplots(figsize=(7.8, 3.8))
fig.subplots_adjust(top=0.80, left=0.09, right=0.99, bottom=0.15)
base(ax)
for i in range(N_LAYERS):
    lx, ly = col(f"train/lambda_mean_L{i}")
    ax.plot(lx, ly, color=BLUES(0.12 + 0.88 * i / (N_LAYERS - 1)), lw=1.7, zorder=3)
ax.set_xlabel("optimizer step")
ax.set_ylabel(u"λ (per-layer mean)")
sm = plt.cm.ScalarMappable(cmap=BLUES, norm=plt.Normalize(vmin=0, vmax=N_LAYERS - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.012, fraction=0.045)
cbar.set_label("layer depth", color=SECOND, fontsize=9.5)
cbar.ax.tick_params(length=0, colors=MUTED, labelsize=9)
cbar.outline.set_edgecolor(AXIS)
titles(ax, "λ separates by depth over training",
       "Each line one layer · deep layers rise to ~0.83, layer 0 drifts down to 0.17")
save(fig, "fig6_lambda_evolution.png")

print("done")
