"""Render the Part 1 blog figures as PNGs from the logged Weights & Biases data.

Regenerates the four charts in docs/blog/assets/ so the Markdown post can embed
them anywhere Markdown renders (GitHub, Medium, dev.to). The HTML artifact draws
the same data as inline SVG; this is the portable-image version.

    python docs/blog/make_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---- palette (matches the blog's research-instrument identity) ----
INK = "#121a1f"
INK_SOFT = "#48565e"
INK_FAINT = "#8a949b"
GRID = "#e9edef"
TEAL = "#0b8f86"
INDIGO = "#5b62d6"
AMBER = "#c2740a"
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

# ---- data (from wandb: New_103/diff-moe-kaggle) ----
dense_train = [[120,4.094],[160,3.469],[200,3.084],[240,2.851],[280,2.674],[320,2.534],[360,2.421],[400,2.337],[440,2.283],[480,2.202],[520,2.145],[560,2.106],[600,2.069],[640,2.011],[680,1.969],[720,1.958],[760,1.922],[800,1.904],[840,1.888],[880,1.858],[920,1.853],[960,1.828],[1000,1.801]]
diff_train = [[20,7.914],[160,4.616],[300,3.757],[440,3.275],[580,3.045],[720,2.814],[860,2.733],[1000,2.601],[1280,2.429],[1560,2.322],[1840,2.184],[2120,2.076],[2400,1.938],[2680,1.836],[2960,1.741],[3240,1.664],[3520,1.551],[3800,1.486],[4080,1.4],[4360,1.326],[4640,1.249],[4920,1.201],[5200,1.134],[5480,1.098],[5620,1.072]]
dense_val = [[500,2.22],[1000,1.866]]
diff_val = [[500,3.975],[1000,3.682],[1500,3.694],[2000,3.805],[2500,3.995],[3000,4.219],[3500,4.467],[4000,4.709],[4500,4.982],[5000,5.245],[5500,5.464]]
lam_learned = [0.206,0.520,0.694,0.632,0.702,0.774,0.722,0.759]
lam_init = [0.200,0.356,0.471,0.556,0.619,0.666,0.701,0.727]
entropy = [(2,0.969),(3,0.990),(4,0.988),(5,0.982),(6,0.998),(7,0.980)]
aux_loss = [[20,6.888],[140,6.071],[380,6.048],[620,6.045],[860,6.045],[1100,6.045],[1340,6.046],[1580,6.044],[1820,6.048],[2060,6.046],[2300,6.046],[2540,6.045],[2780,6.047],[3020,6.043],[3260,6.045],[3500,6.047],[3740,6.042],[3980,6.045],[4220,6.046],[4460,6.047],[4700,6.047],[4940,6.052],[5180,6.047],[5420,6.043],[5660,6.05]]
z_loss = [[20,26.163],[140,1.875],[260,1.008],[500,1.249],[860,1.164],[1220,1.255],[1580,1.44],[1940,1.628],[2300,1.826],[2660,2.006],[3020,2.165],[3380,2.315],[3740,2.458],[4100,2.558],[4460,2.673],[4820,2.797],[5180,2.9],[5660,2.989]]

OUT = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT, exist_ok=True)


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


def xy(pairs):
    return [p[0] for p in pairs], [p[1] for p in pairs]


# Fig 01 — training loss
fig, ax = plt.subplots(figsize=(7.2, 3.1))
style(ax)
ax.plot(*xy(diff_train), color=INDIGO, lw=2.2, label="differential + MoE")
ax.plot(*xy(dense_train), color=TEAL, lw=2.2, label="standard + dense")
for pairs, c in [(diff_train, INDIGO), (dense_train, TEAL)]:
    ax.scatter([pairs[-1][0]], [pairs[-1][1]], color=c, s=26, zorder=5)
ax.set_xlabel("optimizer step")
ax.set_ylabel("train loss (nats)")
ax.set_ylim(1, 8)
ax.set_xlim(0, 5700)
ax.legend(frameon=False, fontsize=10, loc="upper right")
save(fig, "fig01_train_loss.png")

# Fig 02 — lambda by depth
fig, ax = plt.subplots(figsize=(7.2, 3.1))
style(ax)
layers = list(range(8))
ax.plot(layers, lam_init, color=INK_FAINT, lw=1.8, ls="--", label="initialization schedule")
ax.plot(layers, lam_learned, color=TEAL, lw=2.4, marker="o", ms=7,
        markeredgecolor=BG, markeredgewidth=1.4, label="learned (final)")
ax.set_xlabel("layer index")
ax.set_ylabel(u"λ")
ax.set_ylim(0, 1)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.legend(frameon=False, fontsize=10, loc="lower right")
save(fig, "fig02_lambda_depth.png")

# Fig 03 — expert entropy bars (zoomed 0.9-1.0)
fig, ax = plt.subplots(figsize=(7.2, 3.1))
style(ax)
labels = [f"L{l}" for l, _ in entropy]
vals = [v for _, v in entropy]
bars = ax.bar(labels, vals, color=TEAL, width=0.62, zorder=3, alpha=0.9)
ax.axhline(1.0, color=INK_FAINT, lw=1, ls=(0, (2, 3)))
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9.5, color=INK_SOFT)
ax.set_ylim(0.9, 1.008)
ax.set_ylabel("norm. entropy")
ax.text(0.012, 0.955, "1.0 = perfectly uniform", transform=ax.transAxes,
        fontsize=9.5, color=INK_FAINT)
save(fig, "fig03_expert_entropy.png")

# Fig 04 — validation NLL (the overfit)
fig, ax = plt.subplots(figsize=(7.2, 3.1))
style(ax)
ax.plot(*xy(diff_val), color=INDIGO, lw=2.2, marker="o", ms=4.5,
        markeredgecolor=BG, markeredgewidth=1, label="differential + MoE (val)")
ax.plot(*xy(dense_val), color=TEAL, lw=2.2, marker="o", ms=4.5,
        markeredgecolor=BG, markeredgewidth=1, label="standard + dense (val)")
# mark the minimum of diff_val (step 1000)
mx, my = diff_val[1]
ax.axvline(mx, color=AMBER, lw=1, ls=(0, (3, 3)))
ax.annotate("best — then memorizes →", (mx, my), xytext=(mx + 130, my - 0.5),
            color=AMBER, fontsize=10, fontweight="bold", va="top")
ax.scatter([mx], [my], color=INDIGO, s=42, zorder=6, edgecolor=BG, linewidth=1.3)
ax.set_xlabel("optimizer step")
ax.set_ylabel("val NLL (nats)")
ax.set_ylim(1.5, 5.8)
ax.set_xlim(0, 5600)
ax.legend(frameon=False, fontsize=10, loc="upper left")
save(fig, "fig04_val_nll.png")

# Fig 05 — auxiliary losses (load-balance + router z), twin y-axis
fig, ax = plt.subplots(figsize=(7.2, 3.1))
style(ax)
ax.plot(*xy(aux_loss), color=TEAL, lw=2.2, label="load-balance aux (left)")
ax.set_xlabel("optimizer step")
ax.set_ylabel("aux loss  (Σ over 6 MoE layers)", color=TEAL)
ax.set_ylim(5.5, 7.2)
ax.set_xlim(0, 5700)
ax.tick_params(axis="y", colors=TEAL)
ax.spines["left"].set_color(TEAL)
ax.axhline(6.0, color=TEAL, lw=1, ls=(0, (2, 3)), alpha=0.6)
ax.text(3400, 6.05, "6.0 = perfectly balanced (1.0 / layer)", color=TEAL,
        fontsize=9, va="bottom")

ax2 = ax.twinx()
ax2.plot(*xy(z_loss), color=AMBER, lw=2.2, label="router z (right)")
ax2.set_ylabel("router z-loss", color=AMBER)
ax2.set_ylim(0, 28)
ax2.tick_params(axis="y", colors=AMBER, length=0)
ax2.spines["right"].set_color(AMBER)
ax2.spines["top"].set_visible(False)
lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=10, loc="upper center")
save(fig, "fig05_aux_losses.png")

print("done")
