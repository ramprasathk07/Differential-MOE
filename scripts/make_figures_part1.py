"""Render the Part 1 figures: the four cells of the 2x2, side by side.

Reads each run's logged history (no transcription, no smoothing):

  model_checkpoints/checkpoints/s_dense/metrics.csv     local, 2900 steps
  model_checkpoints/checkpoints/s_diff/metrics.csv      local, 2900 steps
  model_checkpoints/checkpoints/s_moe/metrics.csv       local, 2900 steps
  docs/blog/runs_export/s_diffmoe_v2/wandb_history.csv  W&B, 2900 steps

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
BLOG_OUT = os.path.join(ROOT, "docs/blog/assets")   # blogs use RELATIVE assets/ links,
#                                                     which resolve here -- mirror every
#                                                     figure or the blogs show stale PNGs
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


def save(fig, fname):
    """Write to the published assets dir AND mirror into docs/blog/assets, which
    is where the blogs' relative image links actually point."""
    fig.savefig(os.path.join(OUT, fname))
    os.makedirs(BLOG_OUT, exist_ok=True)
    import shutil
    shutil.copy2(os.path.join(OUT, fname), os.path.join(BLOG_OUT, fname))


# ---- load the three histories ----------------------------------------------
def load_curves():
    """Local metrics.csv where the run dir came back from Kaggle; the W&B export
    otherwise (neither s_diffmoe session wrote a metrics.csv back). A run with
    neither is simply absent from every figure."""
    curves = {}
    for run in RUNS:
        local = os.path.join(CK, run, "metrics.csv")
        # a rerun exported as <run>_v2 supersedes both the local metrics of the
        # old checkpoint and the old W&B export (s_diffmoe's first session
        # crashed at 2500 AND ran on a contended T4 -- the v2 run is the one
        # every figure should show)
        v2 = os.path.join(EXPORT, run + "_v2", "wandb_history.csv")
        if os.path.exists(v2):
            local = ""            # force the v2 branch below
            wandb = v2
        else:
            wandb = os.path.join(EXPORT, run, "wandb_history.csv")
        if local and os.path.exists(local):
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
    # all four now end at step 2900 within 0.06 nats -- fan the labels out
    # vertically in the same top-to-bottom order as the endpoints. The whole fan
    # is biased upward so the lowest label clears the x tick labels underneath.
    dy = {"s_dense": 30, "s_diff": 23, "s_moe": 11, "s_diffmoe": 2}
    for run, meta in RUNS.items():
        if run not in C:
            continue
        v = C[run]["val"]
        endlabel(ax, v.step.iloc[-1], v.nll.iloc[-1],
                 f"{meta['label']}  {v.nll.iloc[-1]:.3f}", meta["color"], dx=16,
                 dy=dy.get(run, 0))
    titles(ax, "Validation NLL: the four cells, step for step",
           "Lower is better. Measured every 250 steps on the identical held-out slice.")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation NLL (nats/token)")
    # the four direct labels sit to the right of step 2900 and are ~700 steps
    # wide in data units; the limit leaves room for them, the explicit ticks stop
    # at the last real step so no axis label appears underneath one
    ax.set_xlim(0, 3750)
    ax.set_xticks(range(0, 3001, 500))
    ax.legend(frameon=False, loc="upper right", labelcolor=SECOND, fontsize=10.5)
    source(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "fig_p1_val_nll.png")
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
    save(fig, fname)
    plt.close(fig)
    print("wrote", fname)


# ---- fig 2b: the interaction term -- do the two mechanisms compose? ---------
def fig_interaction():
    """The point of running a 2x2 rather than three separate comparisons: with
    all four cells you can ask whether stacking two mechanisms delivers the sum
    of what each delivers alone. The dashed line is that additive prediction;
    the gap between it and the observed Diff-MoE curve is the interaction."""
    need = ("s_dense", "s_diff", "s_moe", "s_diffmoe")
    if any(r not in C for r in need):
        print("skip interaction: needs all four cells, have", list(C))
        return
    base_ = C["s_dense"]["val"].set_index("step")["nll"]
    d_diff = C["s_diff"]["val"].set_index("step")["nll"] - base_
    d_moe = C["s_moe"]["val"].set_index("step")["nll"] - base_
    d_both = C["s_diffmoe"]["val"].set_index("step")["nll"] - base_
    steps = d_both.dropna().index.intersection(d_diff.index).intersection(d_moe.index)
    additive = (d_diff.loc[steps] + d_moe.loc[steps])

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    base(ax)
    ax.axhline(0, color=AXIS, lw=1.4, zorder=2)
    ax.text(60, 0.003, "dense baseline", color=MUTED, fontsize=9.5, va="bottom")

    for run, series in (("s_diff", d_diff), ("s_moe", d_moe), ("s_diffmoe", d_both)):
        s = series.loc[steps]
        ax.plot(steps, s.values, "-", lw=2, color=RUNS[run]["color"],
                label=f"{RUNS[run]['label']} − Dense", zorder=3)
        ax.plot(steps, s.values, "o", ms=5, color=RUNS[run]["color"],
                mec=SURFACE, mew=1.5, zorder=4)
    # the prediction is not an entity, so it wears neutral ink, not a series hue
    ax.plot(steps, additive.values, "--", lw=2, color=MUTED, zorder=3,
            label="additive prediction (sum of the two alone)")

    gap = d_both.loc[steps].iloc[-1] - additive.iloc[-1]
    x_end = steps[-1]
    ax.annotate("", xy=(x_end, d_both.loc[steps].iloc[-1]), xytext=(x_end, additive.iloc[-1]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.annotate(f"shortfall {gap:+.3f}", (x_end, (d_both.loc[steps].iloc[-1] + additive.iloc[-1]) / 2),
                textcoords="offset points", xytext=(12, 0), color=INK,
                fontsize=10.5, fontweight="600", va="center")

    titles(ax, "Do the two mechanisms compose?",
           "Each mechanism's gain over dense, and what stacking them actually delivers "
           "versus the sum of the parts.")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Δ validation NLL vs dense (nats)")
    # room for the shortfall callout, which hangs off the last point
    ax.set_xlim(0, 3500)
    ax.set_xticks(range(0, 3001, 500))
    ax.legend(frameon=False, loc="lower left", labelcolor=SECOND, fontsize=10)
    source(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "fig_p1_interaction.png")
    plt.close(fig)
    print("wrote fig_p1_interaction.png")


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
    # All four finish within ~3 h and ~0.04 nats of each other, so endpoint labels
    # would overplot each other and the neighbouring curves. The legend already
    # carries name + throughput; a marker alone is enough to find each end.
    for run, meta in RUNS.items():
        if run not in C:
            continue
        v = C[run]["val"]
        hours = v.step * TOK_PER_STEP / C[run]["tok_s"] / 3600
        ax.plot([hours.iloc[-1]], [v.nll.iloc[-1]], "o", ms=8, color=meta["color"],
                mec=SURFACE, mew=2, zorder=5)
    titles(ax, "The same four runs, priced in GPU-hours",
           "Step-for-step is not what a free-tier budget buys. This is quality per hour of 2×T4.")
    ax.set_xlabel("GPU-hours (2×T4, from each run's measured throughput)")
    ax.set_ylabel("validation NLL (nats/token)")
    # the slowest run finishes at 10.3 h; anything beyond ~11 is dead space that
    # flattens every curve and hides the gaps this figure exists to show
    ax.set_xlim(0, 11)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.legend(frameon=False, loc="upper right", labelcolor=SECOND, fontsize=10.5)
    source(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "fig_p1_walltime.png")
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
    # All four end at 2900 in two tight pairs (dense/diff-dense ~2.71, the two
    # MoE cells ~2.64), and 0.009 nats is ~1pt of vertical space -- so the labels
    # are fanned apart by hand, in the same top-to-bottom order as the endpoints.
    dy = {"s_dense": 20, "s_diff": 6, "s_diffmoe": -8, "s_moe": -22}
    for run, meta in RUNS.items():
        if run not in C:
            continue
        t = C[run]["train"]
        endlabel(ax, t.step.iloc[-1], t.loss.iloc[-1],
                 f"{meta['label']}  {t.loss.iloc[-1]:.3f}", meta["color"],
                 dx=16, dy=dy.get(run, 0))
    titles(ax, "Training cross-entropy",
           "Every logged step, unsmoothed. All four fall the way they should; the gaps are small and real.")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training loss (nats/token)")
    ax.set_xlim(0, 3750)
    ax.set_ylim(2.4, 5.0)
    ax.set_xticks(range(0, 3001, 500))
    ax.legend(frameon=False, loc="upper right", labelcolor=SECOND, fontsize=10.5)
    source(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "fig_p1_train_loss.png")
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
           "Held-out test NLL per domain, identical windows for every model. Lower is better.")
    ax.set_xlabel("test NLL (nats/token)")
    # domains are sorted easiest-first and the y-axis is inverted, so the top-right
    # of the plot is empty -- the only place a legend does not sit on a bar
    ax.set_xlim(0, 4.35)
    ax.legend(frameon=False, loc="upper right", labelcolor=SECOND, fontsize=10.5)
    source(fig, "source: scripts/eval_all.py · held-out BabyLM test split · RTX 3060, fp32")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "fig_p1_domain_nll.png")
    plt.close(fig)
    print("wrote fig_p1_domain_nll.png")


# ---- fig 6: does the advantage grow with context position? ------------------
def fig_position():
    """The sharpest available test of differential attention's actual claim.
    If the mechanism cancels attention that leaks onto irrelevant context, its
    benefit must ACCUMULATE with context -- i.e. grow with position index. A
    capacity mechanism like MoE has no such reason to, which makes this a
    discriminating measurement rather than a descriptive one."""
    p = os.path.join(EXPORT, "eval_deep.json")
    if not os.path.exists(p):
        print("skip position: no eval_deep.json (run scripts/eval_deep.py)")
        return
    with open(p) as f:
        deep = json.load(f)
    if "s_dense" not in deep:
        print("skip position: no dense baseline in eval_deep.json")
        return
    dense_pos = np.array(deep["s_dense"]["pos_bucket_nll"])
    n = len(dense_pos)
    pos = np.arange(n) * (512 // n) + (512 // n) // 2

    # Two panels because the difference view alone can only show three series --
    # dense IS the zero line there, which reads as a missing model. The left panel
    # puts all four on an absolute axis; the right panel is the comparison that
    # actually tests the mechanism.
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.6))
    for ax in (axA, axB):
        base(ax)
        ax.set_xlim(0, 512)
        ax.xaxis.set_major_locator(MultipleLocator(128))
        ax.set_xlabel("position in context window (token index)")

    # ---- panel A: absolute NLL, all four cells --------------------------------
    for run, meta in RUNS.items():
        if run not in deep:
            continue
        p = np.array(deep[run]["pos_bucket_nll"])
        axA.plot(pos, p, "-", lw=2, color=meta["color"], label=meta["label"], zorder=3)
        axA.plot(pos, p, "o", ms=3.5, color=meta["color"], mec=SURFACE, mew=1, zorder=4)
    axA.set_ylabel("held-out NLL (nats/token)")
    axA.set_title("All four cells, absolute", color=INK, fontsize=12,
                  fontweight="600", loc="left", pad=10)
    axA.legend(frameon=False, loc="upper right", labelcolor=SECOND, fontsize=10)

    # ---- panel B: the same data as a difference against dense -----------------
    # dense is drawn explicitly as its own flat zero line, in its own hue, so the
    # baseline is a labelled series rather than an unexplained axis rule
    axB.plot(pos, np.zeros_like(pos, dtype=float), "-", lw=2, color=RUNS["s_dense"]["color"],
             zorder=3, label="Dense — baseline")
    for run in ("s_moe", "s_diffmoe", "s_diff"):
        if run not in deep:
            continue
        d = np.array(deep[run]["pos_bucket_nll"]) - dense_pos
        r = np.corrcoef(pos, d)[0, 1]
        axB.plot(pos, d, "-", lw=2, color=RUNS[run]["color"], zorder=3,
                 label=f"{RUNS[run]['label']} − Dense   (r = {r:+.2f})")
        axB.plot(pos, d, "o", ms=3.5, color=RUNS[run]["color"], mec=SURFACE, mew=1, zorder=4)
    axB.set_ylabel("Δ NLL vs dense (nats)")
    axB.set_title("Difference against the dense baseline", color=INK, fontsize=12,
                  fontweight="600", loc="left", pad=10)
    # every curve is at or below the baseline past position ~64, so the band just
    # under the title is the only region no series crosses
    axB.set_ylim(top=0.062)
    axB.legend(frameon=False, loc="upper right", labelcolor=SECOND, fontsize=10,
               ncol=2, columnspacing=1.4, handlelength=1.6)

    fig.suptitle("Differential attention's advantage grows with context; MoE's barely does",
                 color=INK, fontsize=14, fontweight="600", x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.928,
             "Held-out NLL resolved by position in the 512-token window. Every model saw "
             "identical windows. Lower is better; below zero beats dense.",
             color=SECOND, fontsize=10.5, ha="left")
    source(fig, "source: scripts/eval_deep.py · 1,600 held-out windows · identical windows per model")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    save(fig, "fig_p2_position.png")
    plt.close(fig)
    print("wrote fig_p2_position.png")


if __name__ == "__main__":
    fig_val_nll()
    fig_delta()                       # Part 1's pairing: Diff-MoE vs MoE
    fig_delta(baseline="s_dense", others=("s_diff", "s_moe", "s_diffmoe"),
              fname="fig_p1_delta_vs_dense.png",
              title="How much each mechanism buys over a plain dense transformer",
              subtitle=("Validation NLL minus the dense run at the same step, "
                        "at identical active-parameter cost. Below zero = better than dense."))
    fig_interaction()
    fig_walltime()
    fig_train_loss()
    fig_domain()
    fig_position()
