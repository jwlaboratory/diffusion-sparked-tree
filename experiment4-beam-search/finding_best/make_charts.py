"""Charts for the beam width-schedule experiment.

Run:  uv run --python 3.12 --with matplotlib --with numpy python make_charts.py

Outputs (light mode, PNG, into results/):
  schedule_acceptance.png   acceptance by schedule, small multiples (budget x dataset)
  depth_survival.png        P(round accepts >= depth d) per dataset -- why deep slots matter
  branch_distribution.png   (only if results/traces/ exists) accepted-node depth/slot
                            measurement: top-1 hit rate by depth + slots for 95% coverage
                            vs what each schedule actually allocates
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RES = HERE / "results"

# ---- reference palette (light mode) ---------------------------------------- #
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": False, "font.size": 10,
})

ARMS = ["beam.geo50", "beam.geo60", "beam.geo75", "beam.flat",
        "beam.invgeo75", "beam.invgeo50", "beam.depth8", "beam.depth4"]
LABELS = {
    "beam.geo50": "geo 0.50 (front)", "beam.geo60": "geo 0.60 (front)",
    "beam.geo75": "geo 0.75 (front)", "beam.flat": "flat",
    "beam.invgeo75": "inv-geo 0.75 (back)", "beam.invgeo50": "inv-geo 0.50 (back)",
    "beam.depth8": "flat, depth 8", "beam.depth4": "flat, depth 4",
}
# Color by family (identity), fixed order: front / flat / back / truncated.
FAMILY = {
    "beam.geo50": BLUE, "beam.geo60": BLUE, "beam.geo75": BLUE,
    "beam.flat": AQUA,
    "beam.invgeo75": YELLOW, "beam.invgeo50": YELLOW,
    "beam.depth8": GREEN, "beam.depth4": GREEN,
}
DATASETS = ["gsm8k", "humaneval", "mt-bench"]
BUDGETS = ["64", "256"]


def strip_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------------------- #
# Chart 1: acceptance by schedule, budget x dataset small multiples             #
# ---------------------------------------------------------------------------- #

def chart_schedules():
    summary = json.load(open(RES / "summary.json"))
    clean = summary["results"]["clean"]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), sharey=True)
    ypos = np.arange(len(ARMS))[::-1]

    for r, bkey in enumerate(BUDGETS):
        for c, ds in enumerate(DATASETS):
            ax = axes[r][c]
            vals = [clean[bkey][ds][a]["mean_accept"] for a in ARMS]
            ceiling = clean[bkey][ds]["bestfirst.ref"]["mean_accept"]
            colors = [FAMILY[a] for a in ARMS]
            ax.barh(ypos, vals, height=0.55, color=colors, zorder=3)

            best = int(np.argmax(vals))
            ax.text(vals[best] + 0.15, ypos[best], f"{vals[best]:.2f}",
                    va="center", ha="left", fontsize=9, color=INK, fontweight="bold")

            ax.axvline(ceiling, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=4)
            ax.text(ceiling + 0.15, -0.62, f"best-first {ceiling:.2f}",
                    fontsize=7.5, color=MUTED, ha="left", va="center", clip_on=False)
            ax.set_ylim(-1.0, len(ARMS) - 0.4)

            ax.set_xlim(0, 14.2)
            ax.set_yticks(ypos)
            ax.set_yticklabels([LABELS[a] for a in ARMS] if c == 0 else [])
            ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
            ax.set_axisbelow(True)
            strip_axes(ax)
            if r == 0:
                ax.set_title(ds, fontsize=11, color=INK, pad=8)
            if c == 0:
                ax.set_ylabel(f"budget {bkey}", fontsize=10, color=INK2)
            if r == 1:
                ax.set_xlabel("mean accepted tokens / round", fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in (BLUE, AQUA, YELLOW, GREEN)]
    fig.legend(handles, ["front-loaded (geometric)", "flat", "back-loaded (inv-geometric)",
                         "depth-truncated"],
               loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Beam width schedules: acceptance per round (higher = fewer verify rounds)",
                 fontsize=13, color=INK, y=0.99)
    fig.text(0.5, 0.945, "DSpark-b16 + markov head, H100, 4 samples x 512 tokens, temp 0. "
             "Dashed line = adaptive best-first ceiling.",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(RES / "schedule_acceptance.png", dpi=170)
    print("wrote", RES / "schedule_acceptance.png")


# ---------------------------------------------------------------------------- #
# Chart 2: survival curve -- how often is depth d even reached?                 #
# ---------------------------------------------------------------------------- #

def chart_survival():
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = {"gsm8k": BLUE, "humaneval": AQUA, "mt-bench": YELLOW}
    max_d = 17

    for ds in DATASETS:
        recs = json.load(open(RES / "units" / f"b256_{ds}.json"))["clean"]["bestfirst.ref"]
        accepts = np.array([a for r in recs for a in r["acceptance_lengths"]])
        depths = np.arange(1, max_d + 1)
        surv = [(accepts >= d).mean() * 100 for d in depths]
        ax.plot(depths, surv, color=colors[ds], lw=2, zorder=3)
        ax.scatter(depths[-1], surv[-1], s=18, color=colors[ds], zorder=4)
        ax.text(depths[-1] + 0.25, surv[-1], f"{ds}  {surv[-1]:.0f}%",
                va="center", fontsize=9, color=colors[ds])

    ax.set_xlim(1, max_d + 3.6)
    ax.set_ylim(0, 102)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    strip_axes(ax)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("depth d (accepted tokens in the round)", fontsize=9)
    ax.set_ylabel("% of rounds accepting >= d", fontsize=9)
    ax.legend(handles=[plt.Line2D([], [], color=colors[d], lw=2, label=d) for d in DATASETS],
              frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Deep depths are reached often enough to matter\n"
                 "(best-first @ budget 256 -- rounds still alive at each depth)",
                 fontsize=12, color=INK, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(RES / "depth_survival.png", dpi=170)
    print("wrote", RES / "depth_survival.png")


# ---------------------------------------------------------------------------- #
# Chart 3: the branch-distribution measurement (needs results/traces/)          #
# ---------------------------------------------------------------------------- #

def slot_of_accepted(parents, accepted):
    """Slot rank of each accepted non-root node.

    Best-first materializes a parent's children strictly in rank order, so the
    slot of node i is the number of earlier nodes sharing its parent."""
    out = []
    for i in accepted:
        if i == 0:
            continue
        p = parents[i]
        out.append(sum(1 for j in range(1, i) if parents[j] == p))
    return out


def chart_branch_distribution():
    tdir = RES / "traces"
    files = sorted(tdir.glob("*.json")) if tdir.exists() else []
    if not files:
        print("no traces yet -- skipping branch_distribution.png")
        return

    # per-depth: total accepts, top-slot hits, slot values
    per_depth: dict[int, list[int]] = {}
    for f in files:
        data = json.load(open(f))
        for rt in data["round_trees"]:
            parents = rt["tree"]["parents"]
            depths = rt["tree"]["node_depths"]
            accepted = rt["accepted_indices"]
            for i in accepted:
                if i == 0:
                    continue
                d = depths[i - 1]
                p = parents[i]
                slot = sum(1 for j in range(1, i) if parents[j] == p)
                per_depth.setdefault(d, []).append(slot)

    depths = sorted(per_depth)
    top1 = [100 * np.mean([s == 0 for s in per_depth[d]]) for d in depths]
    p95 = [int(np.percentile(per_depth[d], 95)) + 1 for d in depths]
    n = [len(per_depth[d]) for d in depths]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    ax.plot(depths, top1, color=BLUE, lw=2, zorder=3)
    for d, v in zip(depths, top1):
        if d in (depths[0], 4, 8, 12, depths[-1]):
            ax.text(d, v + 3.5, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK2)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    strip_axes(ax)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("depth", fontsize=9)
    ax.set_ylabel("% of accepted nodes that were the top pick", fontsize=9)
    ax.set_title("Top-1 hit rate declines with depth --\nthe drafter never stops needing alternatives",
                 fontsize=12, color=INK, loc="left", pad=8)

    ax = axes[1]
    width = 0.38
    x = np.array(depths, dtype=float)
    ax.bar(x - width / 2, p95, width=width, color=BLUE, zorder=3,
           label="slots needed for 95% coverage (measured)")
    flat = {d: 4 for d in range(1, 17)}
    geo = {}
    for i, w in enumerate([32, 16, 8, 4, 2, 1, 1]):
        geo[i + 1] = w
    ax.bar(x + width / 2, [flat.get(d, 0) for d in depths], width=width, color=AQUA,
           zorder=3, label="flat allocates (budget 64)")
    ax.step(list(x) + [x[-1] + 1], [geo.get(d, 0) for d in depths] + [0], where="mid",
            color=MUTED, lw=1.6, zorder=4, label="geo 0.50 allocates")
    ax.annotate("geo 0.50: 32 slots where 3 are needed", xy=(1.15, 30), xytext=(2.6, 19.5),
                fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("...and 0 slots past depth 7,\nwhere 2-6 are still needed", xy=(11, 1), xytext=(8.6, 12),
                fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    strip_axes(ax)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("depth", fontsize=9)
    ax.set_ylabel("branch slots", fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Slots needed never decay toward 1 --\nevery geometric schedule assumes they do",
                 fontsize=12, color=INK, loc="left", pad=8)

    fig.suptitle("Measured branch distribution: every accepted node's (depth, slot), "
                 f"best-first @ budget 256, {sum(n)} accepted nodes",
                 fontsize=12.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RES / "branch_distribution.png", dpi=170)
    print("wrote", RES / "branch_distribution.png")


if __name__ == "__main__":
    chart_schedules()
    chart_survival()
    chart_branch_distribution()
