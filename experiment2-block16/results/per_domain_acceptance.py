"""Per-domain acceptance: the four canonical arms as lines across datasets, tree budget 64.

Sourced from experiment 3's summary (the only run carrying all four canonical arms
together), so names, colors, and numbers match the other figures. Canonical arm ids,
display names, and the arm config that defines each (exp3 config.methods):

  DFlash             = dflash.chain            (dflash_b16, corrector None, verify chain)
  DSpark             = dspark_b7.chain         (dspark_b7,  corrector None, verify chain)
                        -- chain applies the DSpark markov head intrinsically => markov ON
  DDTree             = ddtree                  (dflash_b16, corrector None, verify "ddtree")
  SparklingTree_b16  = dspark_b16.markov.tree  (dspark_b16 + its markov head + tree)

Colors follow exp3's ARM_COLOR: DFlash family = blues (chain light, tree saturated),
DSpark family = warm (chain light orange, tree red). x-axis = domain, y = mean
acceptance length (tokens/round). Vertical gap labels give the % lift from each line
to the next one above it, per domain (line order can cross between domains).

Usage:
    python per_domain_acceptance.py [path/to/exp3_summary.json]
    # default source: ../../experiment3-timings/results/summary.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Display name -> arm id -> color, in canonical ARM_ORDER (exp3 make_charts.py).
LINES = [
    ("DFlash", "dflash.chain", "#6aa9e9"),                     # light blue (DFlash chain)
    ("DSpark", "dspark_b7.chain", "#f0a15a"),                  # light orange (DSpark chain, markov ON)
    ("DDTree", "ddtree", "#2a78d6"),                           # blue (DFlash-family tree)
    ("SparklingTree_b16", "dspark_b16.markov.tree", "#e34948"),  # red (DSpark tree, block 16)
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e6e3"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

DEFAULT_SRC = Path(__file__).parents[2] / "experiment3-timings" / "results" / "summary.json"
BUDGET = "64"


def chart(res, out):
    """res: {dataset: {arm: {mean_accept, ...}}} for one budget."""
    datasets = list(res)

    ymax = 0.0
    for d in datasets:
        for _, m, _c in LINES:
            e = res.get(d, {}).get(m)
            if e:
                ymax = max(ymax, e["mean_accept"])
    ymax = ymax or 1.0

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    xs = list(range(len(datasets)))

    col = {j: [] for j in range(len(datasets))}   # j -> [(value, color, name)]
    for name, m, color in LINES:
        pts = []
        for j, d in enumerate(datasets):
            e = res.get(d, {}).get(m)
            if e is None:
                continue
            pts.append((j, e["mean_accept"]))
            col[j].append((e["mean_accept"], color, name))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2.4,
                marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=3, label=name)

    # A white halo behind every label so it stays legible where lines cross beneath it.
    halo = dict(boxstyle="square,pad=0.12", fc=SURFACE, ec="none", alpha=0.85)

    # Value labels, to the LEFT of each marker, decluttered bottom-to-top so numbers
    # never overlap however close their markers are (gap labels go to the right).
    min_sep = ymax * 0.05
    for j, items in col.items():
        last = None
        for y, color, _n in sorted(items):
            ty = y if last is None else max(y, last + min_sep)
            last = ty
            ax.text(j - 0.06, ty, f"{y:.1f}", ha="right", va="center",
                    fontsize=9, color=color, fontweight="bold", bbox=halo, zorder=5)

    # Vertical gap labels: % lift from each line to the next one above it, per domain.
    # Arrow sits just right of the markers; the % labels are themselves decluttered
    # bottom-to-top so two small gaps can't overprint into an unreadable blob.
    for j, items in col.items():
        s = sorted(items)
        xg = j + 0.08
        gaps = []
        for lower, upper in zip(s, s[1:]):
            ylo = lower[0]
            yhi, chi = upper[0], upper[1]
            pct = (yhi - ylo) / ylo * 100 if ylo else float("nan")
            ax.annotate("", xy=(xg, yhi), xytext=(xg, ylo),
                        arrowprops=dict(arrowstyle="<->", color=chi, linewidth=1.2,
                                        alpha=0.8, shrinkA=0, shrinkB=0), zorder=4)
            gaps.append(((ylo + yhi) / 2, f"+{pct:.0f}%", chi))
        last = None
        for ymid, txt, chi in gaps:
            ty = ymid if last is None else max(ymid, last + min_sep)
            last = ty
            ax.text(xg + 0.06, ty, txt, ha="left", va="center", fontsize=8.5,
                    color=chi, fontweight="bold", bbox=halo, zorder=5)

    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(xs)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_xlim(-0.5, len(datasets) - 0.4)
    ax.set_ylim(0, ymax * 1.16)
    ax.set_ylabel("mean acceptance length (tokens/round)")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_title("Per-domain acceptance  ·  tree budget 64",
                 fontsize=13.5, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    summary = json.loads(Path(path).read_text())
    res = summary["results"]
    # exp3 nests: results.clean.<budget>.<dataset>.<arm>
    if "clean" in res:
        res = res["clean"]
    if BUDGET in res:
        res = res[BUDGET]
    chart(res, Path(__file__).parent / "per_domain_acceptance.png")


if __name__ == "__main__":
    main()
