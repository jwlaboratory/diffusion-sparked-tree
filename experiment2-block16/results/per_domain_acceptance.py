"""Per-domain acceptance: three methods as lines across datasets, faceted by tree budget.

Three arms tell the progression story on each domain (dataset):
  DSpark             = dspark_b7.tree          (base drafter + tree, markov OFF)
  DDTree             = dspark_b7.markov.tree   (add the markov corrector, block 7)
  SparkingTree_b16   = dspark_b16.markov.tree  (extend the draft horizon to block 16)

x-axis = domain (gsm8k / humaneval / mt-bench), y = mean acceptance length
(tokens accepted per verifier call). One panel per tree budget so the same
domain can be compared across budgets. Colors reuse the repo's validated
categorical palette.

Usage:
    python per_domain_acceptance.py [path/to/summary.json]   # default ./summary.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Display name -> method key in summary.json, in the intended draw/legend order.
LINES = [
    ("DSpark", "dspark_b7.tree", "#4a3aa7"),            # violet  -- base drafter
    ("DDTree", "dspark_b7.markov.tree", "#2a78d6"),     # blue    -- + markov corrector
    ("SparkingTree_b16", "dspark_b16.markov.tree", "#e34948"),  # red -- + block-16 horizon
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


def _budgets(results):
    return sorted(results, key=lambda b: int(b))


def chart(summary, out):
    results = summary["results"]
    budgets = _budgets(results)
    # Datasets: union across budgets, first-seen order.
    datasets = []
    for bk in budgets:
        for d in results[bk]:
            if d not in datasets:
                datasets.append(d)

    # Shared y so the panels are comparable across budgets.
    ymax = 0.0
    for bk in budgets:
        for d in datasets:
            for _, m, _c in LINES:
                e = results[bk].get(d, {}).get(m)
                if e:
                    ymax = max(ymax, e["mean_accept"])
    ymax = ymax or 1.0

    ncols = len(budgets)
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 5.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    xs = list(range(len(datasets)))

    for panel, (bk, ax) in enumerate(zip(budgets, axes)):
        res = results[bk]
        # Collect every method's value at each x so labels can dodge collisions.
        col_labels = {j: [] for j in range(len(datasets))}
        for name, m, color in LINES:
            pts = []
            for j, d in enumerate(datasets):
                e = res.get(d, {}).get(m)
                if e is None:
                    continue
                pts.append((j, e["mean_accept"]))
                col_labels[j].append((e["mean_accept"], color))
            gx = [p[0] for p in pts]
            gy = [p[1] for p in pts]
            ax.plot(gx, gy, color=color, linewidth=2.4, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, label=name)

        # Place value labels per column; when two points nearly coincide, stack one
        # above / one below the marker so the numbers never overprint.
        thresh = ymax * 0.06
        for j, items in col_labels.items():
            items = sorted(items)  # by value, ascending
            for i, (y, color) in enumerate(items):
                below = any(abs(y - y2) < thresh and y2 < y for y2, _ in items[:i])
                if below:
                    ax.text(j, y - ymax * 0.018, f"{y:.1f}", ha="center", va="top",
                            fontsize=8.5, color=color, fontweight="bold")
                else:
                    ax.text(j, y + ymax * 0.018, f"{y:.1f}", ha="center", va="bottom",
                            fontsize=8.5, color=color, fontweight="bold")

        ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(xs)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.set_xlim(-0.35, len(datasets) - 0.65)
        ax.set_ylim(0, ymax * 1.16)
        if panel == 0:
            ax.set_ylabel("mean acceptance length (tokens/round)")
        ax.set_title(f"tree budget {bk}", fontsize=12, fontweight="bold", loc="left")

    axes[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.suptitle("Per-domain acceptance", fontsize=14, fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "summary.json"
    summary = json.loads(Path(path).read_text())
    chart(summary, Path(path).parent / "per_domain_acceptance.png")


if __name__ == "__main__":
    main()
