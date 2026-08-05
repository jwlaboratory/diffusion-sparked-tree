"""Regenerate the "SparklingTree Mogs" headline bar chart — reproducibly.

The original figure was hand-authored (no committed script). This rebuilds it
straight from results_detailed.json so the numbers are auditable, and ADDS the
control bar that was missing: DSpark b16 + markov + NO tree (`dspark_b16.chain`).

Metric = per-round mean acceptance length (flat mean of each method's `lengths`
array), averaged over gsm8k / humaneval / mt-bench, at tree_budget 64 — identical
to how the original four bars were computed.

Why the new bar matters (all five bars share one metric, one run):
  * b7 chain (5.32) -> b16 chain (5.90): extending the draft horizon on a CHAIN
    barely helps -- it stays tied with DFlash (5.40).
  * b16 chain (5.90) -> SparklingTree b16 (8.21) = +39%, and it's the SAME
    checkpoint -- so the gain is the TREE, not the block size and not training data.

Usage: python make_mogs_chart.py [path/to/results_detailed.json]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = ["gsm8k", "humaneval", "mt-bench"]

# (display label, method key in results_detailed.json, bar color)
# Grouped BY BLOCK SIZE so each chain->tree comparison is between ADJACENT bars:
# baseline, then the b7 pair (chain, tree), then the b16 pair (chain, tree).
# Chains use a muted tone, trees the vivid tone, so "add the tree" reads by hue.
BARS = [
    ("DFlash",                             "dflash.chain",           "#4c3fa0"),  # indigo baseline
    ("DSpark b7\n+ markov",                "dspark.chain",           "#8fd3b6"),  # b7 chain (muted green)
    ("SparklingTree b7\n(markov + tree)",  "dspark.markov.tree",     "#f0a500"),  # b7 tree (orange)
    ("DSpark b16\n+ markov",               "dspark_b16.chain",       "#9fc7ea"),  # b16 chain (muted blue)
    ("SparklingTree b16\n(markov + tree)", "dspark_b16.markov.tree", "#2f7fd1"),  # b16 tree (blue)
]

# adjacent chain->tree comparisons to annotate: (chain_idx, tree_idx, tag)
COMPARISONS = [(1, 2, "b7: + tree"), (3, 4, "b16: + tree")]


def mean_accept(pd, key):
    per = [sum(pd[t]["methods"][key]["lengths"]) / len(pd[t]["methods"][key]["lengths"])
           for t in TASKS]
    return sum(per) / len(per)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("results_detailed.json")
    pd = json.loads(src.read_text())["per_dataset"]

    labels = [b[0] for b in BARS]
    colors = [b[2] for b in BARS]
    vals = [mean_accept(pd, b[1]) for b in BARS]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = range(len(vals))
    bars = ax.bar(x, vals, color=colors, width=0.68, zorder=3)

    # value labels
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.12, f"{v:.2f}", ha="center", va="bottom",
                fontsize=15, fontweight="bold", color="#222")

    dflash = vals[0]

    def bracket(i, j, tag):
        """A short bracket spanning ADJACENT bars i,j with the % change + a tag,
        so it is unambiguous which two bars are being compared."""
        top = max(vals[i], vals[j])
        y = top + 0.55
        # square bracket: up from each bar, across the top
        ax.plot([i, i, j, j], [vals[i] + 0.35, y, y, vals[j] + 0.35],
                color="#666", lw=1.6, zorder=4)
        pct = (vals[j] - vals[i]) / vals[i] * 100
        ax.text((i + j) / 2, y + 0.08, f"{tag}   {pct:+.0f}%", ha="center", va="bottom",
                fontsize=12.5, fontweight="bold", color="#444")

    for i, j, tag in COMPARISONS:
        bracket(i, j, tag)

    # +52% vs DFlash callout on the winner, lifted well ABOVE the b16 bracket
    # label so the two never crowd (single line, its own vertical band).
    win = vals[4]
    ax.text(4, win + 2.35, f"+{(win - dflash) / dflash * 100:.0f}% vs DFlash",
            ha="center", va="bottom", fontsize=13.5, fontweight="bold", color="#333")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("mean accepted tokens per verifier call", fontsize=12)
    ax.set_ylim(0, max(vals) + 3.6)
    ax.set_title("SparklingTree Mogs", fontsize=22, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.03, "mean acceptance, avg of gsm8k / humaneval / mt-bench, tree budget 64",
            transform=ax.transAxes, fontsize=12, color="#666")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = src.with_name("mogs_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for l, v in zip(labels, vals):
        print(f"  {l.replace(chr(10), ' '):32s} {v:.2f}")


if __name__ == "__main__":
    main()
