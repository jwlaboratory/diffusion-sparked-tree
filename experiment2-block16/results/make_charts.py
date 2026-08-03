"""Generate charts from an Experiment 2 (block-size) summary.json.

Reads a summary produced by DDTree/run_experiment.py (via modal_benchmark.py) and
writes PNGs next to it:

  block_size.png          headline: mean acceptance length, block 7 vs 16, per arm
  acceptance_by_method.png mean acceptance length, all methods, grouped by dataset
  per_depth_accept.png    conditional accept rate vs tree depth (b7 ends at its block horizon)

The comparison axis here is block size (draft horizon), not exp1's six methods:
two block sizes (7, 16) x corrector on/off. Color encodes that -- a cool hue family
for block-7, a warm one for block-16, so the block-size split reads at a glance;
within a family the plain-tree control is the lighter shade of the markov headline.
The hues are exp1's validated categorical set, kept for cross-experiment consistency.

Usage:
    python make_charts.py [path/to/summary.json]      # default: ./summary.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (light mode), a subset of exp1's. Block size is the
# hue family (cool = b7, warm = b16); the markov headline arm is the saturated shade,
# the markov-off .tree control the paler one. A legend + a markov-solid / tree-dashed
# linestyle carry identity alongside color (relieves the one sub-3:1 slot, the b16 yellow).
METHOD_COLOR = {
    "dspark_b7.tree": "#4a3aa7",          # violet  (b7 control)
    "dspark_b7.markov.tree": "#2a78d6",   # blue    (b7 headline)
    "dspark_b16.tree": "#eda100",         # yellow  (b16 control)
    "dspark_b16.markov.tree": "#e34948",  # red     (b16 headline)
}
# Representative per-block hues for the dumbbell endpoints (small vs large block).
BLOCK_COLOR = {"small": "#2a78d6", "large": "#e34948"}  # blue = b7, red = b16
POS = "#008300"   # positive / a longer horizon helps
NEG = "#e34948"   # negative
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


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def chart_block_size(summary, out):
    """Headline: for each arm, a dumbbell from b7 to b16 mean acceptance length.

    A dumbbell is the right form for a before->after per item: two dots (block 7,
    block 16) connected, one row per arm, so the rise from the longer horizon is
    the thing the eye follows. Averaged across datasets."""
    bs = summary.get("block_size") or {}
    if not bs:
        return
    # Stable order: the markov headline arm on top, its control below.
    order = [a for a in ("markov.tree", "tree") if a in bs] + [a for a in bs if a not in ("markov.tree", "tree")]
    rows = []
    for arm in order:
        o = bs[arm]["overall"]
        rows.append((arm, bs[arm]["small_block"], bs[arm]["large_block"],
                     o["small"], o["large"], o["delta"], o["pct_change"]))

    lo = min(min(r[3], r[4]) for r in rows); hi = max(max(r[3], r[4]) for r in rows)
    span = hi - lo or 1.0

    fig, ax = plt.subplots(figsize=(8.5, 1.15 * len(rows) + 1.8))
    ys = list(range(len(rows)))[::-1]  # first arm at the top
    for y, (arm, sb, lb, s, l, delta, pct) in zip(ys, rows):
        ax.plot([s, l], [y, y], color=INK2, linewidth=2.5, zorder=2,
                solid_capstyle="round")
        ax.scatter([s], [y], s=150, color=BLOCK_COLOR["small"], zorder=3,
                   edgecolor=SURFACE, linewidth=2)
        ax.scatter([l], [y], s=150, color=BLOCK_COLOR["large"], zorder=3,
                   edgecolor=SURFACE, linewidth=2)
        # Endpoint value labels: centered above each dot when they are far enough
        # apart, but pushed outward (left value left, right value right) when the two
        # blocks land near-equal, or the two numbers overprint into mush.
        if abs(l - s) < span * 0.10:
            (xl, vl), (xr, vr) = sorted([(s, s), (l, l)])
            ax.text(xl - span * 0.015, y + 0.22, f"{vl:.2f}", ha="right", va="bottom", fontsize=9, color=INK2)
            ax.text(xr + span * 0.015, y + 0.22, f"{vr:.2f}", ha="left", va="bottom", fontsize=9, color=INK2)
        else:
            ax.text(s, y + 0.22, f"{s:.2f}", ha="center", va="bottom", fontsize=9, color=INK2)
            ax.text(l, y + 0.22, f"{l:.2f}", ha="center", va="bottom", fontsize=9, color=INK2)
        tail = "" if pct is None else f"  ({pct:+.0f}%)"
        ax.text(l + 0.04, y - 0.24, f"+{delta:.2f} tok/round{tail}" if delta >= 0
                else f"{delta:.2f} tok/round{tail}", ha="left", va="top",
                fontsize=10, fontweight="bold", color=(POS if delta >= 0 else NEG))
        ax.text(min(s, l) - 0.06, y, arm, ha="right", va="center", fontsize=11,
                fontweight="bold", color=INK)

    # Legend by block size (identity is not color-alone: labeled dots + this legend).
    b7 = rows[0][1]; b16 = rows[0][2]
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=BLOCK_COLOR["small"], markeredgecolor=SURFACE, label=f"block {b7}"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=BLOCK_COLOR["large"], markeredgecolor=SURFACE, label=f"block {b16}"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="lower right")

    _style(ax); ax.grid(axis="y", linewidth=0)
    ax.set_yticks([]); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(lo - span * 0.28, hi + span * 0.22)
    ax.set_xlabel("mean acceptance length (tokens per verifier call)")
    # Title states what was measured, derived from the headline arm -- never a
    # hardcoded direction. This chart gets screenshotted into writeups; if b16 comes
    # out flat or worse, the headline has to say so.
    head = bs.get("markov.tree") or bs[order[0]]
    pct = head["overall"].get("pct_change")
    if pct is None:
        verdict = "block 7 vs 16"
    elif abs(pct) < 1.0:
        verdict = f"block {b16} is level with block {b7} ({pct:+.1f}%)"
    else:
        verdict = (f"block {b16} {'raises' if pct > 0 else 'lowers'} acceptance length "
                   f"by {abs(pct):.1f}% vs block {b7}")
    ax.set_title(f"Longer draft horizon: {verdict}",
                 fontsize=13, fontweight="bold", loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_acceptance(summary, out):
    results = summary["results"]
    datasets = list(results)
    methods = [m for m in METHOD_COLOR if any(m in results[d] for d in datasets)]
    n = len(methods)
    group_w = 0.82
    bw = group_w / n
    fig, ax = plt.subplots(figsize=(1.9 * len(datasets) + 3.5, 5))
    ymax = 0.0
    for i, m in enumerate(methods):
        xs, ys = [], []
        for j, d in enumerate(datasets):
            e = results[d].get(m)
            if e is None:
                continue
            x = j - group_w / 2 + bw * (i + 0.5)
            xs.append(x); ys.append(e["mean_accept"])
        ymax = max([ymax] + ys)
        ax.bar(xs, ys, width=bw, color=METHOD_COLOR[m], label=m,
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + ymax * 0.012, f"{y:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    _style(ax)
    ax.set_ylim(0, ymax * 1.24)  # headroom so value labels clear the legend
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets)
    ax.set_ylabel("mean acceptance length (tokens/round)")
    ax.set_title("Acceptance length by method", fontsize=13, fontweight="bold", loc="left")
    ax.legend(ncol=2, frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_per_depth(summary, out):
    """Conditional accept rate vs tree depth, one line per method.

    The point of the experiment: the block-7 tree cannot reach past depth 7, so its
    curves stop there (rate = None once no round reaches that depth); the block-16
    curves keep going. The x-axis is forced out to depth_report_limit so the b16
    tail is never truncated."""
    results = summary["results"]
    datasets = list(results)
    cfg = summary.get("config", {})
    limit = cfg.get("depth_report_limit", 16)
    methods = [m for m in METHOD_COLOR
               if any("per_depth_accept" in results[d].get(m, {}) for d in datasets)]
    if not methods:
        return
    # The smaller block size is the horizon the b7 tree cannot draft past.
    blocks = [b.get("block_size") for b in (cfg.get("backbones") or {}).values()
              if b.get("block_size")]
    small_block = min(blocks) if blocks else None

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    # Identity is carried by the legend + a markov-solid / tree-dashed linestyle. No
    # direct end-labels: the two b7 arms share a horizon and the two b16 arms can end
    # at near-identical rates, so end-labels overprint into mush -- the legend is the
    # robust choice for any real data (and avoids doubling the identity channel).
    for m in methods:
        depth_vals = {}
        for d in datasets:
            pd = results[d].get(m, {}).get("per_depth_accept") or {}
            for depth, rate in pd.items():
                if rate is None:
                    continue
                depth_vals.setdefault(int(depth), []).append(rate)
        if not depth_vals:
            continue
        depths = sorted(depth_vals)
        means = [sum(depth_vals[dd]) / len(depth_vals[dd]) for dd in depths]
        # Secondary encoding: markov headline arms solid, .tree controls dashed.
        style = "--" if m.endswith(".tree") and ".markov" not in m else "-"
        ax.plot(depths, means, color=METHOD_COLOR[m], linewidth=2.2, linestyle=style,
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=3, label=m)

    if small_block is not None:
        # Structural marker, not a claim: a block-N drafter cannot draft past depth N,
        # so its curves end here regardless of how the numbers come out.
        ax.axvline(small_block, color=INK2, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
        ax.text(small_block - 0.15, 1.0, f"block-{small_block} horizon ",
                ha="right", va="top", fontsize=9, color=INK2)

    _style(ax)
    ax.set_xlabel("tree depth (draft position within a round)")
    ax.set_ylabel("conditional accept rate  P(reach d+1 | reach d)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0.4, limit + 0.8)  # snug right margin -- no end-labels to clear now
    ax.set_xticks(range(1, limit + 1))
    # Descriptive, not conclusion-shaped: the b7 horizon cutoff is marked on the
    # axis as a structural fact, and the curves are left to say whether the extra
    # depth is actually accepted.
    ax.set_title("Conditional accept rate by tree depth",
                 fontsize=13, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "summary.json"
    summary = json.loads(Path(path).read_text())
    outdir = Path(path).parent
    chart_block_size(summary, outdir / "block_size.png")
    chart_acceptance(summary, outdir / "acceptance_by_method.png")
    chart_per_depth(summary, outdir / "per_depth_accept.png")


if __name__ == "__main__":
    main()
