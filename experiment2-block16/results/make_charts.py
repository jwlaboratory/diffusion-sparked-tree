"""Generate charts from an Experiment 2 (block-size x tree-budget) summary.json.

Reads a summary produced by DDTree/run_experiment.py (via modal_benchmark.py) and
writes PNGs next to it:

  block_size.png          headline: mean acceptance length, block 7 vs 16, per arm, faceted by budget
  block_size_by_dataset.png  the b7->b16 acceptance delta split per dataset (signed bars), both arms, faceted by budget
  acceptance_by_method.png mean acceptance length, all methods, grouped by dataset, faceted by budget
  per_depth_accept.png    conditional accept rate vs tree depth (b7 ends at its block horizon), faceted by budget

Two comparison axes, not exp1's six methods: **block size** (draft horizon, 7 vs 16,
corrector on/off) crossed with **tree budget** (nodes per round, e.g. 64 vs 256). Color
encodes block size + corrector -- a cool hue family for block-7, a warm one for block-16,
so the block-size split reads at a glance; within a family the plain-tree control is the
lighter shade of the markov headline. Budget is the small-multiple axis: one panel per
budget, so the same arm can be compared across budgets side by side. That comparison is
the point of the sweep -- a fixed node budget spread over 16 depths gives a narrower tree
per depth than over 7 depths (b16 trades width for depth), so the reader must be able to
watch whether a larger budget buys the width back. The hues are exp1's validated
categorical set, kept for cross-experiment consistency.

Usage:
    python make_charts.py [path/to/summary.json]      # default: ./summary.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba

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


def _budgets(summary, key):
    """Budget keys (strings) present under summary[key], ordered numerically."""
    return sorted((summary.get(key) or {}), key=lambda b: int(b))


def _arm_verdict(rollup, b7, b16):
    """A compact, direction-free-until-the-data-decides verdict for one arm's change.

    Never hardcodes 'raises'/'lowers' -- the sign comes from pct_change, which
    genuinely differs between arms and between budgets in this experiment. Kept short
    so faceted per-panel titles don't overrun into the neighbouring panel."""
    pct = rollup["overall"].get("pct_change")
    if pct is None:
        return f"block {b7} vs {b16}"
    if abs(pct) < 1.0:
        return f"block {b16} level with block {b7} ({pct:+.1f}%)"
    return f"block {b16} {'raises' if pct > 0 else 'lowers'} by {abs(pct):.0f}%"


def chart_block_size(summary, out):
    """Headline: for each arm, a dumbbell from b7 to b16 mean acceptance length,
    one panel per tree budget.

    A dumbbell is the right form for a before->after per item: two dots (block 7,
    block 16) connected, one row per arm, so the change from the longer horizon is
    the thing the eye follows. Faceting by budget on a shared x-axis lets the reader
    watch the same arm's b7->b16 delta open or close as the node budget grows --
    which is exactly the width-vs-depth question the sweep exists to answer."""
    budgets = _budgets(summary, "block_size")
    if not budgets:
        return

    # Stable arm order, taken from the first budget: markov headline on top, control below.
    first = summary["block_size"][budgets[0]]
    order = ([a for a in ("markov.tree", "tree") if a in first]
             + [a for a in first if a not in ("markov.tree", "tree")])
    b7 = first[order[0]]["small_block"]
    b16 = first[order[0]]["large_block"]

    # Shared x-range across every budget so the deltas are compared on one scale.
    vals = [v for bk in budgets for a in order
            for v in (summary["block_size"][bk][a]["overall"]["small"],
                      summary["block_size"][bk][a]["overall"]["large"])]
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0

    ncols = len(budgets)
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 1.15 * len(order) + 2.2),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    ys = list(range(len(order)))[::-1]  # first arm at the top

    for panel, (bk, ax) in enumerate(zip(budgets, axes)):
        bs = summary["block_size"][bk]
        for y, arm in zip(ys, order):
            o = bs[arm]["overall"]
            s, l, delta, pct = o["small"], o["large"], o["delta"], o["pct_change"]
            ax.plot([s, l], [y, y], color=INK2, linewidth=2.5, zorder=2,
                    solid_capstyle="round")
            ax.scatter([s], [y], s=150, color=BLOCK_COLOR["small"], zorder=3,
                       edgecolor=SURFACE, linewidth=2)
            ax.scatter([l], [y], s=150, color=BLOCK_COLOR["large"], zorder=3,
                       edgecolor=SURFACE, linewidth=2)
            # Endpoint value labels: centered above each dot when far enough apart,
            # pushed outward when the blocks land near-equal so they don't overprint.
            if abs(l - s) < span * 0.10:
                (xl, vl), (xr, vr) = sorted([(s, s), (l, l)])
                ax.text(xl - span * 0.015, y + 0.22, f"{vl:.2f}", ha="right", va="bottom", fontsize=9, color=INK2)
                ax.text(xr + span * 0.015, y + 0.22, f"{vr:.2f}", ha="left", va="bottom", fontsize=9, color=INK2)
            else:
                ax.text(s, y + 0.22, f"{s:.2f}", ha="center", va="bottom", fontsize=9, color=INK2)
                ax.text(l, y + 0.22, f"{l:.2f}", ha="center", va="bottom", fontsize=9, color=INK2)
            tail = "" if pct is None else f"  ({pct:+.0f}%)"
            ax.text(l + span * 0.02, y - 0.24,
                    (f"+{delta:.2f} tok/round{tail}" if delta >= 0 else f"{delta:.2f} tok/round{tail}"),
                    ha="left", va="top", fontsize=10, fontweight="bold",
                    color=(POS if delta >= 0 else NEG))
            # Arm labels only on the leftmost panel (shared y).
            if panel == 0:
                ax.text(lo - span * 0.30, y, arm, ha="left", va="center", fontsize=11,
                        fontweight="bold", color=INK)

        _style(ax); ax.grid(axis="y", linewidth=0)
        ax.set_yticks([]); ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlim(lo - span * 0.34, hi + span * 0.24)
        ax.set_xlabel("mean acceptance length (tokens per verifier call)")
        # Per-panel verdict from that budget's headline arm -- direction is read
        # off the data, never hardcoded, because the sign flips between budgets/arms.
        head = bs.get("markov.tree") or bs[order[0]]
        ax.set_title(f"tree budget {bk}: {_arm_verdict(head, b7, b16)}",
                     fontsize=11.5, fontweight="bold", loc="left")

    # One block-size legend for the whole figure (identity is not color-alone).
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=BLOCK_COLOR["small"], markeredgecolor=SURFACE, label=f"block {b7}"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=BLOCK_COLOR["large"], markeredgecolor=SURFACE, label=f"block {b16}"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=10, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.0))
    # Figure title is structural (no direction): the per-panel titles carry the signs.
    fig.suptitle(f"Longer draft horizon (block {b7} to {b16}) across tree budget",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def _dataset_verdict(bs, arm, datasets, b7, b16):
    """A per-panel verdict for one arm, built from counting the signs of the measured
    per-dataset deltas -- never a hardcoded direction. 'gains'/'loses' only *label*
    the computed positive/negative buckets, so the wording changes with the numbers
    (and genuinely differs between budgets, which is the point of the facet)."""
    pd = (bs.get(arm) or {}).get("per_dataset") or {}
    deltas = [pd[d]["delta"] for d in datasets
              if d in pd and pd[d].get("delta") is not None]
    if not deltas:
        return f"block {b7} vs {b16}"
    n = len(deltas)
    pos = sum(1 for v in deltas if v > 0)
    neg = sum(1 for v in deltas if v < 0)
    parts = []
    if pos:
        parts.append(f"gains on {pos}/{n}")
    if neg:
        parts.append(f"loses on {neg}/{n}")
    if not parts:  # every delta exactly flat
        parts.append(f"flat on {n}/{n}")
    return f"block {b16} vs {b7}: " + ", ".join(parts)


def chart_block_size_by_dataset(summary, out):
    """Per-dataset headline: the b7->b16 acceptance delta for each dataset, one panel
    per tree budget, both arms.

    The block-16 advantage is expected to be task-dependent -- a longer draft horizon
    only pays where completions actually reach deep tree positions, so depths past a
    task's typical length are unused capacity (chat prompts, e.g., rarely reach depth
    8). A cross-dataset average can therefore hide a sign flip: a big math/code gain
    averaged against a chat loss can net out to a bland middle. This is the diverging
    cut that keeps every dataset's sign visible: signed horizontal bars from a zero
    baseline -- extending right (green) where the longer horizon *raises* acceptance,
    left (red) where it *lowers* it. Sign rides three redundant channels (bar
    direction, the diverging colour, and the signed value label), so it survives
    colour-blindness and grayscale; the green/red pair is validated (ΔE 13.3 deutan).
    Arm identity is the *texture* channel, not a hue -- the markov headline is a solid
    saturated bar, the .tree control a paler hatched one -- so colour is free to carry
    sign alone. Faceting by budget lets a dataset's bar flip side between budgets --
    the 'was the tree simply under-budgeted?' question -- at a glance."""
    budgets = _budgets(summary, "block_size")
    if not budgets:
        return
    first = summary["block_size"][budgets[0]]
    order = ([a for a in ("markov.tree", "tree") if a in first]
             + [a for a in first if a not in ("markov.tree", "tree")])
    # Datasets: union across budgets and arms, first-seen order.
    datasets = []
    for bk in budgets:
        for a in order:
            for d in ((summary["block_size"][bk].get(a) or {}).get("per_dataset") or {}):
                if d not in datasets:
                    datasets.append(d)
    if not datasets:
        return
    b7 = first[order[0]]["small_block"]
    b16 = first[order[0]]["large_block"]

    # Shared, zero-anchored x-range so a dataset's delta is compared on one scale
    # across budgets, and 0 (the diverging pivot) is always in frame.
    deltas = []
    for bk in budgets:
        for a in order:
            pd = (summary["block_size"][bk].get(a) or {}).get("per_dataset") or {}
            for d in datasets:
                if d in pd and pd[d].get("delta") is not None:
                    deltas.append(pd[d]["delta"])
    lo = min(deltas + [0.0])
    hi = max(deltas + [0.0])
    span = (hi - lo) or 1.0
    # Pad only the side that actually carries bars+labels, so an all-positive (or
    # all-negative) panel doesn't sit against a wide empty margin.
    left_pad = span * 0.34 if lo < -1e-9 else span * 0.05
    right_pad = span * 0.34 if hi > 1e-9 else span * 0.05

    # Two bars per dataset group (one per arm), markov above its control.
    bar_h = 0.34
    gap = 0.02
    offs = {order[0]: bar_h / 2 + gap}
    if len(order) > 1:
        offs[order[1]] = -(bar_h / 2 + gap)
    for i, a in enumerate(order[2:], start=1):  # extra arms (rare) stack further down
        offs[a] = -(bar_h / 2 + gap) - i * (bar_h + gap)

    ncols = len(budgets)
    fig, axes = plt.subplots(1, ncols, figsize=(6.6 * ncols, 1.5 * len(datasets) + 2.4),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    ys = list(range(len(datasets)))[::-1]  # first dataset at the top

    for panel, (bk, ax) in enumerate(zip(budgets, axes)):
        bs = summary["block_size"][bk]
        # Zero baseline is the diverging pivot -- drawn under the bars.
        ax.axvline(0, color=INK2, linewidth=1.2, zorder=2)
        for y, d in zip(ys, datasets):
            for arm in order:
                pd = (bs.get(arm) or {}).get("per_dataset") or {}
                e = pd.get(d)
                if not e or e.get("delta") is None:
                    continue
                delta, pct = e["delta"], e.get("pct_change")
                yb = y + offs[arm]
                color = POS if delta >= 0 else NEG
                is_head = (arm == order[0])
                if is_head:  # saturated solid = markov headline
                    ax.barh(yb, delta, height=bar_h, color=color, edgecolor=SURFACE,
                            linewidth=1.5, zorder=3)
                else:        # paler + hatched + dashed edge = markov-off control
                    ax.barh(yb, delta, height=bar_h, facecolor=to_rgba(color, 0.28),
                            edgecolor=color, linewidth=1.3, linestyle="--",
                            hatch="////", zorder=3)
                tail = "" if pct is None else f" ({pct:+.0f}%)"
                lab = (f"+{delta:.2f}{tail}" if delta >= 0 else f"{delta:.2f}{tail}")
                # Signed label at the bar's outward end (a third, text channel for sign).
                if delta >= 0:
                    ax.text(delta + span * 0.02, yb, lab, ha="left", va="center",
                            fontsize=9, fontweight=("bold" if is_head else "normal"),
                            color=color)
                else:
                    ax.text(delta - span * 0.02, yb, lab, ha="right", va="center",
                            fontsize=9, fontweight=("bold" if is_head else "normal"),
                            color=color)

        _style(ax); ax.grid(axis="y", linewidth=0)
        ax.set_yticks(list(ys))
        # sharey means panels share one y-formatter, so set the dataset labels once
        # and hide (not clear) them on the other panels -- clearing wipes all of them.
        if panel == 0:
            ax.set_yticklabels(datasets, fontsize=11, fontweight="bold")
        else:
            ax.tick_params(labelleft=False)
        ax.set_ylim(-0.7, len(datasets) - 0.3)
        ax.set_xlim(lo - left_pad, hi + right_pad)
        ax.set_xlabel(f"Δ mean acceptance:  block {b16} − block {b7}  (tokens/round)")
        # Per-panel verdict from the headline arm, sign-counted off the data.
        ax.set_title(f"tree budget {bk}: {_dataset_verdict(bs, order[0], datasets, b7, b16)}",
                     fontsize=11.5, fontweight="bold", loc="left")

    # Legend: arm identity is the texture channel (neutral gray, solid vs hatched);
    # sign is the colour channel -- so identity never rests on colour alone.
    head_label = order[0] if order else "markov.tree"
    ctrl_label = order[1] if len(order) > 1 else "tree"
    handles = [
        Patch(facecolor="#8a8a8a", edgecolor=SURFACE, label=f"{head_label} (headline)"),
        Patch(facecolor="#dcdcda", edgecolor="#8a8a8a", linestyle="--", hatch="////",
              label=f"{ctrl_label} (control)"),
        Patch(facecolor=POS, edgecolor=SURFACE, label=f"Δ > 0  (block {b16} higher)"),
        Patch(facecolor=NEG, edgecolor=SURFACE, label=f"Δ < 0  (block {b16} lower)"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=10, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.0))
    # Structural title (direction-free): the per-panel titles + the signed bars carry it.
    fig.suptitle(f"Per-dataset acceptance change from block {b7} to {b16}, across tree budget",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_acceptance(summary, out):
    """Mean acceptance length, all methods grouped by dataset, one panel per budget."""
    results = summary["results"]
    budgets = _budgets(summary, "results")
    if not budgets:
        return
    # Datasets: union across budgets, first-seen order.
    datasets = []
    for bk in budgets:
        for d in results[bk]:
            if d not in datasets:
                datasets.append(d)
    methods = [m for m in METHOD_COLOR
               if any(m in results[bk].get(d, {}) for bk in budgets for d in datasets)]
    n = len(methods)
    group_w = 0.82
    bw = group_w / max(n, 1)

    # Shared y so panels are comparable across budgets.
    ymax = 0.0
    for bk in budgets:
        for d in datasets:
            for m in methods:
                e = results[bk].get(d, {}).get(m)
                if e:
                    ymax = max(ymax, e["mean_accept"])
    ymax = ymax or 1.0

    ncols = len(budgets)
    fig, axes = plt.subplots(1, ncols, figsize=((1.9 * len(datasets) + 2.2) * ncols, 5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for panel, (bk, ax) in enumerate(zip(budgets, axes)):
        res = results[bk]
        for i, m in enumerate(methods):
            xs, ys = [], []
            for j, d in enumerate(datasets):
                e = res.get(d, {}).get(m)
                if e is None:
                    continue
                x = j - group_w / 2 + bw * (i + 0.5)
                xs.append(x); ys.append(e["mean_accept"])
            ax.bar(xs, ys, width=bw, color=METHOD_COLOR[m], label=m,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
            for x, y in zip(xs, ys):
                ax.text(x, y + ymax * 0.012, f"{y:.1f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK2)
        _style(ax)
        ax.set_ylim(0, ymax * 1.24)  # headroom so value labels clear the legend
        ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets)
        if panel == 0:
            ax.set_ylabel("mean acceptance length (tokens/round)")
        ax.set_title(f"tree budget {bk}", fontsize=12, fontweight="bold", loc="left")

    axes[-1].legend(ncol=2, frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Acceptance length by method and tree budget",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_per_depth(summary, out):
    """Conditional accept rate vs tree depth, one line per method, one panel per budget.

    The point of the experiment: the block-7 tree cannot reach past depth 7, so its
    curves stop there (rate = None once no round reaches that depth); the block-16
    curves keep going. The x-axis is forced out to depth_report_limit so the b16
    tail is never truncated. Faceting by budget shows whether a larger budget (more
    width per depth) lifts the deep-depth accept rates."""
    results = summary["results"]
    budgets = _budgets(summary, "results")
    if not budgets:
        return
    cfg = summary.get("config", {})
    limit = cfg.get("depth_report_limit", 16)
    datasets = []
    for bk in budgets:
        for d in results[bk]:
            if d not in datasets:
                datasets.append(d)
    methods = [m for m in METHOD_COLOR
               if any("per_depth_accept" in results[bk].get(d, {}).get(m, {})
                      for bk in budgets for d in datasets)]
    if not methods:
        return
    # The smaller block size is the horizon the b7 tree cannot draft past.
    blocks = [b.get("block_size") for b in (cfg.get("backbones") or {}).values()
              if b.get("block_size")]
    small_block = min(blocks) if blocks else None

    ncols = len(budgets)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5.5),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for panel, (bk, ax) in enumerate(zip(budgets, axes)):
        res = results[bk]
        # Identity is carried by the legend + a markov-solid / tree-dashed linestyle. No
        # direct end-labels: the two b7 arms share a horizon and the two b16 arms can end
        # at near-identical rates, so end-labels overprint into mush -- the legend is the
        # robust choice for any real data (and avoids doubling the identity channel).
        for m in methods:
            depth_vals = {}
            for d in datasets:
                pd = res.get(d, {}).get(m, {}).get("per_depth_accept") or {}
                for depth, rate in pd.items():
                    if rate is None:
                        continue
                    depth_vals.setdefault(int(depth), []).append(rate)
            if not depth_vals:
                continue
            depths = sorted(depth_vals)
            means = [sum(depth_vals[dd]) / len(depth_vals[dd]) for dd in depths]
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
        if panel == 0:
            ax.set_ylabel("conditional accept rate  P(reach d+1 | reach d)")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0.4, limit + 0.8)
        ax.set_xticks(range(1, limit + 1))
        ax.set_title(f"tree budget {bk}", fontsize=12, fontweight="bold", loc="left")

    axes[-1].legend(frameon=False, fontsize=9, loc="upper right")
    # Descriptive, not conclusion-shaped: the curves are left to say whether the extra
    # depth (or the extra budget) is actually accepted.
    fig.suptitle("Conditional accept rate by tree depth and budget",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "summary.json"
    summary = json.loads(Path(path).read_text())
    outdir = Path(path).parent
    chart_block_size(summary, outdir / "block_size.png")
    chart_block_size_by_dataset(summary, outdir / "block_size_by_dataset.png")
    chart_acceptance(summary, outdir / "acceptance_by_method.png")
    chart_per_depth(summary, outdir / "per_depth_accept.png")


if __name__ == "__main__":
    main()
