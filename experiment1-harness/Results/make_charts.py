"""Generate charts from an Experiment 1 summary.json.

Reads a summary produced by DDTree/run_experiment.py (via modal_benchmark.py) and
writes PNGs next to it:

  transfer.png            headline: within-backbone acceptance % change from the head
  acceptance_by_method.png mean acceptance length, all methods, grouped by dataset
  per_depth_accept.png    conditional accept rate vs tree depth (tree methods)
  corrector_fit.png       tree-free probe: does the head fit each backbone?

Usage:
    python make_charts.py [path/to/summary.json]      # default: ./summary.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (light mode), assigned per method in fixed order.
METHOD_COLOR = {
    "dflash.chain": "#2a78d6",        # blue
    "dflash.tree": "#1baf7a",         # aqua
    "dflash.markov.tree": "#eda100",  # yellow
    "dspark.chain": "#008300",        # green
    "dspark.tree": "#4a3aa7",         # violet
    "dspark.markov.tree": "#e34948",  # red
}
POS = "#008300"   # positive / helps
NEG = "#e34948"   # negative / hurts
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


def chart_transfer(summary, out):
    transfer = summary.get("transfer") or {}
    if not transfer:
        return
    names = list(transfer)
    vals = [transfer[n].get("accept_pct_change") for n in names]
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(names) + 1.6))
    ys = list(range(len(names)))
    for y, n, v in zip(ys, names, vals):
        if v is None:
            continue
        ax.barh(y, v, color=(POS if v >= 0 else NEG), height=0.55,
                edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(v + (1.5 if v >= 0 else -1.5), y, f"{v:+.1f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=12,
                fontweight="bold", color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(ys); ax.set_yticklabels(names)
    ax.set_xlabel("acceptance-length change from adding the markov head (%)")
    ax.set_title("Markov head helps its own backbone, hurts a foreign one",
                 fontsize=13, fontweight="bold", loc="left")
    pad = max(abs(v) for v in vals if v is not None) * 0.25 + 5
    ax.set_xlim(min(0, min(v for v in vals if v is not None)) - pad,
                max(0, max(v for v in vals if v is not None)) + pad)
    ax.grid(axis="x", color=GRID, linewidth=1); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_acceptance(summary, out):
    results = summary["results"]
    datasets = list(results)
    methods = [m for m in METHOD_COLOR if any(m in results[d] for d in datasets)]
    n = len(methods)
    group_w = 0.82
    bw = group_w / n
    fig, ax = plt.subplots(figsize=(1.9 * len(datasets) + 3, 5))
    for i, m in enumerate(methods):
        xs, ys = [], []
        for j, d in enumerate(datasets):
            e = results[d].get(m)
            if e is None:
                continue
            x = j - group_w / 2 + bw * (i + 0.5)
            xs.append(x); ys.append(e["mean_accept"])
        ax.bar(xs, ys, width=bw, color=METHOD_COLOR[m], label=m,
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.08, f"{y:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    _style(ax)
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets)
    ax.set_ylabel("mean acceptance length (tokens/round)")
    ax.set_title("Acceptance length by method", fontsize=13, fontweight="bold", loc="left")
    ax.legend(ncol=2, frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_per_depth(summary, out):
    results = summary["results"]
    datasets = list(results)
    tree_methods = [m for m in METHOD_COLOR
                    if m.endswith(".tree") and any("per_depth_accept" in results[d].get(m, {}) for d in datasets)]
    if not tree_methods:
        return
    # Average each method's per-depth rate across datasets.
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in tree_methods:
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
        ax.plot(depths, means, color=METHOD_COLOR[m], linewidth=2, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.text(depths[-1] + 0.08, means[-1], m, va="center", ha="left",
                fontsize=9, color=METHOD_COLOR[m], fontweight="bold")
    _style(ax)
    ax.set_xlabel("tree depth"); ax.set_ylabel("conditional accept rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-depth acceptance (avg across datasets)", fontsize=13,
                 fontweight="bold", loc="left")
    ax.set_xlim(right=max([d for d in ax.get_xticks()]) + 2)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_corrector_fit(summary, out):
    cf = summary.get("corrector_fit") or {}
    if not cf:
        return
    lim = summary["config"].get("depth_report_limit", 7)
    key = f"overall_depth_le_{lim}"
    rows = []
    for method, d in cf.items():
        o = d.get(key, {})
        rows.append((d.get("backbone", method), o.get("delta_hit"), o.get("delta_ce")))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 0.8 * len(rows) + 2.2))
    ys = list(range(len(rows)))
    labels = [r[0] for r in rows]
    for ax, idx, title, xlabel, good_is_pos in (
        (a1, 1, "Δ top-1 hit rate (higher = head fits)", "delta_hit", True),
        (a2, 2, "Δ cross-entropy (lower = head fits)", "delta_ce", False),
    ):
        for y, r in zip(ys, rows):
            v = r[idx]
            if v is None:
                continue
            helps = (v >= 0) if good_is_pos else (v <= 0)
            ax.barh(y, v, color=(POS if helps else NEG), height=0.5,
                    edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(v, y, f" {v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=10, color=INK)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(ys); ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.grid(axis="x", color=GRID, linewidth=1); ax.set_axisbelow(True)
    fig.suptitle(f"Corrector-fit probe (depth ≤ {lim}, no-corrector runs)",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else 0.0


def chart_distribution(detailed, out):
    """Box plot of per-round acceptance length per method, one panel per dataset."""
    per = detailed["per_dataset"]
    datasets = list(per)
    methods = [m for m in METHOD_COLOR if any(m in per[d]["methods"] for d in datasets)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets) + 1, 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, d in zip(axes, datasets):
        data, colors, labels = [], [], []
        for m in methods:
            md = per[d]["methods"].get(m)
            if not md or not md["lengths"]:
                continue
            data.append(md["lengths"]); colors.append(METHOD_COLOR[m]); labels.append(m)
        bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_edgecolor(SURFACE); patch.set_alpha(0.9)
        for med in bp["medians"]:
            med.set_color(INK); med.set_linewidth(1.5)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(d, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)
    axes[0].set_ylabel("acceptance length per round (tokens)")
    fig.suptitle("Acceptance-length distribution by method", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_speed(detailed, out):
    """Mean decode speed (tokens/sec = 1/tpot) per method, averaged across datasets."""
    per = detailed["per_dataset"]
    datasets = list(per)
    methods = [m for m in METHOD_COLOR if any(m in per[d]["methods"] for d in datasets)]
    speeds = {}
    for m in methods:
        tpots = []
        for d in datasets:
            md = per[d]["methods"].get(m) or {}
            for s in md.get("per_sample", []):
                if s.get("tpot", 0) > 0:
                    tpots.append(s["tpot"])
        if tpots:
            speeds[m] = 1.0 / _mean(tpots)
    if not speeds:
        return
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(speeds) + 2))
    ys = list(range(len(speeds)))
    for y, m in zip(ys, speeds):
        ax.barh(y, speeds[m], color=METHOD_COLOR[m], height=0.6, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(speeds[m], y, f" {speeds[m]:.0f}", va="center", ha="left", fontsize=9, color=INK)
    _style(ax); ax.grid(axis="x", color=GRID, linewidth=1); ax.grid(axis="y", visible=False)
    ax.set_yticks(ys); ax.set_yticklabels(list(speeds))
    ax.set_xlabel("decode speed (output tokens / sec)")
    ax.set_title("Decode speed by method (GPU-dependent; A100-40GB)", fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "summary.json"
    summary = json.loads(Path(path).read_text())
    outdir = Path(path).parent
    chart_transfer(summary, outdir / "transfer.png")
    chart_acceptance(summary, outdir / "acceptance_by_method.png")
    chart_per_depth(summary, outdir / "per_depth_accept.png")
    chart_corrector_fit(summary, outdir / "corrector_fit.png")

    detailed_path = outdir / "results_detailed.json"
    if detailed_path.exists():
        detailed = json.loads(detailed_path.read_text())
        chart_distribution(detailed, outdir / "acceptance_distribution.png")
        chart_speed(detailed, outdir / "decode_speed.png")


if __name__ == "__main__":
    main()
