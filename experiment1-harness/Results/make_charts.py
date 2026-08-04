"""Generate charts from an Experiment 1 summary.json (+ results_detailed.json).

Writes PNGs next to the input:
  transfer.png              headline: within-backbone acceptance % change from the head
  acceptance_by_method.png  mean acceptance length, all methods, grouped by dataset
  per_depth_accept.png      conditional accept rate vs tree depth (tree methods)
  corrector_fit.png         tree-free probe: does the head fit each backbone?
  acceptance_distribution.png  per-round acceptance distribution (needs detailed)
  decode_speed.png          output tokens/sec per method (needs detailed)

Every chart is annotated with which methods use the markov head, because that is
the variable under study (and the naming isn't otherwise obvious):
  * .chain          = linear drafting (no tree). dspark.chain uses DSpark's OWN
                      markov head natively; dflash.chain has no head.
  * .tree           = tree drafting, markov OFF (raw backbone logits).
  * .markov.tree    = tree drafting, markov ON (head conditions each node).

Usage: python make_charts.py [path/to/summary.json]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# method -> (backbone, verify, markov_on)
METHOD_META = {
    "dflash.chain":       ("DFlash-b16", "chain", False),
    "dflash.tree":        ("DFlash-b16", "tree",  False),
    "dflash.markov.tree": ("DFlash-b16", "tree",  True),
    "dspark.chain":       ("DSpark-b7",  "chain", True),
    "dspark.tree":        ("DSpark-b7",  "tree",  False),
    "dspark.markov.tree": ("DSpark-b7",  "tree",  True),
}
METHOD_COLOR = {
    "dflash.chain": "#2a78d6", "dflash.tree": "#1baf7a", "dflash.markov.tree": "#eda100",
    "dspark.chain": "#008300", "dspark.tree": "#4a3aa7", "dspark.markov.tree": "#e34948",
}
POS, NEG = "#008300", "#e34948"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _mk(m):
    return "markov ON" if METHOD_META[m][2] else "markov off"


def _label(m):
    return f"{m}  [{_mk(m)}]"


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def _caption(fig, text):
    fig.text(0.01, 0.005, text, fontsize=8, color=INK2, ha="left", va="bottom", wrap=True)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else 0.0


def chart_transfer(summary, out):
    transfer = summary.get("transfer") or {}
    if not transfer:
        return
    labelmap = {"dspark_b7": "DSpark-b7  (head's OWN backbone)",
                "dflash_b16": "DFlash-b16  (FOREIGN backbone)"}
    names = list(transfer)
    vals = [transfer[n].get("accept_pct_change") for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 0.95 * len(names) + 2.4))
    ys = list(range(len(names)))
    for y, n, v in zip(ys, names, vals):
        if v is None:
            continue
        t = transfer[n]
        ax.barh(y, v, color=(POS if v >= 0 else NEG), height=0.5,
                edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(v + (1.5 if v >= 0 else -1.5), y, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=13, fontweight="bold", color=INK)
        ax.text(0, y + 0.34, f"{t['off_method']} → {t['on_method']}: "
                f"{t['accept_off']:.2f} → {t['accept_on']:.2f} tok/round",
                va="bottom", ha="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(ys); ax.set_yticklabels([labelmap.get(n, n) for n in names])
    ax.set_xlabel("acceptance-length change from adding the DSpark-b7 markov head to the tree (%)")
    ax.set_title("A markov head helps ONLY the backbone it was trained on",
                 fontsize=14, fontweight="bold", loc="left")
    pad = max(abs(v) for v in vals if v is not None) * 0.3 + 6
    lo = min(0, min(v for v in vals if v is not None)); hi = max(0, max(v for v in vals if v is not None))
    ax.set_xlim(lo - pad, hi + pad)
    ax.grid(axis="x", color=GRID, linewidth=1); ax.set_axisbelow(True)
    _caption(fig, "Within-backbone delta: each bar compares  <backbone>.tree (markov off)  vs  <backbone>.markov.tree "
                  "(markov on), averaged over gsm8k/humaneval/mt-bench, temperature 0.")
    fig.tight_layout(rect=[0, 0.06, 1, 1]); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_acceptance(summary, out):
    results = summary["results"]
    datasets = list(results)
    methods = [m for m in METHOD_COLOR if any(m in results[d] for d in datasets)]
    n = len(methods); group_w = 0.82; bw = group_w / n
    fig, ax = plt.subplots(figsize=(2.1 * len(datasets) + 4, 5.4))
    for i, m in enumerate(methods):
        xs, ys = [], []
        for j, d in enumerate(datasets):
            e = results[d].get(m)
            if e is None:
                continue
            xs.append(j - group_w / 2 + bw * (i + 0.5)); ys.append(e["mean_accept"])
        ax.bar(xs, ys, width=bw, color=METHOD_COLOR[m], label=_label(m),
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.08, f"{y:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    _style(ax)
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets)
    ax.set_ylabel("mean acceptance length (tokens accepted / round)")
    ax.set_title("Acceptance length by method  (higher = faster speculative decoding)",
                 fontsize=13, fontweight="bold", loc="left")
    ax.legend(ncol=2, frameon=False, fontsize=9, loc="upper right", title="method [markov status]")
    _caption(fig, "chain = linear (no tree); dspark.chain uses DSpark's native head. tree = markov off; "
                  "markov.tree = markov on. Absolute values not comparable across block sizes (DFlash-b16 vs DSpark-b7).")
    fig.tight_layout(rect=[0, 0.05, 1, 1]); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_per_depth(summary, out):
    results = summary["results"]
    datasets = list(results)
    tree_methods = [m for m in METHOD_COLOR
                    if METHOD_META[m][1] == "tree"
                    and any("per_depth_accept" in results[d].get(m, {}) for d in datasets)]
    if not tree_methods:
        return
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for m in tree_methods:
        depth_vals = {}
        for d in datasets:
            for depth, rate in (results[d].get(m, {}).get("per_depth_accept") or {}).items():
                if rate is not None:
                    depth_vals.setdefault(int(depth), []).append(rate)
        if not depth_vals:
            continue
        depths = sorted(depth_vals)
        means = [_mean(depth_vals[dd]) for dd in depths]
        ls = "-" if METHOD_META[m][2] else "--"
        ax.plot(depths, means, color=METHOD_COLOR[m], linewidth=2.2, linestyle=ls,
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.text(depths[-1] + 0.1, means[-1], _label(m), va="center", ha="left",
                fontsize=8.5, color=METHOD_COLOR[m], fontweight="bold")
    _style(ax)
    ax.set_xlabel("tree depth (token position within a drafted block)")
    ax.set_ylabel("conditional accept rate  P(accept depth d | reached d)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-depth acceptance (avg across datasets); solid = markov on, dashed = off",
                 fontsize=12.5, fontweight="bold", loc="left")
    ax.set_xlim(right=max(ax.get_xticks()) + 3)
    _caption(fig, "Compared at depths the b7 head was trained on, so a foreign penalty is not just depth extrapolation. "
                  "DSpark's markov (solid red) lifts deep acceptance; the same head on DFlash (solid yellow) does not.")
    fig.tight_layout(rect=[0, 0.05, 1, 1]); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_corrector_fit(summary, out):
    cf = summary.get("corrector_fit") or {}
    if not cf:
        return
    lim = summary["config"].get("depth_report_limit", 7)
    key = f"overall_depth_le_{lim}"
    rows = [(d.get("backbone", m), d[key].get("delta_hit"), d[key].get("delta_ce")) for m, d in cf.items()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.9 * len(rows) + 2.6))
    ys = list(range(len(rows)))
    labels = [r[0] for r in rows]
    for ax, idx, title, good_pos in (
        (a1, 1, "Δ top-1 hit rate vs target\n(> 0  ⇒  head helps predict)", True),
        (a2, 2, "Δ cross-entropy vs target\n(< 0  ⇒  head helps predict)", False),
    ):
        for y, r in zip(ys, rows):
            v = r[idx]
            if v is None:
                continue
            helps = (v >= 0) if good_pos else (v <= 0)
            ax.barh(y, v, color=(POS if helps else NEG), height=0.5, edgecolor=SURFACE, linewidth=2, zorder=3)
            verdict = "fits ✓" if helps else "no fit ✗"
            ax.text(v, y, f" {v:+.3f}  {verdict}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=10, color=INK)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(ys); ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.grid(axis="x", color=GRID, linewidth=1); ax.set_axisbelow(True)
    fig.suptitle(f"Corrector-fit probe: does the DSpark-b7 head fit each backbone?  (depth ≤ {lim}, no-tree-corrector runs)",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    _caption(fig, "Confound-free: at each committed position, compare argmax(base) vs argmax(base + head.bias(prev)) "
                  "against the target's real token. Measured on the markov-off runs; no depth extrapolation.")
    fig.tight_layout(rect=[0, 0.05, 1, 0.92]); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_distribution(detailed, out):
    per = detailed["per_dataset"]
    datasets = list(per)
    methods = [m for m in METHOD_COLOR if any(m in per[d]["methods"] for d in datasets)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets) + 1, 5.6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, d in zip(axes, datasets):
        data, colors, labels = [], [], []
        for m in methods:
            md = per[d]["methods"].get(m)
            if md and md["lengths"]:
                data.append(md["lengths"]); colors.append(METHOD_COLOR[m])
                labels.append(f"{m} [{'on' if METHOD_META[m][2] else 'off'}]")
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
    fig.suptitle("Acceptance-length distribution by method  ([on]/[off] = markov)",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    _caption(fig, "Box = IQR, line = median, whiskers = 1.5·IQR (outliers hidden). Higher/tighter-high is better.")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_speed(detailed, out):
    per = detailed["per_dataset"]
    datasets = list(per)
    methods = [m for m in METHOD_COLOR if any(m in per[d]["methods"] for d in datasets)]
    speeds = {}
    for m in methods:
        tpots = [s["tpot"] for d in datasets for s in (per[d]["methods"].get(m) or {}).get("per_sample", [])
                 if s.get("tpot", 0) > 0]
        if tpots:
            speeds[m] = 1.0 / _mean(tpots)
    if not speeds:
        return
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(speeds) + 2.4))
    ys = list(range(len(speeds)))
    for y, m in zip(ys, speeds):
        ax.barh(y, speeds[m], color=METHOD_COLOR[m], height=0.6, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(speeds[m], y, f" {speeds[m]:.0f} tok/s", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels([_label(m) for m in speeds])
    ax.grid(axis="x", color=GRID, linewidth=1); ax.set_axisbelow(True)
    ax.set_xlabel("decode speed (output tokens / sec)")
    ax.set_title("Decode speed by method", fontsize=12.5, fontweight="bold", loc="left")
    _caption(fig, "Hardware-dependent (single A100-40GB); shown for context, not the paper's headline metric. "
                  "Acceptance length (other charts) is the GPU-independent quantity.")
    fig.tight_layout(rect=[0, 0.06, 1, 1]); fig.savefig(out, dpi=150); plt.close(fig)
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
