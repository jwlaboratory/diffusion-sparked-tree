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
    "dflash.chain":              ("DFlash-b16", "chain", False),
    "dflash.markov.chain":       ("DFlash-b16", "chain", True),
    "dflash.tree":               ("DFlash-b16", "tree",  False),
    "dflash.markov.tree":        ("DFlash-b16", "tree",  True),
    "dspark.nomarkov.chain":     ("DSpark-b7",  "chain", False),
    "dspark.chain":              ("DSpark-b7",  "chain", True),
    "dspark.tree":               ("DSpark-b7",  "tree",  False),
    "dspark.markov.tree":        ("DSpark-b7",  "tree",  True),
    "dspark_b16.nomarkov.chain": ("DSpark-b16", "chain", False),
    "dspark_b16.chain":          ("DSpark-b16", "chain", True),
    "dspark_b16.tree":           ("DSpark-b16", "tree",  False),
    "dspark_b16.markov.tree":    ("DSpark-b16", "tree",  True),
}
METHOD_COLOR = {
    "dflash.chain": "#2a78d6", "dflash.markov.chain": "#6b4fc8",
    "dflash.tree": "#1baf7a", "dflash.markov.tree": "#eda100",
    "dspark.nomarkov.chain": "#7a9e3b", "dspark.chain": "#008300",
    "dspark.tree": "#4a3aa7", "dspark.markov.tree": "#e34948",
    "dspark_b16.nomarkov.chain": "#b06a2a", "dspark_b16.chain": "#0e7fa8",
    "dspark_b16.tree": "#8a5a9e", "dspark_b16.markov.tree": "#c2186b",
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
    labelmap = {"dspark_b7": "DSpark-b7  (its OWN b7 head)",
                "dspark_b16": "DSpark-b16  (its OWN b16 head)",
                "dflash_b16": "DFlash-b16  (FOREIGN b7 head)"}
    names = list(transfer)
    vals = [transfer[n].get("accept_pct_change") for n in names]
    fig, ax = plt.subplots(figsize=(10.5, 0.95 * len(names) + 2.4))
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
    ax.set_xlabel("acceptance-length change from adding the markov head to the tree (%)")
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
    ax.legend(ncol=1, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="method [markov status]")
    _caption(fig, "chain = linear (no tree); dspark*.chain uses that DSpark's own native head. tree = markov off; "
                  "markov.tree = markov on. DFlash-b16 and DSpark-b16 share a 16-token horizon; DSpark-b7 stops at 7.")
    fig.tight_layout(rect=[0, 0.05, 1, 1]); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def chart_acceptance_aggregated(summary, out):
    """Mean acceptance length per method, averaged across all datasets. One bar per
    method. No subtitle, no caption (per request)."""
    results = summary["results"]
    datasets = list(results)
    methods = [m for m in METHOD_COLOR if any(m in results[d] for d in datasets)]
    means = {m: _mean([results[d][m]["mean_accept"] for d in datasets if m in results[d]]) for m in methods}
    methods = sorted(methods, key=lambda m: means[m], reverse=True)  # easy comparison
    fig, ax = plt.subplots(figsize=(10, 5.4))
    xs = list(range(len(methods)))
    for x, m in zip(xs, methods):
        ax.bar(x, means[m], width=0.7, color=METHOD_COLOR[m], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(x, means[m] + 0.08, f"{means[m]:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK2)
    _style(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([_label(m) for m in methods], rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("mean acceptance length (tokens accepted / round)")
    ax.set_title("Acceptance length by method", fontsize=13, fontweight="bold", loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_markov_delta(summary, out):
    """Per backbone: the full 2x2 (chain/tree x markov off/on), each bar annotated
    with its % change vs that backbone's no-markov chain baseline. Bars whose
    method has not been run yet are skipped."""
    groups = [  # (group label, [(sub-label, method)], shades light->dark)
        ("DFlash  (block size 16)",
         [("normal", "dflash.chain"), ("normal + markov", "dflash.markov.chain")],
         ["#aac9ec", "#2a78d6"]),
        ("DSpark  (block size 7)",
         [("normal", "dspark.nomarkov.chain"), ("normal + markov", "dspark.chain"),
          ("normal + markov + tree", "dspark.markov.tree")],
         ["#a3cfa3", "#4ba64b", "#008300"]),
    ]
    cross = ("dflash.chain", "dspark.nomarkov.chain")  # bracket: DFlash normal vs DSpark normal
    results = summary["results"]
    datasets = list(results)

    def mean_accept(m):
        vals = [results[d][m]["mean_accept"] for d in datasets if m in results[d]]
        return _mean(vals) if vals else None

    fig, ax = plt.subplots(figsize=(11, 5.6))
    bw, step, gap = 0.62, 0.95, 1.1
    centers, y_max, x_cursor = [], 0.0, 0.0
    bar_pos = {}  # method -> (x, value), for the cross-group bracket
    for label, bars, shades in groups:
        bars = [(sub, m) for sub, m in bars if mean_accept(m) is not None]
        if not bars:
            continue
        xs = [x_cursor + i * step for i in range(len(bars))]
        vals = [mean_accept(m) for _, m in bars]
        base = vals[0]
        for x, v, c, (sub, m) in zip(xs, vals, shades, bars):
            ax.bar(x, v, width=bw, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(x, v + 0.09, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color=INK2)
            if x != xs[0] and base:
                pct = (v - base) / base * 100.0
                ax.text(x, v + 0.42, f"{pct:+.0f}%", ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color=(POS if pct >= 0 else NEG))
            ax.text(x, -0.28, sub, ha="center", va="top", fontsize=9, color=INK2)
            bar_pos[m] = (x, v)
        centers.append((sum(xs) / len(xs), label))
        y_max = max(y_max, max(vals) + 1.0)
        x_cursor = xs[-1] + step + gap
    # cross-group bracket: the two no-markov chain baselines against each other
    if cross[0] in bar_pos and cross[1] in bar_pos:
        (xa, va), (xb, vb) = bar_pos[cross[0]], bar_pos[cross[1]]
        top = y_max + 0.35
        ax.plot([xa, xa, xb, xb], [va + 0.75, top, top, vb + 0.75],
                color=INK2, linewidth=1.2, zorder=3)
        pct = (vb - va) / va * 100.0
        ax.text((xa + xb) / 2, top + 0.07, f"{pct:+.0f}%", ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color=(POS if pct >= 0 else NEG))
        y_max = top + 0.55
    _style(ax)
    ax.set_xticks([c for c, _ in centers])
    ax.set_xticklabels([l for _, l in centers], fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", pad=24, length=0)
    ax.set_ylabel("mean acceptance length (tokens accepted / round)")
    ax.set_ylim(0, y_max + 0.4)
    ax.set_title("Effect of the markov head and tree drafting on acceptance",
                 fontsize=13, fontweight="bold", loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def chart_acceptance_simple(summary, out):
    """4-bar version of the aggregated chart with clean display names only."""
    series = [
        ("dspark.markov.tree", "SparklingTree-blocksize7"),
        ("dflash.chain",       "DFlash-b16"),
        ("dspark.chain",       "DSpark-b7"),
        ("dflash.tree",        "DDTree-b16"),
    ]
    results = summary["results"]
    datasets = list(results)
    means = {m: _mean([results[d][m]["mean_accept"] for d in datasets if m in results[d]])
             for m, _ in series}
    series = sorted(series, key=lambda s: means[s[0]], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(len(series)))
    for x, (m, label) in zip(xs, series):
        ax.bar(x, means[m], width=0.62, color=METHOD_COLOR[m], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(x, means[m] + 0.08, f"{means[m]:.2f}", ha="center", va="bottom", fontsize=10, color=INK2)
    _style(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in series], fontsize=10)
    ax.set_ylabel("mean acceptance length (tokens accepted / round)")
    ax.set_title("Acceptance length by method", fontsize=13, fontweight="bold", loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


# series shown in per_depth_accept.png: (method, display label)
# chain methods have no per_depth_accept in summary.json; their rates are derived
# from per-round `lengths` in results_detailed.json with the same formula
# aggregate.per_depth_accept uses: rate(d) = #(L >= d+1) / #(L >= d).
PER_DEPTH_SERIES = [
    ("dflash.chain",           "DFlash"),
    ("dspark.chain",           "DSpark"),
    ("dflash.tree",            "DDTree"),
    ("dspark.markov.tree",     "SparklingTree BlockSize=7"),
    ("dspark_b16.markov.tree", "SparklingTree BlockSize=16"),
]


def _chain_per_depth(detailed, method, dataset, depth_limit):
    lengths = (detailed["per_dataset"][dataset]["methods"].get(method) or {}).get("lengths") or []
    rates = {}
    for d in range(1, depth_limit + 1):
        reached = sum(1 for a in lengths if a >= d)
        deeper = sum(1 for a in lengths if a >= d + 1)
        rates[d] = (deeper / reached) if reached else None
    return rates


def chart_per_depth(summary, detailed, out):
    results = summary["results"]
    datasets = list(results)
    depth_limit = summary["config"].get("depth_report_limit", 16)
    series = []
    for m, label in PER_DEPTH_SERIES:
        depth_vals = {}
        for d in datasets:
            rates = results[d].get(m, {}).get("per_depth_accept")
            if not rates and METHOD_META[m][1] == "chain" and detailed is not None:
                rates = _chain_per_depth(detailed, m, d, depth_limit)
            for depth, rate in (rates or {}).items():
                if rate is not None:
                    depth_vals.setdefault(int(depth), []).append(rate)
        if not depth_vals:
            continue
        depths = sorted(depth_vals)
        means = [_mean(depth_vals[dd]) for dd in depths]
        # The last depth of a bounded drafter always reads 0 (nothing can be
        # accepted past its horizon) -- that's an artifact, not acceptance decay.
        while means and means[-1] == 0.0:
            depths.pop(); means.pop()
        if depths:
            series.append((m, label, depths, means))
    if not series:
        return
    max_depth = max(s[2][-1] for s in series)
    all_means = [v for s in series for v in s[3]]
    # zoom to the high range, but never clip a series that dips below it
    y_lo = min(0.75, max(0.0, min(all_means) - 0.03))
    # end-labels: nudge apart any that share an end-depth region
    label_y = [min(max(means[-1] + 0.013, y_lo + 0.01), 0.99) for _, _, _, means in series]
    order = sorted(range(len(series)), key=lambda i: label_y[i])
    for a, b in zip(order, order[1:]):
        if abs(series[a][2][-1] - series[b][2][-1]) <= 1 and label_y[b] - label_y[a] < 0.035:
            label_y[b] = label_y[a] + 0.035
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for (m, label, depths, means), ty in zip(series, label_y):
        ax.plot(depths, means, color=METHOD_COLOR[m], linewidth=2.2,
                marker="o", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.text(depths[-1] + 0.15, ty, label, va="center", ha="left",
                fontsize=8.5, color=METHOD_COLOR[m], fontweight="bold")
    _style(ax)
    ax.set_xlabel("depth (token position within a drafted block)")
    ax.set_ylabel("conditional accept rate  P(accept depth d | reached d)")
    ax.set_ylim(y_lo, 1.0)
    ax.set_title("Per-depth acceptance (avg across datasets)",
                 fontsize=12.5, fontweight="bold", loc="left")
    ax.set_xticks(range(1, max_depth + 1))
    ax.set_xlim(0.5, max_depth + 4.5)  # right margin holds the series end-labels
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
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
    detailed_path = outdir / "results_detailed.json"
    detailed = json.loads(detailed_path.read_text()) if detailed_path.exists() else None
    chart_transfer(summary, outdir / "transfer.png")
    chart_acceptance(summary, outdir / "acceptance_by_method.png")
    chart_acceptance_aggregated(summary, outdir / "acceptance_by_method_aggregated.png")
    chart_acceptance_simple(summary, outdir / "acceptance_by_method_simple.png")
    chart_markov_delta(summary, outdir / "acceptance_markov_delta.png")
    chart_per_depth(summary, detailed, outdir / "per_depth_accept.png")
    chart_corrector_fit(summary, outdir / "corrector_fit.png")

    if detailed is not None:
        chart_distribution(detailed, outdir / "acceptance_distribution.png")
        chart_speed(detailed, outdir / "decode_speed.png")


if __name__ == "__main__":
    main()
