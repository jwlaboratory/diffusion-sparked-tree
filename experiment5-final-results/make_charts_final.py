"""Charts for the exp5 final results (works on PARTIAL summaries too, for early peeks).

Two figures per tree budget found in the summary:
  final_speedup_b{B}.png   -- speedup vs AR + mean acceptance, one bar per method,
                              SparklingTree highlighted.
  final_per_domain_b{B}.png -- per-dataset speedup vs AR, grouped bars by method.

Missing methods/datasets are simply skipped, so this renders whatever has completed.

Run:  python make_charts_final.py [results/summary.json] [--baseline Autoregressive]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
OURS = "SparklingTree"
# Muted palette for competitors; SparklingTree pops in red.
COLORS = {"SparklingTree": "#e34948", "DDTree": "#2a78d6", "DSpark": "#e6a13c",
          "DFlash": "#7b5cd6", "Autoregressive": "#9a9a97"}
DEFAULT = "#9aa0a6"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False})


def col(m):
    return COLORS.get(m, DEFAULT)


def accept_agg(s, bkey, name):
    a = r = 0.0
    for ds, arms in s["results"]["clean"][bkey].items():
        e = arms.get(name)
        if e:
            a += e["mean_accept"] * e["rounds"]; r += e["rounds"]
    return a / r if r else None


def tps_agg(s, bkey, name):
    return s.get("timing", {}).get(bkey, {}).get(name, {}).get("tps_clean")


def pd_tps(s, bkey, name, ds):
    return (s.get("timing", {}).get(bkey, {}).get(name, {})
            .get("per_dataset", {}).get(ds, {}).get("tps_clean"))


def grid(ax):
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0); ax.set_axisbelow(True)


def fig_speedup(s, bkey, baseline, out):
    arms = list(s["timing"][bkey])
    base = tps_agg(s, bkey, baseline)
    methods = [m for m in arms if tps_agg(s, bkey, m)]
    methods.sort(key=lambda m: -(tps_agg(s, bkey, m) or 0))
    if not methods or not base:
        return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, valf, ttl, fmt in [
        (axL, lambda m: (tps_agg(s, bkey, m) / base), f"Speedup vs {baseline}", "{:.2f}×"),
        (axR, lambda m: accept_agg(s, bkey, m), "Mean acceptance (tokens/round)", "{:.2f}")]:
        grid(ax)
        vals = [valf(m) for m in methods]
        bars = ax.bar(range(len(methods)), vals, color=[col(m) for m in methods],
                      edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for i, (m, v) in enumerate(zip(methods, vals)):
            ax.text(i, v, fmt.format(v) if v is not None else "-", ha="center", va="bottom",
                    fontsize=9.5, color=INK, fontweight="bold" if m == OURS else "normal")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace("Autoregressive", "AR") for m in methods], rotation=20, ha="right")
        ax.set_title(ttl, fontsize=12.5, fontweight="bold", loc="left")
        ax.set_ylim(0, max(v for v in vals if v is not None) * 1.18)
    ndone = len(s.get("config", {}).get("datasets_present", s["results"]["clean"][bkey]))
    fig.suptitle(f"Final results — budget {bkey}   (SparklingTree = precompute)  [{ndone} datasets]",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
    plt.close(fig)


def fig_per_domain(s, bkey, baseline, out):
    datasets = list(s["results"]["clean"][bkey])
    arms = [m for m in s["timing"][bkey] if m != baseline]
    arms.sort(key=lambda m: -(tps_agg(s, bkey, m) or 0))
    datasets = [d for d in datasets if pd_tps(s, bkey, baseline, d)]
    if not datasets or not arms:
        return
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(datasets)), 4.8))
    grid(ax)
    n = len(arms); w = 0.8 / n
    for j, m in enumerate(arms):
        xs, ys = [], []
        for i, d in enumerate(datasets):
            b = pd_tps(s, bkey, baseline, d); t = pd_tps(s, bkey, m, d)
            if b and t:
                xs.append(i + (j - (n - 1) / 2) * w); ys.append(t / b)
        ax.bar(xs, ys, w, color=col(m), edgecolor=SURFACE, linewidth=0.8, zorder=3,
               label=m.replace("Autoregressive", "AR"))
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel(f"speedup vs {baseline} (×)")
    ax.set_title(f"Per-domain speedup — budget {bkey}", fontsize=13, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9.5, ncol=n)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
    plt.close(fig)


def _opt(flag, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else HERE / "results" / "summary.json"
    baseline = _opt("--baseline", "Autoregressive")
    s = json.loads(path.read_text())
    outdir = HERE / "results"; outdir.mkdir(exist_ok=True)
    for bkey in sorted(s.get("timing", {}), key=int):
        if not s["timing"][bkey]:
            continue
        fig_speedup(s, bkey, baseline, outdir / f"final_speedup_b{bkey}.png")
        fig_per_domain(s, bkey, baseline, outdir / f"final_per_domain_b{bkey}.png")


if __name__ == "__main__":
    main()
