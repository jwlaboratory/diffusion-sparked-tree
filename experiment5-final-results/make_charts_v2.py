"""Final charts from results/summary_t0.json (instrumented pass — sync-on,
DDTree benchmarking practices; the citable run: C=128, max_fanout=64, seed 1).

Run: python3 make_charts_v2.py   -> results/final_speedup_b{64,128,256}.png,
                                    results/final_per_domain.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "results"

METHODS = ["Autoregressive", "DFlash", "DSpark", "DDTree", "SparklingTree"]
SHORT = {"Autoregressive": "AR", "DFlash": "DFlash", "DSpark": "DSpark",
         "DDTree": "DDTree", "SparklingTree": "SparklingTree"}
# Validated palette (dataviz six-checks): color follows the method, fixed order.
COLOR = {"Autoregressive": "#149d8e", "DFlash": "#8460c9", "DSpark": "#bf7d15",
         "DDTree": "#4a7fd4", "SparklingTree": "#e34948"}
SURFACE = "#fcfcfb"
INK = "#333333"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#cccccc", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})


def agg(res, bkey, m):
    tok = t = rounds = 0
    acc_w = 0.0
    for ds, ms in res[bkey].items():
        e = ms.get(m)
        if not e:
            continue
        tok += e["output_tokens"]; t += e["decode_time"]; rounds += e["rounds"]
        acc_w += e["mean_accept"] * e["rounds"]
    return acc_w / rounds, tok / t


def speedup_chart(res, bkey, n_ds):
    ar_acc, ar_tps = agg(res, bkey, "Autoregressive")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.suptitle(f"Final results — budget {bkey}   (C=128, fanout 64; sync-on, one GPU)  [{n_ds} datasets]",
                 fontsize=13.5, fontweight="bold", x=0.02, ha="left")
    axes[0].set_title("Speedup vs Autoregressive", loc="left", fontsize=11)
    axes[1].set_title("Mean acceptance (tokens/round)", loc="left", fontsize=11)
    order = sorted(METHODS, key=lambda m: -agg(res, bkey, m)[1])
    for ax, metric in ((axes[0], "spd"), (axes[1], "acc")):
        for i, m in enumerate(order):
            acc, tps = agg(res, bkey, m)
            v = tps / ar_tps if metric == "spd" else acc
            bold = m == "SparklingTree"
            bar = ax.bar(i, v, 0.68, color=COLOR[m], zorder=3)
            ax.bar_label(bar, labels=[f"{v:.2f}×" if metric == "spd" else f"{v:.2f}"],
                         padding=2, fontsize=10.5, fontweight="bold" if bold else "normal")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([SHORT[m] for m in order], rotation=18, ha="right", fontsize=10)
        ax.grid(axis="y", color="#e8e8e8", zorder=0)
        ax.margins(y=0.16)
    fig.text(0.02, 0.02, "6 datasets × 12 samples × 512 tok, seed 1 · DDTree benchmarking practices (sync ON, C++ compaction ON)",
             fontsize=8.5, color="#777777")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(OUT / f"final_speedup_b{bkey}.png", dpi=160)
    plt.close(fig)


def per_domain_chart(res):
    budgets = sorted(res, key=int)
    datasets = sorted(res[budgets[0]])
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    width = 0.26
    shades = {"64": "#f0a8a7", "128": "#ea7472", "256": "#e34948"}
    for j, bk in enumerate(budgets):
        xs, vs = [], []
        for i, ds in enumerate(datasets):
            try:
                dd = res[bk][ds]["DDTree"]["tps_decode"]
                st = res[bk][ds]["SparklingTree"]["tps_decode"]
            except KeyError:
                continue
            xs.append(i + (j - 1) * width)
            vs.append((st / dd - 1) * 100)
        bars = ax.bar(xs, vs, width * 0.92, color=shades[bk], zorder=3, label=f"budget {bk}")
        for x, v in zip(xs, vs):
            ax.text(x, v + (0.55 if v >= 0 else -0.55), f"{v:+.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8.2)
    ax.axhline(0, color="#999999", linewidth=1, zorder=2)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=10.5)
    ax.set_ylabel("SparklingTree vs DDTree, decode tok/s (%)")
    ax.grid(axis="y", color="#e8e8e8", zorder=0)
    ax.margins(y=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9.5)
    fig.suptitle("Where the wins are: SparklingTree vs DDTree per domain",
                 fontsize=13.5, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.015, "positive = SparklingTree faster · same GPU, same job · final config",
             fontsize=8.5, color="#777777")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(OUT / "final_per_domain.png", dpi=160)
    plt.close(fig)


def main():
    s = json.loads((OUT / "summary_t0.json").read_text())
    res = s["results"]["instrumented"]
    n_ds = len(s["config"].get("datasets_present", list(res["64"])))
    for bk in sorted(res, key=int):
        speedup_chart(res, bk, n_ds)
    per_domain_chart(res)
    print("wrote:", sorted(p.name for p in OUT.glob("final_*.png")))


if __name__ == "__main__":
    main()
