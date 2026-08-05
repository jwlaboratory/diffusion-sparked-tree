"""Budget-64-only per-domain chart: DFlash, DSpark, DDTree, SparklingTree (C=128).

Run: python3 make_per_domain_b64.py  -> results/final_per_domain_b64.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "results"

METHODS = ["DFlash", "DSpark", "DDTree", "SparklingTree"]
LABEL = {"DFlash": "DFlash", "DSpark": "DSpark", "DDTree": "DDTree",
         "SparklingTree": "SparklingTree (C=128)"}
# Validated palette (dataviz six-checks), same as make_charts_v2.py.
COLOR = {"DFlash": "#8460c9", "DSpark": "#bf7d15",
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


def main():
    s = json.loads((OUT / "summary_t0.json").read_text())
    res = s["results"]["instrumented"]["64"]
    datasets = sorted(res)

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    width = 0.2
    for j, m in enumerate(METHODS):
        xs, vs = [], []
        for i, ds in enumerate(datasets):
            e = res[ds].get(m)
            if not e:
                continue
            xs.append(i + (j - 1.5) * width)
            vs.append(e["tps_decode"])
        ax.bar(xs, vs, width * 0.92, color=COLOR[m], zorder=3, label=LABEL[m])
        for x, v in zip(xs, vs):
            ax.text(x, v + 3, f"{v:.0f}", ha="center", va="bottom", fontsize=8.2,
                    fontweight="bold" if m == "SparklingTree" else "normal")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=10.5)
    ax.set_ylabel("decode tok/s")
    ax.grid(axis="y", color="#e8e8e8", zorder=0)
    ax.margins(y=0.18)
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9.5)
    fig.suptitle("Per-domain decode throughput at budget 64",
                 fontsize=13.5, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.015,
             "6 datasets × 12 samples × 512 tok, seed 1 · same GPU, same job · final config (C=128, fanout 64)",
             fontsize=8.5, color="#777777")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(OUT / "final_per_domain_b64.png", dpi=160)
    plt.close(fig)
    print("wrote:", OUT / "final_per_domain_b64.png")


if __name__ == "__main__":
    main()
