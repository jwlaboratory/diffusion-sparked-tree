"""Generate the README results figures from the citable run
(seed 1, 6 datasets x 12 prompts, 512 tok, temp 0, sync ON, compaction ON, C=128, fanout 64).

  results_speedup.png     - aggregate speedup vs AR, grouped by draft budget
  results_by_dataset.png  - per-dataset speedup vs AR at budget 256

Run:  python3 assets/make_results_fig.py

Numbers are the tokens/sec (t) values from the EXACT FULL RESULTS block in the
README; speedup = t_method / t_autoregressive per dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent

# palette (validated defaults, light mode) — fixed categorical order
BLUE = "#2a78d6"      # DFlash
AQUA = "#1baf7a"      # DSpark
YELLOW = "#eda100"    # DDTree
GREEN = "#008300"     # SparklingTree
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

METHODS = ["DFlash", "DSpark", "DDTree", "SparklingTree"]
COLORS = [BLUE, AQUA, YELLOW, GREEN]

DATASETS = ["alpaca", "gsm8k", "humaneval", "math500", "mbpp", "mt-bench"]

# tokens/sec per dataset: {budget: {method: [t per dataset]}}; AR is budget-independent
AR_T = [51.0, 52.2, 52.7, 52.0, 52.4, 51.5]
T = {
    64: {
        "DFlash":        [95.4, 218.0, 207.5, 253.6, 176.6, 105.8],
        "DSpark":        [109.9, 242.6, 200.2, 230.5, 179.8, 114.7],
        "DDTree":        [132.7, 268.7, 267.7, 309.7, 231.6, 145.3],
        "SparklingTree": [149.3, 298.6, 273.1, 287.2, 224.5, 157.0],
    },
    128: {
        "DFlash":        [95.4, 218.0, 207.5, 253.6, 176.6, 105.8],
        "DSpark":        [109.9, 242.6, 200.2, 230.5, 179.8, 114.7],
        "DDTree":        [147.8, 280.9, 284.7, 323.9, 252.9, 158.0],
        "SparklingTree": [169.0, 311.9, 279.8, 302.0, 244.7, 167.1],
    },
    256: {
        "DFlash":        [95.4, 218.0, 207.5, 253.6, 176.6, 105.8],
        "DSpark":        [109.9, 242.6, 200.2, 230.5, 179.8, 114.7],
        "DDTree":        [152.1, 300.8, 286.2, 331.3, 252.5, 158.8],
        "SparklingTree": [168.1, 327.5, 283.7, 316.6, 248.8, 172.9],
    },
}

# aggregate speedups (from the SPEEDUP rows)
AGG = {
    64:  {"DFlash": 3.15, "DSpark": 3.27, "DDTree": 4.12, "SparklingTree": 4.29},
    128: {"DFlash": 3.15, "DSpark": 3.27, "DDTree": 4.45, "SparklingTree": 4.58},
    256: {"DFlash": 3.15, "DSpark": 3.27, "DDTree": 4.55, "SparklingTree": 4.71},
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "font.size": 11,
})


def style_ax(ax, ymax):
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Speedup vs autoregressive (x)")
    ax.axhline(1.0, color=BASELINE, lw=1.2, zorder=1)


def fig_speedup():
    budgets = [64, 128, 256]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(budgets))
    width = 0.19
    for i, (m, c) in enumerate(zip(METHODS, COLORS)):
        vals = [AGG[b][m] for b in budgets]
        bars = ax.bar(x + (i - 1.5) * (width + 0.015), vals, width, color=c,
                      label=m, zorder=3)
        for r, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (r.get_x() + r.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=INK2)
    style_ax(ax, 5.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"budget {b}" for b in budgets], color=INK2)
    ax.set_title("Aggregate speedup vs autoregressive (6 datasets, 512 tok, temp 0)",
                 loc="left", fontsize=12, fontweight="bold", pad=14)
    ax.annotate("AR baseline (1x)", xy=(2.35, 1.0), ha="right", va="bottom",
                fontsize=9, color=MUTED)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.02),
              fontsize=9.5, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "results_speedup.png", dpi=200)
    plt.close(fig)


def fig_by_dataset():
    budget = 256
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    x = np.arange(len(DATASETS))
    width = 0.19
    for i, (m, c) in enumerate(zip(METHODS, COLORS)):
        vals = [T[budget][m][j] / AR_T[j] for j in range(len(DATASETS))]
        ax.bar(x + (i - 1.5) * (width + 0.012), vals, width, color=c,
               label=m, zorder=3)
    style_ax(ax, 7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, color=INK2)
    ax.set_title("Per-dataset speedup at draft budget 256",
                 loc="left", fontsize=12, fontweight="bold", pad=14)
    ax.annotate("AR baseline (1x)", xy=(5.45, 1.0), ha="right", va="bottom",
                fontsize=9, color=MUTED)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.02),
              fontsize=9.5, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "results_by_dataset.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_speedup()
    fig_by_dataset()
    print("wrote", OUT / "results_speedup.png")
    print("wrote", OUT / "results_by_dataset.png")
