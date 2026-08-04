"""Charts for the data-scaling ablation (experiment 2b).

Question: does training the block-16 DSpark drafter on more conversations raise
acceptance, and in particular lift the deep-depth (8-16) conditional accept rate?

Three arms share the _best recipe exactly (A10G-class config: seq 768, 32
anchors, gamma 4, global batch 32, lr 1e-4) and vary ONLY the PerfectBlend row
count, with steps epoch-matched at ~5 epochs:

    data2k4   2,400 rows,   360 steps   (control; reproduces _best)
    data10k  10,000 rows, 1,500 steps
    data24k  24,000 rows, 3,600 steps

Inputs (all repo-local):
    ../summary.json               exp-2 run: dspark_b7 + dspark_b16 (_best)
    ../cache/b*__n4.json          exp-2 raw acceptance lengths (b7/_best)
    data{2k4,10k,24k}_summary.json
    cache/<arm>/b*__n4.json       raw acceptance lengths per arm

Outputs:
    datascale_mean_accept.png     mean acceptance vs training rows (the verdict)
    datascale_per_depth.png       per-depth profile, identical pooling, all arms
    combined_summary.json         merged numbers used by both charts

Per-depth estimator matches ../make_charts.py: rate(d) = #(L >= d+1) / #(L >= d),
pooled over RAW lengths from all 6 (budget, dataset) cells. Pooling raw lengths
(not averaging per-cell rates) matters: deep-depth cells have tiny n, and a mean
of noisy cell rates manufactures dips that are not in the data (the exp-1 chart's
apparent d10-d11 cliff for _best is exactly this artifact). Cells with n < 30 at
a depth are suppressed entirely.
"""
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
EXP2 = HERE.parent

BUDGETS = ["64", "256"]
DATASETS = ["gsm8k", "humaneval", "mt-bench"]
ARMS = [  # (key, rows, method prefix, where lengths live)
    ("data2k4", 2400, "dspark_b16_data2k4"),
    ("data10k", 10000, "dspark_b16_data10k"),
    ("data24k", 24000, "dspark_b16_data24k"),
]
MIN_N = 30
DEPTH_MAX = 16

# Categorical palette (validated: scripts/validate_palette.js, light mode, pass).
# Yellow sits under 3:1 on white -> relief rule: every series is direct-labeled.
C_BLUE, C_AQUA, C_YELLOW, C_VIOLET, C_RED = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
INK, INK2, GRID = "#1a1a19", "#6b6a63", "#e5e4dd"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})


def unit_lengths(arm_key: str, budget: str, ds: str, method: str) -> list:
    if arm_key == "exp2":
        p = EXP2 / "cache" / f"b{budget}__{ds}__n4.json"
    else:
        p = HERE / "cache" / arm_key / f"b{budget}__{ds}__n4.json"
    return json.load(open(p))[method]


def pooled_lengths(arm_key: str, method: str) -> list:
    out = []
    for b in BUDGETS:
        for ds in DATASETS:
            out += unit_lengths(arm_key, b, ds, method)
    return out


def per_depth(lengths: list) -> dict:
    rates = {}
    for d in range(1, DEPTH_MAX + 1):
        reached = sum(1 for a in lengths if a >= d)
        deeper = sum(1 for a in lengths if a >= d + 1)
        rates[d] = (deeper / reached) if reached >= MIN_N else None
    return rates


def cell_mean(arm_key: str, budget: str, ds: str, method: str) -> float:
    return mean(unit_lengths(arm_key, budget, ds, method))


# ---------------------------------------------------------------- chart 1 ----
def chart_mean_accept(combined: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    xs = [rows for _, rows, _ in ARMS]
    ds_color = {"gsm8k": C_BLUE, "humaneval": C_AQUA, "mt-bench": C_YELLOW}

    for ax, budget in zip(axes, BUDGETS):
        for ds in DATASETS:
            ys = [cell_mean(k, budget, ds, f"{p}.markov.tree") for k, _, p in ARMS]
            ax.plot(xs, ys, "-o", color=ds_color[ds], linewidth=2, markersize=8)
            ax.annotate(ds, (xs[-1], ys[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center",
                        color=INK, fontweight="bold", fontsize=10)
            ref = cell_mean("exp2", budget, ds, "dspark_b16.markov.tree")
            ax.axhline(ref, color=ds_color[ds], linewidth=1.2,
                       linestyle=(0, (4, 4)), alpha=0.55)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(["2.4k", "10k", "24k"])
        ax.minorticks_off()
        ax.set_xlim(1900, 90000)
        ax.set_xlabel("training conversations (log scale)")
        ax.set_title(f"tree budget {budget}", loc="left", fontweight="bold")
    axes[0].set_ylabel("mean acceptance length (markov.tree)")
    fig.suptitle("More training data does not move acceptance "
                 "(dashed = shipped _best, same rows as 2.4k arm)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(HERE / "datascale_mean_accept.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- chart 2 ----
def chart_per_depth(combined: dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    series = [
        ("b7 (stock)", "exp2", "dspark_b7.markov.tree", C_VIOLET),
        ("_best (2.4k rows, 8.4 ep)", "exp2", "dspark_b16.markov.tree", C_RED),
        ("2.4k rows", "data2k4", "dspark_b16_data2k4.markov.tree", C_BLUE),
        ("10k rows", "data10k", "dspark_b16_data10k.markov.tree", C_AQUA),
        ("24k rows", "data24k", "dspark_b16_data24k.markov.tree", C_YELLOW),
    ]
    ends = []
    for label, key, method, color in series:
        rates = combined["per_depth"][label]
        xs = [d for d in range(1, DEPTH_MAX + 1) if rates[str(d)] is not None]
        ys = [rates[str(d)] for d in xs]
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=7)
        # anchor the label at the last point INSIDE the y-window; b7's final
        # point is the structural zero at its block horizon, far below ylim
        vis = [(x, y) for x, y in zip(xs, ys) if y >= 0.6]
        ends.append([label, *vis[-1]])

    # stagger end labels that would collide (min vertical separation in data units)
    MIN_SEP = 0.02
    for i, e in enumerate(sorted(ends, key=lambda e: e[2])):
        if i and e[2] - prev < MIN_SEP:
            e[2] = prev + MIN_SEP
        prev = e[2]
    for label, x, y in ends:
        ax.annotate(label, (x, y), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK,
                    fontweight="bold", fontsize=10)
    ax.axvline(7.5, color=INK2, linewidth=1, linestyle=(0, (2, 3)))
    ax.text(7.6, 0.615, "block-7 horizon", color=INK2, fontsize=9)
    ax.set_xticks(range(1, DEPTH_MAX + 1))
    ax.set_xlim(0.5, DEPTH_MAX + 3.5)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("depth d (token position within a drafted block)")
    ax.set_ylabel("conditional accept rate  P(reach d+1 | reach d)")
    ax.set_title("Per-depth acceptance, identical pooling (raw lengths, all 6 cells, min n=30):\n"
                 "no cliff after depth 7, and no data-scale effect",
                 loc="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(HERE / "datascale_per_depth.png", dpi=150)
    plt.close(fig)


def main():
    combined = {"mean_accept": {}, "overall": {}, "per_depth": {}}

    ref_arms = [("b7 (stock)", "exp2", "dspark_b7.markov.tree"),
                ("_best (2.4k rows, 8.4 ep)", "exp2", "dspark_b16.markov.tree")]
    all_arms = ref_arms + [(f"{r//1000}k rows" if r >= 10000 else "2.4k rows",
                            k, f"{p}.markov.tree") for k, r, p in ARMS]

    for label, key, method in all_arms:
        combined["mean_accept"][label] = {
            b: {ds: cell_mean(key, b, ds, method) for ds in DATASETS} for b in BUDGETS
        }
        pooled = pooled_lengths(key, method)
        combined["overall"][label] = {"mean_accept": mean(pooled), "rounds": len(pooled)}
        combined["per_depth"][label] = {str(d): v for d, v in per_depth(pooled).items()}

    (HERE / "combined_summary.json").write_text(json.dumps(combined, indent=2))
    chart_mean_accept(combined)
    chart_per_depth(combined)
    print("wrote", HERE / "combined_summary.json")
    for label, o in combined["overall"].items():
        print(f"  {label:28s} mean_accept={o['mean_accept']:.2f}  rounds={o['rounds']}")


if __name__ == "__main__":
    main()
