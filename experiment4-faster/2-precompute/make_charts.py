"""Charts for the precompute builder: bestfirst.fast vs bestfirst.precompute.

Single-variable comparison -- same backbone (DSpark-b16), same markov head, same
target/verify, same best-first (adaptive) allocation, SAME candidate pool C=512.
The ONLY thing that differs is where the transition arithmetic runs: the fast
builder does a serial per-pop CPU matmul (.expand); the precompute builder hoists
the whole [L-1,C,C] transition stack into one batched GPU matmul before the walk
(lands in .prep), leaving the heap walk as pure CPU indexing (.expand collapses).
So every delta below is attributable to that one move.

Two figures:
  speedup_acceptance.png -- headline: net decode TPS beside mean acceptance (flat).
  phase_collapse.png     -- mechanism: decode ms/round decomposed, fast vs precompute,
                            showing .expand collapse (and .prep absorbing the matmul).

Run:  python make_charts.py [path/to/summary.json]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
FAST, PRE = "bestfirst.fast", "bestfirst.precompute"

# --- style: match exp1/exp2/exp3 -------------------------------------------- #
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
FAST_COLOR, PRE_COLOR = "#9aa0a6", "#e34948"   # fast = muted gray, precompute = story (red)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

PHASE_SEGS = [
    ("draft_forward", "draft fwd",         "#2a78d6"),   # blue
    ("cb_prep",       "cand .prep (matmul+xfer)", "#e6a13c"),   # amber
    ("cb_expand",     "cand .expand (CPU walk)",  "#e34948"),   # red  (the serial per-pop cost)
    ("verify",        "verify",            "#7b5cd6"),   # purple
    ("other",         "other",             "#9a9a97"),   # gray
]


def agg_instr(summary, budget, arm):
    by_ds = summary["results"]["instrumented"][str(budget)]
    rounds, tokens, ph, sub = 0, 0, {}, {}
    for arms in by_ds.values():
        e = arms.get(arm)
        if not e:
            continue
        rounds += e["rounds"]; tokens += e["output_tokens"]
        for p, v in e.get("phases", {}).items():
            ph[p] = ph.get(p, 0.0) + v["sec"]
        for s, v in e.get("subphases", {}).items():
            sub[s] = sub.get(s, 0.0) + v["sec"]
    return rounds, tokens, ph, sub


def ms_round(summary, budget, arm):
    rounds, _, ph, sub = agg_instr(summary, budget, arm)
    total = sum(ph.values())
    draft = ph.get("draft_forward", 0.0)
    verify = ph.get("verify", 0.0)
    prep = sub.get("candidate_build.prep", 0.0)
    expand = sub.get("candidate_build.expand", 0.0)
    other = total - draft - verify - prep - expand
    f = 1000.0 / rounds
    return {"draft_forward": draft * f, "cb_prep": prep * f, "cb_expand": expand * f,
            "verify": verify * f, "other": max(other, 0.0) * f, "_total": total * f}


def tps(summary, budget, arm):
    return summary["timing"][str(budget)][arm]["tps_clean"]


def accept(summary, budget, arm):
    by_ds = summary["results"]["clean"][str(budget)]
    a, r = 0.0, 0
    for arms in by_ds.values():
        e = arms.get(arm)
        if e:
            a += e["mean_accept"] * e["rounds"]; r += e["rounds"]
    return a / r if r else float("nan")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
def fig_speedup_acceptance(summary, budgets, out, suptitle=None, subtext=None):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4.6) if len(budgets) == 1 else (11, 4.6))
    x = range(len(budgets)); w = 0.38

    grid(axL, "y")
    for i, b in enumerate(budgets):
        tb, tn = tps(summary, b, FAST), tps(summary, b, PRE)
        axL.bar(i - w/2, tb, w, color=FAST_COLOR, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        axL.bar(i + w/2, tn, w, color=PRE_COLOR, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        axL.text(i - w/2, tb, f"{tb:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
        axL.text(i + w/2, tn, f"{tn:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
        axL.annotate(f"{tn/tb:.2f}× faster", (i, tn), (i, tn + 14),
                     ha="center", fontsize=12, fontweight="bold", color=PRE_COLOR)
    axL.set_xticks(list(x)); axL.set_xticklabels([f"budget {b}" for b in budgets])
    axL.set_ylabel("net decode throughput (tokens/s, clean)")
    axL.set_ylim(0, max(tps(summary, b, PRE) for b in budgets) * 1.28)
    axL.set_title("Faster", fontsize=13, fontweight="bold", loc="left")

    grid(axR, "y")
    for i, b in enumerate(budgets):
        ab, an = accept(summary, b, FAST), accept(summary, b, PRE)
        axR.bar(i - w/2, ab, w, color=FAST_COLOR, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        axR.bar(i + w/2, an, w, color=PRE_COLOR, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        axR.text(i - w/2, ab, f"{ab:.2f}", ha="center", va="bottom", fontsize=9, color=INK2)
        axR.text(i + w/2, an, f"{an:.2f}", ha="center", va="bottom", fontsize=9, color=INK2)
        d = 100 * (an - ab) / ab
        axR.annotate(f"{d:+.1f}%", (i, max(ab, an)), (i, max(ab, an) + 0.5),
                     ha="center", fontsize=11, fontweight="bold", color=INK2)
    axR.set_xticks(list(x)); axR.set_xticklabels([f"budget {b}" for b in budgets])
    axR.set_ylabel("mean acceptance length (tokens/round)")
    axR.set_ylim(0, max(accept(summary, b, a) for b in budgets for a in (FAST, PRE)) * 1.22)
    axR.set_title("… for free (acceptance unchanged — same tree)", fontsize=13, fontweight="bold", loc="left")

    fig.legend(handles=[Patch(facecolor=FAST_COLOR, label="bestfirst.fast (serial per-pop CPU matmul)"),
                        Patch(facecolor=PRE_COLOR, label="bestfirst.precompute (one batched matmul)")],
               loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle or "Precomputed transition table: net speedup, same accepted tokens",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.90, subtext or ("DSpark-b16 + markov head, H100, gsm8k/humaneval/mt-bench ×4, "
             "both arms C=512. Only where the transition matmul runs differs."), fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def fig_phase_collapse(summary, budgets, out):
    fig, axes = plt.subplots(len(budgets), 1, figsize=(11, 2.0 + 1.6 * len(budgets)))
    if len(budgets) == 1:
        axes = [axes]
    for ax, b in zip(axes, budgets):
        grid(ax, "x")
        rows = [(PRE, "precompute"), (FAST, "fast")]   # fast on top visually
        for y, (arm, lab) in enumerate(rows):
            seg = ms_round(summary, b, arm)
            left = 0.0
            for key, name, col in PHASE_SEGS:
                v = seg[key]
                ax.barh(y, v, left=left, height=0.5, color=col,
                        edgecolor=SURFACE, linewidth=1.5, zorder=3)
                if v > seg["_total"] * 0.045:
                    ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center",
                            fontsize=8.5, color="white", fontweight="bold", zorder=4)
                left += v
            ax.text(left + seg["_total"] * 0.01, y, f"{seg['_total']:.0f} ms/round",
                    ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["bestfirst.precompute", "bestfirst.fast"])
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlim(0, ms_round(summary, b, FAST)["_total"] * 1.14)
        cb_base = ms_round(summary, b, FAST)
        cb_new = ms_round(summary, b, PRE)
        red = 100 * (1 - (cb_new["cb_prep"] + cb_new["cb_expand"]) /
                     (cb_base["cb_prep"] + cb_base["cb_expand"]))
        ax.set_title(f"budget {b}   —   candidate_build (.prep + .expand) cut {red:.0f}%",
                     fontsize=12, fontweight="bold", loc="left")
    axes[-1].set_xlabel("decode time per round (ms), instrumented pass")
    fig.legend(handles=[Patch(facecolor=c, label=n) for _, n, c in PHASE_SEGS],
               loc="lower center", ncol=5, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Where the time went: the serial per-pop matmul (.expand) collapses into "
                 "one batched precompute (.prep)",
                 fontsize=13.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "summary.json"
    summary = json.loads(path.read_text())
    budgets = sorted((int(b) for b in summary["timing"]))
    outdir = HERE / "results"; outdir.mkdir(exist_ok=True)
    fig_speedup_acceptance(summary, budgets, outdir / "speedup_acceptance.png")
    fig_phase_collapse(summary, budgets, outdir / "phase_collapse.png")


if __name__ == "__main__":
    main()
