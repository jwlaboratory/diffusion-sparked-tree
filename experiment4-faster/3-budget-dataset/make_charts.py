"""Budget plateau: clean TPS vs tree_budget, one line per dataset (precompute.c256),
with each dataset's peak marked. The artifact that operationalizes the budget heuristic."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
# categorical, CVD-aware: math=blues, code=greens, chat=warm
COLORS = {"gsm8k": "#2a78d6", "aime24": "#6aa9e9", "humaneval": "#1f9d55",
          "livecodebench": "#7bc47f", "mt-bench": "#e6a13c", "alpaca": "#e34948"}
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID, "xtick.color": INK2,
    "ytick.color": INK2, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

d = json.loads((HERE / "results" / "summary.json").read_text())
tm = d["timing"]; budgets = sorted((int(b) for b in tm))
ARM = "precompute.c256"
def tps(b, ds): return tm[str(b)].get(ARM, {}).get("per_dataset", {}).get(ds, {}).get("tps_clean")
datasets = [ds for ds in COLORS if any(tps(b, ds) for b in budgets)]

fig, ax = plt.subplots(figsize=(10, 5.6))
ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
for ds in datasets:
    ys = [tps(b, ds) for b in budgets]
    ax.plot(budgets, ys, "-o", color=COLORS[ds], lw=2.4, ms=8, mec=SURFACE, mew=1.5, zorder=3, label=ds)
    peak = max(range(len(ys)), key=lambda i: ys[i])
    ax.scatter([budgets[peak]], [ys[peak]], s=190, facecolor="none", edgecolor=COLORS[ds], lw=2.5, zorder=4)
    ax.text(budgets[-1] * 1.03, ys[-1], ds, color=COLORS[ds], fontsize=9.5, va="center", fontweight="bold")
ax.axvspan(96, 272, color="#2a78d6", alpha=0.05, zorder=0)
ax.text(128, ax.get_ylim()[1] if False else 322, "", fontsize=9)
ax.set_xscale("log", base=2); ax.set_xticks(budgets); ax.set_xticklabels(budgets)
ax.set_xlabel("tree_budget"); ax.set_ylabel("net decode throughput (tokens/s, clean)")
ax.set_xlim(14, 380)
ax.set_title("Budget plateau — precompute.c256, one line per dataset (◯ = peak)",
             fontsize=13, fontweight="bold", loc="left")
fig.text(0.012, 0.945, "Every dataset peaks at budget 128 (math/code/chat) or 256 (hardest); "
         "64 is below peak everywhere. Batch-1 H100, bench x4.", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = HERE / "results" / "budget_plateau.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
