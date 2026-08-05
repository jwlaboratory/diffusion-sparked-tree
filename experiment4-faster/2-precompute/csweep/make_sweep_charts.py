"""Sweep evidence: acceptance & speed vs candidate size C, fast vs precompute (b64)."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
FAST_C, PRE_C, REF_C = "#2a78d6", "#e34948", "#9aa0a6"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID, "xtick.color": INK2,
    "ytick.color": INK2, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

d = json.loads((HERE / "results" / "summary.json").read_text())
b = "64"; clean = d["results"][b] if b in d["results"] else d["results"]["clean"][b]
clean = d["results"]["clean"][b]
Cs = [128, 256, 512, 1024, 2048]

def accept(arm):
    a = r = 0.0
    for ds in clean:
        if arm in clean[ds]:
            e = clean[ds][arm]; a += e["mean_accept"] * e["rounds"]; r += e["rounds"]
    return a / r if r else None
def tps(arm):
    return d["timing"][b][arm]["tps_clean"] if arm in d["timing"][b] else None

ref_a, ref_t = accept("bestfirst.ref"), tps("bestfirst.ref")
def series(fam):
    xs, acc, tp = [], [], []
    for c in Cs:
        a = accept(f"{fam}.c{c}")
        if a is None: continue
        xs.append(c); acc.append(a); tp.append(tps(f"{fam}.c{c}"))
    return xs, acc, tp
fx, fa, ft = series("fast"); px, pa, pt = series("precompute")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8))

# Panel A: acceptance vs C, ceiling line.
axA.axhline(ref_a, color=REF_C, ls="--", lw=2, zorder=2)
axA.text(Cs[-1], ref_a, f"  exact best-first ceiling {ref_a:.2f}", va="bottom", ha="right", fontsize=9, color=INK2)
axA.plot(fx, fa, "-o", color=FAST_C, lw=2.4, ms=9, mec=SURFACE, mew=1.5, label="fast (union top-C, per-pop matmul)", zorder=4)
axA.plot(px, pa, "-o", color=PRE_C, lw=2.4, ms=9, mec=SURFACE, mew=1.5, label="precompute (per-depth top-C, table lookup)", zorder=4)
for x, y in zip(px, pa): axA.text(x, y - 0.09, f"{y:.2f}", ha="center", va="top", fontsize=8, color=PRE_C)
for x, y in zip(fx, fa): axA.text(x, y + 0.06, f"{y:.2f}", ha="center", va="bottom", fontsize=8, color=FAST_C)
axA.set_xscale("log", base=2); axA.set_xticks(Cs); axA.set_xticklabels(Cs)
axA.set_xlabel("candidate size C"); axA.set_ylabel("mean acceptance (tokens/round)")
axA.grid(color=GRID, lw=1); axA.set_axisbelow(True)
axA.set_title("Acceptance vs C — fast saturates by 256; precompute never catches up",
              fontsize=11.5, fontweight="bold", loc="left")
axA.legend(fontsize=8.5, frameon=False, loc="lower right")

# Panel B: Pareto — acceptance vs TPS.
axB.plot(ft, fa, "-o", color=FAST_C, lw=2.4, ms=9, mec=SURFACE, mew=1.5, label="fast", zorder=4)
axB.plot(pt, pa, "-o", color=PRE_C, lw=2.4, ms=9, mec=SURFACE, mew=1.5, label="precompute", zorder=4)
axB.scatter([ref_t], [ref_a], s=120, color=REF_C, ec=SURFACE, lw=1.5, zorder=5)
axB.text(ref_t + 4, ref_a, "ref (exact)", fontsize=9, color=INK2, va="center")
for x, y, c in list(zip(ft, fa, fx)): axB.text(x, y + 0.05, f"C{c}", ha="center", fontsize=7.5, color=FAST_C)
for x, y, c in list(zip(pt, pa, px)): axB.text(x, y - 0.09, f"C{c}", ha="center", fontsize=7.5, color=PRE_C)
axB.set_xlabel("net decode throughput (tokens/s, clean)"); axB.set_ylabel("mean acceptance (tokens/round)")
axB.grid(color=GRID, lw=1); axB.set_axisbelow(True)
axB.set_title("Pareto — up-and-right is better; precompute buys speed by spending acceptance",
              fontsize=11.5, fontweight="bold", loc="left")
axB.legend(fontsize=9, frameon=False, loc="lower left")

fig.suptitle("Candidate-size sweep (budget 64, n=8): fast vs precompute", fontsize=14, fontweight="bold", x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = HERE / "results" / "csweep_evidence.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
