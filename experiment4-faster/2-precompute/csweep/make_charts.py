"""Charts for the candidate-size (C) sweep of the fast and precompute best-first builders.

Three figures (budget 64):
  pareto_accept_tps.png  -- mean acceptance (y) vs clean TPS (x). bestfirst.ref is the
                            ceiling point; fast and precompute are C-labeled traces.
                            The knee is where a trace stops climbing in acceptance while
                            still moving in TPS.
  accept_vs_c.png        -- acceptance vs C for both builders, with the ref ceiling as a
                            horizontal line; the shaded band is the ref per-dataset spread
                            (the noise floor a knee must clear).
  prep_expand_vs_c.png   -- candidate_build .prep and .expand ms/round vs C for both
                            builders: fast's .expand climbs ~linearly in C while
                            precompute folds it into .prep (which grows ~quadratically).

Run:  python make_charts.py [path/to/summary.json]
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parent
REF = "bestfirst.ref"
ARM_RE = re.compile(r"^(fast|precompute)\.c(\d+)$")

# --- style: match exp1/exp2/exp3 / the 2-precompute charts ------------------- #
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
FAST_COLOR, PRE_COLOR, REF_COLOR = "#2a78d6", "#e34948", "#52514e"  # fast=blue, precompute=red, ref=ink
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})
BUILDERS = [("fast", FAST_COLOR, "o"), ("precompute", PRE_COLOR, "s")]


def budget_keys(summary):
    return sorted(summary["results"]["instrumented"], key=int)


def sweep_arms(summary, budget):
    arms = summary["results"]["clean"][str(budget)]
    names = {n for d in arms.values() for n in d}
    out = {"fast": {}, "precompute": {}}
    for n in names:
        m = ARM_RE.match(n)
        if m:
            out[m.group(1)][int(m.group(2))] = n
    return out


def agg_accept(summary, budget, arm):
    by_ds = summary["results"]["clean"][str(budget)]
    ta, tr, per = 0.0, 0, {}
    for ds, arms in by_ds.items():
        e = arms.get(arm)
        if not e:
            continue
        per[ds] = e["mean_accept"]
        ta += e["mean_accept"] * e["rounds"]
        tr += e["rounds"]
    return (ta / tr if tr else float("nan")), per


def agg_instr(summary, budget, arm):
    by_ds = summary["results"]["instrumented"][str(budget)]
    rounds, sub = 0, {}
    for arms in by_ds.values():
        e = arms.get(arm)
        if not e:
            continue
        rounds += e["rounds"]
        for s, v in e.get("subphases", {}).items():
            sub[s] = sub.get(s, 0.0) + v["sec"]
    return rounds, sub


def tps_clean(summary, budget, arm):
    return summary.get("timing", {}).get(str(budget), {}).get(arm, {}).get("tps_clean", float("nan"))


def ms_per(sec, rounds):
    return 1000.0 * sec / rounds if rounds else float("nan")


def spread(per):
    vals = [v for v in per.values() if v == v]
    return (max(vals) - min(vals)) if vals else 0.0


def fig_pareto(summary, budget, arms, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ceil_acc, _ = agg_accept(summary, budget, REF)
    ceil_tps = tps_clean(summary, budget, REF)
    ax.scatter([ceil_tps], [ceil_acc], color=REF_COLOR, s=90, marker="*", zorder=5)
    ax.annotate("bestfirst.ref (ceiling)", (ceil_tps, ceil_acc),
                textcoords="offset points", xytext=(8, 6), color=REF_COLOR, fontsize=9)
    for builder, color, marker in BUILDERS:
        cmap = arms.get(builder, {})
        xs, ys, cs = [], [], []
        for C in sorted(cmap):
            acc, _ = agg_accept(summary, budget, cmap[C])
            xs.append(tps_clean(summary, budget, cmap[C]))
            ys.append(acc)
            cs.append(C)
        ax.plot(xs, ys, "-", color=color, marker=marker, label=builder, zorder=4)
        for x, y, C in zip(xs, ys, cs):
            ax.annotate(f"C{C}", (x, y), textcoords="offset points", xytext=(5, -10),
                        color=color, fontsize=8)
    ax.set_xlabel("decode throughput (clean TPS, tok/s)")
    ax.set_ylabel("mean acceptance length")
    ax.set_title(f"Acceptance vs speed Pareto  (budget {budget})", fontsize=12, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "pareto_accept_tps.png", dpi=150)
    plt.close(fig)


def fig_accept_vs_c(summary, budget, arms, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ceil_acc, ceil_per = agg_accept(summary, budget, REF)
    band = spread(ceil_per)
    ax.axhline(ceil_acc, color=REF_COLOR, ls="--", lw=1.2, label="bestfirst.ref (ceiling)")
    ax.axhspan(ceil_acc - band, ceil_acc + band, color=REF_COLOR, alpha=0.08)
    for builder, color, marker in BUILDERS:
        cmap = arms.get(builder, {})
        xs = sorted(cmap)
        ys = [agg_accept(summary, budget, cmap[C])[0] for C in xs]
        ax.plot(xs, ys, "-", color=color, marker=marker, label=builder)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({C for b in arms.values() for C in b}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("candidate size C")
    ax.set_ylabel("mean acceptance length")
    ax.set_title(f"Acceptance vs C  (budget {budget}; shaded = ref per-dataset spread)",
                 fontsize=12, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "accept_vs_c.png", dpi=150)
    plt.close(fig)


def fig_prep_expand_vs_c(summary, budget, arms, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for builder, color, marker in BUILDERS:
        cmap = arms.get(builder, {})
        xs = sorted(cmap)
        prep, expand = [], []
        for C in xs:
            rounds, sub = agg_instr(summary, budget, cmap[C])
            prep.append(ms_per(sub.get("candidate_build.prep", 0.0), rounds))
            expand.append(ms_per(sub.get("candidate_build.expand", 0.0), rounds))
        ax.plot(xs, prep, "-", color=color, marker=marker, label=f"{builder} .prep")
        ax.plot(xs, expand, "--", color=color, marker=marker, alpha=0.7, label=f"{builder} .expand")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({C for b in arms.values() for C in b}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("candidate size C")
    ax.set_ylabel("candidate_build sub-phase (ms / round)")
    ax.set_title(f"Where the time goes vs C  (budget {budget})", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "prep_expand_vs_c.png", dpi=150)
    plt.close(fig)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "summary.json"
    summary = json.loads(path.read_text())
    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)
    for budget in budget_keys(summary):
        arms = sweep_arms(summary, budget)
        fig_pareto(summary, budget, arms, outdir)
        fig_accept_vs_c(summary, budget, arms, outdir)
        fig_prep_expand_vs_c(summary, budget, arms, outdir)
    print(f"wrote charts to {outdir}")


if __name__ == "__main__":
    main()
