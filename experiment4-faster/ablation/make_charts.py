"""Charts for the exp4 ablation ladder (A0 naive -> A1 transfer -> A2 precompute -> A3 launch).

Two charts per optimization step (overall speedup+acceptance, per-phase collapse)
plus one full-ladder hero. Data: results/summary.json (instrumented pass — sync-on,
DDTree benchmarking practices; TPS shown IS the sync-on TPS).

Run: python3 make_charts.py       (needs matplotlib; writes results/*.png)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "results"

ARMS = ["A0.ref", "A1.transfer", "A2.precompute", "A3.launch"]
ARM_LABEL = {
    "A0.ref": "naive best-first",
    "A1.transfer": "+ transfer-less",
    "A2.precompute": "+ precompute",
    "A3.launch": "+ launch batching",
}
STEPS = [
    ("A0.ref", "A1.transfer", "step1_transfer", "Transfer less: ship one small union slice, not 311 MB"),
    ("A1.transfer", "A2.precompute", "step2_precompute", "Precompute: batch the transition table before the walk"),
    ("A2.precompute", "A3.launch", "step3_launch", "Launch batching: ~80 kernel launches/round -> ~5"),
]

# Validated palette (dataviz six-checks, light surface): phase identity, fixed order.
PHASE_COLORS = {
    "draft fwd": "#4a7fd4",
    "cand .prep": "#bf7d15",
    "cand .expand": "#e34948",
    "verify": "#8460c9",
    "other": "#149d8e",
}
BEFORE = "#9a9a9a"   # neutral baseline bar (identity carried by direct labels)
AFTER = "#e34948"    # ours/improved accent (repo convention)
SURFACE = "#fcfcfb"
INK = "#333333"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#cccccc", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})


def load():
    d = json.loads((OUT / "summary.json").read_text())
    return d["results"]["instrumented"]


def agg(res, bkey, arm):
    """Aggregate one arm over datasets: (accept, tps, phase ms/round dict)."""
    tok = t = rounds = 0
    acc_w = 0.0
    ph = {k: 0.0 for k in PHASE_COLORS}
    for ds, ms in res[bkey].items():
        e = ms.get(arm)
        if not e:
            continue
        tok += e["output_tokens"]; t += e["decode_time"]; rounds += e["rounds"]
        acc_w += e["mean_accept"] * e["rounds"]
        sub = e.get("subphases") or {}
        prep = sum(v["sec"] for k, v in sub.items() if k.endswith(".prep") or k.endswith(".markov"))
        expand = sum(v["sec"] for k, v in sub.items() if k.endswith(".expand"))
        phases = e.get("phases") or {}
        total = sum(v["sec"] for v in phases.values() if v)
        draft = (phases.get("draft_forward") or {}).get("sec", 0.0)
        verify = (phases.get("verify") or {}).get("sec", 0.0)
        cb = (phases.get("candidate_build") or {}).get("sec", 0.0)
        ph["draft fwd"] += draft
        ph["cand .prep"] += prep
        ph["cand .expand"] += expand
        ph["verify"] += verify
        ph["other"] += total - draft - verify - prep - expand
    ms_round = {k: v / rounds * 1000 for k, v in ph.items()}
    return acc_w / rounds, tok / t, ms_round


def step_overall(res, before, after, stem, subtitle, budgets=None):
    budgets = budgets or sorted(res, key=int)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.suptitle(f"{ARM_LABEL[before]}  →  {ARM_LABEL[after]}", fontsize=14, fontweight="bold", x=0.02, ha="left")
    axes[0].set_title("Decode throughput (tok/s, sync-on)", loc="left", fontsize=11)
    axes[1].set_title("Mean acceptance (tokens/round)", loc="left", fontsize=11)
    width = 0.36
    for ax, metric in ((axes[0], "tps"), (axes[1], "acc")):
        for i, bk in enumerate(budgets):
            acc_b, tps_b, _ = agg(res, bk, before)
            acc_a, tps_a, _ = agg(res, bk, after)
            vb, va = (tps_b, tps_a) if metric == "tps" else (acc_b, acc_a)
            fmt = (lambda v: f"{v:.0f}") if metric == "tps" else (lambda v: f"{v:.2f}")
            b1 = ax.bar(i - width / 2, vb, width * 0.94, color=BEFORE, zorder=3)
            b2 = ax.bar(i + width / 2, va, width * 0.94, color=AFTER, zorder=3)
            ax.bar_label(b1, labels=[fmt(vb)], padding=2, fontsize=10)
            ax.bar_label(b2, labels=[fmt(va)], padding=2, fontsize=10, fontweight="bold")
            if metric == "tps":
                ax.text(i + width / 2, va * 0.5, f"×{va / vb:.2f}", ha="center",
                        color="white", fontsize=11, fontweight="bold", zorder=4)
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([f"budget {b}" for b in budgets])
        ax.grid(axis="y", color="#e8e8e8", zorder=0)
        ax.margins(y=0.18)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=BEFORE),
                plt.Rectangle((0, 0), 1, 1, color=AFTER)],
               [ARM_LABEL[before], ARM_LABEL[after]],
               frameon=False, loc="upper right", ncol=2, fontsize=9,
               bbox_to_anchor=(0.99, 0.99))
    fig.text(0.02, 0.015, subtitle + "   ·   gsm8k/humaneval/mt-bench, 512 tok, seed 1, one H100", fontsize=8.5, color="#777777")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(OUT / f"{stem}_overall.png", dpi=160)
    plt.close(fig)


def step_phases(res, before, after, stem, budgets=None):
    budgets = budgets or sorted(res, key=int)
    fig, ax = plt.subplots(figsize=(10.5, 3.4 + 0.4 * len(budgets)))
    rows = []
    for bk in budgets:
        rows.append((f"{ARM_LABEL[before]}  (b{bk})", agg(res, bk, before)[2]))
        rows.append((f"{ARM_LABEL[after]}  (b{bk})", agg(res, bk, after)[2]))
    ys = range(len(rows))[::-1]
    for y, (label, ph) in zip(ys, rows):
        x = 0.0
        for name, color in PHASE_COLORS.items():
            v = ph[name]
            ax.barh(y, v, left=x, height=0.62, color=color, zorder=3,
                    edgecolor=SURFACE, linewidth=1.5)
            if v > sum(ph.values()) * 0.06:
                ax.text(x + v / 2, y, f"{v:.1f}", va="center", ha="center",
                        color="white", fontsize=9, zorder=4)
            x += v
        ax.text(x + max(x * 0.01, 0.3), y, f"{x:.1f} ms/round", va="center", fontsize=9.5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlabel("decode time per round (ms), instrumented pass")
    ax.grid(axis="x", color="#e8e8e8", zorder=0)
    ax.margins(x=0.12)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in PHASE_COLORS.values()],
              list(PHASE_COLORS), frameon=False, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), fontsize=9)
    fig.suptitle(f"Where the time goes: {ARM_LABEL[before]} → {ARM_LABEL[after]}",
                 fontsize=13, fontweight="bold", x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}_phases.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def ladder_hero(res, budgets=None, suffix=""):
    budgets = budgets or sorted(res, key=int)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    n = len(ARMS)
    width = 0.34
    shades = ["#c9c9c9", "#f0a8a7", "#ea7472", "#e34948"]  # neutral -> ours ramp
    mults = []
    for j, bk in enumerate(budgets):
        vals = [agg(res, bk, a)[1] for a in ARMS]
        mults.append(vals[-1] / vals[0])
        for i, (a, v) in enumerate(zip(ARMS, vals)):
            x = i + (j - 0.5) * width
            bar = ax.bar(x, v, width * 0.92, color=shades[i], zorder=3)
            ax.bar_label(bar, labels=[f"{v:.0f}"], padding=2, fontsize=9)
    mtxt = " · ".join(f"×{m:.1f} (b{bk})" for m, bk in zip(mults, budgets))
    ax.text(0.03, 0.96, f"total: {mtxt} vs naive",
            transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=10)
    ax.set_ylabel("decode tok/s (sync-on)")
    ax.grid(axis="y", color="#e8e8e8", zorder=0)
    ax.margins(y=0.22)
    title_m = "–".join(f"{m:.1f}" for m in sorted(mults))
    fig.suptitle(f"The speedup ladder: same trees, {title_m}× faster construction",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    group_note = ("per group: left bar = budget 64, right bar = budget 128  ·  "
                  if len(budgets) > 1 else f"budget {budgets[0]}  ·  ")
    fig.text(0.02, 0.015, group_note +
             "acceptance identical across A1–A3 (bit-identical trees); A0 differs only by full-vocab candidates",
             fontsize=8.5, color="#777777")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(OUT / f"ladder_overall{suffix}.png", dpi=160)
    plt.close(fig)


def main():
    res = load()
    for before, after, stem, subtitle in STEPS:
        step_overall(res, before, after, stem, subtitle)
        step_phases(res, before, after, stem)
        # budget-64-only copies
        step_overall(res, before, after, f"{stem}_b64", subtitle, budgets=["64"])
        step_phases(res, before, after, f"{stem}_b64", budgets=["64"])
    ladder_hero(res)
    ladder_hero(res, budgets=["64"], suffix="_b64")
    print("wrote:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
