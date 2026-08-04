"""Generate charts from an Experiment 3 (phase-timing) summary.json.

Reads a summary produced by modal_benchmark.py and writes PNGs next to it:

  tps.png               headline: net decode tok/s per arm (clean pass), one panel per
                        budget; instrumented TPS overlaid as a hollow marker so the
                        instrumentation tax is visible rather than hidden
  phase_breakdown.png   stacked horizontal bars, seconds per OUTPUT TOKEN, one bar per
                        arm, faceted by budget; segments = the 8 canonical phases
  phase_share.png       the same data 100%-normalized -- the hardware-portable view,
                        and the one that shows candidate_build swallowing the naive tree
  tree_cost_scaling.png candidate_build vs verify sec/token, budget 64 -> 256, tree arms
                        only -- separates "the tree got expensive to build" from "...to
                        verify"

This experiment measures WALL CLOCK, not acceptance, so the axes are seconds and
tokens/sec rather than acceptance length. Two things follow from that and shape every
chart here:

  * Two passes. `clean` (timers off) gives the headline throughput; `instrumented`
    (timers on) gives the per-phase split but pays a per-barrier instrumentation tax
    that is *uneven* across arms (tree arms fire more barriers). So absolute phase
    seconds are shown rescaled to the clean-pass throughput -- phase SHARES come from
    the instrumented pass, the total they scale is the clean pass -- and the charts say
    so. `phase_share.png` needs no rescale (it is internally normalized).
  * `null` != `0`. A phase an arm structurally does not have (e.g. a chain arm has no
    `candidate_pack`) is `null` in the summary, meaning "not applicable", not "measured,
    took no time". Null phases are drawn as an ABSENT segment and the arm is flagged
    `n/a` next to its bar -- never plotted as a zero-width sliver.

Palette follows exp1/exp2: cool hues for the DFlash-family arms, warm for DSpark-family;
chain arms hatched, tree arms solid, so arm identity never rests on color alone. Phase
segments use a single sequential colormap ramp ordered by `config.phase_order`, so the
same phase is the same shade in every chart.

Usage:
    python make_charts.py [path/to/summary.json]      # default: ./summary.json
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Palette. Arm identity is a family hue (cool = DFlash backbone, warm = DSpark) plus a
# texture (chain = hatched, tree = solid) -- the exp1/exp2 convention, kept for
# cross-experiment consistency. Within a family the two arms take a light/saturated pair
# of the same hue so the family reads at a glance and the chain/tree split reads within
# it. These are exp1's validated categorical hues.
ARM_COLOR = {
    "dflash.chain":            "#6aa9e9",  # light blue   (DFlash family, chain)
    "ddtree":                  "#2a78d6",  # blue         (DFlash family, tree)
    "dspark_b7.chain":         "#f0a15a",  # light orange (DSpark family, chain)
    "dspark_b16.markov.tree":  "#e34948",  # red          (DSpark family, tree)
}
# Canonical arm order (the reproduce.md arm table): DFlash chain, DSpark chain, then the
# two tree arms. Charts filter this to whatever is actually present.
ARM_ORDER = ["dflash.chain", "dspark_b7.chain", "ddtree", "dspark_b16.markov.tree"]
FALLBACK_ARM = "#8a8a8a"

# Presentation names for the tps ("Decode speeds") chart x-axis; every other chart keeps
# the raw arm ids.
TPS_ARM_NAME = {
    "dflash.chain":           "DFlash",
    "dspark_b7.chain":        "DSpark",
    "ddtree":                 "DDTree",
    "dspark_b16.markov.tree": "SparklingTree_b16",
}

# Canonical phase set, used only if config.phase_order is absent (it should not be).
CANON_PHASES = ["draft_forward", "candidate_build", "candidate_pack", "verify",
                "walk_accept", "kv_update", "state_carry", "unaccounted"]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def _caption(fig, text):
    fig.text(0.01, 0.005, text, fontsize=8, color=INK2, ha="left", va="bottom", wrap=True)


def _is_chain(arm):
    """Texture channel: chain arms are hatched, tree arms solid."""
    return ".chain" in arm


def _arm_color(arm):
    return ARM_COLOR.get(arm, FALLBACK_ARM)


def _budgets(summary):
    """Budget keys (strings) present anywhere under results, ordered numerically.

    Budget-invariant (chain) arms are written under every budget key by the harness, so
    simply iterating the keys treats them uniformly -- no special-casing on read."""
    seen = set()
    for pass_ in (summary.get("results") or {}).values():
        seen.update(pass_ or {})
    return sorted(seen, key=lambda b: int(b))


def _present_arms(summary, budget, pass_="instrumented"):
    """Arms present under a (pass, budget), in canonical order, then any extras."""
    node = (summary.get("results") or {}).get(pass_, {}).get(budget, {})
    present = set()
    for arms in node.values():
        present.update(arms or {})
    ordered = [a for a in ARM_ORDER if a in present]
    ordered += [a for a in sorted(present) if a not in ARM_ORDER]
    return ordered


def _phase_colors(phase_order):
    """One sequential ramp over phase_order -- same phase, same shade, every chart.

    Sampled off the ends of viridis (perceptually uniform, CVD-safe) so the lightest and
    darkest steps stay legible against the surface and the ink labels."""
    cmap = plt.get_cmap("viridis")
    n = max(len(phase_order), 1)
    lo, hi = 0.12, 0.92
    return {p: cmap(lo + (hi - lo) * (i / max(n - 1, 1)))
            for i, p in enumerate(phase_order)}


def _budget_invariant_arms(summary):
    """Arms flagged budget_invariant anywhere in the results -- derived, not assumed.

    The flag rides on the clean entries; chain arms carry it True. Charts use this only
    to *label* the repeated bars, never to change what is read."""
    inv = set()
    for pass_ in (summary.get("results") or {}).values():
        for budget in (pass_ or {}).values():
            for arms in (budget or {}).values():
                for arm, e in (arms or {}).items():
                    if isinstance(e, dict) and e.get("budget_invariant"):
                        inv.add(arm)
    return inv


def _arm_tps(summary, pass_, budget, arm):
    """Output-token-weighted decode throughput for one (pass, budget, arm), aggregated
    across datasets as total_tokens / total_seconds (the physically correct aggregate,
    which is the token-weighted harmonic mean of the per-dataset rates).

    Returns None if the arm has no usable entry under this (pass, budget)."""
    node = (summary.get("results") or {}).get(pass_, {}).get(budget, {})
    tot_tok, tot_sec = 0.0, 0.0
    for arms in node.values():
        e = (arms or {}).get(arm)
        if not e:
            continue
        tps, tok = e.get("tps_decode"), e.get("output_tokens")
        if not tps or not tok:
            continue
        tot_tok += tok
        tot_sec += tok / tps
    return (tot_tok / tot_sec) if tot_sec else None


def _arm_phase_secs(summary, budget, arm, phase_order):
    """Per-phase seconds for one (budget, arm) from the instrumented pass, summed across
    datasets (token-weighting is implicit: seconds and tokens both accumulate).

    Returns (secs, tokens) where secs[phase] is a float, or None when the phase is n/a
    for this arm (null in every dataset, or structurally absent) -- the caller renders
    None as an absent segment, never as a 0. `prefer sec`; falls back to
    sec_per_token * output_tokens if a summary only carried the rate."""
    node = (summary.get("results") or {}).get("instrumented", {}).get(budget, {})
    secs = {p: None for p in phase_order}
    tot_tok = 0.0
    for arms in node.values():
        e = (arms or {}).get(arm)
        if not e:
            continue
        tok = e.get("output_tokens") or 0
        tot_tok += tok
        phases = e.get("phases") or {}
        for p in phase_order:
            v = phases.get(p)          # missing key or explicit null both -> None
            if v is None:
                continue
            sec = v.get("sec")
            if sec is None and v.get("sec_per_token") is not None:
                sec = v["sec_per_token"] * tok
            if sec is None:
                continue
            secs[p] = (secs[p] or 0.0) + sec
    return secs, tot_tok


def _rescaled_spt(summary, budget, arm, phase_order):
    """Absolute seconds-per-OUTPUT-TOKEN per phase, rescaled to the CLEAN throughput.

    Phase shares are taken from the instrumented pass; the total they scale is the clean
    pass's seconds/token (1 / clean tps). This strips the uneven instrumentation tax so
    cross-arm absolute bars are comparable -- the honest absolute view the reproduce.md
    "two passes" section requires. Returns (spt, na_phases, ok):
      spt        {phase: sec_per_token} for phases the arm has,
      na_phases  ordered list of phases that are n/a for this arm,
      ok         False if the arm has no instrumented data at all (skip it)."""
    secs, _ = _arm_phase_secs(summary, budget, arm, phase_order)
    total = sum(v for v in secs.values() if v is not None)
    na = [p for p in phase_order if secs[p] is None]
    if total <= 0:
        return {}, na, False
    clean_tps = _arm_tps(summary, "clean", budget, arm)
    # Fall back to the instrumented total if there is no clean pass to rescale onto.
    clean_spt = (1.0 / clean_tps) if clean_tps else None
    shares = {p: (secs[p] / total) for p in phase_order if secs[p] is not None}
    if clean_spt is None:
        # No clean anchor: report instrumented sec/token directly (rare; note in caption).
        instr_tps = _arm_tps(summary, "instrumented", budget, arm)
        clean_spt = (1.0 / instr_tps) if instr_tps else total  # last resort: raw total
    spt = {p: shares[p] * clean_spt for p in shares}
    return spt, na, True


def _phase_shares(summary, budget, arm, phase_order):
    """100%-normalized instrumented shares per phase. Returns (shares, na_phases, ok).

    No rescale -- shares are internally normalized, so this is the view that survives a
    hardware change unchanged."""
    secs, _ = _arm_phase_secs(summary, budget, arm, phase_order)
    total = sum(v for v in secs.values() if v is not None)
    na = [p for p in phase_order if secs[p] is None]
    if total <= 0:
        return {}, na, False
    return {p: secs[p] / total for p in phase_order if secs[p] is not None}, na, True


# ---------------------------------------------------------------------------

def chart_tps(summary, out):
    """Headline: net decode tokens/sec per arm (clean pass), one panel per budget.

    Bars are the clean-pass throughput (the number a user actually gets); a hollow
    marker on each bar is the instrumented-pass throughput, so the vertical gap between
    bar-top and marker *is* the instrumentation tax, shown rather than hidden. Chain
    arms are budget-invariant -- their bar is identical across panels -- so they are
    labeled `(budget-invariant)` to stop the repetition reading as two measurements."""
    budgets = [b for b in _budgets(summary) if b == "64"]
    if not budgets:
        print("chart_tps: tree budget 64 not in summary; skipping")
        return
    inv = _budget_invariant_arms(summary)

    # Shared y across panels so throughput is comparable budget-to-budget.
    ymax = 0.0
    for bk in budgets:
        for arm in _present_arms(summary, bk, "clean"):
            for p in ("clean", "instrumented"):
                v = _arm_tps(summary, p, bk, arm)
                if v:
                    ymax = max(ymax, v)
    ymax = ymax or 1.0

    ncols = len(budgets)
    fig, axes = plt.subplots(1, ncols, figsize=(1.7 * len(ARM_ORDER) * ncols + 2, 5.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for panel, (bk, ax) in enumerate(zip(budgets, axes)):
        arms = _present_arms(summary, bk, "clean")
        if not arms:
            print(f"chart_tps: budget {bk} has no arms; leaving panel blank")
        xs = list(range(len(arms)))
        for x, arm in zip(xs, arms):
            clean = _arm_tps(summary, "clean", bk, arm)
            if clean is None:
                print(f"chart_tps: {arm} @ budget {bk} missing clean tps; skipping bar")
                continue
            col = _arm_color(arm)
            ax.bar(x, clean, width=0.66, color=col, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3)
            ax.text(x, clean + ymax * 0.015, f"{clean:.0f}", ha="center", va="bottom",
                    fontsize=9, color=INK2)
        _style(ax)
        ax.set_ylim(0, ymax * 1.15)
        ax.set_xticks(xs)
        ax.set_xticklabels([TPS_ARM_NAME.get(a, a) for a in arms],
                           fontsize=9, rotation=0)
        if panel == 0:
            ax.set_ylabel("net decode throughput (output tokens / sec)")
        ax.set_title(f"tree budget {bk}", fontsize=12, fontweight="bold", loc="left")

    fig.suptitle("Decode speeds",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def _stacked_phase_chart(summary, out, *, normalized):
    """Shared engine for phase_breakdown (absolute sec/token) and phase_share (100%).

    Stacked horizontal bars, one bar per arm, faceted by budget. Segments are the
    canonical phases in `phase_order`, each its fixed ramp shade. A phase that is n/a for
    an arm is an ABSENT segment (not a 0), and the arm is annotated `n/a: <phases>` past
    the end of its bar so the missing segment is a stated fact, not a silent gap."""
    cfg = summary.get("config") or {}
    phase_order = cfg.get("phase_order") or CANON_PHASES
    pcolor = _phase_colors(phase_order)
    budgets = _budgets(summary)
    if not budgets:
        print(f"{out.name}: no budgets; skipping")
        return

    # Shared x so bars are comparable across budgets (absolute chart only; the share
    # chart is fixed to [0, 1]).
    xmax = 1.0 if normalized else 0.0
    per_panel = {}   # budget -> [(arm, spt_or_shares, na_phases)]
    for bk in budgets:
        rows = []
        for arm in _present_arms(summary, bk, "instrumented"):
            if normalized:
                vals, na, ok = _phase_shares(summary, bk, arm, phase_order)
            else:
                vals, na, ok = _rescaled_spt(summary, bk, arm, phase_order)
            if not ok:
                print(f"{out.name}: {arm} @ budget {bk} has no instrumented phases; skipping")
                continue
            rows.append((arm, vals, na))
            if not normalized:
                xmax = max(xmax, sum(vals.values()))
        per_panel[bk] = rows
    xmax = xmax or 1.0

    ncols = len(budgets)
    nrows_max = max((len(per_panel[bk]) for bk in budgets), default=1)
    fig, axes = plt.subplots(1, ncols, figsize=(7.2 * ncols, 1.05 * nrows_max + 3.0),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for panel, (bk, ax) in enumerate(zip(budgets, axes)):
        rows = per_panel[bk]
        ys = list(range(len(rows)))[::-1]   # first arm at the top
        for y, (arm, vals, na) in zip(ys, rows):
            left = 0.0
            for p in phase_order:
                if p not in vals:
                    continue            # n/a -> absent segment (never a 0-width sliver)
                w = vals[p]
                ax.barh(y, w, left=left, height=0.62, color=pcolor[p],
                        edgecolor=SURFACE, linewidth=1.2, zorder=3)
                left += w
            # State the missing segments in words, at the end of the bar.
            if na:
                ax.text(left + xmax * 0.01, y, "n/a: " + ", ".join(na),
                        ha="left", va="center", fontsize=8, color=INK2, style="italic")
        _style(ax, axis="x")
        ax.set_yticks(list(ys))
        if panel == 0:
            ax.set_yticklabels([arm for arm, _, _ in rows], fontsize=10, fontweight="bold")
        else:
            ax.tick_params(labelleft=False)
        ax.set_ylim(-0.7, max(len(rows) - 0.3, 0.3))
        ax.set_xlim(0, xmax * (1.28 if not normalized else 1.0))
        if normalized:
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("share of decode wall clock")
        else:
            ax.set_xlabel("seconds per output token (rescaled to clean throughput)")
        ax.set_title(f"tree budget {bk}", fontsize=12, fontweight="bold", loc="left")

    # Phase legend, in ramp order so the legend doubles as the phase ordering key.
    handles = [Patch(facecolor=pcolor[p], edgecolor=SURFACE, label=p) for p in phase_order]
    fig.legend(handles=handles, frameon=False, fontsize=9,
               ncol=min(len(phase_order), 4), loc="lower center", bbox_to_anchor=(0.5, 0.055))

    if normalized:
        fig.suptitle("Where the decode time goes: phase shares by arm and tree budget",
                     fontsize=13.5, fontweight="bold", x=0.01, ha="left")
        _caption(fig, "100%-normalized phase shares from the instrumented pass. Shares are "
                      "internally normalized, so this view is unchanged by the instrumentation "
                      "tax and is the hardware-portable one. Phases an arm does not have are "
                      "n/a (absent), not 0.")
    else:
        fig.suptitle("Decode time per output token by phase, arm and tree budget",
                     fontsize=13.5, fontweight="bold", x=0.01, ha="left")
        _caption(fig, "Per-token seconds. Phase SHARES come from the instrumented pass; the total "
                      "they scale is the CLEAN pass throughput (1 / clean tps), so the uneven "
                      "instrumentation tax is removed and bars are cross-arm comparable. Phases "
                      "an arm does not have are n/a (absent), not 0.")
    fig.tight_layout(rect=[0, 0.17, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def chart_phase_breakdown(summary, out):
    _stacked_phase_chart(summary, out, normalized=False)


def chart_phase_share(summary, out):
    _stacked_phase_chart(summary, out, normalized=True)


def chart_tree_cost_scaling(summary, out):
    """candidate_build vs verify, seconds/output-token, budget 64 -> 256, tree arms only.

    The design question the budget sweep exists to answer: as the node budget grows,
    does the tree get expensive to BUILD (candidate_build, ~linear in nodes) or expensive
    to VERIFY (verify, one wider forward, ~sublinear)? Two lines per tree arm -- color =
    arm (the family hue), linestyle = phase (candidate_build solid, verify dashed) -- so
    each arm's two slopes are directly readable. Chain arms have no budget knob and are
    omitted; this chart is only about the tree arms' scaling."""
    cfg = summary.get("config") or {}
    phase_order = cfg.get("phase_order") or CANON_PHASES
    budgets = [b for b in _budgets(summary)]
    if len(budgets) < 2:
        print("chart_tree_cost_scaling: need >=2 budgets to show scaling; "
              f"have {budgets}, plotting what exists")
    if not budgets:
        print("chart_tree_cost_scaling: no budgets; skipping")
        return

    # Tree arms present anywhere, in canonical order.
    tree_arms = []
    for bk in budgets:
        for arm in _present_arms(summary, bk, "instrumented"):
            if not _is_chain(arm) and arm not in tree_arms:
                tree_arms.append(arm)
    if not tree_arms:
        print("chart_tree_cost_scaling: no tree arms found; skipping")
        return

    PHASES = [p for p in ("candidate_build", "verify") if p in phase_order] or \
             ["candidate_build", "verify"]
    style = {"candidate_build": "-", "verify": "--"}
    xvals = [int(b) for b in budgets]

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ymax = 0.0
    plotted = False
    for arm in tree_arms:
        col = _arm_color(arm)
        for ph in PHASES:
            ys = []
            for bk in budgets:
                spt, _, ok = _rescaled_spt(summary, bk, arm, phase_order)
                ys.append(spt.get(ph) if ok else None)
            pts = [(x, y) for x, y in zip(xvals, ys) if y is not None]
            if not pts:
                continue
            px, py = zip(*pts)
            ax.plot(px, py, color=col, linewidth=2.2, linestyle=style.get(ph, "-"),
                    marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    zorder=3)
            ymax = max(ymax, max(py))
            plotted = True
    if not plotted:
        print("chart_tree_cost_scaling: neither phase had data; skipping")
        plt.close(fig)
        return

    _style(ax)
    ax.set_xticks(xvals)
    ax.set_xlim(min(xvals) - (max(xvals) - min(xvals) or 1) * 0.12,
                max(xvals) + (max(xvals) - min(xvals) or 1) * 0.12)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlabel("tree budget (nodes per round)")
    ax.set_ylabel("seconds per output token (rescaled to clean throughput)")

    # Two legends: arm identity (color) and phase (linestyle) -- identity never rests on
    # color alone.
    arm_handles = [Line2D([0], [0], color=_arm_color(a), linewidth=3, label=a)
                   for a in tree_arms]
    ph_handles = [Line2D([0], [0], color=INK2, linewidth=2.2, linestyle=style.get(p, "-"),
                         label=p) for p in PHASES]
    leg1 = ax.legend(handles=arm_handles, frameon=False, fontsize=9, loc="upper left",
                     title="tree arm")
    ax.add_artist(leg1)
    ax.legend(handles=ph_handles, frameon=False, fontsize=9, loc="upper right",
              title="phase")

    lo, hi = (str(min(xvals)), str(max(xvals)))
    fig.suptitle(f"Tree cost scaling: build vs verify per token, budget {lo} to {hi}",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    _caption(fig, "Per-token seconds, rescaled to clean throughput. Solid = candidate_build "
                  "(building the tree), dashed = verify (the target forward). The two slopes "
                  "say whether growing the budget costs on the drafting side or the verify side.")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# The pies show only the phases the experiment's question is about; everything else
# (pack, walk, KV, carry, unaccounted -- <=4% on every arm) folds into one gray slice.
PIE_MAIN_PHASES = ["draft_forward", "candidate_build", "verify"]
PIE_OTHER_LABEL = "everything else"
PIE_OTHER_COLOR = "#b5b4b0"
# Short presentation names, pie chart only; every other chart keeps the arm ids.
PIE_ARM_NAME = {
    "dflash.chain": "dflash",
    "dspark_b7.chain": "dspark",
    "ddtree": "ddtree",
    "dspark_b16.markov.tree": "SparklingTree_b16",
}


def chart_phase_pies(summary, out, budget="64"):
    """Four side-by-side pies, one per arm, phase shares at a single tree budget.

    Same data as one panel of phase_share.png, re-cut as part-to-whole per arm and
    simplified: only draft_forward / candidate_build / verify keep their own slice (they
    are the experiment's question); the five bookkeeping phases merge into a single gray
    `everything else` slice. Slices >= 3% carry a direct percentage label; the dominant
    slice is also named."""
    cfg = summary.get("config") or {}
    phase_order = cfg.get("phase_order") or CANON_PHASES
    pcolor = _phase_colors(phase_order)
    pcolor[PIE_OTHER_LABEL] = PIE_OTHER_COLOR
    if budget not in _budgets(summary):
        print(f"chart_phase_pies: budget {budget} not in summary; skipping")
        return
    arms = _present_arms(summary, budget, "instrumented")
    if not arms:
        print(f"chart_phase_pies: no arms at budget {budget}; skipping")
        return

    fig, axes = plt.subplots(1, len(arms), figsize=(4.4 * len(arms), 5.4), squeeze=False)
    axes = axes[0]
    for ax, arm in zip(axes, arms):
        shares, _na, ok = _phase_shares(summary, budget, arm, phase_order)
        spt, _na2, spt_ok = _rescaled_spt(summary, budget, arm, phase_order)
        if not ok:
            ax.axis("off")
            print(f"chart_phase_pies: {arm} @ budget {budget} has no instrumented phases")
            continue
        merged = {p: shares[p] for p in PIE_MAIN_PHASES if p in shares}
        other = sum(v for p, v in shares.items() if p not in PIE_MAIN_PHASES)
        if other > 0:
            merged[PIE_OTHER_LABEL] = other
        merged_ms = {p: spt.get(p, 0.0) * 1000 for p in PIE_MAIN_PHASES} if spt_ok else {}
        if spt_ok:
            merged_ms[PIE_OTHER_LABEL] = sum(
                v * 1000 for p, v in spt.items() if p not in PIE_MAIN_PHASES)
        phases = [p for p in PIE_MAIN_PHASES + [PIE_OTHER_LABEL] if p in merged]
        vals = [merged[p] for p in phases]
        wedges, _texts, autotexts = ax.pie(
            vals, colors=[pcolor[p] for p in phases], startangle=90,
            counterclock=False, autopct=lambda pct: f"{pct:.0f}%" if pct >= 3 else "",
            pctdistance=0.7, wedgeprops={"edgecolor": SURFACE, "linewidth": 1.5},
        )
        for w, t, p, v in zip(wedges, autotexts, phases, vals):
            if t.get_text() and spt_ok:
                t.set_text(f"{t.get_text()}\n{merged_ms[p]:.1f} ms/tok")
            if v < 0.10 and t.get_text():
                # Small labeled slices move outside the pie, else they pile up on
                # their neighbors' labels near the crowded top of the circle.
                mid = math.radians((w.theta1 + w.theta2) / 2)
                t.set_position((1.22 * math.cos(mid), 1.22 * math.sin(mid)))
                t.set_ha("left" if math.cos(mid) >= 0 else "right")
                t.set_color(INK)
            else:
                r, g, b, _ = w.get_facecolor()
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                t.set_color(SURFACE if lum < 0.5 else INK)
            t.set_fontsize(9.5)
        # Direct-label the dominant slice with its phase name, not just a percent.
        top = max(phases, key=lambda p: merged[p])
        ang = 90 - 360 * (sum(merged[p] for p in phases[:phases.index(top)]) + merged[top] / 2)
        ax.annotate(top, xy=(0.78 * math.cos(math.radians(ang)),
                             0.78 * math.sin(math.radians(ang))),
                    xytext=(1.25 * math.cos(math.radians(ang)),
                            1.25 * math.sin(math.radians(ang))),
                    ha="center", va="center", fontsize=9, color=INK,
                    arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 1})
        ax.set_title(PIE_ARM_NAME.get(arm, arm) + "\n", fontsize=11, fontweight="bold")

    handles = [Patch(facecolor=pcolor[p], edgecolor=SURFACE, label=p)
               for p in PIE_MAIN_PHASES + [PIE_OTHER_LABEL]]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.04))
    fig.suptitle(f"Phase shares of decode wall clock, tree budget {budget}",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "summary.json"
    summary = json.loads(Path(path).read_text())
    outdir = Path(path).parent
    chart_tps(summary, outdir / "tps.png")
    chart_phase_breakdown(summary, outdir / "phase_breakdown.png")
    chart_phase_share(summary, outdir / "phase_share.png")
    chart_tree_cost_scaling(summary, outdir / "tree_cost_scaling.png")
    chart_phase_pies(summary, outdir / "phase_pies_b64.png", budget="64")


if __name__ == "__main__":
    main()
