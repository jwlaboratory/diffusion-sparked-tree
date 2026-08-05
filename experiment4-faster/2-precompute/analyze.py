"""Compute the three deltas for the precompute builder vs the transfer-less fast one.

Isolation: both arms are best-first at C=512, so the tree (hence acceptance) is
controlled and the deltas read as -- (a) how much candidate_build shrinks and where
(.expand collapsing into .prep), (b) acceptance stays flat (proof the tree is
unchanged), (c) the net wall-clock speedup of the precompute mechanism ALONE.

Loads results/summary.json (schema: pass -> budget -> dataset -> arm) and prints,
for BOTH tree budgets. Aggregates across the 3 datasets by summing seconds and
dividing by summed rounds / tokens (exp3 method).

Run:  python analyze.py [path/to/summary.json]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAST, PRE = "bestfirst.fast", "bestfirst.precompute"
# Subphase -> the native tree_build_* key it corresponds to (for the reader).
SUBPHASES = [
    ("candidate_build.prep", "tree_build_copy", ".prep (GPU precompute+xfer)"),
    ("candidate_build.expand", "tree_build_heap", ".expand (CPU walk)"),
    ("candidate_build.visibility", "tree_build_visibility", ".visibility"),
]


def budget_keys(summary):
    return sorted(summary["results"]["instrumented"], key=int)


def agg_instrumented(summary, budget, arm):
    """Sum seconds / rounds / tokens across the 3 datasets for one arm@budget."""
    by_ds = summary["results"]["instrumented"][str(budget)]
    tot = {"rounds": 0, "tokens": 0, "phase_sec": {}, "sub_sec": {}}
    for ds, arms in by_ds.items():
        e = arms.get(arm)
        if not e:
            continue
        tot["rounds"] += e["rounds"]
        tot["tokens"] += e["output_tokens"]
        for p, v in e.get("phases", {}).items():
            tot["phase_sec"][p] = tot["phase_sec"].get(p, 0.0) + v["sec"]
        for s, v in e.get("subphases", {}).items():
            tot["sub_sec"][s] = tot["sub_sec"].get(s, 0.0) + v["sec"]
    return tot


def agg_accept(summary, budget, arm, pass_name="clean"):
    """Round-weighted mean_accept across datasets, plus per-dataset values."""
    by_ds = summary["results"][pass_name][str(budget)]
    tot_accept, tot_rounds, per = 0.0, 0, {}
    for ds, arms in by_ds.items():
        e = arms.get(arm)
        if not e:
            continue
        per[ds] = e["mean_accept"]
        tot_accept += e["mean_accept"] * e["rounds"]
        tot_rounds += e["rounds"]
    return (tot_accept / tot_rounds if tot_rounds else float("nan")), per


def ms_per(sec, denom):
    return 1000.0 * sec / denom if denom else float("nan")


def pct_red(base, new):
    return 100.0 * (base - new) / base if base else float("nan")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "summary.json"
    summary = json.loads(path.read_text())

    checks = summary.get("checks", {})
    print(f"acceptance_match (clean vs instrumented agree): {checks.get('acceptance_match')}")
    if checks.get("mismatched_units"):
        print("  mismatched units:", checks["mismatched_units"])
    print()

    for budget in budget_keys(summary):
        print("=" * 74)
        print(f"TREE BUDGET {budget}   (baseline=fast, test=precompute, both C=512)")
        print("=" * 74)
        base = agg_instrumented(summary, budget, FAST)
        new = agg_instrumented(summary, budget, PRE)

        # ---- (a) TIME ----------------------------------------------------------
        print("(a) TIME  candidate_build, fast vs precompute (instrumented, ds-summed)")
        cb_base = base["phase_sec"].get("candidate_build", 0.0)
        cb_new = new["phase_sec"].get("candidate_build", 0.0)
        print(f"    {'metric':<40}{'fast':>11}{'precomp':>11}{'reduction':>11}")

        def row(label, b_sec, n_sec, b_den, n_den):
            b_ms, n_ms = ms_per(b_sec, b_den), ms_per(n_sec, n_den)
            print(f"    {label:<40}{b_ms:>10.2f}{n_ms:>11.2f}{pct_red(b_ms, n_ms):>10.1f}%")

        row("candidate_build ms/round", cb_base, cb_new, base["rounds"], new["rounds"])
        row("candidate_build ms/commit-token", cb_base, cb_new, base["tokens"], new["tokens"])
        for skey, native, short in SUBPHASES:
            row(f"  {short} ms/round",
                base["sub_sec"].get(skey, 0.0), new["sub_sec"].get(skey, 0.0),
                base["rounds"], new["rounds"])
        tot_base = sum(base["phase_sec"].values())
        tot_new = sum(new["phase_sec"].values())
        row("TOTAL all-phase ms/round", tot_base, tot_new, base["rounds"], new["rounds"])
        print(f"    (rounds fast={base['rounds']} precomp={new['rounds']}; "
              f"tokens fast={base['tokens']} precomp={new['tokens']})")

        # ---- (b) ACCEPT --------------------------------------------------------
        print("\n(b) ACCEPT  mean acceptance length, fast vs precompute (clean)")
        acc_base, per_base = agg_accept(summary, budget, FAST)
        acc_new, per_new = agg_accept(summary, budget, PRE)
        print(f"    {'dataset':<16}{'fast':>10}{'precomp':>10}{'delta':>10}")
        for ds in sorted(set(per_base) | set(per_new)):
            b, n = per_base.get(ds, float('nan')), per_new.get(ds, float('nan'))
            print(f"    {ds:<16}{b:>10.3f}{n:>10.3f}{n - b:>10.3f}")
        d_abs = acc_new - acc_base
        d_pct = 100.0 * d_abs / acc_base if acc_base else float("nan")
        print(f"    {'AGG (round-wt)':<16}{acc_base:>10.3f}{acc_new:>10.3f}{d_abs:>10.3f}")
        print(f"    accepted tokens/round delta: {d_abs:+.4f}  ({d_pct:+.2f}%)")

        # ---- (c) SPEEDUP -------------------------------------------------------
        print("\n(c) SPEEDUP  tps_clean precompute/fast")
        timing = summary.get("timing", {}).get(str(budget), {})
        tb = timing.get(FAST, {}); tn = timing.get(PRE, {})
        if tb and tn:
            mult = tn["tps_clean"] / tb["tps_clean"] if tb["tps_clean"] else float("nan")
            print(f"    aggregate:  fast={tb['tps_clean']:.2f}  precomp={tn['tps_clean']:.2f}  "
                  f"-> {mult:.2f}x")
            print(f"    fast dominant phase:    {tb['dominant_phase']} ({tb['dominant_share']*100:.0f}%)")
            print(f"    precomp dominant phase: {tn['dominant_phase']} ({tn['dominant_share']*100:.0f}%)")
            pb, pn = tb.get("per_dataset", {}), tn.get("per_dataset", {})
            for ds in sorted(set(pb) | set(pn)):
                bb = pb.get(ds, {}).get("tps_clean", float('nan'))
                nn = pn.get(ds, {}).get("tps_clean", float('nan'))
                m = nn / bb if bb else float('nan')
                print(f"      {ds:<14} fast={bb:>8.2f}  precomp={nn:>8.2f}  -> {m:.2f}x")
        print()


if __name__ == "__main__":
    main()
