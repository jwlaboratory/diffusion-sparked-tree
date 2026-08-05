"""Compute the three deltas for the fast best-first builder vs the slow reference.

Loads results/summary.json (schema: pass -> budget -> dataset -> arm) and prints,
for BOTH tree budgets:

  (a) TIME    candidate_build ms/round and ms/committed-token, ref vs fast, with
              the .prep (tree_build_copy) and .expand (tree_build_heap) subphases
              broken out, plus total ms/round. Aggregated across the 3 datasets by
              summing seconds and dividing by summed rounds / tokens (exp3 method).
  (b) ACCEPT  mean_accept ref vs fast per dataset and round-weighted aggregate;
              the delta in accepted tokens/round and as a %.
  (c) SPEEDUP tps_clean fast/ref per budget and per dataset, stated as a multiple.

Also checks summary["checks"]["acceptance_match"].

Run:  python analyze.py [path/to/summary.json]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF, FAST = "bestfirst.ref", "bestfirst.fast"
# Subphase -> the native tree_build_* key it corresponds to (for the reader).
SUBPHASES = [
    ("candidate_build.prep", "tree_build_copy", ".prep"),
    ("candidate_build.expand", "tree_build_heap", ".expand"),
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


def pct_red(ref, fast):
    return 100.0 * (ref - fast) / ref if ref else float("nan")


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
        print(f"TREE BUDGET {budget}")
        print("=" * 74)
        ref = agg_instrumented(summary, budget, REF)
        fast = agg_instrumented(summary, budget, FAST)

        # ---- (a) TIME ----------------------------------------------------------
        print("(a) TIME  candidate_build, ref vs fast (instrumented, ds-summed)")
        cb_ref = ref["phase_sec"].get("candidate_build", 0.0)
        cb_fast = fast["phase_sec"].get("candidate_build", 0.0)
        print(f"    {'metric':<34}{'ref':>12}{'fast':>12}{'reduction':>12}")

        def row(label, r_sec, f_sec, r_den, f_den):
            r_ms, f_ms = ms_per(r_sec, r_den), ms_per(f_sec, f_den)
            print(f"    {label:<34}{r_ms:>11.2f}{f_ms:>12.2f}{pct_red(r_ms, f_ms):>11.1f}%")

        row("candidate_build ms/round", cb_ref, cb_fast, ref["rounds"], fast["rounds"])
        row("candidate_build ms/commit-token", cb_ref, cb_fast, ref["tokens"], fast["tokens"])
        for skey, native, short in SUBPHASES:
            row(f"  {short} ({native}) ms/round",
                ref["sub_sec"].get(skey, 0.0), fast["sub_sec"].get(skey, 0.0),
                ref["rounds"], fast["rounds"])
        tot_ref = sum(ref["phase_sec"].values())
        tot_fast = sum(fast["phase_sec"].values())
        row("TOTAL all-phase ms/round", tot_ref, tot_fast, ref["rounds"], fast["rounds"])
        print(f"    (rounds ref={ref['rounds']} fast={fast['rounds']}; "
              f"tokens ref={ref['tokens']} fast={fast['tokens']})")

        # ---- (b) ACCEPT --------------------------------------------------------
        print("\n(b) ACCEPT  mean acceptance length, ref vs fast (clean)")
        acc_ref, per_ref = agg_accept(summary, budget, REF)
        acc_fast, per_fast = agg_accept(summary, budget, FAST)
        print(f"    {'dataset':<16}{'ref':>10}{'fast':>10}{'delta':>10}")
        for ds in sorted(set(per_ref) | set(per_fast)):
            r, f = per_ref.get(ds, float('nan')), per_fast.get(ds, float('nan'))
            print(f"    {ds:<16}{r:>10.3f}{f:>10.3f}{f - r:>10.3f}")
        d_abs = acc_fast - acc_ref
        d_pct = 100.0 * d_abs / acc_ref if acc_ref else float("nan")
        print(f"    {'AGG (round-wt)':<16}{acc_ref:>10.3f}{acc_fast:>10.3f}{d_abs:>10.3f}")
        print(f"    accepted tokens/round delta: {d_abs:+.4f}  ({d_pct:+.2f}%)")

        # ---- (c) SPEEDUP -------------------------------------------------------
        print("\n(c) SPEEDUP  tps_clean fast/ref")
        timing = summary.get("timing", {}).get(str(budget), {})
        tr = timing.get(REF, {}); tf = timing.get(FAST, {})
        if tr and tf:
            mult = tf["tps_clean"] / tr["tps_clean"] if tr["tps_clean"] else float("nan")
            print(f"    aggregate:  ref={tr['tps_clean']:.2f}  fast={tf['tps_clean']:.2f}  "
                  f"-> {mult:.2f}x")
            print(f"    ref dominant phase:  {tr['dominant_phase']} ({tr['dominant_share']*100:.0f}%)")
            print(f"    fast dominant phase: {tf['dominant_phase']} ({tf['dominant_share']*100:.0f}%)")
            pr, pf = tr.get("per_dataset", {}), tf.get("per_dataset", {})
            for ds in sorted(set(pr) | set(pf)):
                rr = pr.get(ds, {}).get("tps_clean", float('nan'))
                ff = pf.get(ds, {}).get("tps_clean", float('nan'))
                m = ff / rr if rr else float('nan')
                print(f"      {ds:<14} ref={rr:>8.2f}  fast={ff:>8.2f}  -> {m:.2f}x")
        print()


if __name__ == "__main__":
    main()
