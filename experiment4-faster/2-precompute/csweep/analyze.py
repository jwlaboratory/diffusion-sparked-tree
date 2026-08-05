"""Analyze the candidate-size (C) sweep for the fast and precompute best-first builders.

Loads results/summary.json (schema: pass -> budget -> dataset -> arm) and, per builder
(fast, precompute), reports acceptance and clean TPS as a function of C, plus the
candidate_build .prep/.expand ms/round vs C, against the exact best-first ceiling
(arm `bestfirst.ref`). It then flags a KNEE: the smallest C whose round-weighted
acceptance is within tolerance of BOTH the ref ceiling and the builder's own largest-C
value. Because per-arm acceptance deltas are ~1-2% (near the n=8 noise floor), the
table also prints the per-dataset spread (max-min across the 3 datasets) so a knee is
never claimed on less than that spread.

Run:  python analyze.py [path/to/summary.json] [--tol 0.01]
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = "bestfirst.ref"
SUBPHASES = [
    ("candidate_build.prep", ".prep (GPU precompute+xfer)"),
    ("candidate_build.expand", ".expand (CPU walk)"),
]
ARM_RE = re.compile(r"^(fast|precompute)\.c(\d+)$")


def parse_args(argv):
    path, tol = None, 0.01
    it = iter(argv[1:])
    for a in it:
        if a == "--tol":
            tol = float(next(it))
        else:
            path = Path(a)
    return path, tol


def budget_keys(summary):
    return sorted(summary["results"]["instrumented"], key=int)


def sweep_arms(summary, budget):
    """Return {builder: {C: arm_name}} for the sweep arms present at this budget."""
    arms = summary["results"]["clean"][str(budget)]
    names = {n for by_ds in [arms] for d in arms.values() for n in d}
    out = {"fast": {}, "precompute": {}}
    for n in names:
        m = ARM_RE.match(n)
        if m:
            out[m.group(1)][int(m.group(2))] = n
    return out


def agg_instrumented(summary, budget, arm):
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


def tps_clean(summary, budget, arm):
    return summary.get("timing", {}).get(str(budget), {}).get(arm, {}).get("tps_clean", float("nan"))


def ms_per(sec, denom):
    return 1000.0 * sec / denom if denom else float("nan")


def spread(per):
    vals = [v for v in per.values() if v == v]
    return (max(vals) - min(vals)) if vals else float("nan")


def main():
    path, tol = parse_args(sys.argv)
    path = path or HERE / "results" / "summary.json"
    summary = json.loads(path.read_text())

    checks = summary.get("checks", {})
    print(f"acceptance_match (clean vs instrumented agree): {checks.get('acceptance_match')}")
    if checks.get("mismatched_units"):
        print("  mismatched units:", checks["mismatched_units"])

    for budget in budget_keys(summary):
        print("\n" + "=" * 78)
        print(f"TREE BUDGET {budget}   candidate-size sweep")
        print("=" * 78)

        ceil_acc, ceil_per = agg_accept(summary, budget, REF)
        ceil_tps = tps_clean(summary, budget, REF)
        print(f"CEILING  bestfirst.ref  acceptance={ceil_acc:.3f}  "
              f"(per-ds spread {spread(ceil_per):.3f})  tps_clean={ceil_tps:.2f}")

        arms = sweep_arms(summary, budget)
        for builder in ("fast", "precompute"):
            cmap = arms.get(builder, {})
            if not cmap:
                continue
            print(f"\n-- {builder} --------------------------------------------------------------")
            print(f"   {'C':>6}{'accept':>9}{'ds-spread':>11}{'vs ceil':>9}"
                  f"{'tps':>9}{'.prep ms/rd':>13}{'.expand ms/rd':>15}")
            rows = []
            for C in sorted(cmap):
                arm = cmap[C]
                acc, per = agg_accept(summary, budget, arm)
                ins = agg_instrumented(summary, budget, arm)
                prep = ms_per(ins["sub_sec"].get("candidate_build.prep", 0.0), ins["rounds"])
                expand = ms_per(ins["sub_sec"].get("candidate_build.expand", 0.0), ins["rounds"])
                tps = tps_clean(summary, budget, arm)
                vs_ceil = acc - ceil_acc
                rows.append((C, acc, per, tps, prep, expand))
                print(f"   {C:>6}{acc:>9.3f}{spread(per):>11.3f}{vs_ceil:>+9.3f}"
                      f"{tps:>9.2f}{prep:>13.3f}{expand:>15.3f}")

            # Knee: smallest C within tol of BOTH the ceiling and the builder's max-C
            # acceptance. tol is relative to ceiling acceptance. Also require the gap to
            # ceiling to be no larger than the per-dataset spread (noise guard).
            if rows:
                max_c_acc = rows[-1][1]
                abs_tol = tol * ceil_acc
                knee = None
                for C, acc, per, *_ in rows:
                    gap_ceil = ceil_acc - acc
                    gap_maxc = max_c_acc - acc
                    if gap_ceil <= max(abs_tol, spread(per)) and gap_maxc <= abs_tol:
                        knee = C
                        break
                if knee is not None:
                    kacc = next(r[1] for r in rows if r[0] == knee)
                    print(f"   KNEE (heuristic, tol={tol:.1%}): C={knee}  "
                          f"acceptance={kacc:.3f}  ({kacc - ceil_acc:+.3f} vs ceiling)")
                    print("   NB: acceptance deltas are within ~1 per-dataset-spread; treat the")
                    print("       knee as indicative, not a hard threshold (n=8 per dataset).")
                else:
                    print(f"   KNEE: none of the swept C reach within tol={tol:.1%} of the ceiling.")


if __name__ == "__main__":
    main()
