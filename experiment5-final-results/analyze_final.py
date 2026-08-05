"""Final-results tables: DFlash / DSpark / DDTree / SparklingTree.

For each tree budget, prints:
  1. AGGREGATE: per method -- mean acceptance, clean-pass TPS, speedup vs baseline
     (default Autoregressive), dominant phase; then SparklingTree's advantage %.
  2. PER-DOMAIN: the same acceptance / TPS / speedup broken out per dataset, since
     the markov corrector's edge varies a lot by domain (code vs math vs chat).

Run:  python analyze_final.py [results/summary.json] [--baseline Autoregressive] [--ours SparklingTree]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def accept(summary, bkey, name, pass_name="clean"):
    by_ds = summary["results"][pass_name][bkey]
    a, r, per = 0.0, 0, {}
    for ds, arms in by_ds.items():
        e = arms.get(name)
        if e:
            per[ds] = e["mean_accept"]
            a += e["mean_accept"] * e["rounds"]; r += e["rounds"]
    return (a / r if r else float("nan")), per


def tps(summary, bkey, name):
    return summary.get("timing", {}).get(bkey, {}).get(name, {}).get("tps_clean")


def pd_tps(summary, bkey, name, ds):
    """Per-dataset clean TPS for one method."""
    return (summary.get("timing", {}).get(bkey, {}).get(name, {})
            .get("per_dataset", {}).get(ds, {}).get("tps_clean"))


def pd_accept(summary, bkey, name, ds, pass_name="clean"):
    """Per-dataset mean acceptance for one method."""
    e = summary["results"][pass_name][bkey].get(ds, {}).get(name)
    return e["mean_accept"] if e else None


def dataset_names(summary):
    # Preserve config order (results dicts are insertion-ordered from the run).
    for bkey in summary["results"]["clean"]:
        return list(summary["results"]["clean"][bkey].keys())
    return []


def print_per_domain(summary, baseline, ours):
    """Per-dataset acceptance / TPS / speedup, one block per budget."""
    for bkey in sorted(summary.get("timing", {}), key=int):
        arms = list(summary["timing"][bkey])
        print("\n" + "=" * 80)
        print(f"PER-DOMAIN — tree budget {bkey}   (speedup vs {baseline})")
        print("=" * 80)
        for ds in dataset_names(summary):
            base_t = pd_tps(summary, bkey, baseline, ds)
            print(f"\n  {ds}")
            print(f"    {'method':<16}{'acceptance':>12}{'TPS(clean)':>13}{'speedup':>10}")
            rows = sorted(arms, key=lambda n: -((pd_tps(summary, bkey, n, ds)) or 0))
            for name in rows:
                a = pd_accept(summary, bkey, name, ds)
                t = pd_tps(summary, bkey, name, ds)
                if a is None or t is None:
                    continue
                spd = (t / base_t) if base_t else float("nan")
                mark = "  <-- ours" if name == ours else ""
                print(f"    {name:<16}{a:>12.3f}{t:>13.2f}{spd:>9.2f}×{mark}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else HERE / "results" / "summary.json"
    baseline = _opt("--baseline", "Autoregressive")
    ours = _opt("--ours", "SparklingTree")
    summary = json.loads(path.read_text())

    checks = summary.get("checks", {})
    print(f"acceptance_match: {checks.get('acceptance_match')}"
          + (f"   mismatched: {checks['mismatched_units']}" if checks.get("mismatched_units") else ""))
    methods = list(summary["config"]["methods"])
    print(f"methods: {', '.join(methods)}   |   speedup baseline: {baseline}")

    for bkey in sorted(summary.get("timing", {}), key=int):
        arms = summary["timing"][bkey]
        base_tps = tps(summary, bkey, baseline)
        print("\n" + "=" * 80)
        print(f"TREE BUDGET {bkey}")
        print("=" * 80)
        print(f"  {'method':<16}{'acceptance':>12}{'TPS(clean)':>13}{'speedup':>10}{'dominant phase':>18}")
        rows = sorted(arms, key=lambda n: -(arms[n]["tps_clean"] or 0))
        for name in rows:
            r = arms[name]
            acc, _ = accept(summary, bkey, name)
            t = r["tps_clean"]
            spd = (t / base_tps) if base_tps else float("nan")
            dom = r.get("dominant_phase") or "-"
            share = r.get("dominant_share")
            dom_s = f"{dom} ({share*100:.0f}%)" if share is not None else dom
            print(f"  {name:<16}{acc:>12.3f}{t:>13.2f}{spd:>9.2f}×{dom_s:>18}")

        # SparklingTree advantage over each competitor
        if ours in arms:
            our_acc, _ = accept(summary, bkey, ours)
            our_t = arms[ours]["tps_clean"]
            print(f"\n  {ours} advantage:")
            for name in rows:
                if name == ours:
                    continue
                a2, _ = accept(summary, bkey, name)
                t2 = arms[name]["tps_clean"]
                dacc = 100 * (our_acc - a2) / a2 if a2 else float("nan")
                dtps = 100 * (our_t - t2) / t2 if t2 else float("nan")
                print(f"    vs {name:<14} acceptance {dacc:+6.1f}%   wall-clock {dtps:+6.1f}%")

    print_per_domain(summary, baseline, ours)


def _opt(flag, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


if __name__ == "__main__":
    main()
