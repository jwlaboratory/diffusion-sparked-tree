"""Metric aggregation: native stage vocabularies -> the canonical exp3 phase set.

The three generator families report three different stage tuples. This module maps
each into ONE canonical vocabulary so "time building the tree" is readable across
arms that have no tree at all. Rules (see experiment3-timings/reproduce.md):

  * A phase an arm structurally does not have is None, never 0.0 -- 0 reads as
    "measured, took no time", which is a different claim.
  * Sub-phases live in a separate namespace and are NEVER summed with the parents
    (the native tree_build total already contains its three sub-keys; summing all
    native keys double-counts ~2x).
  * Phases exclude the cold round (reported separately); unaccounted is the
    residual against the decode wall clock, kept signed so the shares stay honest.
"""

from statistics import mean

PHASE_ORDER = (
    "draft_forward",
    "candidate_build",
    "candidate_pack",
    "verify",
    "walk_accept",
    "kv_update",
    "state_carry",
    "unaccounted",
)

_COMMON_TAIL = ("verify", "walk_accept", "kv_update", "state_carry")


def canonical_phases(verify: str, kind: str, st: dict) -> tuple[dict, dict]:
    """Map one summed native stage_times dict -> ({phase: sec|None}, {subphase: sec})."""
    if verify == "autoregressive":
        # AR has no drafter/tree/spec tail: the only real phase is the per-token
        # target forward (bucketed as "verify"); everything else is structurally
        # absent -> None (never 0.0, per the module rule).
        phases = {"draft_forward": None, "candidate_build": None, "candidate_pack": None,
                  "verify": st.get("verify", 0.0), "walk_accept": None,
                  "kv_update": None, "state_carry": None}
        return phases, {}
    tail = {k: st[k] for k in _COMMON_TAIL}
    if verify == "chain" and kind == "dflash":
        phases = {"draft_forward": st["draft"], "candidate_build": st["candidate_build"],
                  "candidate_pack": None, **tail}
        sub = {}
    elif verify == "chain" and kind == "dspark":
        phases = {"draft_forward": st["draft_backbone"],
                  "candidate_build": st["draft_markov"] + st["draft_confidence"],
                  "candidate_pack": None, **tail}
        sub = {"candidate_build.markov": st["draft_markov"],
               "candidate_build.confidence": st["draft_confidence"]}
    else:  # "tree" | "ddtree"
        phases = {"draft_forward": st["draft"], "candidate_build": st["tree_build"],
                  "candidate_pack": st["tree_compile"], **tail}
        sub = {"candidate_build.prep": st["tree_build_copy"],
               "candidate_build.expand": st["tree_build_heap"],
               "candidate_build.visibility": st["tree_build_visibility"]}
    return phases, sub


def sample_record(result) -> dict:
    """The per-generation raw record persisted in a unit cache file."""
    return {
        "acceptance_lengths": [int(a) for a in result.acceptance_lengths],
        "n_out": int(result.num_output_tokens),
        "rounds": int(result.decode_rounds),
        "ttft": float(result.time_to_first_token),
        "decode_time": float(result.total_decode_time),
        "cold_round": float(result.cold_round_time),
        "stage_times": {k: float(v) for k, v in result.stage_times.items()},
        "round_timestamps": [float(t) for t in result.round_timestamps],
    }


def build_entry(records: list[dict], verify: str, kind: str,
                instrumented: bool, budget_invariant: bool) -> dict:
    """Aggregate one method's sample records (one unit) into a summary entry.

    Totals are token-weighted (summed seconds / summed tokens), not means of
    per-sample ratios -- a short sample must not carry the weight of a long one.
    """
    n_out = sum(r["n_out"] for r in records)
    decode = sum(r["decode_time"] for r in records)
    all_accepts = [a for r in records for a in r["acceptance_lengths"]]
    entry = {
        "tps_decode": (n_out / decode) if decode else None,
        "ttft": mean(r["ttft"] for r in records),
        "output_tokens": n_out,
        "rounds": sum(r["rounds"] for r in records),
        "decode_time": decode,
        "mean_accept": mean(all_accepts) if all_accepts else 0.0,
        "budget_invariant": budget_invariant,
    }
    if not instrumented:
        return entry

    summed = {}
    for r in records:
        for k, v in r["stage_times"].items():
            summed[k] = summed.get(k, 0.0) + v
    cold = sum(r["cold_round"] for r in records)

    phases, sub = canonical_phases(verify, kind, summed)
    accounted = sum(v for v in phases.values() if v is not None)
    # Signed on purpose: a negative residual means the stage clocks overlapped the
    # wall clock (shouldn't happen with barriers on) and must be visible, not hidden.
    phases["unaccounted"] = decode - cold - accounted

    entry["phases"] = {
        k: None if v is None else {
            "sec": v,
            "sec_per_token": (v / n_out) if n_out else None,
            "share": (v / decode) if decode else None,
        }
        for k, v in phases.items()
    }
    entry["subphases"] = {
        k: {"sec": v, "sec_per_token": (v / n_out) if n_out else None}
        for k, v in sub.items()
    }
    entry["cold_round"] = {"sec": cold, "share": (cold / decode) if decode else None}
    return entry


def build_timing_rollup(results: dict, budgets: list[int], method_names: list[str],
                        datasets: list[str]) -> dict:
    """timing[budget][method]: clean vs instrumented TPS, tax, dominant phase.

    Everything here is derived from the measurement -- dominant_phase is computed,
    never asserted, so no chart title can claim a direction the data doesn't show.
    """
    rollup: dict[str, dict] = {}
    for budget in budgets:
        bkey = str(budget)
        rollup[bkey] = {}
        for name in method_names:
            def _tot(pass_name, field):
                vals = [results[pass_name][bkey][d][name][field]
                        for d in datasets if name in results[pass_name][bkey].get(d, {})]
                return sum(vals) if vals else None

            clean_tokens = _tot("clean", "output_tokens")
            clean_decode = _tot("clean", "decode_time")
            inst_tokens = _tot("instrumented", "output_tokens")
            inst_decode = _tot("instrumented", "decode_time")
            if not clean_decode or not inst_decode:
                continue
            tps_clean = clean_tokens / clean_decode
            tps_inst = inst_tokens / inst_decode

            # Dominant phase from the instrumented pass, summed across datasets.
            phase_secs: dict[str, float] = {}
            for d in datasets:
                e = results["instrumented"][bkey].get(d, {}).get(name)
                if not e:
                    continue
                for ph, v in e["phases"].items():
                    if v is not None:
                        phase_secs[ph] = phase_secs.get(ph, 0.0) + v["sec"]
            dominant = max(phase_secs, key=phase_secs.get) if phase_secs else None

            rollup[bkey][name] = {
                "tps_clean": tps_clean,
                "tps_instrumented": tps_inst,
                "instrumentation_tax": 1.0 - tps_inst / tps_clean,
                "dominant_phase": dominant,
                "dominant_share": (phase_secs[dominant] / sum(phase_secs.values()))
                                  if dominant else None,
                "per_dataset": {
                    d: {"tps_clean": (results["clean"][bkey][d][name]["output_tokens"]
                                      / results["clean"][bkey][d][name]["decode_time"])}
                    for d in datasets if name in results["clean"][bkey].get(d, {})
                    and results["clean"][bkey][d][name]["decode_time"]
                },
            }
    return rollup
