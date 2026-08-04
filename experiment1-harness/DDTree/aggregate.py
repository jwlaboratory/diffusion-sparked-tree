"""Pure-Python aggregation for the transfer experiment.

No torch / transformers imports on purpose: this module runs both inside the GPU
container (to build the final summary) AND locally in the Modal launcher's entry
point (which has no GPU/torch) to assemble per-dataset results produced in
parallel. Keep it dependency-free.

The unit of work is one dataset's *raw* result:

    raw = {
      "n": <num samples>,
      "methods": { method_name: {"lengths": [int, ...], "probe": {depth: {...}}} },
    }

`build_summary` merges a {dataset: raw} map into the final summary (results /
corrector_fit / transfer), identical in shape to what the old monolithic run
produced.
"""

import hashlib
import json
from statistics import mean

# cfg keys that change the numbers -> the checkpoint fingerprint. Includes
# "code_version" so a change to the harness LOGIC (not just config) invalidates
# stale caches -- bump cfg["code_version"] whenever the captured data changes.
FINGERPRINT_KEYS = (
    "code_version", "target", "backbones", "methods", "tree_budget", "temperature",
    "max_new_tokens", "seed", "probe_corrector", "measure_corrector_fit",
    "measure_per_depth", "depth_report_limit",
)


def fingerprint(cfg: dict) -> str:
    """Short stable hash of the result-affecting config (dataset-independent)."""
    payload = {k: cfg.get(k) for k in FINGERPRINT_KEYS}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def per_depth_accept(acceptance_lengths: list, depth_limit: int) -> dict:
    """Conditional accept rate at depth d = P(reach d+1 | reach d).

    acceptance_lengths store L = new tokens per round; a round reaches tree depth d
    when L >= d, so rate(d) = #(L >= d+1) / #(L >= d)."""
    rates = {}
    for d in range(1, depth_limit + 1):
        reached = sum(1 for a in acceptance_lengths if a >= d)
        deeper = sum(1 for a in acceptance_lengths if a >= d + 1)
        rates[d] = (deeper / reached) if reached else None
    return rates


def _norm_probe(probe: dict) -> dict:
    """JSON turns int depth keys into strings; normalize back to int."""
    out = {}
    for depth, b in (probe or {}).items():
        out[int(depth)] = {k: b[k] for k in ("n", "base_hit", "corr_hit", "ce_base", "ce_corr")}
    return out


def merge_probe(dst: dict, src: dict) -> None:
    for depth, b in _norm_probe(src).items():
        acc = dst.setdefault(depth, {"n": 0, "base_hit": 0, "corr_hit": 0, "ce_base": 0.0, "ce_corr": 0.0})
        for k in acc:
            acc[k] += b[k]


def summarize_probe(probe: dict, depth_limit: int) -> dict:
    by_depth, roll = {}, {"n": 0, "base_hit": 0, "corr_hit": 0, "ce_base": 0.0, "ce_corr": 0.0}
    for depth in sorted(probe):
        b = probe[depth]
        n = max(b["n"], 1)
        by_depth[depth] = {
            "n": b["n"],
            "base_hit_rate": b["base_hit"] / n,
            "corr_hit_rate": b["corr_hit"] / n,
            "delta_hit": (b["corr_hit"] - b["base_hit"]) / n,
            "mean_ce_base": b["ce_base"] / n,
            "mean_ce_corr": b["ce_corr"] / n,
            "delta_ce": (b["ce_corr"] - b["ce_base"]) / n,  # negative = corrector helps
        }
        if depth <= depth_limit:
            for k in roll:
                roll[k] += b[k]
    nn = max(roll["n"], 1)
    overall = {
        "n": roll["n"],
        "base_hit_rate": roll["base_hit"] / nn,
        "corr_hit_rate": roll["corr_hit"] / nn,
        "delta_hit": (roll["corr_hit"] - roll["base_hit"]) / nn,
        "mean_ce_base": roll["ce_base"] / nn,
        "mean_ce_corr": roll["ce_corr"] / nn,
        "delta_ce": (roll["ce_corr"] - roll["ce_base"]) / nn,
    }
    return {"by_depth": by_depth, f"overall_depth_le_{depth_limit}": overall}


def build_summary(cfg: dict, backbones_meta: dict, per_dataset_raw: dict) -> dict:
    """Assemble the final summary from {dataset: raw} produced (in any order)."""
    methods_meta = {m["name"]: m for m in cfg["methods"]}
    depth_limit = cfg["depth_report_limit"]

    summary = {
        "config": {
            **{k: cfg[k] for k in (
                "target", "tasks", "tree_budget", "temperature", "max_new_tokens",
                "seed", "probe_corrector", "depth_report_limit",
            )},
            "backbones": backbones_meta,
            "methods": methods_meta,
            "metric": "mean_acceptance_length",
        },
        "results": {},
        "corrector_fit": {},
        "transfer": {},
    }

    probe_accum: dict = {}
    for dataset_name, _ in cfg["tasks"]:
        raw = per_dataset_raw.get(dataset_name)
        if raw is None:
            continue
        summary["results"][dataset_name] = {}
        for mname, md in raw["methods"].items():
            lengths = md["lengths"]
            entry = {"mean_accept": (mean(lengths) if lengths else 0.0), "rounds": len(lengths)}
            if cfg["measure_per_depth"] and methods_meta[mname]["verify"] == "tree":
                entry["per_depth_accept"] = per_depth_accept(lengths, depth_limit)
            summary["results"][dataset_name][mname] = entry
            if md.get("probe"):
                merge_probe(probe_accum.setdefault(mname, {}), md["probe"])

    for mname, probe in probe_accum.items():
        summary["corrector_fit"][mname] = {
            "backbone": methods_meta[mname]["backbone"],
            "probe_corrector": cfg.get("probe_corrector"),
            **summarize_probe(probe, depth_limit),
        }

    def method_by(backbone, corrected):
        for m in cfg["methods"]:
            if m["verify"] == "tree" and m["backbone"] == backbone and (m["corrector"] is not None) == corrected:
                return m["name"]
        return None

    datasets = [d for d, _ in cfg["tasks"] if d in summary["results"]]
    for backbone in {m["backbone"] for m in cfg["methods"]}:
        off, on = method_by(backbone, False), method_by(backbone, True)
        if off and on and datasets:
            off_mean = mean([summary["results"][d][off]["mean_accept"] for d in datasets])
            on_mean = mean([summary["results"][d][on]["mean_accept"] for d in datasets])
            summary["transfer"][backbone] = {
                "off_method": off, "on_method": on,
                "accept_off": off_mean, "accept_on": on_mean,
                "accept_pct_change": ((on_mean - off_mean) / off_mean * 100.0) if off_mean else None,
            }
    return summary


def print_rollups(summary: dict) -> None:
    depth_limit = summary["config"]["depth_report_limit"]
    if summary["transfer"]:
        print("\n" + "=" * 60)
        print("Transfer: acceptance-length % change from adding the corrector")
        print("=" * 60)
        for bb, t in summary["transfer"].items():
            pct = t["accept_pct_change"]
            if pct is None:
                print(f"  {bb}: n/a")
            else:
                print(f"  {bb:<16} {t['accept_off']:.3f} -> {t['accept_on']:.3f}  ({pct:+.1f}%)")
    if summary["corrector_fit"]:
        print("\n" + "=" * 60)
        print(f"Corrector fit (probe head, depth<={depth_limit}): does the head fit this backbone?")
        print("=" * 60)
        for name, cf in summary["corrector_fit"].items():
            o = cf[f"overall_depth_le_{depth_limit}"]
            print(f"  {cf['backbone']:<16} delta_hit={o['delta_hit']:+.4f}  "
                  f"delta_ce={o['delta_ce']:+.4f}  (n={o['n']})")
