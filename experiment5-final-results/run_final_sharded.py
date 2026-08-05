"""Experiment 5 FINAL RESULTS -- fan the run across N H100s, robustly.

Same experiment as run_final.py, sharded for speed. Tunables live in run_final.py;
this launcher just partitions the datasets across N GPUs and reassembles.

WHY shard by dataset. All methods for a given dataset run in the SAME container, so
every per-dataset comparison (SparklingTree vs DDTree, and vs the Autoregressive
1x) is on one machine -- clean. Only the cross-dataset AGGREGATE blends slightly
different H100 clocks (~±12%), which is acceptable and noted.

ROBUSTNESS -- "never crash and lose all work":
  1. Every (budget,dataset) unit is checkpointed to the shared volume and committed
     the instant it finishes (driver's on_checkpoint). A killed container keeps all
     units it already wrote.
  2. Shards are independent: one crash never touches another shard's work.
  3. Modal Retries auto-restart a failed shard; the per-unit cache makes the restart
     resume, not recompute.
  4. Re-running this launcher is idempotent: finished units are skipped from cache.
  5. `--merge` rebuilds summary.json DIRECTLY FROM THE PER-UNIT CACHE, so even a
     shard that died before writing its own summary contributes every unit it
     checkpointed. Nothing completed is ever lost.

All run artifacts are namespaced by a RUN TAG (a hash of the config minus the
dataset list), so a smoke run, or a run with a different C/K/budget, never mixes
its units into another run's merge.

Usage:
  modal run --detach run_final_sharded.py --shards 10   # launch all shards, detached
  modal run run_final_sharded.py --status-only          # units done so far
  modal run run_final_sharded.py --merge                # assemble results/summary.json
  modal run run_final_sharded.py --smoke --shards 2     # tiny end-to-end validation
  (add --smoke to --status-only / --merge to target the smoke run's tag)
"""

import hashlib
import json
from pathlib import Path

import modal

import run_final as cfgmod   # tunables + config builders + the results image/volumes

app = modal.App("ddtree-exp5-final-sharded")
# run_final.py is a sibling module imported at top level; make it importable inside
# the container too (Modal only auto-mounts the entrypoint file + the harness dir).
image = cfgmod.image.add_local_file(
    str(cfgmod.HERE / "run_final.py"), remote_path="/root/run_final.py")
hf_cache = cfgmod.hf_cache
results_vol = cfgmod.results_vol
secrets = cfgmod.secrets

CACHE_ROOT = cfgmod.CACHE_DIR          # "/results/final/cache"
RUN_ROOT = "/results/final/runs"       # per-tag: {tag}/shards/, {tag}/summary.json
GPU = cfgmod.GPU
CPU = cfgmod.CPU
TIMEOUT_SECONDS = cfgmod.TIMEOUT_SECONDS

# Config keys that define "the same run" -- everything the driver fingerprints
# EXCEPT tasks (shards differ only by tasks). A change to any of these -> new tag.
_TAG_KEYS = ("target", "backbones", "methods", "tree_budgets", "temperature",
             "max_new_tokens", "seed", "confidence_threshold", "warmup_tokens",
             "discard_first_sample")


def _base_cfg(smoke: bool) -> dict:
    base = cfgmod.build_run_config()
    if smoke:
        base = {**base, "tasks": [["gsm8k", 1], ["humaneval", 1]],
                "max_new_tokens": 64, "tree_budgets": [64], "warmup_tokens": 32}
    return base


def run_tag(base: dict) -> str:
    payload = {k: base[k] for k in _TAG_KEYS}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def _paths(tag: str) -> tuple[str, str, str]:
    cache_dir = f"{CACHE_ROOT}/{tag}"          # driver appends its own fingerprint below
    shard_dir = f"{RUN_ROOT}/{tag}/shards"
    summary = f"{RUN_ROOT}/{tag}/summary.json"
    return cache_dir, shard_dir, summary


# --------------------------------------------------------------------------- #
# Balancing                                                                    #
# --------------------------------------------------------------------------- #

def balance_datasets(datasets: list, n: int) -> list[list]:
    """Longest-processing-time bin-packing of [name, count] by count into <=n shards.

    Keeps whole datasets together (never splits one across GPUs). The biggest
    dataset sets the wall-clock floor -- e.g. humaneval (164) on 10 GPUs."""
    shards = [[] for _ in range(n)]
    loads = [0] * n
    for name, cnt in sorted(datasets, key=lambda d: -d[1]):
        i = min(range(n), key=lambda j: loads[j])
        shards[i].append([name, cnt])
        loads[i] += cnt
    return [s for s in shards if s]


# --------------------------------------------------------------------------- #
# Remote: one shard                                                            #
# --------------------------------------------------------------------------- #

@app.function(
    image=image, gpu=GPU, cpu=CPU, timeout=TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache, "/results": results_vol}, secrets=secrets,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=10.0),
)
def run_shard(cfg: dict, shard_id: int, shard_dir: str) -> dict:
    import sys
    sys.path.insert(0, "/root/harness/ddtree")
    sys.path.insert(0, "/root/harness/runner")
    import driver

    tasks = ", ".join(f"{n}:{c}" for n, c in cfg["tasks"])
    print(f"[shard {shard_id}] datasets: {tasks}", flush=True)
    summary = driver.run(cfg, on_checkpoint=results_vol.commit)   # per-unit commit

    out = Path(shard_dir) / f"shard_{shard_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    print(f"[shard {shard_id}] done -> {out}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
# Remote: merge (rebuilds summary.json from the per-unit CACHE -- max robust)  #
# --------------------------------------------------------------------------- #

@app.function(
    image=image, cpu=2, timeout=30 * 60,
    volumes={"/cache": hf_cache, "/results": results_vol}, secrets=secrets,
)
def merge(tag: str, smoke: bool) -> dict:
    import sys
    sys.path.insert(0, "/root/harness/ddtree")
    sys.path.insert(0, "/root/harness/runner")
    from metrics import build_entry, build_timing_rollup, PHASE_ORDER

    results_vol.reload()
    cache_dir, _, summary_path = _paths(tag)
    base = _base_cfg(smoke)

    methods = cfgmod.build_methods()
    method_names = [m["name"] for m in methods]
    verify_of = {m["name"]: m["verify"] for m in methods}
    kind_of = {m["name"]: cfgmod._BACKBONE_DEFS[m["backbone"]]["kind"] for m in methods}
    tree_names = [m["name"] for m in methods if m["verify"] in ("tree", "ddtree")]
    chain_names = [m["name"] for m in methods if m["verify"] not in ("tree", "ddtree")]
    budgets = base["tree_budgets"]
    passes = base["passes"]

    # ---- scan this run's per-unit cache files (across its shard fingerprint dirs) --
    # filename: both__b{blabel}__{dataset}__n{n}.json ; contents {pass:{method:[recs]}}
    raw = {p: {} for p in passes}
    seen, n_units = set(), 0
    for unit_path in sorted(Path(cache_dir).glob("*/both__*.json")):
        parts = unit_path.stem.split("__")
        if len(parts) < 4:
            continue
        blabel, dataset = parts[1][1:], parts[2]
        try:
            recs = json.loads(unit_path.read_text())
        except Exception as e:                # half-written unit -> skip, never crash
            print(f"[merge] skipping unreadable {unit_path.name}: {e}", flush=True)
            continue
        for p in passes:
            if p in recs:
                raw[p].setdefault(blabel, {})[dataset] = recs[p]
        seen.add(dataset)
        n_units += 1
    datasets = sorted(seen)
    print(f"[merge] tag={tag}: {n_units} unit files, {len(datasets)} datasets: {datasets}", flush=True)

    # ---- assemble (mirror of driver's assembly) ------------------------------- #
    results = {}
    for p in passes:
        instrumented = p == "instrumented"
        results[p] = {}
        for budget in budgets:
            bkey = str(budget)
            results[p][bkey] = {}
            for ds in datasets:
                entries = {}
                for name in chain_names:
                    r = raw[p].get("na", {}).get(ds, {}).get(name)
                    if r:
                        entries[name] = build_entry(r, verify_of[name], kind_of[name], instrumented, True)
                for name in tree_names:
                    r = raw[p].get(bkey, {}).get(ds, {}).get(name)
                    if r:
                        entries[name] = build_entry(r, verify_of[name], kind_of[name], instrumented, False)
                results[p][bkey][ds] = entries

    # ---- acceptance parity across passes (temp 0 -> must match) --------------- #
    mismatches = []
    if "clean" in raw and "instrumented" in raw:
        for blabel, by_ds in raw["clean"].items():
            for ds, by_m in by_ds.items():
                for name, recs in by_m.items():
                    a = [x for r in recs for x in r["acceptance_lengths"]]
                    inst = raw["instrumented"].get(blabel, {}).get(ds, {}).get(name, [])
                    b = [x for r in inst for x in r["acceptance_lengths"]]
                    if a != b:
                        mismatches.append(f"b{blabel}/{ds}/{name}")

    timing = {}
    if "clean" in results and "instrumented" in results:
        timing = build_timing_rollup(results, budgets, method_names, datasets)

    summary = {
        "config": {
            "target": cfgmod.TARGET, "tasks": base["tasks"], "tree_budgets": budgets,
            "passes": passes, "temperature": base["temperature"],
            "max_new_tokens": base["max_new_tokens"], "seed": base["seed"],
            "methods": {m["name"]: {k: m.get(k) for k in ("backbone", "corrector", "verify", "tree_kwargs")}
                        for m in methods},
            "phase_order": list(PHASE_ORDER), "sharded": True, "run_tag": tag,
            "datasets_present": datasets, "units_merged": n_units,
        },
        "results": results,
        "timing": timing,
        "checks": {"acceptance_match": not mismatches, "mismatched_units": mismatches},
    }
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    print(f"[merge] wrote {summary_path}  (acceptance_match={not mismatches})", flush=True)
    return summary


@app.function(
    image=image, cpu=2, timeout=10 * 60,
    volumes={"/results": results_vol}, secrets=secrets,
)
def status(tag: str) -> dict:
    results_vol.reload()
    cache_dir, shard_dir, _ = _paths(tag)
    per_ds: dict[str, int] = {}
    total = 0
    for unit_path in Path(cache_dir).glob("*/both__*.json"):
        parts = unit_path.stem.split("__")
        if len(parts) >= 3:
            per_ds[parts[2]] = per_ds.get(parts[2], 0) + 1
            total += 1
    shards_done = sorted(p.name for p in Path(shard_dir).glob("shard_*.json")) if Path(shard_dir).exists() else []
    print(f"run tag {tag}: {total} units checkpointed")
    for ds in sorted(per_ds):
        print(f"  {ds:<16} {per_ds[ds]}")
    print(f"shard summaries written: {len(shards_done)}  {shards_done}")
    return {"tag": tag, "total_units": total, "per_dataset": per_ds, "shards_done": shards_done}


# --------------------------------------------------------------------------- #
# Local entrypoint                                                             #
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def main(shards: int = 10, merge: bool = False, status_only: bool = False, smoke: bool = False):
    base = _base_cfg(smoke)
    tag = run_tag(base)
    cache_dir, shard_dir, _ = _paths(tag)

    if status_only:
        globals()["status"].remote(tag)
        return

    if merge:
        summary = globals()["merge"].remote(tag, smoke)
        out_dir = cfgmod.HERE / "results"
        out_dir.mkdir(exist_ok=True)
        name = "summary_smoke.json" if smoke else "summary.json"
        (out_dir / name).write_text(json.dumps(summary, indent=2))
        print(f"\nSaved merged summary to {out_dir / name}  (tag {tag})")
        if not summary.get("checks", {}).get("acceptance_match", True):
            print("WARNING: acceptance mismatch:", summary["checks"]["mismatched_units"])
        cfgmod.print_final_table(summary)
        return

    groups = balance_datasets(base["tasks"], shards)
    print(f"run tag {tag}: launching {len(groups)} shards over {shards} requested GPUs")
    for i, group in enumerate(groups):
        cfg = {**base, "tasks": group, "cache_dir": cache_dir}
        h = run_shard.spawn(cfg, i, shard_dir)
        print(f"  shard {i}: {h.object_id}   [{', '.join(f'{n}:{c}' for n, c in group)}]")
    print("\nunits checkpoint to the volume as they finish; re-run this launcher")
    print("  anytime to resume/fill gaps (idempotent).")
    print(f"progress:  modal run run_final_sharded.py --status-only{' --smoke' if smoke else ''}")
    print(f"assemble:  modal run run_final_sharded.py --merge{' --smoke' if smoke else ''}")
