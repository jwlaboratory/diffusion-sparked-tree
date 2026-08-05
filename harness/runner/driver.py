"""Shared experiment driver: run a backbone x corrector x verify matrix, twice.

This is the exp3 timing driver, built as the merge the two per-experiment drivers
were converging on: exp1's dflash support and timing capture, exp2's RoPE/block
guards, budget sweep, and per-unit resume cache.

Two passes per unit (see experiment3-timings/reproduce.md):
  clean        -- timing.set_timing(False): no per-stage barriers, full CPU/GPU
                  overlap. Yields the headline tps_decode.
  instrumented -- set_timing(True): honest per-phase attribution, slower. Yields
                  the phase breakdown (and its own, discounted, TPS).

The two passes are INTERLEAVED PER SAMPLE, alternating which runs first on
even/odd samples. Running the passes sequentially (v1 of this driver) produced a
consistently NEGATIVE instrumentation tax (-11..-36%): the second pass saw a
warmer machine -- sustained boost clocks, warm allocator, datasets already in
memory -- so it measured machine state, not timer overhead. Back-to-back
execution with alternating order cancels both the warm-state drift and the
residual within-pair order effect.

The checkpoint unit is (budget-label, dataset) and holds BOTH passes; chain arms
have no budget knob so they run once per dataset under the label "na" and are copied
under every budget key at assembly time with budget_invariant=True. Each finished
unit is written to the cache dir and the volume committed, so an interruption
costs at most one unit. The cache directory is fingerprinted over the config AND
CODE_VERSION: a schema or code change misses the old cache instead of silently
resuming from stale, incompatible units (exp2 lacked this and it bit).

Standalone: `python driver.py` (needs harness/ddtree and harness/runner on
sys.path). From Modal: `import driver; driver.run(cfg, on_checkpoint=...)`.
"""

import argparse
import hashlib
import json
import os
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ddtree import maybe_enable_cpp_compact, load_cpp_compact_module
from timing import set_timing
from model import load_and_process_dataset

from backbones import load_backbones, build_correctors
from methods import Method, build_method_callable, needs_budget
from metrics import PHASE_ORDER, sample_record, build_entry, build_timing_rollup

# Bump on any change to the unit-record schema or the decode/instrumentation code:
# the fingerprint below folds this in, so old cache units are missed, not reused.
# v2: passes interleaved per sample (v1's sequential passes had warm-state bias).
# v3: sparked_tree gained tree_mode="beam" (+ Method.tree_kwargs pass-through).
# v4: sparked_tree gained tree_mode="best-first-fast" (build_sparked_tree_fast).
# v5: sparked_tree gained tree_mode="best-first-precompute" (build_sparked_tree_precompute).
# v5-csweep: candidate-size (C) sweep over the fast + precompute builders (exp4/2-precompute/csweep).
CODE_VERSION = "harness-5-csweep"

DEFAULT_BACKBONES = [
    {"name": "dflash_b16", "model_id": "z-lab/Qwen3-4B-DFlash-b16", "kind": "dflash"},
    {"name": "dspark_b7", "model_id": "deepseek-ai/dspark_qwen3_4b_block7", "kind": "dspark"},
    {"name": "dspark_b16", "model_id": "shreybirmiwal/Qwen3-4B-DSpark-b16", "kind": "dspark"},
]
DEFAULT_METHODS = [
    {"name": "dflash.chain", "backbone": "dflash_b16", "corrector": None, "verify": "chain"},
    {"name": "dspark_b7.chain", "backbone": "dspark_b7", "corrector": None, "verify": "chain"},
    {"name": "ddtree", "backbone": "dflash_b16", "corrector": None, "verify": "ddtree"},
    {"name": "dspark_b16.markov.tree", "backbone": "dspark_b16", "corrector": "dspark_b16_markov", "verify": "tree"},
]


def default_config() -> dict:
    return {
        "target": "Qwen/Qwen3-4B",
        "backbones": DEFAULT_BACKBONES,
        "methods": DEFAULT_METHODS,
        "tasks": [["gsm8k", 4], ["humaneval", 4], ["mt-bench", 4]],
        "tree_budgets": [64, 256],
        "passes": ["clean", "instrumented"],
        "temperature": 0.0,
        "max_new_tokens": 512,
        "seed": 0,
        "confidence_threshold": 0.0,   # dspark chain; inert at 0
        "measure_corrector_fit": False,  # load-bearing: the probe sits inside the decode window
        "warmup_tokens": 256,          # must reach deep tree positions, 32 does not
        "discard_first_sample": True,  # belt-and-braces on top of warmup
        "cache_dir": None,             # if set, each unit is checkpointed here (resume)
        "force": False,                # ignore cache and recompute
    }


def fingerprint(cfg: dict) -> str:
    """Cache identity: everything that changes what a unit measures."""
    keys = ("target", "backbones", "methods", "tasks", "tree_budgets", "temperature",
            "max_new_tokens", "seed", "confidence_threshold", "warmup_tokens",
            "discard_first_sample")
    payload = {k: cfg[k] for k in keys}
    payload["code_version"] = CODE_VERSION
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Main run                                                                     #
# --------------------------------------------------------------------------- #

def run(cfg: dict, on_checkpoint=None) -> dict:
    assert cfg.get("measure_corrector_fit") is not True, (
        "the corrector-fit probe runs inside the decode window and poisons timings; "
        "it must stay off in the timing harness"
    )
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device("cuda:0")

    # Which passes to run (interleaved per sample). Default: both. Kept as a list so
    # a caller could run clean-only, but the default preserves the original behavior.
    passes = list(cfg.get("passes") or ["clean", "instrumented"])

    # Compile the inline C++ KV-compaction extension OUTSIDE the timed region, and
    # record which path is live -- it changes kv_update.
    maybe_enable_cpp_compact(True)
    cpp_compact_live = load_cpp_compact_module() is not None
    print(f"cpp_compact_live={cpp_compact_live}", flush=True)

    target = AutoModelForCausalLM.from_pretrained(
        cfg["target"], attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["target"])

    backbones = load_backbones(cfg, target, device)
    correctors = build_correctors(backbones)
    methods = [Method(**m) for m in cfg["methods"]]
    chain_methods = [m for m in methods if not needs_budget(m)]
    tree_methods = [m for m in methods if needs_budget(m)]

    def encode(prompt: str) -> torch.Tensor:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return tokenizer.encode(text, return_tensors="pt").to(device)

    # ---- callables: (method.name, budget-or-None) -> generate fn ------------- #
    fns: dict[tuple[str, Optional[int]], Callable] = {}
    for m in chain_methods:
        fns[(m.name, None)] = build_method_callable(
            m, backbones, correctors, target, tokenizer.eos_token_id, cfg, None)
    for m in tree_methods:
        for budget in cfg["tree_budgets"]:
            fns[(m.name, budget)] = build_method_callable(
                m, backbones, correctors, target, tokenizer.eos_token_id, cfg, budget)

    # ---- warmup: every (method, budget) pair, deep enough to reach the tree's
    # far positions and JIT every kernel the measured runs will touch ---------- #
    warmup_cfg = {**cfg, "max_new_tokens": cfg["warmup_tokens"]}
    warmup_ids = encode("Warmup")
    for (name, budget) in fns:
        m = next(mm for mm in methods if mm.name == name)
        fn = build_method_callable(
            m, backbones, correctors, target, tokenizer.eos_token_id, warmup_cfg, budget)
        _ = fn(warmup_ids)
        print(f"[warmup] {name} budget={budget}", flush=True)

    # ---- units ---------------------------------------------------------------- #
    kind_of = {m.name: backbones[m.backbone].kind for m in methods}
    verify_of = {m.name: m.verify for m in methods}
    datasets_cache: dict[str, list] = {}

    def get_rows(dataset_name: str, max_samples: int) -> list:
        key = f"{dataset_name}:{max_samples}"
        if key not in datasets_cache:
            dataset = load_and_process_dataset(dataset_name)
            if max_samples is not None and len(dataset) > max_samples:
                dataset = dataset.shuffle(seed=cfg["seed"]).select(range(max_samples))
            datasets_cache[key] = list(dataset)
        return datasets_cache[key]

    fp = fingerprint(cfg)
    cache_dir = os.path.join(cfg["cache_dir"], fp) if cfg.get("cache_dir") else None
    print(f"config fingerprint: {fp}", flush=True)

    def run_unit(dataset_name: str, max_samples: int,
                 budget: Optional[int], unit_methods: list[Method]) -> dict:
        """One (budget-label, dataset) unit, BOTH passes, interleaved per sample.

        Each sample runs clean and instrumented back-to-back, alternating which
        goes first on even/odd samples -- sequential whole-passes measured warm
        machine state (boost clocks, allocator), not timer overhead.
        """
        blabel = "na" if budget is None else str(budget)
        unit = f"both__b{blabel}__{dataset_name}__n{max_samples}"
        cpath = os.path.join(cache_dir, unit + ".json") if cache_dir else None

        if cpath and not cfg.get("force") and os.path.exists(cpath):
            print(f"[resume] {unit} loaded from cache", flush=True)
            return json.load(open(cpath))

        records = {p: {m.name: [] for m in unit_methods} for p in passes}
        rows = get_rows(dataset_name, max_samples)
        for i, row in enumerate(rows):
            input_ids = encode(row["turns"][0])
            # Interleave passes per sample, alternating order (kills warm-state drift).
            # With a single pass (e.g. clean-only) this is just [pass] every sample.
            order = passes if i % 2 == 0 else passes[::-1]
            for m in unit_methods:
                fn = fns[(m.name, budget)]
                if i == 0 and cfg.get("discard_first_sample"):
                    _ = fn(input_ids)  # discarded: absorbs any residual cold state
                for pass_name in order:
                    set_timing(pass_name == "instrumented")
                    records[pass_name][m.name].append(sample_record(fn(input_ids)))
        set_timing(True)

        if cpath:
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            json.dump(records, open(cpath, "w"))
            if on_checkpoint:
                on_checkpoint()
            print(f"[checkpoint] {unit} saved", flush=True)
        return records

    # Flat unit list. Every unit is independent and checkpointed, so parallel
    # containers can each take a rotated slice (cfg["shard_offset"]) and fill the
    # SHARED cache dir; any container that iterates the full list assembles a
    # complete summary from cache (run_unit loads a done unit instead of recomputing).
    # shard_offset is intentionally NOT in fingerprint(), so all shards share cache.
    unit_specs = []
    for dataset_name, max_samples in cfg["tasks"]:
        if chain_methods:
            unit_specs.append((dataset_name, max_samples, None, chain_methods))
        for budget in cfg["tree_budgets"]:
            unit_specs.append((dataset_name, max_samples, budget, tree_methods))
    if unit_specs:
        off = int(cfg.get("shard_offset", 0)) % len(unit_specs)
        unit_specs = unit_specs[off:] + unit_specs[:off]

    # pass -> budget-label -> dataset -> {method: [records]}
    raw: dict[str, dict[str, dict[str, dict]]] = {p: {} for p in passes}
    for dataset_name, max_samples, budget, umethods in unit_specs:
        both = run_unit(dataset_name, max_samples, budget, umethods)
        blabel = "na" if budget is None else str(budget)
        for p in passes:
            raw[p].setdefault(blabel, {})[dataset_name] = both[p]

    # ---- assembly ------------------------------------------------------------- #
    dataset_names = [d for d, _ in cfg["tasks"]]
    results: dict[str, dict] = {}
    for pass_name in passes:
        instrumented = pass_name == "instrumented"
        results[pass_name] = {}
        for budget in cfg["tree_budgets"]:
            bkey = str(budget)
            results[pass_name][bkey] = {}
            for dataset_name in dataset_names:
                entries = {}
                for m in chain_methods:
                    entries[m.name] = build_entry(
                        raw[pass_name]["na"][dataset_name][m.name],
                        verify_of[m.name], kind_of[m.name], instrumented, True)
                for m in tree_methods:
                    entries[m.name] = build_entry(
                        raw[pass_name][bkey][dataset_name][m.name],
                        verify_of[m.name], kind_of[m.name], instrumented, False)
                results[pass_name][bkey][dataset_name] = entries

    # ---- acceptance must be identical across passes (temp 0; timers do not
    # touch the math). A mismatch means the two-pass setup itself is broken. ---- #
    mismatches = []
    if "clean" in raw and "instrumented" in raw:
        for blabel, by_dataset in raw["clean"].items():
            for dataset_name, by_method in by_dataset.items():
                for name, recs in by_method.items():
                    a = [x for r in recs for x in r["acceptance_lengths"]]
                    b = [x for r in raw["instrumented"][blabel][dataset_name][name]
                         for x in r["acceptance_lengths"]]
                    if a != b:
                        mismatches.append(f"b{blabel}/{dataset_name}/{name}")
        if mismatches:
            print(f"WARNING: acceptance mismatch between passes: {mismatches}", flush=True)

    summary = {
        "config": {
            **{k: cfg[k] for k in (
                "target", "tasks", "tree_budgets", "passes", "temperature",
                "max_new_tokens", "seed", "warmup_tokens", "discard_first_sample",
            )},
            **{k: cfg[k] for k in ("gpu", "cpu") if k in cfg},
            "measure_corrector_fit": False,
            "backbones": {b.name: {"model_id": b.model_id, "kind": b.kind, "block_size": b.eff_block_size}
                          for b in backbones.values()},
            "methods": {m.name: {"backbone": m.backbone, "corrector": m.corrector,
                                 "verify": m.verify, "tree_kwargs": m.tree_kwargs}
                        for m in methods},
            "phase_order": list(PHASE_ORDER),
            "cpp_compact_live": cpp_compact_live,
            "code_version": CODE_VERSION,
            "fingerprint": fp,
            "metric": "decode timings (two-pass) + acceptance",
        },
        "results": results,
        "timing": {},
        "checks": {"acceptance_match": not mismatches, "mismatched_units": mismatches},
    }
    if "clean" in results and "instrumented" in results:
        summary["timing"] = build_timing_rollup(
            results, cfg["tree_budgets"], [m.name for m in methods], dataset_names)

    _print_rollup(summary)
    return summary


def _print_rollup(summary: dict) -> None:
    if not summary["timing"]:
        return
    print("\n" + "=" * 72)
    print("Decode throughput (clean pass) and dominant phase (instrumented pass)")
    print("=" * 72)
    for bkey in sorted(summary["timing"], key=int):
        print(f"  tree_budget {bkey}:")
        for name, r in summary["timing"][bkey].items():
            print(f"    {name:<26} tps={r['tps_clean']:>8.2f}  "
                  f"(instrumented {r['tps_instrumented']:>8.2f}, "
                  f"tax {r['instrumentation_tax']*100:>5.1f}%)  "
                  f"dominant={r['dominant_phase']} ({r['dominant_share']*100:.0f}%)")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=str, default=None, help="e.g. gsm8k:4,humaneval:4")
    p.add_argument("--tree-budgets", type=str, default=None, help="e.g. 64,256")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--save-json", type=str, default=None)
    a = p.parse_args()

    cfg = default_config()
    if a.tasks:
        cfg["tasks"] = [[s.split(":")[0], int(s.split(":")[1])] for s in a.tasks.split(",") if s.strip()]
    if a.tree_budgets:
        cfg["tree_budgets"] = [int(x) for x in a.tree_budgets.split(",") if x.strip()]
    if a.cache_dir:
        cfg["cache_dir"] = a.cache_dir
    if a.force:
        cfg["force"] = True
    if a.max_new_tokens is not None:
        cfg["max_new_tokens"] = a.max_new_tokens
    cfg["_save_json"] = a.save_json
    return cfg


def main() -> None:
    cfg = parse_args()
    save_json = cfg.pop("_save_json", None)
    summary = run(cfg)
    if save_json:
        with open(save_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved results to {save_json}")


if __name__ == "__main__":
    main()
