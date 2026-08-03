"""Transfer experiment: does a markov corrector help the backbone it was trained
on, but not a foreign one?

This is the single experiment driver. It loads the target verifier and a set of
draft backbones once, then runs any `backbone x corrector x verify` cell you ask
for. The comparison that carries the claim is a 2x2 (temperature 0, greedy):

                     corrector=off             corrector=<dspark head>
    backbone=dflash  dflash.tree               dflash.markov.tree     (FOREIGN)
    backbone=dspark  dspark.tree               dspark.markov.tree     (OWN)

The within-backbone delta is the head's effect. The claim predicts OWN >> 0 and
FOREIGN <= 0. Three outputs support it:

  * mean acceptance length per method,
  * per-depth acceptance rate (report depths <= min block size to depth-match),
  * a tree-free corrector-fit probe: on a *no-corrector* run, at each committed
    position, does argmax(base + head.bias(prev)) match the target better than
    argmax(base)? Real positions, real prev tokens, no depth extrapolation -- this
    is the confound-free proof that the head is a residual fitted to one backbone.

Backbones are parameterized by (checkpoint, kind, block_size). `kind` is intrinsic
to the checkpoint: "dflash" = in-place indexing, borrows the target's embed/lm_head,
drafts block_size-1 tokens; "dspark" = next-token indexing, owns embed/lm_head,
drafts block_size tokens, and carries a markov head. `block_size` is a runtime knob
(default = the checkpoint's config): running a b16 checkpoint at block_size=7 is how
you depth-match a foreign backbone to a b7 head -- no separate depth cap needed.

Correctors are auto-derived from any dspark-kind backbone that has a markov head:
backbone "dspark_b7" exposes corrector "dspark_b7_markov". A corrector is token-only
(bias = W2 @ W1[prev]) so it can be spliced onto any backbone.

Standalone: `python run_experiment.py`  (runs the default 2x2 on gsm8k:8, ...).
From Modal: `import run_experiment; run_experiment.run(cfg)`.
"""

import argparse
import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, DSparkDraftModel, load_and_process_dataset
from dflash import dflash_generate
from dspark import dspark_generate
from ddtree import maybe_enable_cpp_compact
from sparked_tree import sparked_tree_generate


# --------------------------------------------------------------------------- #
# Registries                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class Backbone:
    name: str
    model_id: str
    kind: str                         # "dflash" | "dspark"
    block_size: Optional[int] = None  # runtime override; None -> checkpoint config
    # filled in by load():
    model: object = None
    eff_block_size: int = 0
    markov_head: object = None


@dataclass
class Method:
    name: str
    backbone: str                     # key into backbones
    corrector: Optional[str] = None   # corrector name, or None
    verify: str = "tree"              # "tree" | "chain"


# The default 2x2 that proves the claim, plus optional chain anchors.
DEFAULT_BACKBONES = [
    {"name": "dflash_b16", "model_id": "z-lab/Qwen3-4B-DFlash-b16", "kind": "dflash"},
    {"name": "dspark_b7", "model_id": "deepseek-ai/dspark_qwen3_4b_block7", "kind": "dspark"},
]
DEFAULT_METHODS = [
    {"name": "dflash.tree", "backbone": "dflash_b16", "corrector": None, "verify": "tree"},
    {"name": "dflash.markov.tree", "backbone": "dflash_b16", "corrector": "dspark_b7_markov", "verify": "tree"},
    {"name": "dspark.tree", "backbone": "dspark_b7", "corrector": None, "verify": "tree"},
    {"name": "dspark.markov.tree", "backbone": "dspark_b7", "corrector": "dspark_b7_markov", "verify": "tree"},
]
# Optional reference anchors (off by default): native chain drafting.
DEFAULT_CHAIN_ANCHORS = [
    {"name": "dflash.chain", "backbone": "dflash_b16", "corrector": None, "verify": "chain"},
    {"name": "dspark.chain", "backbone": "dspark_b7", "corrector": None, "verify": "chain"},
]


def default_config() -> dict:
    return {
        "target": "Qwen/Qwen3-4B",
        "backbones": DEFAULT_BACKBONES,
        "methods": DEFAULT_METHODS,
        "probe_corrector": "dspark_b7_markov",  # head used for the fit probe
        "tasks": [["gsm8k", 8], ["humaneval", 8], ["mt-bench", 8]],
        "tree_budget": 64,
        "temperature": 0.0,
        "max_new_tokens": 512,
        "seed": 0,
        "confidence_threshold": 0.0,
        "measure_per_depth": True,
        "measure_corrector_fit": True,
        "depth_report_limit": 7,  # per-depth / probe reporting horizon
    }


# --------------------------------------------------------------------------- #
# Model loading                                                                #
# --------------------------------------------------------------------------- #

def load_backbones(cfg: dict, target, device) -> dict[str, Backbone]:
    backbones: dict[str, Backbone] = {}
    for spec in cfg["backbones"]:
        bb = Backbone(**spec)
        if bb.kind == "dflash":
            bb.model = DFlashDraftModel.from_pretrained(
                bb.model_id, attn_implementation="flash_attention_2", dtype=torch.bfloat16
            ).to(device).eval()
            bb.markov_head = None
        elif bb.kind == "dspark":
            bb.model = DSparkDraftModel.from_pretrained(
                bb.model_id, attn_implementation="flash_attention_2", dtype=torch.bfloat16
            ).to(device).eval()
            bb.markov_head = bb.model.markov_head
        else:
            raise ValueError(f"backbone {bb.name!r}: unknown kind {bb.kind!r}")
        bb.eff_block_size = int(bb.block_size) if bb.block_size else int(bb.model.block_size)
        backbones[bb.name] = bb
    return backbones


def build_correctors(backbones: dict[str, Backbone]) -> dict[str, object]:
    """Every dspark backbone with a markov head exposes '<name>_markov'."""
    correctors: dict[str, object] = {}
    for bb in backbones.values():
        if bb.markov_head is not None:
            correctors[f"{bb.name}_markov"] = bb.markov_head
    return correctors


# --------------------------------------------------------------------------- #
# Method dispatch                                                              #
# --------------------------------------------------------------------------- #

def build_method_callable(
    method: Method,
    backbones: dict[str, Backbone],
    correctors: dict[str, object],
    target,
    eos_id: int,
    cfg: dict,
    probe_head: Optional[object],
) -> Callable:
    bb = backbones[method.backbone]
    model = bb.model
    bs = bb.eff_block_size
    common = dict(
        max_new_tokens=cfg["max_new_tokens"],
        stop_token_ids=[eos_id],
        temperature=cfg["temperature"],
    )

    corrector_head = None
    if method.corrector is not None:
        if method.corrector not in correctors:
            raise ValueError(f"method {method.name!r}: unknown corrector {method.corrector!r}")
        corrector_head = correctors[method.corrector]

    if method.verify == "chain":
        if bb.kind == "dflash":
            if corrector_head is not None:
                raise ValueError(f"{method.name}: dflash chain has no corrector slot")
            return lambda ids: dflash_generate(
                model=model, target=target, input_ids=ids,
                mask_token_id=model.mask_token_id, block_size=bs, **common,
            )
        # dspark chain uses its own markov head intrinsically.
        return lambda ids: dspark_generate(
            model=model, target=target, input_ids=ids, block_size=bs,
            confidence_threshold=cfg["confidence_threshold"], **common,
        )

    if method.verify == "tree":
        # Probe only on no-corrector runs (the corrector isn't live there).
        this_probe = probe_head if (corrector_head is None and cfg["measure_corrector_fit"]) else None
        return lambda ids: sparked_tree_generate(
            model=model, target=target, input_ids=ids, mask_token_id=model.mask_token_id,
            block_size=bs, tree_budget=cfg["tree_budget"],
            markov_head=corrector_head, draft_mode=bb.kind, probe_markov_head=this_probe,
            **common,
        )

    raise ValueError(f"method {method.name!r}: unknown verify {method.verify!r}")


# --------------------------------------------------------------------------- #
# Aggregation helpers                                                          #
# --------------------------------------------------------------------------- #

def per_depth_accept(acceptance_lengths: list[int], depth_limit: int) -> dict[int, float]:
    """Conditional accept rate at depth d = P(reach d+1 | reach d).

    acceptance_lengths store L = new tokens per round; a round reaches tree depth d
    when L >= d. So rate(d) = #(L >= d+1) / #(L >= d)."""
    rates = {}
    for d in range(1, depth_limit + 1):
        reached = sum(1 for a in acceptance_lengths if a >= d)
        deeper = sum(1 for a in acceptance_lengths if a >= d + 1)
        rates[d] = (deeper / reached) if reached else None
    return rates


def merge_probe(dst: dict, src: dict) -> None:
    for depth, b in src.items():
        acc = dst.setdefault(depth, {"n": 0, "base_hit": 0, "corr_hit": 0, "ce_base": 0.0, "ce_corr": 0.0})
        for k in acc:
            acc[k] += b[k]


def summarize_probe(probe: dict, depth_limit: int) -> dict:
    """Turn raw depth sums into hit rates / CE and deltas, plus a depth<=limit rollup."""
    by_depth = {}
    roll = {"n": 0, "base_hit": 0, "corr_hit": 0, "ce_base": 0.0, "ce_corr": 0.0}
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


# --------------------------------------------------------------------------- #
# Main run                                                                     #
# --------------------------------------------------------------------------- #

def run(cfg: dict) -> dict:
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device("cuda:0")
    maybe_enable_cpp_compact(True)

    target = AutoModelForCausalLM.from_pretrained(
        cfg["target"], attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["target"])

    backbones = load_backbones(cfg, target, device)
    correctors = build_correctors(backbones)
    probe_head = correctors.get(cfg.get("probe_corrector")) if cfg["measure_corrector_fit"] else None

    methods = [Method(**m) for m in cfg["methods"]]
    method_fns = {
        m.name: build_method_callable(m, backbones, correctors, target, tokenizer.eos_token_id, cfg, probe_head)
        for m in methods
    }

    # Warmup each method once (kernels, cache-compaction extension).
    warmup_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    warmup_ids = tokenizer.encode(warmup_text, return_tensors="pt").to(device)
    for fn in method_fns.values():
        _ = fn(warmup_ids)

    depth_limit = cfg["depth_report_limit"]
    summary = {
        "config": {
            **{k: cfg[k] for k in (
                "target", "tasks", "tree_budget", "temperature", "max_new_tokens",
                "seed", "probe_corrector", "depth_report_limit",
            )},
            "backbones": {b.name: {"model_id": b.model_id, "kind": b.kind, "block_size": b.eff_block_size}
                          for b in backbones.values()},
            "methods": {m.name: {"backbone": m.backbone, "corrector": m.corrector, "verify": m.verify}
                        for m in methods},
            "metric": "mean_acceptance_length",
        },
        "results": {},        # results[dataset][method]
        "corrector_fit": {},  # corrector_fit[method] (only no-corrector tree runs)
        "transfer": {},       # transfer[backbone]
    }

    # method -> accumulated probe across datasets, for the corrector-fit rollup.
    probe_accum: dict[str, dict] = {}

    for dataset_name, max_samples in cfg["tasks"]:
        dataset = load_and_process_dataset(dataset_name)
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.shuffle(seed=cfg["seed"]).select(range(max_samples))

        lengths = {m.name: [] for m in methods}
        for row in dataset:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["turns"][0]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
            for m in methods:
                result = method_fns[m.name](input_ids)
                lengths[m.name].extend(result.acceptance_lengths)
                probe = getattr(result, "probe_by_depth", None)
                if probe:
                    merge_probe(probe_accum.setdefault(m.name, {}), probe)

        summary["results"][dataset_name] = {}
        for m in methods:
            v = lengths[m.name]
            entry = {"mean_accept": (mean(v) if v else 0.0), "rounds": len(v)}
            if cfg["measure_per_depth"] and m.verify == "tree":
                entry["per_depth_accept"] = per_depth_accept(v, depth_limit)
            summary["results"][dataset_name][m.name] = entry

        print(f"\nDataset {dataset_name} ({len(dataset)} samples, tree_budget {cfg['tree_budget']})")
        print(f"{'method':<26}{'mean_accept':>12}{'rounds':>9}")
        print("-" * 47)
        for m in methods:
            e = summary["results"][dataset_name][m.name]
            print(f"{m.name:<26}{e['mean_accept']:>12.3f}{e['rounds']:>9}")

    # Corrector-fit rollup (per no-corrector tree method, keyed by its backbone).
    for m in methods:
        if m.name in probe_accum:
            summary["corrector_fit"][m.name] = {
                "backbone": m.backbone,
                "probe_corrector": cfg.get("probe_corrector"),
                **summarize_probe(probe_accum[m.name], depth_limit),
            }

    # Transfer rollup: within-backbone acceptance % change from adding the corrector.
    def method_by(backbone, corrected):
        for m in methods:
            if m.verify == "tree" and m.backbone == backbone and (m.corrector is not None) == corrected:
                return m.name
        return None

    for bb in backbones:
        off = method_by(bb, corrected=False)
        on = method_by(bb, corrected=True)
        if off and on:
            off_mean = mean([summary["results"][d][off]["mean_accept"] for d, _ in cfg["tasks"]])
            on_mean = mean([summary["results"][d][on]["mean_accept"] for d, _ in cfg["tasks"]])
            summary["transfer"][bb] = {
                "off_method": off, "on_method": on,
                "accept_off": off_mean, "accept_on": on_mean,
                "accept_pct_change": ((on_mean - off_mean) / off_mean * 100.0) if off_mean else None,
            }

    _print_rollups(summary, depth_limit)
    return summary


def _print_rollups(summary: dict, depth_limit: int) -> None:
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


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--tasks", type=str, default=None, help="e.g. gsm8k:8,humaneval:8,mt-bench:8")
    p.add_argument("--tree-budget", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--chain-anchors", action="store_true", help="also run native chain references")
    p.add_argument("--save-json", type=str, default=None)
    a = p.parse_args()

    cfg = default_config()
    if a.target: cfg["target"] = a.target
    if a.tree_budget is not None: cfg["tree_budget"] = a.tree_budget
    if a.temperature is not None: cfg["temperature"] = a.temperature
    if a.max_new_tokens is not None: cfg["max_new_tokens"] = a.max_new_tokens
    if a.seed is not None: cfg["seed"] = a.seed
    if a.tasks:
        cfg["tasks"] = [[s.split(":")[0], int(s.split(":")[1])] for s in a.tasks.split(",") if s.strip()]
    if a.chain_anchors:
        cfg["methods"] = cfg["methods"] + DEFAULT_CHAIN_ANCHORS
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
