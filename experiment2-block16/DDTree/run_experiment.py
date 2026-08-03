"""Block-size experiment: does a longer draft horizon raise acceptance length?

This is the single experiment driver. It loads the target verifier and a set of
DSpark draft backbones once, then runs every `backbone x corrector x verify`
method in the config (temperature 0, greedy). The two backbones differ only in
`block_size` -- the checkpoints are architecturally identical (5 draft layers,
hidden 2560, markov_rank 256, same target_layer_ids / mask_token / confidence
head); one drafts 7 tokens per block, the other 16. The question is whether the
longer horizon lets the tree commit more accepted tokens per round.

Each backbone carries its own markov head, so the headline arm is `<bb>.markov.tree`
(tree decoding with the backbone's own corrector live). The matched `<bb>.tree`
control turns the corrector off, which makes the block-size effect readable in
isolation from the corrector's contribution.

Two outputs support the claim:
  * mean acceptance length per method (per dataset and overall),
  * per-depth acceptance rate reported out to `depth_report_limit` -- this MUST
    reach 16, or the b16 model's entire advantage (depths 8..16 the b7 tree can
    never reach) is invisible.

Backbones are parameterized by (checkpoint, kind, block_size). `kind` is "dspark"
for both here: next-token indexing, block_size drafts, owns embed/lm_head, carries
a markov head. `block_size` defaults to the checkpoint's config; the b7 and b16
checkpoints supply 7 and 16 respectively.

Correctors are auto-derived from any dspark-kind backbone that has a markov head:
backbone "dspark_b7" exposes corrector "dspark_b7_markov". A corrector is token-only
(bias = W2 @ W1[prev]).

Standalone: `python run_experiment.py`  (runs the default arms on gsm8k:8, ...).
From Modal: `import run_experiment; run_experiment.run(cfg)`.
"""

import argparse
import json
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from model import DSparkDraftModel, load_and_process_dataset
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
    kind: str                         # "dspark" (both checkpoints here)
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


# What the checkpoints must actually be once loaded. See load_backbones().
EXPECTED_ROPE_THETA = 1000000.0
EXPECTED_BLOCK_SIZE = {"dspark_b7": 7, "dspark_b16": 16}


# The two headline arms are the markov ones (`*.markov.tree`); the plain `*.tree`
# arms are the markov-off controls that isolate the block-size effect.
DEFAULT_BACKBONES = [
    {"name": "dspark_b7", "model_id": "deepseek-ai/dspark_qwen3_4b_block7", "kind": "dspark"},
    {"name": "dspark_b16", "model_id": "shreybirmiwal/Qwen3-4B-DSpark-b16", "kind": "dspark"},
]
DEFAULT_METHODS = [
    {"name": "dspark_b7.tree", "backbone": "dspark_b7", "corrector": None, "verify": "tree"},
    {"name": "dspark_b7.markov.tree", "backbone": "dspark_b7", "corrector": "dspark_b7_markov", "verify": "tree"},
    {"name": "dspark_b16.tree", "backbone": "dspark_b16", "corrector": None, "verify": "tree"},
    {"name": "dspark_b16.markov.tree", "backbone": "dspark_b16", "corrector": "dspark_b16_markov", "verify": "tree"},
]


def default_config() -> dict:
    return {
        "target": "Qwen/Qwen3-4B",
        "backbones": DEFAULT_BACKBONES,
        "methods": DEFAULT_METHODS,
        "tasks": [["gsm8k", 8], ["humaneval", 8], ["mt-bench", 8]],
        "tree_budget": 64,
        "temperature": 0.0,
        "max_new_tokens": 512,
        "seed": 0,
        "confidence_threshold": 0.0,
        "measure_per_depth": True,
        "depth_report_limit": 16,  # must reach the b16 horizon or its advantage is invisible
    }


# --------------------------------------------------------------------------- #
# Model loading                                                                #
# --------------------------------------------------------------------------- #

def load_config(model_id: str):
    """Load a DSpark checkpoint config, normalizing RoPE across transformers majors.

    Both DSpark checkpoints were serialized by transformers v5, which moved RoPE
    settings into a nested `rope_parameters` dict and stopped emitting a top-level
    `rope_theta`. We pin transformers 4.57.1, whose Qwen3Config knows nothing about
    that field: it silently falls back to its own default `rope_theta=10000.0` while
    these drafters were trained at 1000000.0. Nothing raises -- the drafts just get
    quietly worse, on both arms, which would look like a plausible result. So read
    the nested value across and let the assertion below hold the line.

    Do NOT "clean this up" by deleting the branch: it is load-bearing for as long as
    the image pins transformers < 5.
    """
    config = AutoConfig.from_pretrained(model_id)
    params = getattr(config, "rope_parameters", None)
    if params:
        rope_type = params.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(
                f"{model_id}: rope_type={rope_type!r} needs an explicit scaling port; "
                "only 'default' is handled here."
            )
        config.rope_theta = float(params["rope_theta"])
    return config


def load_backbones(cfg: dict, target, device) -> dict[str, Backbone]:
    backbones: dict[str, Backbone] = {}
    for spec in cfg["backbones"]:
        bb = Backbone(**spec)
        if bb.kind == "dspark":
            config = load_config(bb.model_id)
            bb.model = DSparkDraftModel.from_pretrained(
                bb.model_id, config=config,
                attn_implementation="flash_attention_2", dtype=torch.bfloat16,
            ).to(device).eval()
            bb.markov_head = bb.model.markov_head
        else:
            raise ValueError(f"backbone {bb.name!r}: unknown kind {bb.kind!r}")
        bb.eff_block_size = int(bb.block_size) if bb.block_size else int(bb.model.block_size)

        # Both of these are silent-corruption guards, not sanity theater. A wrong
        # rotary base degrades every draft without raising; a block_size that failed
        # to come across from the config would collapse the one axis under test.
        #
        # Recover the base the model will ACTUALLY use straight from the rotary
        # embedding's inv_freq buffer, not from config.rope_theta (the field
        # load_config() just wrote). The default rope init sets
        #   inv_freq[k] = base ** (-2k / dim),  dim = 2 * len(inv_freq),
        # so base = inv_freq[1] ** (-len(inv_freq)). Reading the computed buffer is
        # what makes this independent: it fires even if load_config()'s normalization
        # is later deleted or a transformers change stops consuming config.rope_theta.
        inv_freq = bb.model.rotary_emb.inv_freq.detach().double()
        base = float(inv_freq[1] ** (-inv_freq.numel()))
        if abs(base - EXPECTED_ROPE_THETA) / EXPECTED_ROPE_THETA > 1e-3:
            raise ValueError(
                f"backbone {bb.name!r} ({bb.model_id}): effective rotary base is {base:g}, "
                f"expected {EXPECTED_ROPE_THETA:g}. The transformers v4/v5 rope_parameters "
                "split is not being handled -- see load_config()."
            )
        expected_bs = EXPECTED_BLOCK_SIZE.get(bb.name)
        if expected_bs is not None and bb.eff_block_size != expected_bs:
            raise ValueError(
                f"backbone {bb.name!r} ({bb.model_id}): block_size is "
                f"{bb.eff_block_size}, expected {expected_bs}."
            )
        print(f"loaded {bb.name}: block_size={bb.eff_block_size} rope_theta={base:g} "
              f"markov_head={'yes' if bb.markov_head is not None else 'no'}")

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
        # dspark chain uses its own markov head intrinsically.
        return lambda ids: dspark_generate(
            model=model, target=target, input_ids=ids, block_size=bs,
            confidence_threshold=cfg["confidence_threshold"], **common,
        )

    if method.verify == "tree":
        return lambda ids: sparked_tree_generate(
            model=model, target=target, input_ids=ids, mask_token_id=model.mask_token_id,
            block_size=bs, tree_budget=cfg["tree_budget"],
            markov_head=corrector_head, draft_mode=bb.kind,
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

    methods = [Method(**m) for m in cfg["methods"]]
    method_fns = {
        m.name: build_method_callable(m, backbones, correctors, target, tokenizer.eos_token_id, cfg)
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
                "seed", "depth_report_limit",
            )},
            "backbones": {b.name: {"model_id": b.model_id, "kind": b.kind, "block_size": b.eff_block_size}
                          for b in backbones.values()},
            "methods": {m.name: {"backbone": m.backbone, "corrector": m.corrector, "verify": m.verify}
                        for m in methods},
            "metric": "mean_acceptance_length",
        },
        "results": {},      # results[dataset][method]
        "block_size": {},   # block_size[arm]
    }

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

    # Block-size rollup: for each arm (same corrector on/off + verify, differing
    # only in backbone block size), compare the smaller vs larger block: per-dataset
    # and overall mean acceptance, absolute delta and % change.
    datasets = [d for d, _ in cfg["tasks"]]
    arms: dict[str, dict[int, str]] = {}
    for m in methods:
        arm = ("markov." if m.corrector is not None else "") + m.verify
        arms.setdefault(arm, {})[backbones[m.backbone].eff_block_size] = m.name

    for arm, by_block in arms.items():
        if len(by_block) < 2:
            continue  # nothing to compare against
        small_bs, large_bs = min(by_block), max(by_block)
        small, large = by_block[small_bs], by_block[large_bs]

        per_dataset = {}
        for d in datasets:
            s = summary["results"][d][small]["mean_accept"]
            l = summary["results"][d][large]["mean_accept"]
            per_dataset[d] = {
                "small": s, "large": l, "delta": l - s,
                "pct_change": ((l - s) / s * 100.0) if s else None,
            }
        s_all = mean([summary["results"][d][small]["mean_accept"] for d in datasets])
        l_all = mean([summary["results"][d][large]["mean_accept"] for d in datasets])
        summary["block_size"][arm] = {
            "small_block": small_bs, "large_block": large_bs,
            "small_method": small, "large_method": large,
            "per_dataset": per_dataset,
            "overall": {
                "small": s_all, "large": l_all, "delta": l_all - s_all,
                "pct_change": ((l_all - s_all) / s_all * 100.0) if s_all else None,
            },
        }

    _print_rollups(summary)
    return summary


def _print_rollups(summary: dict) -> None:
    if summary["block_size"]:
        print("\n" + "=" * 64)
        print("Block size: acceptance-length change from a longer draft horizon")
        print("=" * 64)
        for arm, r in summary["block_size"].items():
            o = r["overall"]
            pct = o["pct_change"]
            tail = "n/a" if pct is None else f"({pct:+.1f}%)"
            print(f"  {arm:<14} b{r['small_block']}={o['small']:.3f} -> "
                  f"b{r['large_block']}={o['large']:.3f}  delta={o['delta']:+.3f}  {tail}")


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
