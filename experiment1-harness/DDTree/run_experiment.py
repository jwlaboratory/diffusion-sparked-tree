"""Transfer experiment: does a markov corrector help the backbone it was trained
on, but not a foreign one?

This is the single experiment driver. It loads the target verifier and a set of
draft backbones once, then runs every `backbone x corrector x verify` method in the
config (temperature 0, greedy). By default that is all of them: native chain
drafting for each backbone plus the tree matrix (corrector off / on per backbone).

The head's effect is the within-backbone delta between a backbone's tree runs with
the corrector off vs on. The claim predicts it is large-positive on the head's own
backbone and <= 0 on a foreign one. Three outputs support it:

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

Structure (so the work is checkpointable and parallelizable):
  * load_context(cfg, device)          -> load target + backbones + method fns once
  * run_one_dataset_raw(ctx, cfg, ...)  -> one dataset's raw per-method lengths/probe
  * run(cfg)                            -> loop datasets (resumable via cfg["cache_dir"]),
                                           then aggregate.build_summary(...)
The per-dataset raw dict is the checkpoint unit; the Modal launcher fans these out
across containers and caches each on a Volume so a re-run never recomputes a
finished dataset. Aggregation lives in aggregate.py (pure Python, no torch).

Standalone: `python run_experiment.py --cache-dir runs/cache`  (resumable).
From Modal:  see ../modal_benchmark.py (parallel + volume checkpointing).
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import aggregate
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


# All methods run by default (chain references + the tree matrix).
DEFAULT_BACKBONES = [
    {"name": "dflash_b16", "model_id": "z-lab/Qwen3-4B-DFlash-b16", "kind": "dflash"},
    {"name": "dspark_b7", "model_id": "deepseek-ai/dspark_qwen3_4b_block7", "kind": "dspark"},
]
DEFAULT_METHODS = [
    {"name": "dflash.chain", "backbone": "dflash_b16", "corrector": None, "verify": "chain"},
    {"name": "dflash.tree", "backbone": "dflash_b16", "corrector": None, "verify": "tree"},
    {"name": "dflash.markov.tree", "backbone": "dflash_b16", "corrector": "dspark_b7_markov", "verify": "tree"},
    {"name": "dspark.chain", "backbone": "dspark_b7", "corrector": None, "verify": "chain"},
    {"name": "dspark.tree", "backbone": "dspark_b7", "corrector": None, "verify": "tree"},
    {"name": "dspark.markov.tree", "backbone": "dspark_b7", "corrector": "dspark_b7_markov", "verify": "tree"},
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
        "cache_dir": None,        # if set, per-dataset raw results are cached here (resume)
        "force": False,           # ignore cache and recompute
    }


# --------------------------------------------------------------------------- #
# Model loading                                                                #
# --------------------------------------------------------------------------- #

EXPECTED_ROPE_THETA = 1000000.0  # every Qwen3-4B-derived checkpoint here


def load_config(model_id: str):
    """Load a draft checkpoint's config, normalizing RoPE across transformers majors.

    The DSpark checkpoints were serialized by transformers v5, which moved RoPE into
    a nested `rope_parameters` dict and stopped emitting a top-level `rope_theta`. We
    pin 4.57.1, whose Qwen3Config knows nothing about that field and silently falls
    back to its own default 10000.0 -- while these drafters were trained at 1000000.0.
    Nothing raises; the drafts just get quietly worse.

    That asymmetry is what makes it dangerous *here*: the DFlash checkpoint keeps a
    top-level rope_theta and loads fine, so without this an experiment comparing the
    two backbones would run DFlash correctly and DSpark broken, and read the
    difference as a property of the markov head.

    Do NOT delete this while the image pins transformers < 5.
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


def load_backbones(cfg: dict, target, device) -> dict:
    backbones: dict = {}
    for spec in cfg["backbones"]:
        bb = Backbone(**spec)
        config = load_config(bb.model_id)
        if bb.kind == "dflash":
            bb.model = DFlashDraftModel.from_pretrained(
                bb.model_id, config=config,
                attn_implementation="flash_attention_2", dtype=torch.bfloat16,
            ).to(device).eval()
            bb.markov_head = None
        elif bb.kind == "dspark":
            bb.model = DSparkDraftModel.from_pretrained(
                bb.model_id, config=config,
                attn_implementation="flash_attention_2", dtype=torch.bfloat16,
            ).to(device).eval()
            bb.markov_head = bb.model.markov_head
        else:
            raise ValueError(f"backbone {bb.name!r}: unknown kind {bb.kind!r}")
        bb.eff_block_size = int(bb.block_size) if bb.block_size else int(bb.model.block_size)

        # Recover the base the model will ACTUALLY rotate with, from the computed
        # inv_freq buffer rather than the field load_config() just wrote, so this
        # still fires if the normalization above rots. Default rope init sets
        #   inv_freq[k] = base ** (-2k / dim),  dim = 2 * len(inv_freq)
        # hence base = inv_freq[1] ** (-len(inv_freq)).
        inv_freq = bb.model.rotary_emb.inv_freq.detach().double()
        base = float(inv_freq[1] ** (-inv_freq.numel()))
        if abs(base - EXPECTED_ROPE_THETA) / EXPECTED_ROPE_THETA > 1e-3:
            raise ValueError(
                f"backbone {bb.name!r} ({bb.model_id}): effective rotary base is {base:g}, "
                f"expected {EXPECTED_ROPE_THETA:g}. The transformers v4/v5 rope_parameters "
                "split is not being handled -- see load_config()."
            )
        print(f"loaded {bb.name}: block_size={bb.eff_block_size} rope_theta={base:g}")

        backbones[bb.name] = bb
    return backbones


def build_correctors(backbones: dict) -> dict:
    """Every dspark backbone with a markov head exposes '<name>_markov'."""
    correctors: dict = {}
    for bb in backbones.values():
        if bb.markov_head is not None:
            correctors[f"{bb.name}_markov"] = bb.markov_head
    return correctors


# --------------------------------------------------------------------------- #
# Method dispatch                                                              #
# --------------------------------------------------------------------------- #

def build_method_callable(
    method: Method,
    backbones: dict,
    correctors: dict,
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
# Load / per-dataset run (the checkpointable, parallelizable units)           #
# --------------------------------------------------------------------------- #

def load_context(cfg: dict, device) -> dict:
    """Load target + backbones + built method callables once, and warm them up."""
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

    # Warmup each method once (kernels, cache-compaction extension). Cheap: short prompt.
    warmup_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    warmup_ids = tokenizer.encode(warmup_text, return_tensors="pt").to(device)
    for fn in method_fns.values():
        _ = fn(warmup_ids)

    return {
        "device": device,
        "target": target,
        "tokenizer": tokenizer,
        "methods": methods,
        "method_fns": method_fns,
        "backbones_meta": {
            b.name: {"model_id": b.model_id, "kind": b.kind, "block_size": b.eff_block_size}
            for b in backbones.values()
        },
    }


def run_one_dataset_raw(ctx: dict, cfg: dict, dataset_name: str, max_samples) -> dict:
    """Run every method on one dataset; return JSON-serializable raw results.

    raw = {"n": int, "methods": {name: {"lengths": [...], "probe": {depth: {...}}}}}
    This is the checkpoint unit written to the cache and merged by aggregate.py."""
    tokenizer, methods, method_fns = ctx["tokenizer"], ctx["methods"], ctx["method_fns"]
    dataset = load_and_process_dataset(dataset_name)
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=cfg["seed"]).select(range(max_samples))

    lengths = {m.name: [] for m in methods}
    probes: dict = {}
    for row in dataset:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["turns"][0]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        input_ids = tokenizer.encode(text, return_tensors="pt").to(ctx["device"])
        for m in methods:
            result = method_fns[m.name](input_ids)
            lengths[m.name].extend(result.acceptance_lengths)
            probe = getattr(result, "probe_by_depth", None)
            if probe:
                aggregate.merge_probe(probes.setdefault(m.name, {}), probe)

    return {
        "n": len(dataset),
        "methods": {
            m.name: {"lengths": lengths[m.name], "probe": probes.get(m.name, {})}
            for m in methods
        },
    }


def _print_dataset(name: str, raw: dict, cfg: dict) -> None:
    print(f"\nDataset {name} ({raw['n']} samples, tree_budget {cfg['tree_budget']})")
    print(f"{'method':<26}{'mean_accept':>12}{'rounds':>9}")
    print("-" * 47)
    for mname, md in raw["methods"].items():
        L = md["lengths"]
        ma = (sum(L) / len(L)) if L else 0.0
        print(f"{mname:<26}{ma:>12.3f}{len(L):>9}")


# --------------------------------------------------------------------------- #
# Sequential run (resumable via cfg["cache_dir"])                              #
# --------------------------------------------------------------------------- #

def run(cfg: dict) -> dict:
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device("cuda:0")
    maybe_enable_cpp_compact(True)

    ctx = load_context(cfg, device)

    cache_dir = cfg.get("cache_dir")
    fp = aggregate.fingerprint(cfg)
    per_dataset_raw = {}
    for dataset_name, max_samples in cfg["tasks"]:
        raw, cpath = None, None
        if cache_dir:
            cpath = os.path.join(cache_dir, fp, f"{dataset_name}__n{max_samples}.json")
            if not cfg.get("force") and os.path.exists(cpath):
                raw = json.load(open(cpath))
                print(f"[resume] {dataset_name} loaded from {cpath}")
        if raw is None:
            raw = run_one_dataset_raw(ctx, cfg, dataset_name, max_samples)
            if cpath:
                os.makedirs(os.path.dirname(cpath), exist_ok=True)
                json.dump(raw, open(cpath, "w"))
        per_dataset_raw[dataset_name] = raw
        _print_dataset(dataset_name, raw, cfg)

    summary = aggregate.build_summary(cfg, ctx["backbones_meta"], per_dataset_raw)
    aggregate.print_rollups(summary)
    return summary


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
    p.add_argument("--cache-dir", type=str, default=None, help="per-dataset resume cache dir")
    p.add_argument("--force", action="store_true", help="ignore cache and recompute")
    p.add_argument("--save-json", type=str, default=None)
    a = p.parse_args()

    cfg = default_config()
    if a.target: cfg["target"] = a.target
    if a.tree_budget is not None: cfg["tree_budget"] = a.tree_budget
    if a.temperature is not None: cfg["temperature"] = a.temperature
    if a.max_new_tokens is not None: cfg["max_new_tokens"] = a.max_new_tokens
    if a.seed is not None: cfg["seed"] = a.seed
    if a.cache_dir: cfg["cache_dir"] = a.cache_dir
    if a.force: cfg["force"] = True
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
