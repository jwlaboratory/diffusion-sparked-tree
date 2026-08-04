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

# =========================================================================== #
# METHOD SETTINGS (EXACT) -- verified against this file's code, not assumed.   #
# =========================================================================== #
#
# All four default arms are DSpark backbones verified with TREE decoding. There
# are NO `.chain` arms and NO DFlash arms in experiment 2 (the chain path and
# draft_mode="dflash" exist in the code but are never configured here). Both
# backbones carry markov_rank=256 vanilla markov heads, so both models
# architecturally OWN a markov head at all times -- but "owns a head" is not the
# same as "the head is applied". Whether the head is LIVE is decided per method:
#
#   verify="tree"  -> sparked_tree_generate(..., markov_head=corrector_head)
#                     corrector_head = the backbone's OWN markov head when the
#                     method sets corrector="<backbone>_markov", else None.
#                     markov_head=None  => build_sparked_tree branches
#                       per-depth-independent -> markov head OFF (dormant).
#                     markov_head=<head> => each node's children are top-k of
#                       base_logits + W2@W1[parent_token] -> markov head ON.
#   verify="chain" -> dspark_generate(...) uses model.sample_draft_tokens, which
#                     applies self.markov_head INTRINSICALLY (always ON if the
#                     model has one). No chain arm is configured in exp 2.
#
# So the naming maps to behavior as:  ".tree" = head OFF, ".markov.tree" = head
# ON (the backbone's OWN head, native -- never a foreign/spliced head here).
#
# The confidence head (present on both checkpoints) is NEVER applied by any exp 2
# arm: only dspark_generate (chain) uses it, and only when confidence_threshold
# > 0. cfg default confidence_threshold=0.0 and there are no chain arms.
#
# Backbone block_size is intrinsic to each checkpoint (no runtime override); the
# guard in load_backbones() asserts b7=7 / b16=16 (EXPECTED_BLOCK_SIZE). For a
# dspark tree, depth_limit = block_size, so b7 trees reach depth 7 and b16 trees
# reach depth 16.
#
# Shared by every arm: target=Qwen/Qwen3-4B (sdpa, bf16); draft_mode="dspark";
# temperature=0.0 (greedy; tree build is greedy top-k regardless);
# max_new_tokens=512; seed=0; tree_budget SWEPT over {64, 256} (each arm runs at
# both); depth_report_limit=16.
#
#   method                    backbone / model_id                          eff  markov head            corrector           verify
#                                                                           bs   (applied?)             (source)
#   ------------------------  -------------------------------------------  ---  --------------------  ------------------  ------
#   dspark_b7.tree            dspark_b7 / deepseek-ai/                       7   OFF (owned, dormant)  None                tree    <- markov-off control
#                                        dspark_qwen3_4b_block7
#   dspark_b7.markov.tree     dspark_b7 / deepseek-ai/                       7   ON  (native)          dspark_b7_markov    tree    <- HEADLINE
#                                        dspark_qwen3_4b_block7                    b7's OWN head        (=b7's own head)
#   dspark_b16.tree           dspark_b16 / shreybirmiwal/                   16   OFF (owned, dormant)  None                tree    <- markov-off control
#                                         Qwen3-4B-DSpark-b16
#   dspark_b16.markov.tree    dspark_b16 / shreybirmiwal/                   16   ON  (native)          dspark_b16_markov   tree    <- HEADLINE
#                                         Qwen3-4B-DSpark-b16                      b16's OWN head       (=b16's own head)
#
# NAME-vs-BEHAVIOR asymmetry to note: a `.tree` DSpark arm still LOADS a model
# that owns a markov head -- the head just is not passed in, so it does not touch
# the tree. "DSpark with markov off" means dormant head, not absent head.
# =========================================================================== #

import argparse
import json
import os
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
        "tree_budgets": [64, 256],
        "temperature": 0.0,
        "max_new_tokens": 512,
        "seed": 0,
        "confidence_threshold": 0.0,
        "measure_per_depth": True,
        "depth_report_limit": 16,  # must reach the b16 horizon or its advantage is invisible
        "warmup_tokens": 32,       # warmup only touches code paths; no need for a full generation
        "cache_dir": None,         # if set, each (budget, dataset) unit is cached here (resume)
        "force": False,            # ignore cache and recompute
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
    tree_budget: int,
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
            block_size=bs, tree_budget=tree_budget,
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

def run(cfg: dict, on_checkpoint=None) -> dict:
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

    # Warm up every (method, budget) pair once: kernels, the cache-compaction
    # extension, and the tree builder's allocation path all differ by budget.
    #
    # A few rounds is enough to touch every code path; warming up at the full
    # max_new_tokens means N_methods x N_budgets complete 512-token generations
    # before the first measurement, which on the b16/budget-256 cells is several
    # minutes of GPU time that measures nothing.
    warmup_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    warmup_ids = tokenizer.encode(warmup_text, return_tensors="pt").to(device)
    warmup_cfg = {**cfg, "max_new_tokens": cfg.get("warmup_tokens", 32)}

    fns_by_budget: dict[int, dict[str, Callable]] = {}
    for budget in cfg["tree_budgets"]:
        for m in methods:
            fn = build_method_callable(
                m, backbones, correctors, target, tokenizer.eos_token_id, warmup_cfg, budget
            )
            _ = fn(warmup_ids)
        fns_by_budget[budget] = {
            m.name: build_method_callable(
                m, backbones, correctors, target, tokenizer.eos_token_id, cfg, budget
            )
            for m in methods
        }

    depth_limit = cfg["depth_report_limit"]
    summary = {
        "config": {
            **{k: cfg[k] for k in (
                "target", "tasks", "tree_budgets", "temperature", "max_new_tokens",
                "seed", "depth_report_limit",
            )},
            "backbones": {b.name: {"model_id": b.model_id, "kind": b.kind, "block_size": b.eff_block_size}
                          for b in backbones.values()},
            "methods": {m.name: {"backbone": m.backbone, "corrector": m.corrector, "verify": m.verify}
                        for m in methods},
            "metric": "mean_acceptance_length",
        },
        "results": {},      # results[budget][dataset][method]
        "block_size": {},   # block_size[budget][arm]
    }

    # (budget, dataset) is the checkpoint unit -- roughly 5-20 GPU-minutes each. Each
    # finished unit is written to cache_dir and the volume committed, so a timeout or
    # a killed container costs at most one unit, and a re-run resumes. Learned the
    # hard way: without this, an hour of work vanishes on any interruption.
    cache_dir = cfg.get("cache_dir")
    for budget in cfg["tree_budgets"]:
        bkey = str(budget)
        summary["results"][bkey] = {}
        for dataset_name, max_samples in cfg["tasks"]:
            unit = f"b{budget}__{dataset_name}__n{max_samples}"
            cpath = os.path.join(cache_dir, unit + ".json") if cache_dir else None

            lengths = None
            if cpath and not cfg.get("force") and os.path.exists(cpath):
                lengths = json.load(open(cpath))
                print(f"[resume] {unit} loaded from cache", flush=True)

            if lengths is None:
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
                        result = fns_by_budget[budget][m.name](input_ids)
                        lengths[m.name].extend(int(a) for a in result.acceptance_lengths)
                if cpath:
                    os.makedirs(os.path.dirname(cpath), exist_ok=True)
                    json.dump(lengths, open(cpath, "w"))
                    if on_checkpoint:
                        on_checkpoint()
                    print(f"[checkpoint] {unit} saved", flush=True)

            entries = {}
            for m in methods:
                v = lengths[m.name]
                e = {"mean_accept": (mean(v) if v else 0.0), "rounds": len(v)}
                if cfg["measure_per_depth"] and m.verify == "tree":
                    e["per_depth_accept"] = per_depth_accept(v, depth_limit)
                entries[m.name] = e
            summary["results"][bkey][dataset_name] = entries

            print(f"\nDataset {dataset_name} (n={max_samples}, tree_budget {budget})")
            print(f"{'method':<26}{'mean_accept':>12}{'rounds':>9}")
            print("-" * 47)
            for m in methods:
                e = entries[m.name]
                print(f"{m.name:<26}{e['mean_accept']:>12.3f}{e['rounds']:>9}", flush=True)

    # Block-size rollup, computed per budget: for each arm (same corrector on/off +
    # verify, differing only in backbone block size), compare smaller vs larger block.
    datasets = [d for d, _ in cfg["tasks"]]
    arms: dict[str, dict[int, str]] = {}
    for m in methods:
        arm = ("markov." if m.corrector is not None else "") + m.verify
        arms.setdefault(arm, {})[backbones[m.backbone].eff_block_size] = m.name

    for budget in cfg["tree_budgets"]:
        bkey = str(budget)
        res = summary["results"][bkey]
        summary["block_size"][bkey] = {}
        for arm, by_block in arms.items():
            if len(by_block) < 2:
                continue  # nothing to compare against
            small_bs, large_bs = min(by_block), max(by_block)
            small, large = by_block[small_bs], by_block[large_bs]

            per_dataset = {}
            for d in datasets:
                s = res[d][small]["mean_accept"]
                l = res[d][large]["mean_accept"]
                per_dataset[d] = {
                    "small": s, "large": l, "delta": l - s,
                    "pct_change": ((l - s) / s * 100.0) if s else None,
                }
            s_all = mean([res[d][small]["mean_accept"] for d in datasets])
            l_all = mean([res[d][large]["mean_accept"] for d in datasets])
            summary["block_size"][bkey][arm] = {
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
        for bkey in sorted(summary["block_size"], key=int):
            print(f"  tree_budget {bkey}:")
            for arm, r in summary["block_size"][bkey].items():
                o = r["overall"]
                pct = o["pct_change"]
                tail = "n/a" if pct is None else f"({pct:+.1f}%)"
                print(f"    {arm:<16} b{r['small_block']}={o['small']:.3f} -> "
                      f"b{r['large_block']}={o['large']:.3f}  delta={o['delta']:+.3f}  {tail}")
                # Per-dataset, because the effect is expected to be task-dependent:
                # a longer horizon only pays where completions actually reach deep
                # tree positions, so chat should gain least and math/code most. The
                # average across a mixed task set can hide a sign flip underneath it.
                for d, pd in r["per_dataset"].items():
                    dpct = pd["pct_change"]
                    dtail = "n/a" if dpct is None else f"({dpct:+.1f}%)"
                    print(f"      {d:<14} {pd['small']:.3f} -> {pd['large']:.3f}  "
                          f"delta={pd['delta']:+.3f}  {dtail}")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--tasks", type=str, default=None, help="e.g. gsm8k:8,humaneval:8,mt-bench:8")
    p.add_argument("--tree-budgets", type=str, default=None, help="e.g. 64,256")
    p.add_argument("--cache-dir", type=str, default=None, help="resume dir for (budget,dataset) units")
    p.add_argument("--force", action="store_true", help="ignore cache and recompute")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save-json", type=str, default=None)
    a = p.parse_args()

    cfg = default_config()
    if a.target: cfg["target"] = a.target
    if a.tree_budgets: cfg["tree_budgets"] = [int(x) for x in a.tree_budgets.split(",") if x.strip()]
    if a.cache_dir: cfg["cache_dir"] = a.cache_dir
    if a.force: cfg["force"] = True
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
