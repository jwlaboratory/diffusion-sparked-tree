"""
Modal launcher for Experiment 1: markov-corrector transfer across drafters.

PARALLEL + CHECKPOINTED. Each dataset runs in its OWN GPU container (via .map),
and every finished dataset is cached on the `ddtree-results` Volume keyed by a
config fingerprint. So:
  * wall-clock ~= slowest single dataset (not the sum), and
  * a re-run never recomputes a dataset that already finished -- a crash, timeout,
    or added backbone only recomputes what's missing.

Aggregation (transfer / corrector-fit / per-depth rollups) is pure Python
(DDTree/aggregate.py) and runs in the local entrypoint, so no GPU is needed to
assemble the final summary from the per-dataset pieces.

The official `DDTree/benchmark.py` is not on this path.

Default methods (backbone x corrector x verify):
    dflash.chain, dflash.tree, dflash.markov.tree,
    dspark.chain, dspark.tree, dspark.markov.tree

Usage:
    modal run modal_benchmark.py                 # parallel, resumes from volume cache
    modal run modal_benchmark.py --force         # recompute, ignore cache

All knobs are the UPPERCASE constants below. Nothing here needs a GPU locally.
"""

import json
import sys
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Experiment configuration (all tunables here)                                 #
# --------------------------------------------------------------------------- #

# Bump when the captured/raw schema changes -> invalidates the volume cache.
# Keep in sync with run_experiment.CODE_VERSION.
CODE_VERSION = "v2-detail"

TARGET = "Qwen/Qwen3-4B"

# `kind` is intrinsic to the checkpoint. `block_size` is an optional runtime
# override (default = checkpoint config); e.g. add the DFlash checkpoint again at
# block_size 7 to depth-match a foreign backbone to the b7 head.
BACKBONES = [
    {"name": "dflash_b16", "model_id": "z-lab/Qwen3-4B-DFlash-b16", "kind": "dflash"},
    {"name": "dspark_b7", "model_id": "deepseek-ai/dspark_qwen3_4b_block7", "kind": "dspark"},
    # Our fine-tuned block-16 DSpark from experiment 2 (warm-started from the b7
    # checkpoint; carries its own jointly-trained markov head).
    {"name": "dspark_b16", "model_id": "shreybirmiwal/Qwen3-4B-DSpark-b16", "kind": "dspark"},
]

# All methods run by default. corrector=None = no correction; "<dspark>_markov" is
# auto-derived. On a dflash CHAIN the corrector is swept serially over the block
# (chain analogue of markov.tree). markov="off" ablates a dspark chain's intrinsic
# head (parallel argmax), giving the true no-markov chain baseline.
METHODS = [
    {"name": "dflash.chain", "backbone": "dflash_b16", "corrector": None, "verify": "chain"},
    {"name": "dflash.markov.chain", "backbone": "dflash_b16", "corrector": "dspark_b7_markov", "verify": "chain"},
    {"name": "dflash.tree", "backbone": "dflash_b16", "corrector": None, "verify": "tree"},
    {"name": "dflash.markov.tree", "backbone": "dflash_b16", "corrector": "dspark_b7_markov", "verify": "tree"},
    {"name": "dspark.nomarkov.chain", "backbone": "dspark_b7", "corrector": None, "verify": "chain", "markov": "off"},
    {"name": "dspark.chain", "backbone": "dspark_b7", "corrector": None, "verify": "chain"},
    {"name": "dspark.tree", "backbone": "dspark_b7", "corrector": None, "verify": "tree"},
    {"name": "dspark.markov.tree", "backbone": "dspark_b7", "corrector": "dspark_b7_markov", "verify": "tree"},
    {"name": "dspark_b16.nomarkov.chain", "backbone": "dspark_b16", "corrector": None, "verify": "chain", "markov": "off"},
    {"name": "dspark_b16.chain", "backbone": "dspark_b16", "corrector": None, "verify": "chain"},
    {"name": "dspark_b16.tree", "backbone": "dspark_b16", "corrector": None, "verify": "tree"},
    {"name": "dspark_b16.markov.tree", "backbone": "dspark_b16", "corrector": "dspark_b16_markov", "verify": "tree"},
]

PROBE_CORRECTOR = "dspark_b7_markov"   # head used for the tree-free fit probe

TASKS = [
    ["gsm8k", 8],
    ["humaneval", 8],
    ["mt-bench", 8],
]

TREE_BUDGET = 64
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
SEED = 0
CONFIDENCE_THRESHOLD = 0.0
MEASURE_PER_DEPTH = True
MEASURE_CORRECTOR_FIT = True
DEPTH_REPORT_LIMIT = 16   # must reach the b16 horizon or its whole advantage is invisible
FORCE = False   # True -> ignore the volume cache and recompute every dataset

GPU = "A100-40GB"
TIMEOUT_SECONDS = 4 * 60 * 60   # per dataset-container (12 methods each now)

TORCH_VERSION = "2.5.1"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-"
    "cp311-cp311-linux_x86_64.whl"
)

HERE = Path(__file__).parent
DDTREE_DIR = HERE / "DDTree"


def build_run_config(force: bool = FORCE) -> dict:
    return {
        "code_version": CODE_VERSION,
        "target": TARGET,
        "backbones": BACKBONES,
        "methods": METHODS,
        "probe_corrector": PROBE_CORRECTOR,
        "tasks": TASKS,
        "tree_budget": TREE_BUDGET,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "measure_per_depth": MEASURE_PER_DEPTH,
        "measure_corrector_fit": MEASURE_CORRECTOR_FIT,
        "depth_report_limit": DEPTH_REPORT_LIMIT,
        "force": force,
    }


# --------------------------------------------------------------------------- #
# Image                                                                        #
# --------------------------------------------------------------------------- #

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(f"torch=={TORCH_VERSION}", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install(
        "transformers==4.57.1", "datasets==3.6.0",
        "numpy", "loguru", "tqdm", "ninja", "typing_extensions", "hf_transfer",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(DDTREE_DIR.as_posix(), remote_path="/root/DDTree")
)

app = modal.App("ddtree-markov-transfer")

hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]

CACHE_ROOT = "/results/cache"


# --------------------------------------------------------------------------- #
# Remote: one dataset per container, cached on the volume                      #
# --------------------------------------------------------------------------- #

@app.function(
    image=image, gpu=GPU, timeout=TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache, "/results": results_vol}, secrets=secrets,
)
def run_one(payload: dict) -> dict:
    """payload = {"cfg": cfg, "dataset": name, "max_samples": n}.

    Returns {"dataset", "raw", "backbones_meta"}. Resumes from the volume cache
    unless cfg["force"]; writes the cache + commits when it computes fresh."""
    import os

    sys.path.insert(0, "/root/DDTree")
    import torch
    import aggregate
    import run_experiment as exp
    from ddtree import maybe_enable_cpp_compact

    cfg, dataset, max_samples = payload["cfg"], payload["dataset"], payload["max_samples"]
    fp = aggregate.fingerprint(cfg)
    cache_file = os.path.join(CACHE_ROOT, fp, f"{dataset}__n{max_samples}.json")

    results_vol.reload()
    if not cfg.get("force") and os.path.exists(cache_file):
        cached = json.load(open(cache_file))
        print(f"[resume] {dataset}: loaded from cache {cache_file}")
        return {"dataset": dataset, "raw": cached["raw"], "backbones_meta": cached["backbones_meta"]}

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    maybe_enable_cpp_compact(True)
    device = torch.device("cuda:0")

    ctx = exp.load_context(cfg, device)
    raw = exp.run_one_dataset_raw(ctx, cfg, dataset, max_samples)
    exp._print_dataset(dataset, raw, cfg)

    out = {"raw": raw, "backbones_meta": ctx["backbones_meta"]}
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    json.dump(out, open(cache_file, "w"))
    results_vol.commit()
    print(f"[cache] {dataset}: wrote {cache_file}")
    return {"dataset": dataset, **out}


@app.function(
    image=image, gpu=GPU, timeout=30 * 60,
    volumes={"/cache": hf_cache, "/results": results_vol}, secrets=secrets,
)
def verify_equiv(n: int = 4, tree_budget: int = 64, max_new_tokens: int = 256) -> dict:
    """External validation: our dflash.tree (sparked_tree, markov=None) must equal
    the OFFICIAL ddtree_generate round-for-round. Writes result to the volume."""
    import os
    sys.path.insert(0, "/root/DDTree")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model import DFlashDraftModel, load_and_process_dataset
    from ddtree import ddtree_generate, maybe_enable_cpp_compact
    from sparked_tree import sparked_tree_generate
    import run_experiment as exp  # for load_config (RoPE normalization)

    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    maybe_enable_cpp_compact(True)
    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, attn_implementation="sdpa", dtype=torch.bfloat16).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(
        "z-lab/Qwen3-4B-DFlash-b16", config=exp.load_config("z-lab/Qwen3-4B-DFlash-b16"),
        attn_implementation="flash_attention_2", dtype=torch.bfloat16).to(device).eval()
    tok = AutoTokenizer.from_pretrained(TARGET)
    ds = load_and_process_dataset("gsm8k").shuffle(seed=0).select(range(n))
    common = dict(mask_token_id=draft.mask_token_id, block_size=draft.block_size,
                  max_new_tokens=max_new_tokens, stop_token_ids=[tok.eos_token_id],
                  temperature=0.0, tree_budget=tree_budget)
    max_diff, mismatch = 0, False
    for row in ds:
        text = tok.apply_chat_template([{"role": "user", "content": row["turns"][0]}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, return_tensors="pt").to(device)
        a = ddtree_generate(model=draft, target=target, input_ids=ids, **common).acceptance_lengths
        b = sparked_tree_generate(model=draft, target=target, input_ids=ids,
                                  markov_head=None, draft_mode="dflash", **common).acceptance_lengths
        if len(a) != len(b):
            mismatch = True
        else:
            max_diff = max(max_diff, max((abs(x - y) for x, y in zip(a, b)), default=0))
    result = {"n": n, "tree_budget": tree_budget, "max_abs_diff": max_diff,
              "length_mismatch": mismatch, "pass": (not mismatch and max_diff == 0)}
    json.dump(result, open("/results/verify_equiv.json", "w"), indent=2)
    results_vol.commit()
    print("verify_equiv:", result)
    return result


@app.local_entrypoint()
def main(force: bool = FORCE, methods: str = ""):
    """`--methods a,b,c` runs ONLY those methods (their own volume-cache
    fingerprint) and merges the fresh raws into the existing local
    Results/results_detailed.json before rebuilding the summary, so a new method
    never forces a recompute of the ones already measured."""
    cfg = build_run_config(force=force)
    only = [s.strip() for s in methods.split(",") if s.strip()]
    if only:
        known = {m["name"] for m in METHODS}
        unknown = set(only) - known
        if unknown:
            raise SystemExit(f"unknown methods: {sorted(unknown)}")
        cfg["methods"] = [m for m in METHODS if m["name"] in only]
    sys.path.insert(0, DDTREE_DIR.as_posix())
    import aggregate
    fp = aggregate.fingerprint(cfg)

    out_dir = HERE / "Results"
    out_dir.mkdir(exist_ok=True)
    # Write run metadata BEFORE dispatch so an out-of-band aggregator can assemble
    # the summary from the volume caches even if this local client disconnects
    # (the remote, run with --detach, keeps going and checkpoints each dataset).
    (out_dir / "_run_meta.json").write_text(json.dumps({
        "fingerprint": fp,
        "cfg": cfg,
        "cache_files": [f"cache/{fp}/{d}__n{n}.json" for d, n in TASKS],
    }, indent=2))

    specs = [{"cfg": cfg, "dataset": d, "max_samples": n} for d, n in TASKS]
    print(f"Dispatching {len(specs)} datasets in parallel: {[s['dataset'] for s in specs]}")
    outs = list(run_one.map(specs))

    per_dataset_raw = {o["dataset"]: o["raw"] for o in outs}
    backbones_meta = next(o["backbones_meta"] for o in outs)

    if only:
        prev = json.loads((out_dir / "results_detailed.json").read_text())
        for k in ("target", "tree_budget", "temperature", "max_new_tokens", "seed", "tasks"):
            if prev["cfg"][k] != cfg[k]:
                raise SystemExit(f"cannot merge: existing results_detailed.json has {k}={prev['cfg'][k]!r}, "
                                 f"this run has {cfg[k]!r}")
        for d, raw in per_dataset_raw.items():
            prev["per_dataset"][d]["methods"].update(raw["methods"])
        # keep method order canonical (METHODS order) for stable charts/tables
        for d in prev["per_dataset"]:
            ms = prev["per_dataset"][d]["methods"]
            prev["per_dataset"][d]["methods"] = {m["name"]: ms[m["name"]] for m in METHODS if m["name"] in ms}
        cfg = {**cfg, "methods": [m for m in METHODS
                                  if m["name"] in prev["per_dataset"][TASKS[0][0]]["methods"]]}
        per_dataset_raw = prev["per_dataset"]
        backbones_meta = prev["backbones_meta"]

    summary = aggregate.build_summary(cfg, backbones_meta, per_dataset_raw)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # Full detail (per-round distributions, per-sample timing, stage breakdown,
    # probe-by-depth) for rich charting.
    (out_dir / "results_detailed.json").write_text(json.dumps(
        {"cfg": cfg, "backbones_meta": backbones_meta, "per_dataset": per_dataset_raw}, indent=2))
    aggregate.print_rollups(summary)
    print(f"\nSaved summary + results_detailed to {out_dir}")
