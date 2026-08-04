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
]

# All methods run by default. corrector=None = no correction; "<dspark>_markov" is auto-derived.
METHODS = [
    {"name": "dflash.chain", "backbone": "dflash_b16", "corrector": None, "verify": "chain"},
    {"name": "dflash.tree", "backbone": "dflash_b16", "corrector": None, "verify": "tree"},
    {"name": "dflash.markov.tree", "backbone": "dflash_b16", "corrector": "dspark_b7_markov", "verify": "tree"},
    {"name": "dspark.chain", "backbone": "dspark_b7", "corrector": None, "verify": "chain"},
    {"name": "dspark.tree", "backbone": "dspark_b7", "corrector": None, "verify": "tree"},
    {"name": "dspark.markov.tree", "backbone": "dspark_b7", "corrector": "dspark_b7_markov", "verify": "tree"},
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
DEPTH_REPORT_LIMIT = 7
FORCE = False   # True -> ignore the volume cache and recompute every dataset

GPU = "A100-40GB"
TIMEOUT_SECONDS = 2 * 60 * 60   # per dataset-container (only one dataset each now)

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


@app.local_entrypoint()
def main(force: bool = FORCE):
    cfg = build_run_config(force=force)
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
    summary = aggregate.build_summary(cfg, backbones_meta, per_dataset_raw)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # Full detail (per-round distributions, per-sample timing, stage breakdown,
    # probe-by-depth) for rich charting.
    (out_dir / "results_detailed.json").write_text(json.dumps(
        {"cfg": cfg, "backbones_meta": backbones_meta, "per_dataset": per_dataset_raw}, indent=2))
    aggregate.print_rollups(summary)
    print(f"\nSaved summary + results_detailed to {out_dir}")
