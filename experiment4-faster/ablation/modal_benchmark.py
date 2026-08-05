"""Experiment 4-faster / ablation — the full speedup ladder, redone under DDTree
benchmarking practices (sync ON via instrumented-only pass, C++ KV compaction ON).

Four arms, ONE GPU, same job — each arm adds exactly one optimization:

    A0  bestfirst.ref     naive best-first tree (full-vocab per-pop markov rerun)
    A1  +transfer         build_sparked_tree_fast: GPU top-k, ship one small
                          union slice once (exp4/1-transfer-less)
    A2  +precompute       union transition table batched BEFORE the walk,
                          per-depth kernel loop (exp4/2-precompute, pre-4-launch:
                          batch_depths=False)
    A3  +launch           adaptive depth-batching, ~5 kernel launches per round
                          instead of ~80 (exp4/4-launch: batch_depths=True)

All markov arms at C=128 (the final exp5 config). A0->A1 isolates the transfer
fix, A1->A2 the precompute, A2->A3 the launch reduction.

Run:
    modal run --detach modal_benchmark.py --spawn
    modal volume get ddtree-results ablation/summary.json
"""

import json
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Experiment configuration                                                     #
# --------------------------------------------------------------------------- #

TARGET = "Qwen/Qwen3-4B"

BACKBONES = [
    {"name": "dspark_b16", "model_id": "shreybirmiwal/Qwen3-4B-DSpark-b16", "kind": "dspark"},
]

CORRECTOR = "dspark_b16_markov"
C = 128   # union shortlist, matches the final exp5 config

METHODS = [
    {"name": "A0.ref", "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree",
     "tree_kwargs": {"tree_mode": "best-first"}},
    {"name": "A1.transfer", "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree",
     "tree_kwargs": {"tree_mode": "best-first-fast", "beam_candidates": C}},
    {"name": "A2.precompute", "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree",
     "tree_kwargs": {"tree_mode": "best-first-precompute", "beam_candidates": C,
                     "batch_depths": False}},
    {"name": "A3.launch", "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree",
     "tree_kwargs": {"tree_mode": "best-first-precompute", "beam_candidates": C,
                     "batch_depths": True}},
]

# Original exp4 trio of datasets; 4 samples (first discarded -> 3 measured).
TASKS = [
    ["gsm8k", 4],
    ["humaneval", 4],
    ["mt-bench", 4],
]

TREE_BUDGETS = [64, 128]
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
SEED = 1
WARMUP_TOKENS = 256
DISCARD_FIRST_SAMPLE = True

CACHE_DIR = "/results/ablation/cache"

GPU = "H100"
CPU = 8
TIMEOUT_SECONDS = 6 * 60 * 60

TORCH_VERSION = "2.5.1"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-"
    "cp311-cp311-linux_x86_64.whl"
)

HERE = Path(__file__).parent
HARNESS_DIR = HERE.parent.parent / "harness"


def build_run_config() -> dict:
    return {
        "target": TARGET,
        "backbones": BACKBONES,
        "methods": list(METHODS),
        "tasks": TASKS,
        "tree_budgets": TREE_BUDGETS,
        # Instrumented ONLY -- sync-on timing (DDTree benchmarking practices),
        # which also yields the per-phase breakdown the per-section charts need.
        "passes": ["instrumented"],
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "confidence_threshold": 0.0,
        "measure_corrector_fit": False,
        "warmup_tokens": WARMUP_TOKENS,
        "discard_first_sample": DISCARD_FIRST_SAMPLE,
        "cache_dir": CACHE_DIR,
        "force": False,
        "gpu": GPU,
        "cpu": CPU,
    }


# --------------------------------------------------------------------------- #
# Image                                                                        #
# --------------------------------------------------------------------------- #

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .pip_install(f"torch=={TORCH_VERSION}", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install(
        "transformers==4.57.1",
        "datasets==3.6.0",
        "numpy", "loguru", "tqdm", "ninja", "typing_extensions", "hf_transfer",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(HARNESS_DIR.as_posix(), remote_path="/root/harness")
)

app = modal.App("ddtree-exp4-ablation")

hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


@app.function(
    image=image,
    gpu=GPU,
    cpu=CPU,
    timeout=TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=secrets,
)
def run_experiment(cfg: dict) -> dict:
    import sys

    sys.path.insert(0, "/root/harness/ddtree")
    sys.path.insert(0, "/root/harness/runner")
    import driver

    summary = driver.run(cfg, on_checkpoint=results_vol.commit)

    out = Path("/results/ablation/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main(smoke: bool = False, spawn: bool = False):
    cfg = build_run_config()
    if smoke:
        cfg["tasks"] = [["gsm8k", 1]]
        cfg["max_new_tokens"] = 64
        cfg["tree_budgets"] = [64]
        cfg["warmup_tokens"] = 32

    if spawn:
        call = run_experiment.spawn(cfg)
        print(f"spawned: {call.object_id}")
        print("fetch:   modal volume get ddtree-results ablation/summary.json")
        return

    summary = run_experiment.remote(cfg)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    name = "summary_smoke.json" if smoke else "summary.json"
    (out_dir / name).write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_dir / name}")
