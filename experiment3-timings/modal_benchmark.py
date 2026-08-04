"""Modal launcher for Experiment 3: where does the time actually go?

ONE stage. The remote GPU function loads the target verifier + all three draft
backbones once and runs the four decoding arms through the SHARED harness driver
(harness/runner + harness/ddtree -- no per-experiment DDTree copy).

Every (budget, dataset) unit runs TWICE:
    clean        -- stage timers off -> the headline decode tokens/sec
    instrumented -- stage timers on  -> the per-phase breakdown

Arms:
    dflash.chain             DFlash-b16, linear chain, parallel argmax
    dspark_b7.chain          DSpark-b7, chain, intrinsic markov sweep
    ddtree                   DFlash-b16 + the official ddtree_generate tree
    dspark_b16.markov.tree   DSpark-b16 + own markov + naive sparked tree

UNLIKE exp1/exp2, the numbers here are hardware-bound: GPU **and** CPU matter (a
large share of tree cost is single-threaded CPU), so both are pinned below.

Usage:
    modal run modal_benchmark.py            # add --detach to survive disconnects

All knobs are the UPPERCASE constants below. Nothing here needs a GPU locally.
"""

import json
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Experiment configuration (all tunables live here)                            #
# --------------------------------------------------------------------------- #

TARGET = "Qwen/Qwen3-4B"

BLOCK16_MODEL_ID = "shreybirmiwal/Qwen3-4B-DSpark-b16"
BACKBONES = [
    {"name": "dflash_b16", "model_id": "z-lab/Qwen3-4B-DFlash-b16", "kind": "dflash"},
    {"name": "dspark_b7", "model_id": "deepseek-ai/dspark_qwen3_4b_block7", "kind": "dspark"},
    {"name": "dspark_b16", "model_id": BLOCK16_MODEL_ID, "kind": "dspark"},
]

METHODS = [
    {"name": "dflash.chain", "backbone": "dflash_b16", "corrector": None, "verify": "chain"},
    {"name": "dspark_b7.chain", "backbone": "dspark_b7", "corrector": None, "verify": "chain"},
    {"name": "ddtree", "backbone": "dflash_b16", "corrector": None, "verify": "ddtree"},
    {"name": "dspark_b16.markov.tree", "backbone": "dspark_b16", "corrector": "dspark_b16_markov", "verify": "tree"},
]

# Small representative slice: one math, one code, one chat. Timing gets many
# measurements per sample (30-150 rounds each), so n=4 stabilizes phase shares;
# bump to 8 before quoting cross-arm speedups as results.
TASKS = [
    ["gsm8k", 4],
    ["humaneval", 4],
    ["mt-bench", 4],
]

# Swept for the tree arms only; chain arms have no budget knob and are reported
# as budget-invariant. candidate_build is ~linear in budget, verify sublinear --
# the sweep separates "tree expensive to build" from "tree expensive to verify".
TREE_BUDGETS = [64, 256]
PASSES = ["clean", "instrumented"]
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
SEED = 0
CONFIDENCE_THRESHOLD = 0.0   # dspark chain only; inert at 0
WARMUP_TOKENS = 256          # 32 never reaches deep tree positions
DISCARD_FIRST_SAMPLE = True

# Namespaced away from exp1 (/results) and exp2 (/results/block16). The driver
# adds a config-fingerprint subdirectory, so stale-schema resume is impossible.
CACHE_DIR = "/results/timings/cache"

# BOTH pinned, BOTH load-bearing: tree_build_heap is single-threaded CPU, so the
# container's CPU shapes the tree arms as much as the GPU shapes the chain arms.
GPU = "H100"
CPU = 8
TIMEOUT_SECONDS = 6 * 60 * 60

# Pinned versions. torch 2.5.1 (cu124) + a matching prebuilt flash-attn wheel.
TORCH_VERSION = "2.5.1"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-"
    "cp311-cp311-linux_x86_64.whl"
)

HERE = Path(__file__).parent
HARNESS_DIR = HERE.parent / "harness"


def build_run_config() -> dict:
    return {
        "target": TARGET,
        "backbones": BACKBONES,
        "methods": list(METHODS),
        "tasks": TASKS,
        "tree_budgets": TREE_BUDGETS,
        "passes": PASSES,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
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

app = modal.App("ddtree-exp3-timings")

hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


# --------------------------------------------------------------------------- #
# Remote function (single stage, in-process)                                   #
# --------------------------------------------------------------------------- #

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

    out = Path("/results/timings/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main(smoke: bool = False, spawn: bool = False):
    cfg = build_run_config()
    if smoke:
        # End-to-end validation at minimal GPU cost: one sample per dataset kind,
        # short generations, one budget. Uses a separate fingerprint (different
        # cfg), so it never contaminates the full run's cache.
        cfg["tasks"] = [["gsm8k", 1]]
        cfg["max_new_tokens"] = 64
        cfg["tree_budgets"] = [64]
        cfg["warmup_tokens"] = 32

    if spawn:
        # Fire-and-forget: survives ANY local failure (a crashing client can cancel
        # an awaited .remote() input even under --detach -- it happened). Results
        # land on the volume; fetch with:
        #   modal volume get ddtree-results timings/summary.json results/summary.json
        call = run_experiment.spawn(cfg)
        print(f"spawned: {call.object_id}")
        print("progress:  modal volume ls ddtree-results timings/cache/<fingerprint>")
        print("fetch:     modal volume get ddtree-results timings/summary.json")
        return

    summary = run_experiment.remote(cfg)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    name = "summary_smoke.json" if smoke else "summary.json"
    (out_dir / name).write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {out_dir / name}")

    if not summary.get("checks", {}).get("acceptance_match", True):
        print("WARNING: acceptance mismatch between clean and instrumented passes!")
        print("         mismatched units:", summary["checks"]["mismatched_units"])

    if summary.get("timing"):
        print("\nDecode TPS (clean) per tree budget:")
        for bkey in sorted(summary["timing"], key=int):
            print(f"  tree_budget {bkey}:")
            for arm, r in summary["timing"][bkey].items():
                print(f"    {arm:<26} {r['tps_clean']:>8.2f} tok/s   "
                      f"dominant phase: {r['dominant_phase']} ({r['dominant_share']*100:.0f}%)")
