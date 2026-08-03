"""
Modal launcher for Experiment 1: acceptance lengths across drafter/corrector/tree.

Two stages run remotely on a single GPU at temperature 0 (greedy):

STAGE A - official reference (unmodified `DDTree/benchmark.py`, sdpa target):
  - "baseline"      : block_size=1 (autoregressive-style draft) reference
  - "dflash"        : DFlash block diffusion draft
  - "ddtree_tbN"    : DFlash + DDTree draft tree, one per tree budget N
  This validates the harness and cross-checks configs 0/4 below.

STAGE B - the six experiment configs (`DDTree/run_acceptance.py`), which drive
DSpark and the naive sparked tree the official harness cannot produce:
  0 dflash_b16            DFlash-b16 drafter, chain              (== reference dflash)
  2 dspark_b7             DSpark-b7 drafter + markov, chain
  3 dflash+markov+tree    DFlash-b16 drafter, DSpark markov corrector, tree
  4 dflash+tree(ddtree)   DFlash-b16 drafter, no corrector, tree (== reference ddtree)
  5 dspark+tree           DSpark-b7 drafter, no corrector, tree
  6 dspark+markov+tree    DSpark-b7 drafter, DSpark markov corrector, tree

Configs 3-6 share one tree code path; the only variables are drafter and whether
the DSpark markov head is passed as corrector. Both drafters share the same target
(Qwen3-4B), target_layer_ids, and mask token, so the splice (config 3) is well-posed.

We do not run the `--flash-attn` pass: it only affects the timing "best-of" pick,
not acceptance length, the quantity we care about. See reproduce.md for details.

Usage:
    modal run modal_benchmark.py

Config knobs are the UPPERCASE constants below. Nothing here needs a GPU locally.
"""

import json
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Benchmark configuration (this is the "smaller subset")                       #
# --------------------------------------------------------------------------- #

# One model/draft pair. Qwen3-4B is the smallest pair in the paper's sweep, so
# it is the cheapest faithful reproduction. All are public on the HF Hub.
MODEL_NAME = "Qwen/Qwen3-4B"
DRAFT_NAME = "z-lab/Qwen3-4B-DFlash-b16"           # DFlash drafter (block 16)
DSPARK_DRAFT_NAME = "deepseek-ai/dspark_qwen3_4b_block7"  # DSpark drafter + markov (block 7)

# Stage A (official DDTree/benchmark.py) gives baseline + dflash + ddtree and
# cross-validates configs 0/4. Set False to run only the six experiment configs.
RUN_OFFICIAL_REFERENCE = True

# Single tree budget for the six experiment configs (Stage B). Kept in the Stage A
# sweep below too, so config 4 lines up with ddtree_tb{SPARKED_TREE_BUDGET}.
SPARKED_TREE_BUDGET = 64

# A small, representative slice of the paper's suite: one math, one code, one
# chat dataset. (dataset_name, max_samples)
TASKS = [
    ("gsm8k", 8),
    ("humaneval", 8),
    ("mt-bench", 8),
]

# Subset of the paper's tree-budget sweep (paper: 16,32,64,128,256,512,1024).
# Two budgets are enough to show DDTree's acceptance-length scaling vs DFlash.
TREE_BUDGET = "64,256"

TEMPERATURE = 0.0          # greedy / deterministic
MAX_NEW_TOKENS = 512       # paper uses 2048; smaller here to keep the sample cheap

# Acceptance length at temperature 0 is deterministic and GPU-independent, so an
# A100-40GB reproduces the paper's acceptance numbers exactly. Timing/speedup
# numbers would require matching the paper's H100 hardware (set GPU = "H100").
GPU = "A100-40GB"
TIMEOUT_SECONDS = 60 * 60  # 1 hour is plenty for this subset

# Pinned versions for reproducibility. torch 2.5.1 (cu124) + a matching prebuilt
# flash-attn wheel avoids a ~30 min from-source flash-attn build.
TORCH_VERSION = "2.5.1"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-"
    "cp311-cp311-linux_x86_64.whl"
)

HERE = Path(__file__).parent
DDTREE_DIR = HERE / "DDTree"

# --------------------------------------------------------------------------- #
# Image                                                                        #
# --------------------------------------------------------------------------- #

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    # git is needed by some HF dataset loaders; build tools support DDTree's
    # optional inline C++ KV-cache compaction (falls back to Python if absent).
    .apt_install("git", "build-essential")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install(
        # >=4.56: harness uses the newer `dtype=` from_pretrained kwarg (was torch_dtype)
        "transformers==4.57.1",
        "datasets==3.6.0",
        "numpy",
        "loguru",
        "tqdm",
        "ninja",
        "typing_extensions",
        "hf_transfer",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # single-process run: benchmark.py's dist layer no-ops without RANK
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    # Ship the official harness verbatim.
    .add_local_dir(DDTREE_DIR.as_posix(), remote_path="/root/DDTree")
)

app = modal.App("ddtree-dflash-repro")

# Persist HF downloads (models + datasets) and benchmark outputs across runs.
hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)

# Attach an HF token in case any dataset/model ends up gated. The models and
# datasets used here are public, so this is belt-and-suspenders. Uses the
# existing "huggingface" secret in this workspace.
secrets = [modal.Secret.from_name("huggingface")]


# --------------------------------------------------------------------------- #
# Remote benchmark function                                                    #
# --------------------------------------------------------------------------- #

@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=secrets,
)
def run_benchmark() -> dict:
    import subprocess
    import sys

    import numpy as np
    import torch

    runs_dir = Path("/results/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    def slug(v: str) -> str:
        return v.replace("/", "_").replace(":", "_").replace(" ", "_")

    summary = {
        "config": {
            "model": MODEL_NAME,
            "dflash_draft": DRAFT_NAME,
            "dspark_draft": DSPARK_DRAFT_NAME,
            "tasks": TASKS,
            "tree_budget": TREE_BUDGET,
            "sparked_tree_budget": SPARKED_TREE_BUDGET,
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "gpu": GPU,
            "metric": "mean_acceptance_length",
        },
        "results": {},    # the six experiment configs (Stage B)
        "reference": {},  # official baseline/dflash/ddtree (Stage A)
    }

    # ------------------------------------------------------------------ #
    # STAGE A - official reference (unmodified benchmark.py).            #
    # ------------------------------------------------------------------ #
    if RUN_OFFICIAL_REFERENCE:
        save_paths = {}
        for dataset_name, max_samples in TASKS:
            run_name = (
                f"{dataset_name}__{slug(MODEL_NAME)}__{slug(DRAFT_NAME)}"
                f"__temp{slug(str(TEMPERATURE))}__sdpa.pt"
            )
            save_path = runs_dir / run_name
            save_paths[dataset_name] = save_path
            if save_path.exists():
                print(f"[skip] {save_path} already exists", flush=True)
                continue
            cmd = [
                sys.executable, "benchmark.py",
                "--dataset", dataset_name,
                "--max-samples", str(max_samples),
                "--model-name-or-path", MODEL_NAME,
                "--draft-name-or-path", DRAFT_NAME,
                "--tree-budget", TREE_BUDGET,
                "--temperature", str(TEMPERATURE),
                "--max-new-tokens", str(MAX_NEW_TOKENS),
                "--save-path", save_path.as_posix(),
                # NOTE: no --flash-attn -> target uses sdpa -> DFlash + DDTree both run.
            ]
            print(f"\n{'='*70}\n[stage A] {dataset_name} (max_samples={max_samples})\n{'='*70}", flush=True)
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, cwd="/root/DDTree", check=True)
            results_vol.commit()

        def mean_accept(run_data, key):
            vals = [
                float(np.mean(r[key].acceptance_lengths))
                for r in run_data["responses"]
                if key in r and r[key].acceptance_lengths
            ]
            return float(np.mean(vals)) if vals else None

        ref_keys = ["baseline", "dflash"] + [f"ddtree_tb{b}" for b in TREE_BUDGET.split(",")]
        for dataset_name, _ in TASKS:
            run_data = torch.load(save_paths[dataset_name], weights_only=False, map_location="cpu")
            summary["reference"][dataset_name] = {k: mean_accept(run_data, k) for k in ref_keys}

    # ------------------------------------------------------------------ #
    # STAGE B - the six experiment configs (run_acceptance.py).          #
    # ------------------------------------------------------------------ #
    sparked_json = runs_dir / (
        f"sparked__{slug(MODEL_NAME)}__{slug(DRAFT_NAME)}__{slug(DSPARK_DRAFT_NAME)}"
        f"__tb{SPARKED_TREE_BUDGET}__temp{slug(str(TEMPERATURE))}.json"
    )
    tasks_arg = ",".join(f"{name}:{n}" for name, n in TASKS)
    if sparked_json.exists():
        print(f"[skip] {sparked_json} already exists", flush=True)
    else:
        cmd = [
            sys.executable, "run_acceptance.py",
            "--target", MODEL_NAME,
            "--dflash-draft", DRAFT_NAME,
            "--dspark-draft", DSPARK_DRAFT_NAME,
            "--tasks", tasks_arg,
            "--tree-budget", str(SPARKED_TREE_BUDGET),
            "--temperature", str(TEMPERATURE),
            "--max-new-tokens", str(MAX_NEW_TOKENS),
            "--save-json", sparked_json.as_posix(),
        ]
        print(f"\n{'='*70}\n[stage B] six configs: {tasks_arg}\n{'='*70}", flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd="/root/DDTree", check=True)
        results_vol.commit()

    sparked = json.loads(sparked_json.read_text())
    method_ids = list(sparked["config"]["methods"].keys())
    method_labels = sparked["config"]["methods"]
    for dataset_name, _ in TASKS:
        summary["results"][dataset_name] = {
            mid: sparked["results"][dataset_name][mid]["mean_accept"] for mid in method_ids
        }

    (Path("/results") / "summary.json").write_text(json.dumps(summary, indent=2))
    results_vol.commit()

    # Pretty-print the six-config table to the container log.
    print("\n\n" + "=" * 78)
    print("Mean acceptance length (tokens/round), temperature 0, greedy")
    print("=" * 78)
    header = ["dataset"] + [f"{mid}:{method_labels[mid]}" for mid in method_ids]
    print("  ".join(f"{h:>22}" for h in header))
    for dataset_name, _ in TASKS:
        row = [dataset_name] + [f"{summary['results'][dataset_name][mid]:.3f}" for mid in method_ids]
        print("  ".join(f"{c:>22}" for c in row))

    return summary


@app.local_entrypoint()
def main():
    summary = run_benchmark.remote()

    out_dir = HERE / "Results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2))

    print(f"\nSaved summary to {out_file}")
    print(json.dumps(summary["results"], indent=2))
