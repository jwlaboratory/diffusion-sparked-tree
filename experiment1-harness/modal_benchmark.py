"""
Modal launcher for Experiment 1: markov-corrector transfer across drafters.

ONE stage. The remote GPU function loads the target verifier + a set of draft
backbones once and runs a `backbone x corrector x verify` matrix via
`DDTree/run_experiment.py` (imported in-process -- no subprocess, no shell-out).
The official `DDTree/benchmark.py` is NOT on this path; its equivalence to our
tree loop is a one-off check, not part of the run.

The question this proves (temperature 0, greedy): a markov head is a residual
correction fitted to the backbone it was trained on. Same head should raise
acceptance a lot on its OWN backbone and not at all (or hurt) on a FOREIGN one.

Default matrix (the 2x2):
    dflash.tree          DFlash-b16, no corrector, tree        (foreign OFF, == ddtree)
    dflash.markov.tree   DFlash-b16 + DSpark-b7 markov, tree   (foreign ON)
    dspark.tree          DSpark-b7,  no corrector, tree        (own OFF)
    dspark.markov.tree   DSpark-b7  + DSpark-b7 markov, tree   (own ON)

Backbones are parameterized by (checkpoint, kind, block_size). Add more, or run a
b16 checkpoint at block_size=7 to depth-match a foreign backbone to a b7 head --
that is why there is no separate depth cap. See reproduce.md.

Usage:
    modal run modal_benchmark.py

All knobs are the UPPERCASE constants below. Nothing here needs a GPU locally.
"""

import json
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Experiment configuration (all tunables live here)                            #
# --------------------------------------------------------------------------- #

TARGET = "Qwen/Qwen3-4B"

# Draft backbones. `kind` is intrinsic to the checkpoint ("dflash" | "dspark").
# `block_size` is an optional runtime override (default = checkpoint config); e.g.
# add {"name": "dflash_b7", "model_id": "z-lab/Qwen3-4B-DFlash-b16",
#      "kind": "dflash", "block_size": 7} to depth-match DFlash to the b7 head.
BACKBONES = [
    {"name": "dflash_b16", "model_id": "z-lab/Qwen3-4B-DFlash-b16", "kind": "dflash"},
    {"name": "dspark_b7", "model_id": "deepseek-ai/dspark_qwen3_4b_block7", "kind": "dspark"},
]

# The 2x2. Each entry is backbone x corrector x verify. corrector=None means no
# correction; a corrector name is "<dspark-backbone>_markov" (auto-derived).
METHODS = [
    {"name": "dflash.tree", "backbone": "dflash_b16", "corrector": None, "verify": "tree"},
    {"name": "dflash.markov.tree", "backbone": "dflash_b16", "corrector": "dspark_b7_markov", "verify": "tree"},
    {"name": "dspark.tree", "backbone": "dspark_b7", "corrector": None, "verify": "tree"},
    {"name": "dspark.markov.tree", "backbone": "dspark_b7", "corrector": "dspark_b7_markov", "verify": "tree"},
]
# Native chain references, on by default:
#   dflash.chain = DFlash-b16 block drafting, no tree, no markov (old config 0)
#   dspark.chain = DSpark-b7 drafting with its own markov head, native serial (old config 2)
CHAIN_ANCHORS = True

# Markov head used for the tree-free corrector-fit probe (measured on no-corrector runs).
PROBE_CORRECTOR = "dspark_b7_markov"

# Small representative slice: one math, one code, one chat. (dataset, max_samples)
TASKS = [
    ["gsm8k", 8],
    ["humaneval", 8],
    ["mt-bench", 8],
]

TREE_BUDGET = 64
TEMPERATURE = 0.0          # greedy / deterministic (tree build is greedy top-k regardless)
MAX_NEW_TOKENS = 512
SEED = 0
CONFIDENCE_THRESHOLD = 0.0  # dspark chain only
MEASURE_PER_DEPTH = True
MEASURE_CORRECTOR_FIT = True
DEPTH_REPORT_LIMIT = 7      # per-depth / probe reporting horizon (depth-match to b7)

# Acceptance length at temperature 0 is deterministic and GPU-independent, so an
# A100-40GB reproduces the numbers exactly. Timing/speedup would need matching H100.
GPU = "A100-40GB"
TIMEOUT_SECONDS = 60 * 60

# Pinned versions. torch 2.5.1 (cu124) + a matching prebuilt flash-attn wheel.
TORCH_VERSION = "2.5.1"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-"
    "cp311-cp311-linux_x86_64.whl"
)

HERE = Path(__file__).parent
DDTREE_DIR = HERE / "DDTree"


def build_run_config() -> dict:
    methods = list(METHODS)
    if CHAIN_ANCHORS:
        methods = methods + [
            {"name": "dflash.chain", "backbone": "dflash_b16", "corrector": None, "verify": "chain"},
            {"name": "dspark.chain", "backbone": "dspark_b7", "corrector": None, "verify": "chain"},
        ]
    return {
        "target": TARGET,
        "backbones": BACKBONES,
        "methods": methods,
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
        # >=4.56: harness uses the newer `dtype=` from_pretrained kwarg (was torch_dtype)
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
    .add_local_dir(DDTREE_DIR.as_posix(), remote_path="/root/DDTree")
)

app = modal.App("ddtree-markov-transfer")

hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


# --------------------------------------------------------------------------- #
# Remote function (single stage, in-process)                                   #
# --------------------------------------------------------------------------- #

@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=secrets,
)
def run_experiment(cfg: dict) -> dict:
    import sys

    sys.path.insert(0, "/root/DDTree")
    import run_experiment as exp  # the driver in DDTree/

    summary = exp.run(cfg)

    out = Path("/results/summary.json")
    out.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main():
    summary = run_experiment.remote(build_run_config())

    out_dir = HERE / "Results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nSaved summary to {out_dir / 'summary.json'}")
    if summary.get("transfer"):
        print("\nTransfer (acceptance % change from adding the corrector):")
        for bb, t in summary["transfer"].items():
            pct = t.get("accept_pct_change")
            if pct is None:
                print(f"  {bb}: n/a")
            else:
                print(f"  {bb}: {t['accept_off']:.3f} -> {t['accept_on']:.3f} ({pct:+.1f}%)")
    if summary.get("corrector_fit"):
        lim = summary["config"]["depth_report_limit"]
        print(f"\nCorrector fit (depth<={lim}):")
        for name, cf in summary["corrector_fit"].items():
            o = cf[f"overall_depth_le_{lim}"]
            print(f"  {cf['backbone']}: delta_hit={o['delta_hit']:+.4f} delta_ce={o['delta_ce']:+.4f}")
