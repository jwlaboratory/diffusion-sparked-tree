"""Modal launcher for Experiment 2: does a longer draft horizon raise acceptance?

ONE stage. The remote GPU function loads the target verifier + both DSpark draft
backbones once and runs a `backbone x corrector x verify` matrix via
`DDTree/run_experiment.py` (imported in-process -- no subprocess, no shell-out).

The question this proves (temperature 0, greedy): our fine-tuned block-16 DSpark
drafter drafts 16 tokens per block versus the stock block-7 checkpoint's 7. If the
longer horizon is worth training, the tree should commit more accepted tokens per
round. The two checkpoints are architecturally identical, so this is a clean
block-size ablation.

Arms (backbone x corrector x verify):
    dspark_b7.tree           block-7,  no corrector, tree   (markov-off control)
    dspark_b7.markov.tree    block-7  + its own markov, tree
    dspark_b16.tree          block-16, no corrector, tree   (markov-off control)
    dspark_b16.markov.tree   block-16 + its own markov, tree

The two `*.markov.tree` arms are the headline comparison; the `*.tree` arms isolate
the block-size effect from the corrector's contribution. The `block_size` rollup
reports per-dataset and overall acceptance for b7 vs b16, absolute delta and %
change, per arm.

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

# Draft backbones. Both are `kind="dspark"`; they differ only in block_size, which
# is intrinsic to each checkpoint's config (7 vs 16) -- no runtime override needed.
# BLOCK16_MODEL_ID is our fine-tuned checkpoint; kept as a single constant so it is
# trivially swappable if the HF repo id changes.
BLOCK16_MODEL_ID = "shreybirmiwal/Qwen3-4B-DSpark-b16"
BACKBONES = [
    {"name": "dspark_b7", "model_id": "deepseek-ai/dspark_qwen3_4b_block7", "kind": "dspark"},
    {"name": "dspark_b16", "model_id": BLOCK16_MODEL_ID, "kind": "dspark"},
]

# Each method is backbone x corrector x verify. corrector=None means the markov-off
# control; a corrector name is "<dspark-backbone>_markov" (auto-derived from the
# backbone's own head). The two headline arms are the `*.markov.tree` ones.
METHODS = [
    {"name": "dspark_b7.tree", "backbone": "dspark_b7", "corrector": None, "verify": "tree"},
    {"name": "dspark_b7.markov.tree", "backbone": "dspark_b7", "corrector": "dspark_b7_markov", "verify": "tree"},
    {"name": "dspark_b16.tree", "backbone": "dspark_b16", "corrector": None, "verify": "tree"},
    {"name": "dspark_b16.markov.tree", "backbone": "dspark_b16", "corrector": "dspark_b16_markov", "verify": "tree"},
]

# Small representative slice: one math, one code, one chat. (dataset, max_samples)
# Small on purpose: this is a smoke-sized slice, and the
# (budget x dataset) matrix is 6 units. Raise it once the direction is established
# -- at n=4 only large effects are resolvable (see reproduce.md).
TASKS = [
    ["gsm8k", 4],
    ["humaneval", 4],
    ["mt-bench", 4],
]

# Swept, not fixed. At a fixed budget a longer block spreads the same nodes over
# more depths, so b16 trades width for depth; 256 buys back the width and separates
# "block size helps" from "we under-budgeted the tree".
TREE_BUDGETS = [64, 256]
TEMPERATURE = 0.0          # greedy / deterministic (tree build is greedy top-k regardless)
MAX_NEW_TOKENS = 512
SEED = 0
CONFIDENCE_THRESHOLD = 0.0  # dspark chain only (unused by the tree arms here)
MEASURE_PER_DEPTH = True
DEPTH_REPORT_LIMIT = 16    # must reach the b16 horizon or its whole advantage is invisible

# Each (budget, dataset) unit is checkpointed here on the results volume and the
# volume committed, so an interruption costs at most one unit and a re-run resumes.
CACHE_DIR = "/results/block16/cache"

# Acceptance length at temperature 0 is deterministic and GPU-independent, so an
# A100-40GB reproduces the numbers exactly. Timing/speedup would need matching H100.
GPU = "A100-40GB"
TIMEOUT_SECONDS = 6 * 60 * 60   # generous: units checkpoint, so a long run is cheap to resume

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
    return {
        "target": TARGET,
        "backbones": BACKBONES,
        "methods": list(METHODS),
        "tasks": TASKS,
        "tree_budgets": TREE_BUDGETS,
        "cache_dir": CACHE_DIR,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "measure_per_depth": MEASURE_PER_DEPTH,
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

app = modal.App("dspark-block16-compare")

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

    # commit after every checkpointed unit so progress survives a timeout/kill
    summary = exp.run(cfg, on_checkpoint=results_vol.commit)

    # Namespaced: the `ddtree-results` volume is shared with experiment 1, which
    # writes /results/summary.json. Writing there would clobber its results.
    out = Path("/results/block16/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main():
    summary = run_experiment.remote(build_run_config())

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nSaved summary to {out_dir / 'summary.json'}")
    if summary.get("block_size"):
        print("\nBlock size (acceptance-length change from b7 -> b16), per tree budget:")
        for bkey in sorted(summary["block_size"], key=int):
            print(f"  tree_budget {bkey}:")
            for arm, r in summary["block_size"][bkey].items():
                o = r["overall"]
                pct = o.get("pct_change")
                if pct is None:
                    print(f"    {arm}: n/a")
                else:
                    print(f"    {arm}: b{r['small_block']}={o['small']:.3f} -> "
                          f"b{r['large_block']}={o['large']:.3f} ({pct:+.1f}%)")
