"""Modal launcher for Experiment 4-faster / 2 (precompute): isolate the precompute
mechanism from the transfer-less win already measured in 1-transfer-less.

1-transfer-less removed build_sparked_tree's two implementation costs (the ~311 MB
static-weight transfer and the full-vocab per-pop compute) and left a flag: at
budget 256 the fast arm is STILL candidate_build-dominant (56%), because .expand
(tree_build_heap) is a SERIAL per-pop CPU matmul, ~linear in budget (6.6 ms/round
@b64 -> 36.7 ms/round @b256).

build_sparked_tree_precompute attacks exactly that: a node at depth d always holds
an element of the depth d-1 candidate set, so the whole [L-1, C, C] transition stack
is ONE batched GPU matmul with no dependence on the walk. Precompute it, take the
per-row top-k, ship it in one transfer -- the heap walk becomes pure CPU indexing.
Strict best-first pop order is untouched, so the tree is unchanged (verified 100%
identical to the fast builder at C=512 in test_precompute_builder.py).

ISOLATION DESIGN. Both arms use the SAME builder family (best-first), the SAME
candidate pool C=512, the SAME everything -- only the transition arithmetic moves
(serial per-pop CPU matmul -> one batched GPU matmul before the walk). Holding C
constant controls acceptance (candidate set ~identical), so:
  * acceptance delta  -> should be ~0 (float noise): confirms the tree is unchanged
  * candidate_build   -> the mechanism's speed win, .expand collapsing into .prep
  * net TPS           -> the wall-clock effect of the precompute solution ALONE

  bestfirst.fast        build_sparked_tree_fast        (tree_mode best-first-fast, C=512)
  bestfirst.precompute  build_sparked_tree_precompute  (tree_mode best-first-precompute, C=512)

NB: exp1 ran the fast arm at C=2048 (its per-pop matmul is only linear in C). Here
both arms run C=512 -- the precompute builder is quadratic in C, and equal C is what
makes the head-to-head clean. The fast@512 acceptance is the correct baseline for
THIS comparison; the exp1 fast@2048 number is a different (transfer-less) question.

Usage:
    modal run modal_benchmark.py --smoke          # end-to-end validation, minutes
    modal run modal_benchmark.py --spawn --detach # full run, detached (survives CLI)
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
    {"name": "dflash_b16", "model_id": "z-lab/Qwen3-4B-DFlash-b16", "kind": "dflash"},
]

CORRECTOR = "dspark_b16_markov"
BEAM_CANDIDATES = 128  # union builder's per-depth shortlist (post harness-6-union fix)


# Lever test at large budget: can we rescue the b256 column? ONE GPU, DDTree
# benchmarking practices (sync ON instrumented-only, compaction ON).
#   ST.pack      - v8 packed transfer, fanout = budget (the current default)
#   ST.fanout64  - + max_fanout=64: transferred slice /4 at b256; old data says
#                  a fanout cap this size costs ~0.4% acceptance
#   ST.beam      - level-synchronous beam builder (all-GPU, near-free build) --
#                  trades adaptive allocation (acceptance) for build cost
# Quant ladder: price the accuracy-vs-speed trade of quantizing the transition
# table (production config: union C=128, max_fanout=64). Local tree agreement vs
# exact: fp16 99.2%, bf16c 29.4%(!), fp8 66.3% -- this run prices the other side
# (real acceptance + TPS).
_ST_BASE = {"tree_mode": "best-first-precompute", "beam_candidates": BEAM_CANDIDATES,
            "max_fanout": 64}
METHODS = [
    {"name": "DDTree", "backbone": "dflash_b16", "corrector": None, "verify": "ddtree"},
    {"name": "ST.exact", "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree",
     "tree_kwargs": {**_ST_BASE}},
    {"name": "ST.fp16", "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree",
     "tree_kwargs": {**_ST_BASE, "table_quant": "fp16"}},
    {"name": "ST.bf16c", "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree",
     "tree_kwargs": {**_ST_BASE, "table_quant": "bf16c"}},
    {"name": "ST.fp8", "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree",
     "tree_kwargs": {**_ST_BASE, "table_quant": "fp8"}},
]

# 6 datasets x 3 samples (first discarded -> 2 measured) x 512 tokens.
TASKS = [
    ["humaneval", 3], ["mbpp", 3], ["gsm8k", 3],
    ["math500", 3], ["mt-bench", 3], ["alpaca", 3],
]

TREE_BUDGETS = [64]
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
SEED = 0
WARMUP_TOKENS = 256
DISCARD_FIRST_SAMPLE = True

# Fresh cache namespaced for this experiment. Driver folds CODE_VERSION
# ("harness-5-precompute") into the fingerprint, so stale fastbf units cannot be
# reused against the new builder.
CACHE_DIR = "/results/precompute/cache"

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
        # Instrumented ONLY — sync-on timing, the DDTree repo's own methodology.
        # (No clean pass: per user decision we report one set of numbers, measured
        # exactly as DDTree measures theirs.)
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

app = modal.App("ddtree-exp4-precompute")

hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


# --------------------------------------------------------------------------- #
# Remote function                                                              #
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

    out = Path("/results/precompute/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


# --------------------------------------------------------------------------- #
# Local entrypoint                                                             #
# --------------------------------------------------------------------------- #

def print_speed_table(summary: dict) -> None:
    """Headline: clean-pass TPS fast vs precompute, per budget, plus dominant phase."""
    timing = summary.get("timing", {})
    if not timing:
        return
    print("\n" + "=" * 72)
    print("Decode TPS (clean) fast vs precompute, per tree budget")
    print("=" * 72)
    for bkey in sorted(timing, key=int):
        print(f"  tree_budget {bkey}:")
        for arm, r in timing[bkey].items():
            print(f"    {arm:<22} {r['tps_clean']:>8.2f} tok/s   "
                  f"dominant phase: {r['dominant_phase']} ({r['dominant_share']*100:.0f}%)")


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
        print("progress:  modal volume ls ddtree-results precompute/cache")
        print("fetch:     modal volume get ddtree-results precompute/summary.json")
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

    print_speed_table(summary)
