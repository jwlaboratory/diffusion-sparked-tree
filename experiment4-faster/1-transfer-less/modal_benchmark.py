"""Modal launcher for Experiment 4b (1-transfer-less): does the candidate-union
restriction make the best-first markov tree fast WITHOUT losing acceptance?

Exp3 localized build_sparked_tree's cost to two sub-costs inside candidate_build:
  * .prep   (tree_build_copy): copying the STATIC full-vocab markov matrices
            W1/W2 (~311 MB) GPU->CPU every round -- budget-invariant ~120 ms.
  * .expand (tree_build_heap): every popped node ran a full-vocab log_softmax +
            top-k + full-vocab bias -- 178 ms/round @b64 -> 733 ms/round @b256.

build_sparked_tree_fast ports the beam builder's candidate-union trick into the
best-first heap: top-k on GPU, gather only the ~U active columns/rows to CPU once,
and run every per-pop compute on that length-U slice. The heap's adaptive
(best-first) allocation is unchanged, so acceptance should be preserved (the
restriction is lossy only if a bias-promoted token falls outside the top-2048 base
candidates).

Two arms on the SAME backbone + corrector (DSpark-b16 + its markov head), run in
ONE job so they see the same machine (fair TPS):

  bestfirst.ref   the existing slow builder (tree_mode default "best-first")
  bestfirst.fast  build_sparked_tree_fast (tree_mode "best-first-fast", C=2048)

Everything else is matched to exp3/exp4 for comparability.

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
]

CORRECTOR = "dspark_b16_markov"
BEAM_CANDIDATES = 256  # optimal top-C (=K) shortlist; knee of the C-sweep


METHODS = [
    # Reference: the existing slow best-first builder (no tree_kwargs -> default
    # tree_mode="best-first", full-vocab per-pop compute, full-matrix transfer).
    {"name": "bestfirst.ref", "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree"},
    # Fast: same adaptive heap, restricted to the candidate union.
    {"name": "bestfirst.fast", "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree",
     "tree_kwargs": {"tree_mode": "best-first-fast", "beam_candidates": BEAM_CANDIDATES}},
]

TASKS = [
    ["gsm8k", 4],
    ["humaneval", 4],
    ["mt-bench", 4],
]

TREE_BUDGETS = [64]  # optimal budget for this DSpark-b16/H100 setup
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
SEED = 0
WARMUP_TOKENS = 256
DISCARD_FIRST_SAMPLE = True

# Fresh cache namespaced for this experiment. Driver adds a config-fingerprint
# subdirectory (folding in CODE_VERSION="harness-4-fastbf"), so stale exp3/exp4
# units cannot be resumed against the new builder.
CACHE_DIR = "/results/fastbf/cache"

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
        "passes": ["clean", "instrumented"],
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

app = modal.App("ddtree-exp4-fastbf")

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

    out = Path("/results/fastbf/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


# --------------------------------------------------------------------------- #
# Local entrypoint                                                             #
# --------------------------------------------------------------------------- #

def print_speed_table(summary: dict) -> None:
    """Headline: clean-pass TPS ref vs fast, per budget, plus tree_build share."""
    timing = summary.get("timing", {})
    if not timing:
        return
    print("\n" + "=" * 72)
    print("Decode TPS (clean) ref vs fast, per tree budget")
    print("=" * 72)
    for bkey in sorted(timing, key=int):
        print(f"  tree_budget {bkey}:")
        for arm, r in timing[bkey].items():
            print(f"    {arm:<18} {r['tps_clean']:>8.2f} tok/s   "
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
        print("progress:  modal volume ls ddtree-results fastbf/cache")
        print("fetch:     modal volume get ddtree-results fastbf/summary.json")
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
