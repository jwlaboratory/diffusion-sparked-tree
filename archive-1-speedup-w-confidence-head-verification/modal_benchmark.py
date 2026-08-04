"""Experiment 4, stage 1: can we verify less?

The tree verifies 65-129 positions per round to accept ~8 tokens; a chain verifies
17 to accept ~6. That ratio is what makes trees lose above concurrency ~4
(`old-experiments/RESULTS.md` section 9). This experiment asks whether the width
can be spent only on the rounds that convert.

`investigate/FINDINGS.md` established offline, from logs already in the repo:

  * whether width pays THIS round is predictable, but from the current round's
    drafter state -- history is at chance (AUC ~0.5 within dataset).
  * an estimator only needs +-1.5-2 tokens of accuracy to capture the benefit.
  * the gate must shrink on CONFIDENT rounds. Rounds where the chain accepts <=3
    are 64% of rounds and carry 71% of everything tree width produces, so cutting
    budget on uncertain rounds destroys exactly the value it is trying to save.

This stage measures rather than gates. It runs the tree arm with
`measure_confidence=True`, which emits `confidence_by_round` paired positionally
with `acceptance_lengths` -- both from the SAME run, so there is no cross-run
alignment step. (Alignment is what killed finding 1: methods diverge in output by
up to 113 tokens, so a shared token index stops meaning a shared context.)

Output feeds two things:
  * `investigate/price_gate.py` -- go/no-go on the free estimator. Pass/fail is
    its error in tokens: <=2.0 captures the benefit, >=3.0 buys almost nothing.
  * `train_head.py` -- training data for the tree-aware head, if the free one misses.

    modal run experiment4-speedup-verification/modal_benchmark.py

Stage 2 (gated vs ungated, on H100 for honest wall-clock) is a separate run and is
blocked on the threshold this stage produces.
"""

import json
from pathlib import Path

import modal

TARGET = "Qwen/Qwen3-4B"
BLOCK16_MODEL_ID = "shreybirmiwal/Qwen3-4B-DSpark-b16"

BACKBONES = [
    {"name": "dspark_b16", "model_id": BLOCK16_MODEL_ID, "kind": "dspark"},
]

# One arm. This stage is pure measurement -- adding arms would multiply GPU time
# without adding anything the analysis needs, since the (confidence, accepted)
# pairing comes from within a single run.
METHODS = [
    {
        "name": "dspark_b16.markov.tree",
        "backbone": "dspark_b16",
        "corrector": "dspark_b16_markov",
        "verify": "tree",
        "measure_confidence": True,
    },
]

# 8 prompts x 3 datasets x ~60 rounds each is ~1500 labelled rounds, which is
# ample for a 6-input head and enough to resolve an estimator error of ~0.3 tokens.
# One math, one code, one chat: FINDINGS.md finding 5 shows the gate's value is
# strongly workload-dependent, so a single-dataset read would not generalise.
TASKS = [
    ["gsm8k", 8],
    ["humaneval", 8],
    ["mt-bench", 8],
]

# Budget 64 only. The question here is "which rounds need width", not "how much
# width" -- and 64 is where the offline analysis was calibrated.
TREE_BUDGETS = [64]
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
SEED = 0
CONFIDENCE_THRESHOLD = 0.0
MEASURE_PER_DEPTH = True
DEPTH_REPORT_LIMIT = 16

CACHE_DIR = "/results/exp4/cache"

# Acceptance and confidence at temperature 0 are deterministic and GPU-independent,
# so an A100 measures them exactly. Stage 2 needs an H100 because it claims
# wall-clock, and cross-GPU timing comparisons are a standing methodology error.
GPU = "A100-40GB"
TIMEOUT_SECONDS = 6 * 60 * 60

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
    .add_local_dir(DDTREE_DIR.as_posix(), remote_path="/root/DDTree")
)

app = modal.App("exp4-speedup-verification")

hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache, "/results": results_vol},
    secrets=secrets,
)
def run_measurement(cfg: dict) -> dict:
    import sys

    sys.path.insert(0, "/root/DDTree")
    import run_experiment as exp

    summary = exp.run(cfg, on_checkpoint=results_vol.commit)

    # The summary carries only aggregates. The per-round (confidence, accepted)
    # pairs -- the entire point of this stage -- live in the per-unit cache files,
    # so bundle them into one artifact the local analysis can consume directly.
    raw = {}
    cache = Path(cfg["cache_dir"])
    for unit in sorted(cache.glob("*.json")):
        for method_name, rec in json.loads(unit.read_text()).items():
            if not isinstance(rec, dict) or not rec.get("conf"):
                continue
            key = f"{unit.stem}::{method_name}"
            raw[key] = {"acc": rec["acc"], "conf": rec["conf"]}

    out_dir = Path("/results/exp4")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "rounds.json").write_text(json.dumps(raw))
    results_vol.commit()
    return {"summary": summary, "raw": raw}


@app.local_entrypoint()
def main():
    result = run_measurement.remote(build_run_config())

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2))
    (out_dir / "rounds.json").write_text(json.dumps(result["raw"]))

    n = sum(len(v["acc"]) for v in result["raw"].values())
    print(f"\nSaved {n} labelled rounds across {len(result['raw'])} units")
    print(f"  {out_dir / 'rounds.json'}")
    print("\nNext:")
    print(f"  python3 investigate/price_gate.py {out_dir / 'rounds.json'} --ceiling 16")
