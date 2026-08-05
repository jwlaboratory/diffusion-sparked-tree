"""Modal launcher for Experiment 4-faster / 2-precompute / csweep: sweep the
candidate size C on BOTH restricted best-first builders.

WHY sweep both. `build_sparked_tree_fast` restricts to a deduped UNION of the
per-depth top-C tokens; `build_sparked_tree_precompute` restricts to PER-DEPTH
top-C (it needs a fixed [L,C] table). Same C therefore means slightly different
candidate coverage, and the two also differ in where the float reduction happens
(serial per-pop CPU matmul vs one batched GPU baddbmm). So their acceptance-vs-C
knees -- the smallest C at which acceptance saturates against the exact best-first
ceiling -- need not coincide. We sweep C directly on each and read the knee off.

Arms (all DSpark-b16 + dspark_b16_markov corrector, verify="tree", ONE job so TPS
is comparable):
  bestfirst.ref                     exact best-first (C=inf acceptance ceiling)
  fast.c{128,256,512,1024,2048}     tree_mode best-first-fast,       beam_candidates=C
  precompute.c{128,256,512,1024}    tree_mode best-first-precompute, beam_candidates=C
(precompute is O(L C^2 R), so it is swept only up to 1024; fast is ~linear in C.)

Budget 64 only. gsm8k/humaneval/mt-bench x 8 (up from 4: acceptance deltas are
~1-2%, near the noise floor, so more samples). Everything else identical to the
prior runs: temp 0, max_new_tokens 512, seed 0, warmup 256, discard-first, passes
clean+instrumented, H100 + 8 CPU.

Usage:
    modal run modal_benchmark.py --smoke          # end-to-end validation, minutes
    modal run --detach modal_benchmark.py --spawn # full run, detached (survives CLI)
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

FAST_CS = [256]
PRECOMPUTE_CS = [256]


def _sweep_arm(builder_tag: str, tree_mode: str, C: int) -> dict:
    return {
        "name": f"{builder_tag}.c{C}", "backbone": "dspark_b16", "corrector": CORRECTOR,
        "verify": "tree",
        "tree_kwargs": {"tree_mode": tree_mode, "beam_candidates": C},
    }


METHODS = (
    # Final-config head-to-head at budget 128, C=256: naive ref -> transfer-less
    # fast -> precompute. One run feeds both the 1-transfer-less and 2-precompute charts.
    [{"name": "bestfirst.ref", "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree"}]
    + [_sweep_arm("fast", "best-first-fast", C) for C in FAST_CS]
    + [_sweep_arm("precompute", "best-first-precompute", C) for C in PRECOMPUTE_CS]
)

# Holistic dataset mix: easy+hard math, standard+hard code, two chat styles.
TASKS = [
    ["gsm8k", 8],
    ["humaneval", 8],
    ["mt-bench", 8],
]

TREE_BUDGETS = [128]
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
SEED = 0
WARMUP_TOKENS = 256
DISCARD_FIRST_SAMPLE = True

# Fresh cache namespaced for the C-sweep. Driver folds CODE_VERSION
# ("harness-5-csweep") into the fingerprint, so no fastbf/precompute unit resumes here.
CACHE_DIR = "/results/final128/cache"

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
# final128/ -> 2-precompute/ -> experiment4-faster/ -> repo root; harness lives at root.
HARNESS_DIR = HERE.parent.parent / "harness"  # 3-budget-dataset/ -> experiment4-faster/ -> repo root


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

app = modal.App("ddtree-exp4-final128")

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

    out = Path("/results/final128/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


# --------------------------------------------------------------------------- #
# Local entrypoint                                                             #
# --------------------------------------------------------------------------- #

def print_sweep_table(summary: dict) -> None:
    """Headline: clean-pass TPS and dominant phase per arm (the whole C sweep)."""
    timing = summary.get("timing", {})
    if not timing:
        return
    print("\n" + "=" * 72)
    print("Decode TPS (clean) per arm, budget 64")
    print("=" * 72)
    for bkey in sorted(timing, key=int):
        print(f"  tree_budget {bkey}:")
        for arm, r in timing[bkey].items():
            print(f"    {arm:<22} {r['tps_clean']:>8.2f} tok/s   "
                  f"dominant: {r['dominant_phase']} ({r['dominant_share']*100:.0f}%)")


@app.local_entrypoint()
def main(smoke: bool = False, spawn: bool = False, parallel: int = 0):
    cfg = build_run_config()
    if smoke:
        # Exercise the new-dataset loaders (livecodebench multi-file download is the
        # real risk) and both budget extremes.
        cfg["tasks"] = [["aime24", 1], ["livecodebench", 1], ["alpaca", 1]]
        cfg["max_new_tokens"] = 64
        cfg["tree_budgets"] = [16, 256]
        cfg["warmup_tokens"] = 32

    if spawn:
        # parallel>1: fan out N containers over the SHARED checkpoint dir, each
        # starting on a rotated slice of the unit grid (shard_offset, excluded from
        # fingerprint). They resume already-done units; any that finishes assembles
        # the full summary from cache. ~N x faster wall-clock.
        n_units = len(cfg["tasks"]) * len(cfg["tree_budgets"])
        n = max(1, int(parallel))
        offsets = [i * n_units // n for i in range(n)] if n > 1 else [0]
        ids = []
        for off in offsets:
            call = run_experiment.spawn({**cfg, "shard_offset": off})
            ids.append(call.object_id)
        print(f"spawned {len(ids)} shard(s) offsets={offsets}: {ids}")
        print("progress:  modal volume ls ddtree-results final128/cache")
        print("fetch:     modal volume get ddtree-results final128/summary.json")
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

    print_sweep_table(summary)
