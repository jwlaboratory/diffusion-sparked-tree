"""Modal launcher for Experiment 4a: which beam width schedule is best?

The level-synchronous beam builder fixes the tree SHAPE in advance -- widths[d]
nodes at depth d -- which is what kills the ~budget dependent GPU round-trips
that make the naive best-first tree ~20x more expensive than DDTree's. But it
forces the question the old experiments answered with "flat": HOW should the
budget be distributed across depths?

This run sweeps the schedule families through the shared harness driver, one arm
per schedule, all on the SAME backbone + corrector (DSpark-b16 + its markov head):

  shape axis (full-depth trees, opposite orientations):
    beam.geo50      [32,16,8,4,2,1,1]        heavy front-load (intuition says this)
    beam.geo60      [26,15,9,6,3,2,1,1,1]    old default decay
    beam.geo75      [16,12,9,7,5,4,3,2,2,..] mild front-load
    beam.flat       [4]*16                    uniform
    beam.invgeo75   reversed geo75            mild back-load (same width multiset)
    beam.invgeo50   reversed geo50            heavy back-load

  depth-vs-width axis (uniform shape, truncated horizon):
    beam.depth8     [8]*8                     twice as wide, half as deep
    beam.depth4     [16]*4                    4x as wide, quarter depth

  reference ceiling:
    bestfirst.ref   the naive best-first builder (adaptive allocation; the
                    acceptance any fixed schedule is trying to approach)

Primary metric: mean acceptance length (resolves at ~0.5%). Secondary: clean-pass
TPS and the tree_build phase share (per-cell speed noise is ~16%; do not rank
schedules on TPS alone).

Usage:
    modal run modal_benchmark.py --smoke        # end-to-end validation, minutes
    modal run modal_benchmark.py --spawn        # full run, fire-and-forget
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
BEAM_CANDIDATES = 2048  # union top-C restriction; the old beam sweep's setting


def _beam(name: str, schedule: dict) -> dict:
    return {
        "name": name, "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree",
        "tree_kwargs": {"tree_mode": "beam", "beam_schedule": schedule,
                        "beam_candidates": BEAM_CANDIDATES},
    }


METHODS = [
    _beam("beam.geo50", {"kind": "geometric", "decay": 0.5}),
    _beam("beam.geo60", {"kind": "geometric", "decay": 0.6}),
    _beam("beam.geo75", {"kind": "geometric", "decay": 0.75}),
    _beam("beam.flat", {"kind": "flat"}),
    _beam("beam.invgeo75", {"kind": "inv_geometric", "decay": 0.75}),
    _beam("beam.invgeo50", {"kind": "inv_geometric", "decay": 0.5}),
    _beam("beam.depth8", {"kind": "flat_depth", "depth": 8}),
    _beam("beam.depth4", {"kind": "flat_depth", "depth": 4}),
    # Acceptance ceiling: adaptive best-first (the slow naive builder).
    {"name": "bestfirst.ref", "backbone": "dspark_b16", "corrector": CORRECTOR, "verify": "tree"},
]

TASKS = [
    ["gsm8k", 4],
    ["humaneval", 4],
    ["mt-bench", 4],
]

TREE_BUDGETS = [64, 256]
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
SEED = 0
WARMUP_TOKENS = 256
DISCARD_FIRST_SAMPLE = True

# Namespaced away from exp3 (/results/timings). Driver adds a config-fingerprint
# subdirectory, so stale-schema resume is impossible.
CACHE_DIR = "/results/beamsched/cache"

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

app = modal.App("ddtree-exp4-beamsched")

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

    out = Path("/results/beamsched/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


# --------------------------------------------------------------------------- #
# Local entrypoint                                                             #
# --------------------------------------------------------------------------- #

def print_acceptance_table(summary: dict) -> None:
    """The experiment's headline: mean acceptance per schedule, per budget,
    aggregated across datasets weighted by rounds (sum accepts / sum rounds)."""
    clean = summary.get("results", {}).get("clean", {})
    if not clean:
        return
    print("\n" + "=" * 72)
    print("Mean acceptance length per schedule (clean pass, round-weighted)")
    print("=" * 72)
    for bkey in sorted(clean, key=int):
        by_ds = clean[bkey]
        names = sorted({n for d in by_ds.values() for n in d})
        rows = []
        for name in names:
            tot_accept, tot_rounds = 0.0, 0
            for d in by_ds.values():
                e = d.get(name)
                if e:
                    tot_accept += e["mean_accept"] * e["rounds"]
                    tot_rounds += e["rounds"]
            if tot_rounds:
                rows.append((tot_accept / tot_rounds, name))
        print(f"  tree_budget {bkey}:")
        for acc, name in sorted(rows, reverse=True):
            print(f"    {name:<18} acceptance={acc:.3f}")


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
        print("progress:  modal volume ls ddtree-results beamsched/cache")
        print("fetch:     modal volume get ddtree-results beamsched/summary.json")
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

    print_acceptance_table(summary)

    if summary.get("timing"):
        print("\nDecode TPS (clean) per tree budget:")
        for bkey in sorted(summary["timing"], key=int):
            print(f"  tree_budget {bkey}:")
            for arm, r in summary["timing"][bkey].items():
                print(f"    {arm:<18} {r['tps_clean']:>8.2f} tok/s   "
                      f"dominant phase: {r['dominant_phase']} ({r['dominant_share']*100:.0f}%)")
