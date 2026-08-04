"""Modal launcher: benchmark ONE data-scaling checkpoint (_data2k4/_data10k/_data24k).

Companion to modal_benchmark.py for the data-scaling arms trained by
training/modal_train.py::pipeline_data. Those checkpoints live on the
`ddtree-train` volume (published by the pipeline, not pushed to HF), so this
launcher mounts that volume and loads the backbone from a path.

One arm per invocation, each in its own results namespace
(/results/block16_datascale/<arm>/). Deliberate: run_experiment's (budget,
dataset) cache units store ALL methods of a run, so growing the method list
later would poison resume. Per-arm runs also let each arm benchmark as soon as
its training finishes. Config (tasks, budgets, seed, temperature) matches
modal_benchmark.py exactly, so results are directly comparable with the
existing exp-2 numbers for dspark_b7 / dspark_b16 (_best).

Usage (run the FUNCTION directly -- do not go through the local entrypoint):
    modal run --detach modal_benchmark_datascale.py::run_experiment --arm data2k4
    modal run --detach modal_benchmark_datascale.py::run_experiment --arm data10k
    modal run --detach modal_benchmark_datascale.py::run_experiment --arm data24k

A `local_entrypoint` that calls `.remote()` is driven from the local machine, so
the remote call is cancelled if the client dies -- even with --detach (a wifi
drop killed two runs this way). Invoking the @app.function itself makes the run
fully server-side; results persist on the ddtree-results volume:
    modal volume get ddtree-results block16_datascale/<arm>/summary.json
"""

import json
from pathlib import Path

import modal

TARGET = "Qwen/Qwen3-4B"

# Must match modal_benchmark.py for comparability.
TASKS = [
    ["gsm8k", 4],
    ["humaneval", 4],
    ["mt-bench", 4],
]
TREE_BUDGETS = [64, 256]
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
SEED = 0
CONFIDENCE_THRESHOLD = 0.0
MEASURE_PER_DEPTH = True
DEPTH_REPORT_LIMIT = 16

# H100 by default for wall-clock; acceptance at temperature 0 is deterministic
# and GPU-independent (see modal_benchmark.py / reproduce.md), so GPU choice
# affects speed only. Override with DDTREE_BENCH_GPU=A100-40GB for exact-repro runs.
import os as _os
GPU = _os.environ.get("DDTREE_BENCH_GPU", "H100")
TIMEOUT_SECONDS = 6 * 60 * 60

TORCH_VERSION = "2.5.1"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-"
    "cp311-cp311-linux_x86_64.whl"
)

HERE = Path(__file__).parent
DDTREE_DIR = HERE / "DDTree"

VALID_ARMS = ("data2k4", "data10k", "data24k")


def build_run_config(arm: str) -> dict:
    name = f"dspark_b16_{arm}"
    return {
        "target": TARGET,
        "backbones": [
            {"name": name,
             "model_id": f"/train/ckpt_final/dspark_block16_{arm}",
             "kind": "dspark"},
        ],
        "methods": [
            {"name": f"{name}.tree", "backbone": name, "corrector": None, "verify": "tree"},
            {"name": f"{name}.markov.tree", "backbone": name,
             "corrector": f"{name}_markov", "verify": "tree"},
        ],
        "tasks": TASKS,
        "tree_budgets": TREE_BUDGETS,
        "cache_dir": f"/results/block16_datascale/{arm}/cache",
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

app = modal.App("dspark-block16-datascale")

hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)
train_vol = modal.Volume.from_name("ddtree-train")
secrets = [modal.Secret.from_name("huggingface")]


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache, "/results": results_vol, "/train": train_vol},
    secrets=secrets,
)
def run_experiment(arm: str) -> dict:
    import os
    import sys

    assert arm in VALID_ARMS, f"arm must be one of {VALID_ARMS}"
    cfg = build_run_config(arm)

    sys.path.insert(0, "/root/DDTree")
    import run_experiment as exp

    ckpt = cfg["backbones"][0]["model_id"]
    assert os.path.exists(os.path.join(ckpt, "config.json")), \
        f"checkpoint not published: {ckpt}"
    # keep the block-size guard active for the path-loaded backbone
    exp.EXPECTED_BLOCK_SIZE[cfg["backbones"][0]["name"]] = 16

    summary = exp.run(cfg, on_checkpoint=results_vol.commit)

    out = Path(f"/results/block16_datascale/{arm}/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache, "/results": results_vol, "/train": train_vol},
    secrets=secrets,
)
def run_unit(arm: str, budget: int, dataset: str) -> None:
    """Compute ONE (budget, dataset) unit for one arm and cache it.

    Parallelism lever: the sequential run_experiment walks 6 units in one
    container; fanning units out to one container each cuts wall-clock to
    boot + model load + the single unit. Unit results are deterministic and
    independent, and cache files are per-unit, so concurrent writers cannot
    conflict. After all 6 units are cached, run_experiment(arm) assembles the
    summary (its unit loop resumes entirely from cache).
    """
    import sys

    assert arm in VALID_ARMS, f"arm must be one of {VALID_ARMS}"
    cfg = build_run_config(arm)
    assert any(dataset == t[0] for t in cfg["tasks"]), f"unknown dataset {dataset}"
    assert budget in cfg["tree_budgets"], f"unknown budget {budget}"
    cfg["tree_budgets"] = [budget]
    cfg["tasks"] = [t for t in cfg["tasks"] if t[0] == dataset]

    sys.path.insert(0, "/root/DDTree")
    import run_experiment as exp

    exp.EXPECTED_BLOCK_SIZE[cfg["backbones"][0]["name"]] = 16
    exp.run(cfg, on_checkpoint=results_vol.commit)


@app.local_entrypoint()
def main(arm: str = "data2k4"):
    """Interactive convenience only -- for detached runs invoke ::run_experiment
    directly (see module docstring). This entrypoint is client-driven and will
    cancel the remote call if the local machine disconnects."""
    summary = run_experiment.remote(arm)

    out_dir = HERE / "results" / "datascale"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{arm}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {out_dir / f'{arm}_summary.json'}")

    for bkey, datasets in summary["results"].items():
        print(f"tree_budget {bkey}:")
        for ds, entries in datasets.items():
            for mname, e in entries.items():
                print(f"  {ds:10s} {mname}: mean_accept={e['mean_accept']:.2f} (n={e['rounds']})")
