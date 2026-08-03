"""Final benchmark: DSpark vs DDTree vs sparked-tree, one locked configuration.

    modal run --detach final_benchmark/run_final.py::final

One GPU container per dataset, all arms inside each container so a slow container
cannot make one method look worse than another. Results land on the volume and in
`report.py`-readable JSON.

Every setting comes from config.py, which records why each was chosen. Nothing is
swept here — this run exists to produce one comparison, not to explore.
"""

import os
import sys
from pathlib import Path

import modal

# Resolvable both locally (where this file sits next to config.py) and inside the
# container (where the directory is mounted at a fixed path and __file__ is not
# necessarily its sibling).
for _candidate in (Path(__file__).resolve().parent, Path("/root/final_benchmark")):
    if (_candidate / "config.py").exists():
        sys.path.insert(0, str(_candidate))
        break
import config as cfg  # noqa: E402

app = modal.App("sparked-tree-final")

REPO_ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "datasets",
                 "safetensors", "loguru", "tqdm", "numpy")
    .add_local_dir(REPO_ROOT / "ddtree", remote_path="/root/ddtree")
    .add_local_dir(REPO_ROOT / "final_benchmark", remote_path="/root/final_benchmark")
)

vol = modal.Volume.from_name("ddtree-train", create_if_missing=True)
hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)

GPU = os.environ.get("FINAL_GPU", "H100")


def _run(cmd):
    import subprocess
    env = dict(os.environ, HF_HOME="/hfcache", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd="/root/ddtree", env=env, check=True)


def _aggregate(save_path):
    """Per-method acceptance, tpot and stage times, summed over prompts."""
    import torch

    data = torch.load(save_path, weights_only=False)
    agg = {}
    for response in data["responses"]:
        for method, result in response.items():
            row = agg.setdefault(method, {"accept": [], "tpot": [], "stage": {},
                                          "tokens": 0, "rounds": 0})
            row["accept"].extend(result.acceptance_lengths)
            row["tpot"].append(result.time_per_output_token)
            row["tokens"] += result.num_output_tokens
            row["rounds"] += result.decode_rounds
            for stage, elapsed in getattr(result, "stage_times", {}).items():
                row["stage"][stage] = row["stage"].get(stage, 0.0) + elapsed
    for row in agg.values():
        row["mean_accept"] = sum(row["accept"]) / max(len(row["accept"]), 1)
        row["mean_tpot_ms"] = 1e3 * sum(row["tpot"]) / max(len(row["tpot"]), 1)
        del row["accept"], row["tpot"]
    return agg


@app.function(image=image, gpu=GPU, timeout=14400, volumes={"/vol": vol, "/hfcache": hf_cache})
def one_dataset(dataset: str, arm: str = "sparked"):
    """All methods on one dataset, one tree-builder arm, one container."""
    import os

    settings = {"sparked": cfg.SPARKED, "beam_graphed": cfg.BEAM_GRAPHED,
                "dspark_paper": cfg.DSPARK_PAPER}[arm]
    checkpoint = settings.get("checkpoint", cfg.CHECKPOINT)
    if checkpoint.startswith("/"):
        assert os.path.exists(f"{checkpoint}/config.json"), f"{checkpoint} missing"
    save_path = f"/root/final_{dataset}_{arm}.pt"

    cmd = [
        "python", "benchmark.py",
        "--model-name-or-path", cfg.TARGET,
        "--draft-name-or-path", cfg.DFLASH_DRAFT,
        "--dspark-name-or-path", checkpoint,
        "--dataset", dataset,
        "--max-samples", str(cfg.MAX_SAMPLES),
        "--max-new-tokens", str(cfg.MAX_NEW_TOKENS),
        "--tree-budget", cfg.TREE_BUDGETS,
        "--tree-mode", settings["tree_mode"],
        "--beam-widths", "flat",
        "--beam-candidates", str(settings["beam_candidates"]),
        "--beam-min-width", str(settings["beam_min_width"]),
        "--beam-max-fanout", str(settings.get("max_fanout", 0)),
        "--minimal",
        "--temperature", "0.0",
        "--disable-cpp-compact-cache",
        "--save-path", save_path,
    ]
    if settings.get("cuda_graph"):
        cmd.append("--beam-cuda-graph")
    _run(cmd)
    return _aggregate(save_path)


@app.function(image=image, timeout=28800, volumes={"/vol": vol, "/hfcache": hf_cache})
def final(arms: str = "sparked,beam_graphed"):
    """Fan out one container per (dataset, arm); gather into a single report JSON."""
    import json
    import time

    arm_list = [a.strip() for a in arms.split(",")]
    calls = {
        (ds, arm): one_dataset.spawn(dataset=ds, arm=arm)
        for ds in cfg.DATASETS for arm in arm_list
    }
    print(f"spawned {len(calls)} containers: {len(cfg.DATASETS)} datasets x {len(arm_list)} arms",
          flush=True)

    results = {}
    for (ds, arm), call in calls.items():
        try:
            results.setdefault(arm, {})[ds] = call.get()
            print(f"[done] {ds} / {arm}", flush=True)
        except Exception as exc:
            print(f"!! {ds} / {arm} FAILED: {exc}", flush=True)

    payload = {
        "gpu": GPU,
        "checkpoint": cfg.CHECKPOINT,
        "datasets": cfg.DATASETS,
        "tree_budgets": cfg.TREE_BUDGETS,
        "max_samples": cfg.MAX_SAMPLES,
        "max_new_tokens": cfg.MAX_NEW_TOKENS,
        "arms": {a: {"sparked": cfg.SPARKED, "beam_graphed": cfg.BEAM_GRAPHED,
                     "dspark_paper": cfg.DSPARK_PAPER}[a] for a in arm_list},
        "results": results,
    }
    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/FINAL_{GPU}_{int(time.time())}.json"
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2)
    vol.commit()
    print(f"\nsaved {out}\nFINAL BENCHMARK COMPLETE", flush=True)
    return payload
