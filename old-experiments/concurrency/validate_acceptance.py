"""Does the acceptance ratio the crossover prediction rests on actually hold here?

    modal run --detach concurrency/validate_acceptance.py::run

PREDICTION.md multiplies a measured cost ratio by an acceptance ratio taken from
RESULTS.md section 1:

    sparked_tb64 / dspark_chain  =  7.842 / 6.075  =  1.291

That number is the single largest unvalidated input to the crossover, and it is
spliced across two different setups. The cost side of the prediction was measured
on **sharegpt** with the **published block-7** drafter; the acceptance side comes
from **our block-16 checkpoint** on **task datasets**. Different checkpoint,
different workload, different harness.

So re-measure the ratio in the setting the cost was measured in: the published
block-7 drafter, on the two chat-shaped datasets this repo supports (alpaca,
mt-bench -- the closest available proxies for sharegpt). Absolute acceptance will
differ from RESULTS.md and that is expected and fine; the prediction consumes
only the ratio.

  * ratio holds  -> the crossover stands as reported.
  * ratio lower  -> trees are worse than PREDICTION.md says; crossover moves left.
  * ratio higher -> crossover moves right, and by how much is computable.

Deliberately reuses ddtree/benchmark.py unchanged rather than reimplementing
acceptance, so this measures the same quantity the paper numbers do.
"""

import os
from pathlib import Path

import modal

app = modal.App("sparked-acceptance-validate")
REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "datasets",
                 "safetensors", "loguru", "tqdm", "numpy")
    .add_local_dir(str(REPO / "ddtree"), remote_path="/root/ddtree")
)
vol = modal.Volume.from_name("ddtree-train", create_if_missing=True)
hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)

GPU = os.environ.get("ACCEPT_GPU", "H100")

TARGET = "Qwen/Qwen3-4B"
DFLASH_DRAFT = "z-lab/Qwen3-4B-DFlash-b16"
# The SAME drafter the concurrency cost sweep served, so the two halves of the
# prediction finally come from one checkpoint.
DSPARK_DRAFT = "deepseek-ai/dspark_qwen3_4b_block7"
DATASETS = ["alpaca", "mt-bench"]
TREE_BUDGETS = "64,128"
MAX_SAMPLES = 12
MAX_NEW_TOKENS = 512


@app.function(image=image, gpu=GPU, timeout=14400,
              volumes={"/vol": vol, "/hfcache": hf_cache})
def one_dataset(dataset: str) -> dict:
    import subprocess
    import torch

    save_path = f"/root/accept_{dataset}.pt"
    cmd = [
        "python", "benchmark.py",
        "--model-name-or-path", TARGET,
        "--draft-name-or-path", DFLASH_DRAFT,
        "--dspark-name-or-path", DSPARK_DRAFT,
        "--dataset", dataset,
        "--max-samples", str(MAX_SAMPLES),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--tree-budget", TREE_BUDGETS,
        "--tree-mode", "exact-precomputed",
        "--beam-candidates", "512",
        "--beam-min-width", "2",
        "--minimal",
        "--temperature", "0.0",
        "--disable-cpp-compact-cache",
        "--save-path", save_path,
    ]
    print("$ " + " ".join(cmd), flush=True)
    env = dict(os.environ, HF_HOME="/hfcache", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run(cmd, cwd="/root/ddtree", env=env, check=True)

    data = torch.load(save_path, weights_only=False)
    agg = {}
    for response in data["responses"]:
        for method, result in response.items():
            row = agg.setdefault(method, [])
            row.extend(result.acceptance_lengths)
    return {m: (sum(v) / len(v) if v else None) for m, v in agg.items()}


@app.function(image=image, timeout=28800, volumes={"/vol": vol})
def run() -> dict:
    import json
    import time

    calls = {ds: one_dataset.spawn(dataset=ds) for ds in DATASETS}
    results = {}
    for ds, call in calls.items():
        try:
            results[ds] = call.get()
            print(f"[done] {ds}", flush=True)
        except Exception as exc:
            print(f"!! {ds} FAILED: {exc}", flush=True)

    payload = {"gpu": GPU, "target": TARGET, "dspark_draft": DSPARK_DRAFT,
               "datasets": DATASETS, "results": results}
    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/ACCEPT_{GPU}_{int(time.time())}.json"
    with open(out, "w") as h:
        json.dump(payload, h, indent=2)
    vol.commit()
    print(f"saved {out}", flush=True)

    # The comparison this run exists for.
    print("\n=== acceptance ratio vs the 1.291 PREDICTION.md assumes ===", flush=True)
    for ds, agg in results.items():
        chain = agg.get("dspark")
        for budget, label in ((64, "sparked_tb64"), (128, "sparked_tb128")):
            tree = agg.get(f"dsparktree_markov_tb{budget}")
            if chain and tree:
                print(f"  {ds:9s} {label:14s} {tree:.3f} / {chain:.3f} = "
                      f"{tree / chain:.3f}", flush=True)
    return payload


@app.local_entrypoint()
def main():
    import json

    out = run.remote()
    print("\n=== measured acceptance (block-7 drafter, chat datasets) ===")
    for ds, agg in out["results"].items():
        print(f"\n{ds}:")
        for method in sorted(agg):
            if agg[method] is not None:
                print(f"  {method:28s} {agg[method]:.3f}")
    print("\nraw:", json.dumps(out["results"], indent=2)[:2000])
