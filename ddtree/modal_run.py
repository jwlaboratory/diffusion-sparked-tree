"""Run the repo's benchmark.py on a Modal GPU (cursory experiment runner).

Detached suite (survives local client disconnect; results in Modal logs + volume):

    modal run --detach ddtree/modal_run.py::suite

Single dataset:

    modal run ddtree/modal_run.py::bench --dataset humaneval --max-samples 2

Methods compared on Qwen3-4B:
    baseline            plain autoregressive (block_size=1)
    dflash              DFlash chain (z-lab/Qwen3-4B-DFlash-b16)
    ddtree_tb{B}        DDTree on the DFlash drafter
    dspark              DSpark chain (deepseek-ai/dspark_qwen3_4b_block7)
    dsparktree_nomkv    tree on DSpark drafter, independence-assumed (ablation)
    dsparktree_markov   tree on DSpark drafter, markov-guided branches (the experiment)

Fetch saved results later:
    modal volume get ddtree-hf-cache results/<file>.json .
"""

from pathlib import Path

import modal

app = modal.App("ddtree-markov-bench")

REPO_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "safetensors",
        "loguru",
        "tqdm",
        "numpy",
    )
    .add_local_dir(REPO_DIR, remote_path="/root/ddtree")
)

hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)

TARGET = "Qwen/Qwen3-4B"
DFLASH_DRAFT = "z-lab/Qwen3-4B-DFlash-b16"
DSPARK_DRAFT = "deepseek-ai/dspark_qwen3_4b_block7"


def _run_one(dataset: str, max_samples: int, max_new_tokens: int, tree_budget: str, temperature: float) -> dict:
    import os
    import subprocess

    import torch

    env = dict(os.environ, HF_HOME="/hfcache", TOKENIZERS_PARALLELISM="false")
    save_path = f"/root/out_{dataset}.pt"
    cmd = [
        "python", "benchmark.py",
        "--model-name-or-path", TARGET,
        "--draft-name-or-path", DFLASH_DRAFT,
        "--dspark-name-or-path", DSPARK_DRAFT,
        "--dataset", dataset,
        "--max-samples", str(max_samples),
        "--max-new-tokens", str(max_new_tokens),
        "--tree-budget", tree_budget,
        "--temperature", str(temperature),
        "--disable-cpp-compact-cache",
        "--save-path", save_path,
    ]
    subprocess.run(cmd, cwd="/root/ddtree", env=env, check=True)

    data = torch.load(save_path, weights_only=False)
    summary: dict[str, dict] = {}
    for response in data["responses"]:
        for method, result in response.items():
            row = summary.setdefault(
                method,
                {"accept": [], "tpot": [], "rounds": 0, "tokens": 0, "stage_times": {}},
            )
            row["accept"].extend(result.acceptance_lengths)
            row["tpot"].append(result.time_per_output_token)
            row["rounds"] += result.decode_rounds
            row["tokens"] += result.num_output_tokens
            for stage, elapsed in getattr(result, "stage_times", {}).items():
                row["stage_times"][stage] = row["stage_times"].get(stage, 0.0) + elapsed

    for row in summary.values():
        row["mean_accept"] = sum(row["accept"]) / max(len(row["accept"]), 1)
        row["mean_tpot_ms"] = 1e3 * sum(row["tpot"]) / max(len(row["tpot"]), 1)
        del row["accept"], row["tpot"]
    return summary


def _print_summary(dataset: str, summary: dict) -> None:
    baseline_tpot = summary.get("baseline", {}).get("mean_tpot_ms")
    print(f"\n===== {dataset} =====")
    print(f"{'method':28s} {'mean_accept':>11s} {'tpot_ms':>8s} {'speedup':>8s} {'tokens':>7s} {'rounds':>7s}")
    for method, row in summary.items():
        speedup = f"{baseline_tpot / row['mean_tpot_ms']:.2f}x" if baseline_tpot else "-"
        print(
            f"{method:28s} {row['mean_accept']:11.3f} {row['mean_tpot_ms']:8.2f} {speedup:>8s}"
            f" {row['tokens']:7d} {row['rounds']:7d}"
        )
    print("stage times (s, summed over prompts):")
    for method, row in summary.items():
        stages = {k: round(v, 2) for k, v in row["stage_times"].items() if v > 0.005}
        print(f"  {method:28s} {stages}")


@app.function(image=image, gpu="A10G", timeout=10800, volumes={"/hfcache": hf_cache})
def suite(
    datasets: str = "humaneval,gsm8k",
    max_samples: int = 2,
    max_new_tokens: int = 256,
    tree_budget: str = "16,64",
    temperature: float = 0.0,
) -> dict:
    import json
    import os
    import time

    all_results: dict[str, dict] = {}
    for dataset in datasets.split(","):
        dataset = dataset.strip()
        print(f"\n########## running {dataset} ##########", flush=True)
        summary = _run_one(dataset, max_samples, max_new_tokens, tree_budget, temperature)
        _print_summary(dataset, summary)
        all_results[dataset] = summary

        os.makedirs("/hfcache/results", exist_ok=True)
        out_path = f"/hfcache/results/suite_{int(time.time())}_{dataset}.json"
        with open(out_path, "w") as handle:
            json.dump({"dataset": dataset, "args": {
                "max_samples": max_samples, "max_new_tokens": max_new_tokens,
                "tree_budget": tree_budget, "temperature": temperature,
            }, "summary": summary}, handle, indent=2)
        hf_cache.commit()
        print(f"saved {out_path}", flush=True)

    print("\nSUITE COMPLETE", flush=True)
    return all_results


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/hfcache": hf_cache})
def bench(
    dataset: str = "humaneval",
    max_samples: int = 2,
    max_new_tokens: int = 256,
    tree_budget: str = "64",
    temperature: float = 0.0,
) -> dict:
    return _run_one(dataset, max_samples, max_new_tokens, tree_budget, temperature)


@app.function(image=image, timeout=14400, volumes={"/hfcache": hf_cache})
def wide(
    datasets: str = "humaneval,mbpp,gsm8k,math500,mt-bench,alpaca",
    max_samples: int = 16,
    max_new_tokens: int = 512,
    tree_budget: str = "16,64",
    temperature: float = 0.0,
) -> dict:
    """Fan out one GPU container per dataset in parallel, gather all tables."""
    import json
    import os
    import time

    names = [d.strip() for d in datasets.split(",")]
    calls = [
        bench.spawn(
            dataset=name,
            max_samples=max_samples,
            max_new_tokens=max_new_tokens,
            tree_budget=tree_budget,
            temperature=temperature,
        )
        for name in names
    ]
    print(f"spawned {len(calls)} parallel bench containers: {names}", flush=True)

    all_results = {}
    for name, call in zip(names, calls):
        try:
            summary = call.get()
        except Exception as exc:  # one dataset failing shouldn't sink the rest
            print(f"!! {name} FAILED: {exc}", flush=True)
            continue
        all_results[name] = summary
        _print_summary(name, summary)

    os.makedirs("/hfcache/results", exist_ok=True)
    out = f"/hfcache/results/wide_{int(time.time())}.json"
    json.dump(
        {
            "args": {
                "datasets": names,
                "max_samples": max_samples,
                "max_new_tokens": max_new_tokens,
                "tree_budget": tree_budget,
                "temperature": temperature,
            },
            "results": all_results,
        },
        open(out, "w"),
        indent=2,
    )
    hf_cache.commit()
    print(f"saved {out}\nWIDE COMPLETE", flush=True)
    return all_results


@app.local_entrypoint()
def main(
    dataset: str = "humaneval",
    max_samples: int = 2,
    max_new_tokens: int = 256,
    tree_budget: str = "16,64",
    temperature: float = 0.0,
):
    summary = bench.remote(
        dataset=dataset,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        tree_budget=tree_budget,
        temperature=temperature,
    )
    _print_summary(dataset, summary)
