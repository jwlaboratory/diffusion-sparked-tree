"""Detached Modal pipeline: fine-tune a block-16 DSpark drafter for Qwen3-4B,
warm-started from deepseek-ai/dspark_qwen3_4b_block7, then benchmark it.

    modal run --detach training/modal_train.py::pipeline          # data + cache + train
    modal run --detach training/modal_train.py::bench16           # after training

Everything persists in the `ddtree-train` volume:
    /vol/data/train.jsonl            PerfectBlend subset (chat conversations)
    /vol/target_cache_block16/       precomputed target hidden-state cache
    /vol/checkpoints/                trainer checkpoints (HF format per step)
    /vol/ckpt_final/dspark_block16/  latest checkpoint, ready for from_pretrained
"""

from pathlib import Path

import modal

app = modal.App("ddtree-train-block16")

REPO_ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "safetensors",
        "huggingface_hub",
        "tensorboard",
        "loguru",
        "tqdm",
        "numpy",
    )
    .run_commands("git clone --depth 1 https://github.com/deepseek-ai/DeepSpec /root/DeepSpec")
    .add_local_dir(REPO_ROOT / "training", remote_path="/root/training")
    .add_local_dir(REPO_ROOT / "ddtree", remote_path="/root/ddtree")
)

vol = modal.Volume.from_name("ddtree-train", create_if_missing=True)
hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)

NUM_SAMPLES = 2400          # PerfectBlend rows before the 5% eval split
CACHE_DIR = "/vol/target_cache_block16"
CKPT_FINAL = "/vol/ckpt_final/dspark_block16"


def _sh(cmd, cwd="/root/DeepSpec", env=None):
    import os
    import subprocess

    full_env = dict(os.environ, HF_HOME="/hfcache", TOKENIZERS_PARALLELISM="false", PYTHONPATH="/root/DeepSpec")
    if env:
        full_env.update(env)
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=full_env, check=True)


def _stage_files():
    """Copy our config + warm-start entry next to DeepSpec's train.py."""
    import shutil

    shutil.copy("/root/training/dspark_block16_qwen3_4b.py", "/root/DeepSpec/config_block16.py")
    shutil.copy("/root/training/train_warmstart.py", "/root/DeepSpec/train_warmstart.py")


@app.function(
    image=image,
    gpu="A10G",
    timeout=43200,
    volumes={"/vol": vol, "/hfcache": hf_cache},
)
def pipeline():
    import os

    _stage_files()

    # checkpoints/tensorboard land in ~/... inside the container; point them at the volume
    os.makedirs("/vol/checkpoints", exist_ok=True)
    os.makedirs("/vol/tensorboard", exist_ok=True)
    for link, target in (("/root/checkpoints", "/vol/checkpoints"), ("/root/tensorboard", "/vol/tensorboard")):
        if not os.path.exists(link):
            os.symlink(target, link)

    # ---- stage 1: chat data ----
    if not os.path.exists("/vol/data/train.jsonl"):
        _sh([
            "python", "scripts/data/download_and_split.py",
            "--sample-size", str(NUM_SAMPLES),
            "--train-output-path", "/vol/data/train.jsonl",
            "--test-output-dir", "/vol/data",
            "--skip-existing",
        ])
        vol.commit()
    else:
        print("stage 1: /vol/data/train.jsonl exists, skipping", flush=True)

    # ---- stage 2: target hidden-state cache (block-16 config: layer ids, seq len) ----
    done_marker = f"{CACHE_DIR}/.done"
    if not os.path.exists(done_marker):
        _sh([
            "python", "scripts/data/prepare_target_cache.py",
            "--config", "config_block16.py",
            "--train-data-path", "/vol/data/train.jsonl",
            "--output-dir", CACHE_DIR,
            "--local-batch-size", "8",
        ])
        open(done_marker, "w").write("ok")
        vol.commit()
    else:
        print("stage 2: target cache exists, skipping", flush=True)

    # ---- stage 3: warm-start fine-tune at block 16 ----
    _sh(["python", "train_warmstart.py", "--config", "config_block16.py"])
    vol.commit()

    # ---- stage 4: publish latest checkpoint for from_pretrained ----
    import glob
    import shutil

    step_dirs = sorted(
        glob.glob("/vol/checkpoints/deepspec/dspark_block16_qwen3_4b_warmstart/step_*"),
        key=lambda p: int(p.rsplit("_", 1)[-1]),
    )
    assert step_dirs, "no checkpoints written"
    latest = step_dirs[-1]
    print(f"publishing {latest} -> {CKPT_FINAL}", flush=True)
    if os.path.exists(CKPT_FINAL):
        shutil.rmtree(CKPT_FINAL)
    os.makedirs(os.path.dirname(CKPT_FINAL), exist_ok=True)
    shutil.copytree(latest, CKPT_FINAL)
    vol.commit()
    print("PIPELINE COMPLETE", flush=True)
    return latest


@app.function(
    image=image,
    gpu="A10G",
    timeout=10800,
    volumes={"/vol": vol, "/hfcache": hf_cache},
)
def bench16(
    datasets: str = "humaneval,gsm8k",
    max_samples: int = 2,
    max_new_tokens: int = 256,
    tree_budget: str = "16,64",
):
    """Benchmark the freshly trained block-16 drafter with the repo's benchmark.py."""
    import json
    import os
    import subprocess
    import time

    import torch

    assert os.path.exists(f"{CKPT_FINAL}/config.json"), "run ::pipeline first"
    all_results = {}
    for dataset in datasets.split(","):
        dataset = dataset.strip()
        save_path = f"/root/out_{dataset}.pt"
        _sh([
            "python", "benchmark.py",
            "--model-name-or-path", "Qwen/Qwen3-4B",
            "--draft-name-or-path", "z-lab/Qwen3-4B-DFlash-b16",
            "--dspark-name-or-path", CKPT_FINAL,
            "--dataset", dataset,
            "--max-samples", str(max_samples),
            "--max-new-tokens", str(max_new_tokens),
            "--tree-budget", tree_budget,
            "--temperature", "0.0",
            "--disable-cpp-compact-cache",
            "--save-path", save_path,
        ], cwd="/root/ddtree")

        data = torch.load(save_path, weights_only=False)
        summary = {}
        for response in data["responses"]:
            for method, result in response.items():
                row = summary.setdefault(method, {"accept": [], "tpot": [], "tokens": 0, "rounds": 0})
                row["accept"].extend(result.acceptance_lengths)
                row["tpot"].append(result.time_per_output_token)
                row["tokens"] += result.num_output_tokens
                row["rounds"] += result.decode_rounds
        for method, row in summary.items():
            row["mean_accept"] = sum(row["accept"]) / max(len(row["accept"]), 1)
            row["mean_tpot_ms"] = 1e3 * sum(row["tpot"]) / max(len(row["tpot"]), 1)
            del row["accept"], row["tpot"]
            print(f"{dataset:12s} {method:28s} accept={row['mean_accept']:.3f} tpot={row['mean_tpot_ms']:.2f}ms", flush=True)
        all_results[dataset] = summary

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/block16_{int(time.time())}.json"
    json.dump(all_results, open(out, "w"), indent=2)
    vol.commit()
    print(f"saved {out}\nBENCH16 COMPLETE", flush=True)
    return all_results
