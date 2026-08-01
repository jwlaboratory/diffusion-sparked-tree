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

import os
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

# GPU tier for the large-memory pipeline. Modal validates EVERY function in the
# app at deploy time, so hardcoding H100 makes the whole file undeployable on a
# workspace without H100 access - even for CPU-only functions like publish().
BIG_GPU = os.environ.get("DDTREE_BIG_GPU", "H100")

NUM_SAMPLES = 2400          # PerfectBlend rows before the 5% eval split
CACHE_DIR = "/vol/target_cache_block16"
CKPT_FINAL = "/vol/ckpt_final/dspark_block16"


def _latest_checkpoint(exp_suffix: str = "") -> str:
    """Newest checkpoint dir. The trainer writes both step_<int> and a step_latest
    alias, so numeric sorting must skip the alias (and prefer it when present)."""
    import glob
    import os

    base = f"/vol/checkpoints/deepspec/dspark_block16_qwen3_4b_warmstart{exp_suffix}"
    alias = os.path.join(base, "step_latest")
    if os.path.exists(os.path.join(alias, "config.json")):
        return alias
    numbered = sorted(
        (p for p in glob.glob(f"{base}/step_*") if p.rsplit("_", 1)[-1].isdigit()),
        key=lambda p: int(p.rsplit("_", 1)[-1]),
    )
    assert numbered, f"no checkpoints under {base}"
    return numbered[-1]


@app.function(image=image, gpu=None, timeout=1800, volumes={"/vol": vol, "/hfcache": hf_cache})
def publish(exp_suffix: str = ""):
    """Stage 4 only: copy the newest trained checkpoint to a from_pretrained path."""
    import os
    import shutil

    latest = _latest_checkpoint(exp_suffix)
    dest = f"{CKPT_FINAL}{exp_suffix}"
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(latest, dest)
    vol.commit()
    print(f"published {latest} -> {dest}", flush=True)
    print("files:", sorted(os.listdir(dest)), flush=True)
    return dest


def _sh(cmd, cwd="/root/DeepSpec", env=None):
    import os
    import subprocess

    full_env = dict(
        os.environ,
        HF_HOME="/hfcache",
        TOKENIZERS_PARALLELISM="false",
        PYTHONPATH="/root/DeepSpec",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    if env:
        full_env.update(env)
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=full_env, check=True)


def _stage_files(gamma: float = 4.0, exp_suffix: str = "", overrides: dict | None = None):
    """Copy our config + warm-start entry next to DeepSpec's train.py.

    gamma overrides loss_decay_gamma: exp(-p/gamma) position weighting. The stock
    4.0 was tuned for block 7; at block 16 it gives the 8 new deep positions only
    ~12% of total gradient weight (gamma=8 -> ~27%).
    """
    import shutil

    src = open("/root/training/dspark_block16_qwen3_4b.py").read()
    src = src.replace("loss_decay_gamma=4.0", f"loss_decay_gamma={float(gamma)}")
    for key, value in (overrides or {}).items():
        import re as _re
        src = _re.sub(rf"^(\s*{key}=)[^,]+,", lambda m: f"{m.group(1)}{value},", src, count=1, flags=_re.M)
    if exp_suffix:
        src = src.replace(
            'exp_name = "dspark_block16_qwen3_4b_warmstart"',
            f'exp_name = "dspark_block16_qwen3_4b_warmstart{exp_suffix}"',
        )
    open("/root/DeepSpec/config_block16.py", "w").write(src)
    shutil.copy("/root/training/train_warmstart.py", "/root/DeepSpec/train_warmstart.py")


@app.function(
    image=image,
    gpu="A10G",
    timeout=43200,
    volumes={"/vol": vol, "/hfcache": hf_cache},
)
def pipeline(gamma: float = 4.0, exp_suffix: str = ""):
    import os

    _stage_files(gamma=gamma, exp_suffix=exp_suffix)

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

    latest = _latest_checkpoint(exp_suffix)
    dest = f"{CKPT_FINAL}{exp_suffix}"
    print(f"publishing {latest} -> {dest}", flush=True)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(latest, dest)
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
    exp_suffix: str = "",
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

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    assert os.path.exists(f"{ckpt}/config.json"), "run ::pipeline first"
    all_results = {}
    for dataset in datasets.split(","):
        dataset = dataset.strip()
        save_path = f"/root/out_{dataset}.pt"
        _sh([
            "python", "benchmark.py",
            "--model-name-or-path", "Qwen/Qwen3-4B",
            "--draft-name-or-path", "z-lab/Qwen3-4B-DFlash-b16",
            "--dspark-name-or-path", ckpt,
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
    out = f"/vol/results/block16{exp_suffix}_{int(time.time())}.json"
    json.dump(all_results, open(out, "w"), indent=2)
    vol.commit()
    print(f"saved {out}\nBENCH16 COMPLETE", flush=True)
    return all_results


@app.function(
    image=image,
    gpu=BIG_GPU,
    timeout=43200,
    volumes={"/vol": vol, "/hfcache": hf_cache},
)
def pipeline_h100(
    gamma: float = 8.0,
    exp_suffix: str = "_h100",
    num_anchors: int = 128,
    max_length: int = 1024,
    max_train_steps: int = 1000,
    global_batch_size: int = 64,
):
    """Same pipeline on an H100 80GB.

    The A10G run had to cut num_anchors 256 -> 32 because the DSpark loss
    materializes float32 [anchors, block, vocab] probability tensors (2.5GB each
    at 256 anchors x block 16). 80GB lets us restore a real anchor count, which is
    the main lever on gradient signal per sample.
    """
    import os

    _stage_files(
        gamma=gamma,
        exp_suffix=exp_suffix,
        overrides={
            "num_anchors": num_anchors,
            "max_length": max_length,
            "max_train_steps": max_train_steps,
            "global_batch_size": global_batch_size,
        },
    )
    print(open("/root/DeepSpec/config_block16.py").read(), flush=True)

    os.makedirs("/vol/checkpoints", exist_ok=True)
    os.makedirs("/vol/tensorboard", exist_ok=True)
    for link, target in (("/root/checkpoints", "/vol/checkpoints"), ("/root/tensorboard", "/vol/tensorboard")):
        if not os.path.exists(link):
            os.symlink(target, link)

    if not os.path.exists("/vol/data/train.jsonl"):
        _sh(["python", "scripts/data/download_and_split.py",
             "--sample-size", str(NUM_SAMPLES),
             "--train-output-path", "/vol/data/train.jsonl",
             "--test-output-dir", "/vol/data", "--skip-existing"])
        vol.commit()

    done_marker = f"{CACHE_DIR}/.done"
    if not os.path.exists(done_marker):
        _sh(["python", "scripts/data/prepare_target_cache.py",
             "--config", "config_block16.py",
             "--train-data-path", "/vol/data/train.jsonl",
             "--output-dir", CACHE_DIR, "--local-batch-size", "16"])
        open(done_marker, "w").write("ok")
        vol.commit()

    _sh(["python", "train_warmstart.py", "--config", "config_block16.py"])
    vol.commit()

    import glob
    import shutil
    step_dirs = [_latest_checkpoint(exp_suffix)]
    dest = f"{CKPT_FINAL}{exp_suffix}"
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(step_dirs[-1], dest)
    vol.commit()
    print(f"published {step_dirs[-1]} -> {dest}", flush=True)
    print("PIPELINE COMPLETE", flush=True)
    return step_dirs[-1]


@app.function(image=image, gpu=BIG_GPU, timeout=10800, volumes={"/vol": vol, "/hfcache": hf_cache})
def bench16_h100(
    exp_suffix: str = "_h100",
    datasets: str = "humaneval,gsm8k,alpaca",
    max_samples: int = 8,
    max_new_tokens: int = 512,
    tree_budget: str = "16,64",
):
    return bench16.local(
        exp_suffix=exp_suffix, datasets=datasets, max_samples=max_samples,
        max_new_tokens=max_new_tokens, tree_budget=tree_budget,
    )


@app.function(image=image, gpu="A10G", timeout=10800, volumes={"/vol": vol, "/hfcache": hf_cache})
def bench_builder(
    exp_suffix: str = "",
    dataset: str = "gsm8k",
    max_samples: int = 4,
    max_new_tokens: int = 384,
    tree_budget: str = "64",
    candidate_sweep: str = "0,4096,2048,512",
):
    """Isolate tree-builder cost: same model/data/budget, vary candidate restriction.

    candidates=0 is the full-vocab path (each table reads all of markov_w2, ~78MB);
    smaller values gather only the union of per-depth top-C rows. Reports the
    tree_build stage time so the net wall-clock effect is attributable.
    """
    import json
    import os
    import time

    import torch

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    assert os.path.exists(f"{ckpt}/config.json"), "run ::pipeline / ::publish first"

    rows = {}
    for cand in [int(c) for c in candidate_sweep.split(",")]:
        save_path = f"/root/out_c{cand}.pt"
        _sh([
            "python", "benchmark.py",
            "--model-name-or-path", "Qwen/Qwen3-4B",
            "--draft-name-or-path", "z-lab/Qwen3-4B-DFlash-b16",
            "--dspark-name-or-path", ckpt,
            "--dataset", dataset,
            "--max-samples", str(max_samples),
            "--max-new-tokens", str(max_new_tokens),
            "--tree-budget", tree_budget,
            "--tree-candidates", str(cand),
            "--temperature", "0.0",
            "--disable-cpp-compact-cache",
            "--save-path", save_path,
        ], cwd="/root/ddtree")

        data = torch.load(save_path, weights_only=False)
        agg = {}
        for response in data["responses"]:
            for method, result in response.items():
                r = agg.setdefault(method, {"accept": [], "tpot": [], "stage": {}, "tokens": 0})
                r["accept"].extend(result.acceptance_lengths)
                r["tpot"].append(result.time_per_output_token)
                r["tokens"] += result.num_output_tokens
                for k, v in getattr(result, "stage_times", {}).items():
                    r["stage"][k] = r["stage"].get(k, 0.0) + v
        for method, r in agg.items():
            r["mean_accept"] = sum(r["accept"]) / max(len(r["accept"]), 1)
            r["mean_tpot_ms"] = 1e3 * sum(r["tpot"]) / max(len(r["tpot"]), 1)
            del r["accept"], r["tpot"]
        rows[cand] = agg

        m = agg.get("dsparktree_markov_tb64") or agg.get("dsparktree_markov_tb16") or {}
        print(f"[candidates={cand:5d}] accept={m.get('mean_accept', 0):.3f} "
              f"tpot={m.get('mean_tpot_ms', 0):.2f}ms tree_build={m.get('stage', {}).get('tree_build', 0):.2f}s",
              flush=True)

    base = next(iter(rows.values())).get("baseline", {}).get("mean_tpot_ms")
    print(f"\n{'cand':>6s} {'method':24s} {'accept':>7s} {'tpot_ms':>8s} {'speedup':>8s} {'tree_build_s':>12s} {'verify_s':>9s}")
    for cand, agg in rows.items():
        for method, r in agg.items():
            if "markov" not in method and method != "ddtree_tb64":
                continue
            sp = f"{base / r['mean_tpot_ms']:.2f}x" if base else "-"
            print(f"{cand:6d} {method:24s} {r['mean_accept']:7.3f} {r['mean_tpot_ms']:8.2f} {sp:>8s}"
                  f" {r['stage'].get('tree_build', 0):12.2f} {r['stage'].get('verify', 0):9.2f}")

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/builder_{dataset}_{int(time.time())}.json"
    json.dump({str(k): v for k, v in rows.items()}, open(out, "w"), indent=2)
    vol.commit()
    print(f"saved {out}\nBUILDER SWEEP COMPLETE", flush=True)
    return rows


@app.function(image=image, gpu="A10G", timeout=10800, volumes={"/vol": vol, "/hfcache": hf_cache})
def bench_beam(
    exp_suffix: str = "",
    dataset: str = "gsm8k",
    max_samples: int = 4,
    max_new_tokens: int = 384,
    tree_budget: str = "64",
    configs: str = "exact:0.6,beam:0.6,beam:0.5,beam:0.75",
):
    """Best-first vs level-synchronous beam builder: acceptance AND tree_build cost.

    The beam builder fixes tree shape up front, so each depth expands in one batched
    call and surviving tokens stay on GPU - 1 sync per round instead of ~budget.
    Question is whether the fixed schedule costs acceptance.
    """
    import json
    import os
    import time

    import torch

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    assert os.path.exists(f"{ckpt}/config.json"), "run ::publish first"

    rows = {}
    for cfg in configs.split(","):
        mode, decay = cfg.split(":")
        tag = f"{mode}_d{decay}"
        save_path = f"/root/out_{tag}.pt"
        _sh([
            "python", "benchmark.py",
            "--model-name-or-path", "Qwen/Qwen3-4B",
            "--draft-name-or-path", "z-lab/Qwen3-4B-DFlash-b16",
            "--dspark-name-or-path", ckpt,
            "--dataset", dataset,
            "--max-samples", str(max_samples),
            "--max-new-tokens", str(max_new_tokens),
            "--tree-budget", tree_budget,
            "--tree-mode", mode,
            "--beam-decay", decay,
            "--temperature", "0.0",
            "--disable-cpp-compact-cache",
            "--save-path", save_path,
        ], cwd="/root/ddtree")

        data = torch.load(save_path, weights_only=False)
        agg = {}
        for response in data["responses"]:
            for method, result in response.items():
                r = agg.setdefault(method, {"accept": [], "tpot": [], "stage": {}})
                r["accept"].extend(result.acceptance_lengths)
                r["tpot"].append(result.time_per_output_token)
                for k, v in getattr(result, "stage_times", {}).items():
                    r["stage"][k] = r["stage"].get(k, 0.0) + v
        for method, r in agg.items():
            r["mean_accept"] = sum(r["accept"]) / max(len(r["accept"]), 1)
            r["mean_tpot_ms"] = 1e3 * sum(r["tpot"]) / max(len(r["tpot"]), 1)
            del r["accept"], r["tpot"]
        rows[tag] = agg
        m = agg.get("dsparktree_markov_tb64", {})
        print(f"[{tag:12s}] accept={m.get('mean_accept',0):.3f} tpot={m.get('mean_tpot_ms',0):.2f}ms "
              f"tree_build={m.get('stage',{}).get('tree_build',0):.2f}s", flush=True)

    base = next(iter(rows.values())).get("baseline", {}).get("mean_tpot_ms")
    print(f"\n{'config':14s} {'method':24s} {'accept':>7s} {'tpot_ms':>8s} {'speedup':>8s} {'build_s':>8s} {'verify_s':>9s}")
    for tag, agg in rows.items():
        for method, r in agg.items():
            if "markov" not in method and method != "ddtree_tb64":
                continue
            sp = f"{base / r['mean_tpot_ms']:.2f}x" if base else "-"
            print(f"{tag:14s} {method:24s} {r['mean_accept']:7.3f} {r['mean_tpot_ms']:8.2f} {sp:>8s}"
                  f" {r['stage'].get('tree_build',0):8.2f} {r['stage'].get('verify',0):9.2f}")

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/beam_{dataset}_{int(time.time())}.json"
    json.dump(rows, open(out, "w"), indent=2)
    vol.commit()
    print(f"saved {out}\nBEAM SWEEP COMPLETE", flush=True)
    return rows


@app.function(image=image, gpu="A10G", timeout=14400, volumes={"/vol": vol, "/hfcache": hf_cache})
def collect_tree_stats(
    exp_suffix: str = "",
    datasets: str = "humaneval,mbpp,gsm8k,math500,mt-bench,alpaca",
    max_samples: int = 6,
    max_new_tokens: int = 384,
    tree_budget: int = 256,
):
    """Record (depth, slot) of every accepted node across real workloads.

    Run with a WIDE budget so the measurement is not truncated by the very schedule
    we are trying to tune - with 256 nodes the tree is broad enough that the
    accepted token is nearly always present, so the observed slot indices reflect
    the drafter's actual ranking rather than the tree's cutoff.
    """
    import json
    import os
    import time

    import torch

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    assert os.path.exists(f"{ckpt}/config.json"), "run ::publish first"

    out_rows = {}
    for dataset in [d.strip() for d in datasets.split(",")]:
        save_path = f"/root/stats_{dataset}.pt"
        _sh([
            "python", "benchmark.py",
            "--model-name-or-path", "Qwen/Qwen3-4B",
            "--draft-name-or-path", "z-lab/Qwen3-4B-DFlash-b16",
            "--dspark-name-or-path", ckpt,
            "--dataset", dataset,
            "--max-samples", str(max_samples),
            "--max-new-tokens", str(max_new_tokens),
            "--tree-budget", str(tree_budget),
            "--tree-mode", "exact",
            "--collect-tree-stats",
            "--temperature", "0.0",
            "--disable-cpp-compact-cache",
            "--save-path", save_path,
        ], cwd="/root/ddtree")

        data = torch.load(save_path, weights_only=False)
        slots, reached, accepts = [], [], []
        for response in data["responses"]:
            result = response.get(f"dsparktree_markov_tb{tree_budget}")
            if result is None:
                continue
            slots.extend(getattr(result, "accepted_slots", []))
            reached.extend(getattr(result, "depth_reached", []))
            accepts.extend(result.acceptance_lengths)
        out_rows[dataset] = {
            "accepted_slots": [list(x) for x in slots],
            "depth_reached": reached,
            "mean_accept": sum(accepts) / max(len(accepts), 1),
            "tree_budget": tree_budget,
        }
        print(f"[{dataset:12s}] {len(slots)} accepted nodes, mean_accept={out_rows[dataset]['mean_accept']:.3f}", flush=True)

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/tree_slots_{int(time.time())}.json"
    json.dump(out_rows, open(out, "w"), indent=2)
    vol.commit()
    print(f"saved {out}\nSTATS COMPLETE", flush=True)
    return out


@app.function(image=image, gpu="A10G", timeout=14400, volumes={"/vol": vol, "/hfcache": hf_cache})
def bench_horizon(
    exp_suffix: str = "",
    datasets: str = "gsm8k,humaneval,alpaca",
    max_samples: int = 6,
    max_new_tokens: int = 384,
    tree_budget: str = "64",
    tree_mode: str = "exact",
):
    """Block-7 (released) vs block-16 (ours) on IDENTICAL prompts.

    benchmark.py picks prompts with dataset.shuffle(seed=0).select(range(n)), so
    two runs agree only if max_samples matches - previous block-7/block-16 numbers
    came from runs with different n and are not comparable. ddtree_tb64 is the
    control here: it never touches the DSpark model, so it must come out identical
    for the comparison to be valid.
    """
    import json
    import os
    import time

    import torch

    trained = f"{CKPT_FINAL}{exp_suffix}"
    assert os.path.exists(f"{trained}/config.json"), "run ::publish first"
    arms = {"block7_released": "deepseek-ai/dspark_qwen3_4b_block7", "block16_ours": trained}

    rows = {}
    for dataset in [d.strip() for d in datasets.split(",")]:
        rows[dataset] = {}
        for arm, ckpt in arms.items():
            save_path = f"/root/hz_{dataset}_{arm}.pt"
            _sh([
                "python", "benchmark.py",
                "--model-name-or-path", "Qwen/Qwen3-4B",
                "--draft-name-or-path", "z-lab/Qwen3-4B-DFlash-b16",
                "--dspark-name-or-path", ckpt,
                "--dataset", dataset,
                "--max-samples", str(max_samples),
                "--max-new-tokens", str(max_new_tokens),
                "--tree-budget", tree_budget,
                "--tree-mode", tree_mode,
                "--temperature", "0.0",
                "--disable-cpp-compact-cache",
                "--save-path", save_path,
            ], cwd="/root/ddtree")

            data = torch.load(save_path, weights_only=False)
            agg = {}
            for response in data["responses"]:
                for method, result in response.items():
                    r = agg.setdefault(method, {"accept": [], "tpot": []})
                    r["accept"].extend(result.acceptance_lengths)
                    r["tpot"].append(result.time_per_output_token)
            for method, r in agg.items():
                r["mean_accept"] = sum(r["accept"]) / max(len(r["accept"]), 1)
                r["mean_tpot_ms"] = 1e3 * sum(r["tpot"]) / max(len(r["tpot"]), 1)
                del r["accept"], r["tpot"]
            rows[dataset][arm] = agg
            print(f"[{dataset:10s} {arm:16s}] block={data['block_size']} "
                  f"dspark={agg['dspark']['mean_accept']:.3f} "
                  f"tree={agg.get('dsparktree_markov_tb64', {}).get('mean_accept', 0):.3f}", flush=True)

    print(f"\n{'dataset':10s} {'method':24s} {'block7':>8s} {'block16':>8s} {'delta':>8s}")
    for dataset, arms_data in rows.items():
        b7, b16 = arms_data["block7_released"], arms_data["block16_ours"]
        for method in ("dspark", "dsparktree_markov_tb64", "ddtree_tb64", "dflash"):
            if method not in b7 or method not in b16:
                continue
            a7, a16 = b7[method]["mean_accept"], b16[method]["mean_accept"]
            tag = "  <-- control" if method in ("ddtree_tb64", "dflash") else ""
            print(f"{dataset:10s} {method:24s} {a7:8.3f} {a16:8.3f} {100*(a16/a7-1):+7.1f}%{tag}")
        print()

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/horizon_{int(time.time())}.json"
    json.dump(rows, open(out, "w"), indent=2)
    vol.commit()
    print(f"saved {out}\nHORIZON COMPARE COMPLETE", flush=True)
    return rows


@app.function(image=image, gpu="A10G", timeout=10800, volumes={"/vol": vol, "/hfcache": hf_cache})
def bench_schedule(
    exp_suffix: str = "",
    dataset: str = "gsm8k",
    max_samples: int = 4,
    max_new_tokens: int = 384,
    tree_budget: str = "64",
    schedules: str = "exact:|beam:2,3,7,7,5,6,6,4,4,3,3,3,2,3,4,2|beam:4,4,5,5,5,5,4,4,4,4,4,4,4,4,2,2|beam-decay:0.75",
):
    """Measured width schedule vs geometric decay vs best-first.

    Measured schedule comes from experiments/analyze_tree_slots.py over real
    workloads: the drafter is confident near the root (95% coverage in 3 slots at
    depth 1) and uncertain deep (42 slots at depth 16), so budget belongs deeper
    than any decaying curve puts it.
    """
    import json
    import os
    import time

    import torch

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    assert os.path.exists(f"{ckpt}/config.json"), "run ::publish first"

    rows = {}
    for spec in schedules.split("|"):
        kind, _, value = spec.partition(":")
        extra, tag = [], kind
        if kind == "exact":
            extra = ["--tree-mode", "exact"]
        elif kind == "beam":
            extra = ["--tree-mode", "beam", "--beam-widths", value]
            tag = f"beam[{value.split(',')[0]},{value.split(',')[1]}..]"
        elif kind == "beam-decay":
            extra = ["--tree-mode", "beam", "--beam-decay", value]
            tag = f"beam_decay{value}"

        save_path = f"/root/sched_{abs(hash(spec)) % 10**8}.pt"
        _sh([
            "python", "benchmark.py",
            "--model-name-or-path", "Qwen/Qwen3-4B",
            "--draft-name-or-path", "z-lab/Qwen3-4B-DFlash-b16",
            "--dspark-name-or-path", ckpt,
            "--dataset", dataset,
            "--max-samples", str(max_samples),
            "--max-new-tokens", str(max_new_tokens),
            "--tree-budget", tree_budget,
            "--temperature", "0.0",
            "--disable-cpp-compact-cache",
            "--save-path", save_path,
        ] + extra, cwd="/root/ddtree")

        data = torch.load(save_path, weights_only=False)
        agg = {}
        for response in data["responses"]:
            for method, result in response.items():
                r = agg.setdefault(method, {"accept": [], "tpot": [], "stage": {}})
                r["accept"].extend(result.acceptance_lengths)
                r["tpot"].append(result.time_per_output_token)
                for k, v in getattr(result, "stage_times", {}).items():
                    r["stage"][k] = r["stage"].get(k, 0.0) + v
        for method, r in agg.items():
            r["mean_accept"] = sum(r["accept"]) / max(len(r["accept"]), 1)
            r["mean_tpot_ms"] = 1e3 * sum(r["tpot"]) / max(len(r["tpot"]), 1)
            del r["accept"], r["tpot"]
        rows[tag] = agg
        m = agg.get("dsparktree_markov_tb64", {})
        print(f"[{tag:26s}] accept={m.get('mean_accept',0):.3f} tpot={m.get('mean_tpot_ms',0):.2f}ms "
              f"build={m.get('stage',{}).get('tree_build',0):.2f}s", flush=True)

    base = next(iter(rows.values()))["baseline"]["mean_tpot_ms"]
    print(f"\n{'schedule':26s} {'method':24s} {'accept':>7s} {'tpot_ms':>8s} {'speedup':>8s} {'build_s':>8s}")
    for tag, agg in rows.items():
        for method in ("dsparktree_markov_tb64", "ddtree_tb64", "dspark"):
            r = agg.get(method)
            if not r:
                continue
            print(f"{tag:26s} {method:24s} {r['mean_accept']:7.3f} {r['mean_tpot_ms']:8.2f} "
                  f"{base/r['mean_tpot_ms']:7.2f}x {r['stage'].get('tree_build',0):8.2f}")
        print()

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/schedule_{dataset}_{int(time.time())}.json"
    json.dump(rows, open(out, "w"), indent=2)
    vol.commit()
    print(f"saved {out}\nSCHEDULE SWEEP COMPLETE", flush=True)
    return rows


FINAL_WIDTHS = "4,4,5,5,5,5,4,4,4,4,4,4,4,4,2,2"   # flat schedule, sums to 64


@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/vol": vol, "/hfcache": hf_cache})
def final_one(dataset: str, exp_suffix: str = "", max_samples: int = 12,
              max_new_tokens: int = 512, tree_budget: str = "64"):
    """Final configuration on one dataset: block-16 drafter + flat-schedule beam tree."""
    import os
    import torch

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    assert os.path.exists(f"{ckpt}/config.json")
    save_path = f"/root/final_{dataset}.pt"
    _sh([
        "python", "benchmark.py",
        "--model-name-or-path", "Qwen/Qwen3-4B",
        "--draft-name-or-path", "z-lab/Qwen3-4B-DFlash-b16",
        "--dspark-name-or-path", ckpt,
        "--dataset", dataset,
        "--max-samples", str(max_samples),
        "--max-new-tokens", str(max_new_tokens),
        "--tree-budget", tree_budget,
        "--tree-mode", "beam",
        "--beam-widths", FINAL_WIDTHS,
        "--temperature", "0.0",
        "--disable-cpp-compact-cache",
        "--save-path", save_path,
    ], cwd="/root/ddtree")

    data = torch.load(save_path, weights_only=False)
    agg = {}
    for response in data["responses"]:
        for method, result in response.items():
            r = agg.setdefault(method, {"accept": [], "tpot": [], "stage": {}, "tokens": 0})
            r["accept"].extend(result.acceptance_lengths)
            r["tpot"].append(result.time_per_output_token)
            r["tokens"] += result.num_output_tokens
            for k, v in getattr(result, "stage_times", {}).items():
                r["stage"][k] = r["stage"].get(k, 0.0) + v
    for method, r in agg.items():
        r["mean_accept"] = sum(r["accept"]) / max(len(r["accept"]), 1)
        r["mean_tpot_ms"] = 1e3 * sum(r["tpot"]) / max(len(r["tpot"]), 1)
        del r["accept"], r["tpot"]
    return agg


@app.function(image=image, timeout=14400, volumes={"/vol": vol, "/hfcache": hf_cache})
def final_wide(datasets: str = "humaneval,mbpp,gsm8k,math500,mt-bench,alpaca",
               exp_suffix: str = "", max_samples: int = 12, max_new_tokens: int = 512):
    """Validate the final configuration across all benchmarks, one container each."""
    import json
    import os
    import time

    names = [d.strip() for d in datasets.split(",")]
    calls = [final_one.spawn(dataset=n, exp_suffix=exp_suffix, max_samples=max_samples,
                             max_new_tokens=max_new_tokens) for n in names]
    print(f"spawned {len(calls)} containers: {names}", flush=True)

    results = {}
    for name, call in zip(names, calls):
        try:
            results[name] = call.get()
        except Exception as exc:
            print(f"!! {name} FAILED: {exc}", flush=True)

    KEY = ["baseline", "dflash", "ddtree_tb64", "dspark", "dsparktree_markov_tb64"]
    print(f"\n{'dataset':11s} {'method':24s} {'accept':>7s} {'tpot_ms':>8s} {'speedup':>8s}")
    totals = {}
    for name, agg in results.items():
        base = agg.get("baseline", {}).get("mean_tpot_ms")
        for m in KEY:
            r = agg.get(m)
            if not r:
                continue
            sp = base / r["mean_tpot_ms"] if base else 0
            totals.setdefault(m, {"a": [], "s": []})
            totals[m]["a"].append(r["mean_accept"])
            totals[m]["s"].append(sp)
            print(f"{name:11s} {m:24s} {r['mean_accept']:7.3f} {r['mean_tpot_ms']:8.2f} {sp:7.2f}x")
        print()

    print(f"{'MEAN across datasets':36s} {'accept':>7s} {'speedup':>8s}")
    for m, v in totals.items():
        print(f"{m:36s} {sum(v['a'])/len(v['a']):7.3f} {sum(v['s'])/len(v['s']):7.2f}x")

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/final_{int(time.time())}.json"
    json.dump(results, open(out, "w"), indent=2)
    vol.commit()
    print(f"\nsaved {out}\nFINAL VALIDATION COMPLETE", flush=True)
    return results


@app.function(image=image, gpu="A10G", timeout=10800, volumes={"/vol": vol, "/hfcache": hf_cache})
def sweep_one(dataset: str, exp_suffix: str = "", max_samples: int = 8,
              max_new_tokens: int = 512, tree_budget: str = "16,32,64,128,256"):
    """DSpark vs DDTree vs sparked-tree across tree budgets, on one dataset.

    Tree budget is the verify batch: how many draft nodes the target scores in a
    single forward pass. Flat beam schedule scales with budget (budget/depth per
    level), per the measured hit data.
    """
    import os
    import torch

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    assert os.path.exists(f"{ckpt}/config.json")
    save_path = f"/root/sweep_{dataset}.pt"
    _sh([
        "python", "benchmark.py",
        "--model-name-or-path", "Qwen/Qwen3-4B",
        "--draft-name-or-path", "z-lab/Qwen3-4B-DFlash-b16",
        "--dspark-name-or-path", ckpt,
        "--dataset", dataset,
        "--max-samples", str(max_samples),
        "--max-new-tokens", str(max_new_tokens),
        "--tree-budget", tree_budget,
        "--tree-mode", "beam",
        "--beam-widths", "flat",
        "--minimal",
        "--temperature", "0.0",
        "--disable-cpp-compact-cache",
        "--save-path", save_path,
    ], cwd="/root/ddtree")

    data = torch.load(save_path, weights_only=False)
    agg = {}
    for response in data["responses"]:
        for method, result in response.items():
            r = agg.setdefault(method, {"accept": [], "tpot": [], "stage": {}, "tokens": 0})
            r["accept"].extend(result.acceptance_lengths)
            r["tpot"].append(result.time_per_output_token)
            r["tokens"] += result.num_output_tokens
            for k, v in getattr(result, "stage_times", {}).items():
                r["stage"][k] = r["stage"].get(k, 0.0) + v
    for method, r in agg.items():
        r["mean_accept"] = sum(r["accept"]) / max(len(r["accept"]), 1)
        r["mean_tpot_ms"] = 1e3 * sum(r["tpot"]) / max(len(r["tpot"]), 1)
        del r["accept"], r["tpot"]
    return agg


@app.function(image=image, timeout=21600, volumes={"/vol": vol, "/hfcache": hf_cache})
def sweep_wide(datasets: str = "humaneval,mbpp,gsm8k,math500,mt-bench,alpaca",
               exp_suffix: str = "", max_samples: int = 8, max_new_tokens: int = 512,
               tree_budget: str = "16,32,64,128,256"):
    """Full comparison: one GPU container per dataset, all budgets in each."""
    import json
    import os
    import time

    names = [d.strip() for d in datasets.split(",")]
    budgets = [int(b) for b in tree_budget.split(",")]
    calls = [sweep_one.spawn(dataset=n, exp_suffix=exp_suffix, max_samples=max_samples,
                             max_new_tokens=max_new_tokens, tree_budget=tree_budget) for n in names]
    print(f"spawned {len(calls)} containers x {len(budgets)} budgets: {names}", flush=True)

    results = {}
    for name, call in zip(names, calls):
        try:
            results[name] = call.get()
            print(f"[done] {name}", flush=True)
        except Exception as exc:
            print(f"!! {name} FAILED: {exc}", flush=True)

    def spd(agg, method):
        base = agg.get("baseline", {}).get("mean_tpot_ms")
        r = agg.get(method)
        return (base / r["mean_tpot_ms"], r["mean_accept"]) if base and r else (None, None)

    print(f"\n{'budget':>7s} {'method':22s} {'accept':>7s} {'speedup':>8s}   (mean over datasets)")
    summary = {}
    for budget in budgets:
        for label, key in (("ddtree", f"ddtree_tb{budget}"), ("sparked-tree", f"dsparktree_markov_tb{budget}")):
            accs, sps = [], []
            for agg in results.values():
                sp, ac = spd(agg, key)
                if sp:
                    sps.append(sp)
                    accs.append(ac)
            if not sps:
                continue
            summary[f"{label}_tb{budget}"] = {"accept": sum(accs) / len(accs), "speedup": sum(sps) / len(sps)}
            print(f"{budget:7d} {label:22s} {summary[f'{label}_tb{budget}']['accept']:7.3f} "
                  f"{summary[f'{label}_tb{budget}']['speedup']:7.2f}x")
    for label, key in (("dspark (chain)", "dspark"), ("dflash (chain)", "dflash")):
        accs, sps = [], []
        for agg in results.values():
            sp, ac = spd(agg, key)
            if sp:
                sps.append(sp)
                accs.append(ac)
        if sps:
            summary[key] = {"accept": sum(accs) / len(accs), "speedup": sum(sps) / len(sps)}
            print(f"{'-':>7s} {label:22s} {summary[key]['accept']:7.3f} {summary[key]['speedup']:7.2f}x")

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/sweep_{int(time.time())}.json"
    json.dump({"per_dataset": results, "summary": summary}, open(out, "w"), indent=2)
    vol.commit()
    print(f"\nsaved {out}\nSWEEP COMPLETE", flush=True)
    return {"summary": summary, "per_dataset": results}
