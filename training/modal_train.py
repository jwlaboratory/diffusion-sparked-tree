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

# The benchmark GPU is separately selectable so the headline sweep can be run on
# the reference hardware (NEXT.md: H100) without moving the training pipeline.
SWEEP_GPU = os.environ.get("DDTREE_SWEEP_GPU", "A10G")

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
                row = summary.setdefault(method, {"accept": [], "tpot": [], "tokens": 0, "rounds": 0, "stage": {}})
                row["accept"].extend(result.acceptance_lengths)
                row["tpot"].append(result.time_per_output_token)
                row["tokens"] += result.num_output_tokens
                row["rounds"] += result.decode_rounds
                # kept for the stage-time table: which stage the win comes from
                for stage, elapsed in getattr(result, "stage_times", {}).items():
                    row["stage"][stage] = row["stage"].get(stage, 0.0) + elapsed
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
    num_samples: int = NUM_SAMPLES,
    cache_tag: str = "",
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
            "target_cache_path": f'"{CACHE_DIR}{cache_tag}"',
        },
    )
    print(open("/root/DeepSpec/config_block16.py").read(), flush=True)

    os.makedirs("/vol/checkpoints", exist_ok=True)
    os.makedirs("/vol/tensorboard", exist_ok=True)
    for link, target in (("/root/checkpoints", "/vol/checkpoints"), ("/root/tensorboard", "/vol/tensorboard")):
        if not os.path.exists(link):
            os.symlink(target, link)

    data_path = f"/vol/data/train{cache_tag}.jsonl"
    cache_dir = f"{CACHE_DIR}{cache_tag}"
    if not os.path.exists(data_path):
        # download_and_split errors on PARTIAL existence, so the eval split needs a
        # tag-specific filename or it collides with a previous run's leftover.
        _sh(["python", "scripts/data/download_and_split.py",
             "--sample-size", str(num_samples),
             "--train-output-path", data_path,
             "--test-output-dir", "/vol/data",
             "--test-output-name", f"perfectblend{cache_tag}.jsonl",
             "--skip-existing"])
        vol.commit()

    done_marker = f"{cache_dir}/.done"
    if not os.path.exists(done_marker):
        _sh(["python", "scripts/data/prepare_target_cache.py",
             "--config", "config_block16.py",
             "--train-data-path", data_path,
             "--output-dir", cache_dir, "--local-batch-size", "16"])
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


@app.function(image=image, gpu=SWEEP_GPU, timeout=10800, volumes={"/vol": vol, "/hfcache": hf_cache})
def sweep_one(dataset: str, exp_suffix: str = "", max_samples: int = 8,
              max_new_tokens: int = 512, tree_budget: str = "16,32,64,128,256",
              beam_min_width: int = 2, beam_candidates: int = 512, cuda_graph: bool = False,
              tree_mode: str = "beam", max_fanout: int = 0, precompute: bool = True,
              tree_candidates: int = 2048, temperature: float = 0.0):
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
        "--tree-mode", tree_mode,
        "--beam-widths", "flat",
        "--beam-min-width", str(beam_min_width),
        "--beam-candidates", str(beam_candidates),
        "--beam-max-fanout", str(max_fanout),
        "--tree-candidates", str(tree_candidates),
        *([] if precompute else ["--no-beam-precompute"]),
        *(["--beam-cuda-graph"] if cuda_graph else []),
        "--minimal",
        "--temperature", str(temperature),
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
               tree_budget: str = "16,32,64,128,256", beam_min_width: int = 2,
               beam_candidates: int = 512, cuda_graph: bool = False,
               tree_mode: str = "beam", max_fanout: int = 0, precompute: bool = True,
               tree_candidates: int = 2048, temperature: float = 0.0):
    """Full comparison: one GPU container per dataset, all budgets in each."""
    import json
    import os
    import time

    names = [d.strip() for d in datasets.split(",")]
    budgets = [int(b) for b in tree_budget.split(",")]
    calls = [sweep_one.spawn(dataset=n, exp_suffix=exp_suffix, max_samples=max_samples,
                             max_new_tokens=max_new_tokens, tree_budget=tree_budget,
                             beam_min_width=beam_min_width, beam_candidates=beam_candidates,
                             cuda_graph=cuda_graph, tree_mode=tree_mode,
                             max_fanout=max_fanout, precompute=precompute,
                             tree_candidates=tree_candidates,
                             temperature=temperature) for n in names]
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


@app.function(image=image, timeout=28800, volumes={"/vol": vol, "/hfcache": hf_cache})
def finalize(exp_suffix: str = "_bigdata", datasets: str = "humaneval,mbpp,gsm8k,math500,mt-bench,alpaca",
             max_samples: int = 12, max_new_tokens: int = 512, tree_budget: str = "32,64,128,256"):
    """Stage 2 of experiments/NEXT.md: publish the trained drafter, then run the
    full transformers-harness sweep on it.

    One command to go from "training finished" to final single-sequence results.
    Concurrency is out of scope here - that needs the SGLang build (stage 3).
    """
    import os

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    if not os.path.exists(f"{ckpt}/config.json"):
        print(f"publishing checkpoint for {exp_suffix}...", flush=True)
        publish.local(exp_suffix=exp_suffix)
    else:
        print(f"{ckpt} already published", flush=True)

    print(f"\n=== full sweep: {datasets} x budgets {tree_budget} ===", flush=True)
    return sweep_wide.local(
        datasets=datasets, exp_suffix=exp_suffix, max_samples=max_samples,
        max_new_tokens=max_new_tokens, tree_budget=tree_budget,
    )


@app.function(image=image, timeout=36000, volumes={"/vol": vol, "/hfcache": hf_cache})
def queued_final(
    exp_suffix: str = "_bigdata",
    datasets: str = "humaneval,mbpp,gsm8k,math500,mt-bench,alpaca",
    max_samples: int = 12,
    max_new_tokens: int = 512,
    tree_budget: str = "64,128",
    poll_seconds: int = 120,
    max_wait_hours: float = 8.0,
):
    """Wait for the bigdata drafter to finish training, then run the final sweep.

    Runs server-side and detached so it survives the local client. Polls the
    volume for the published checkpoint rather than chaining off the training
    app, so it is robust to the trainer being restarted or resumed.
    """
    import os
    import time

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    trained_dir = f"/vol/checkpoints/deepspec/dspark_block16_qwen3_4b_warmstart{exp_suffix}"
    deadline = max_wait_hours * 3600
    waited = 0.0

    while True:
        vol.reload()
        if os.path.exists(f"{ckpt}/config.json"):
            print(f"[queue] checkpoint published at {ckpt}", flush=True)
            break
        # trainer publishes at the end; if it died mid-run we can still salvage
        # the newest step_* directory rather than waiting out the full deadline.
        if os.path.exists(trained_dir):
            steps = [d for d in os.listdir(trained_dir) if d.startswith("step_")]
            print(f"[queue] waiting... {waited/60:.0f} min, checkpoints so far: {sorted(steps)}", flush=True)
        else:
            print(f"[queue] waiting... {waited/60:.0f} min, training dir not created yet", flush=True)

        if waited >= deadline:
            if os.path.exists(trained_dir):
                print("[queue] deadline hit but step checkpoints exist - publishing newest", flush=True)
                publish.local(exp_suffix=exp_suffix)
                break
            raise TimeoutError(f"no checkpoint after {max_wait_hours}h")
        time.sleep(poll_seconds)
        waited += poll_seconds

    if not os.path.exists(f"{ckpt}/config.json"):
        publish.local(exp_suffix=exp_suffix)

    print(f"\n=== FINAL BENCHMARK: {datasets} x budgets {tree_budget} ===", flush=True)
    return sweep_wide.local(
        datasets=datasets, exp_suffix=exp_suffix, max_samples=max_samples,
        max_new_tokens=max_new_tokens, tree_budget=tree_budget,
    )


@app.function(image=image, gpu="A10G", timeout=14400, volumes={"/vol": vol, "/hfcache": hf_cache})
def bench_mode_one(dataset: str, exp_suffix: str = "_bigdata", mode: str = "beam",
                   max_samples: int = 10, max_new_tokens: int = 512, tree_budget: str = "64,128",
                   max_fanout: int = 0):
    """One dataset, one tree-construction mode."""
    import os
    import torch

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    save_path = f"/root/mode_{dataset}_{mode}_{max_fanout}.pt"
    _sh([
        "python", "benchmark.py",
        "--beam-max-fanout", str(max_fanout),
        "--model-name-or-path", "Qwen/Qwen3-4B",
        "--draft-name-or-path", "z-lab/Qwen3-4B-DFlash-b16",
        "--dspark-name-or-path", ckpt,
        "--dataset", dataset,
        "--max-samples", str(max_samples),
        "--max-new-tokens", str(max_new_tokens),
        "--tree-budget", tree_budget,
        "--tree-mode", mode,
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
            r = agg.setdefault(method, {"accept": [], "tpot": [], "stage": {}})
            r["accept"].extend(result.acceptance_lengths)
            r["tpot"].append(result.time_per_output_token)
            for k, v in getattr(result, "stage_times", {}).items():
                r["stage"][k] = r["stage"].get(k, 0.0) + v
    for method, r in agg.items():
        r["mean_accept"] = sum(r["accept"]) / max(len(r["accept"]), 1)
        r["mean_tpot_ms"] = 1e3 * sum(r["tpot"]) / max(len(r["tpot"]), 1)
        del r["accept"], r["tpot"]
    return agg


@app.function(image=image, timeout=36000, volumes={"/vol": vol, "/hfcache": hf_cache})
def queued_confidence(
    exp_suffix: str = "_bigdata",
    datasets: str = "humaneval,gsm8k,math500,alpaca",
    max_samples: int = 10,
    max_new_tokens: int = 512,
    tree_budget: str = "64,128",
    poll_seconds: int = 120,
    max_wait_hours: float = 9.0,
):
    """Does DSpark's confidence head make a better tree than a fixed schedule?

    The flat schedule is the AVERAGE optimal allocation (measured over many
    rounds). The confidence head predicts P(accept) per position for the CURRENT
    round, so width can adapt: concentrate shallow when the continuation looks
    hard, spread deep when it looks easy.

    Arms (identical otherwise): beam+flat vs confidence-adaptive widths.
    """
    import json
    import os
    import time

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    waited = 0.0
    while not os.path.exists(f"{ckpt}/config.json"):
        vol.reload()
        if waited >= max_wait_hours * 3600:
            raise TimeoutError(f"no checkpoint after {max_wait_hours}h")
        print(f"[queue] waiting for {ckpt}... {waited/60:.0f} min", flush=True)
        time.sleep(poll_seconds)
        waited += poll_seconds
    print(f"[queue] checkpoint ready, starting confidence experiment", flush=True)

    names = [d.strip() for d in datasets.split(",")]
    calls = {
        (n, m): bench_mode_one.spawn(dataset=n, exp_suffix=exp_suffix, mode=m,
                                     max_samples=max_samples, max_new_tokens=max_new_tokens,
                                     tree_budget=tree_budget)
        for n in names for m in ("beam", "confidence")
    }
    results = {}
    for (n, m), call in calls.items():
        try:
            results.setdefault(n, {})[m] = call.get()
        except Exception as exc:
            print(f"!! {n}/{m} FAILED: {exc}", flush=True)

    budgets = [int(b) for b in tree_budget.split(",")]
    print(f"\n{'dataset':11s} {'budget':>6s} {'flat acc':>9s} {'conf acc':>9s} {'delta':>7s}  "
          f"{'flat spd':>9s} {'conf spd':>9s} {'delta':>7s}")
    deltas = {"acc": [], "spd": []}
    for n, arms in results.items():
        if "beam" not in arms or "confidence" not in arms:
            continue
        for B in budgets:
            key = f"dsparktree_markov_tb{B}"
            f_, c_ = arms["beam"].get(key), arms["confidence"].get(key)
            if not f_ or not c_:
                continue
            bf = arms["beam"]["baseline"]["mean_tpot_ms"]
            bc = arms["confidence"]["baseline"]["mean_tpot_ms"]
            sf, sc = bf / f_["mean_tpot_ms"], bc / c_["mean_tpot_ms"]
            da = 100 * (c_["mean_accept"] / f_["mean_accept"] - 1)
            ds = 100 * (sc / sf - 1)
            deltas["acc"].append(da)
            deltas["spd"].append(ds)
            print(f"{n:11s} {B:6d} {f_['mean_accept']:9.3f} {c_['mean_accept']:9.3f} {da:+6.1f}%  "
                  f"{sf:8.2f}x {sc:8.2f}x {ds:+6.1f}%")
    if deltas["acc"]:
        print(f"\nMEAN  accept {sum(deltas['acc'])/len(deltas['acc']):+.1f}%   "
              f"speed {sum(deltas['spd'])/len(deltas['spd']):+.1f}%")

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/confidence_{int(time.time())}.json"
    json.dump(results, open(out, "w"), indent=2)
    vol.commit()
    print(f"\nsaved {out}\nCONFIDENCE EXPERIMENT COMPLETE", flush=True)
    return results


@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/vol": vol, "/hfcache": hf_cache})
def bench_tree_build(
    exp_suffix: str = "_bigdata",
    dataset: str = "gsm8k",
    budgets: str = "16,64,128,256",
    candidates: str = "256,512,2048",
    rounds: int = 60,
    min_width: int = 2,
    gpu_label: str = "A10G",
):
    """Isolated builder microbenchmark - the only measurement that can see this stage.

    RESULTS.md §12: tree_build is ~1-4 ms/round against a ~500 ms run, and the
    per-dataset sweep carries ~±12% container variance, so it moved in the WRONG
    direction on half the datasets while the isolated number was unambiguous. Any
    claim about builder cost comes from here; the sweep only has to show that
    end-to-end acceptance and speed did not regress.

    Reports in-loop matmul vs precomputed C x C table vs CUDA-graph replay, at each
    candidate pool size, plus fidelity of every restricted pool against the exact
    full-vocab beam.
    """
    import json
    import os
    import time

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    assert os.path.exists(f"{ckpt}/config.json"), f"{ckpt} not published"

    save_path = "/root/tree_build.json"
    _sh([
        "python", "bench_tree_build.py",
        "--model-name-or-path", "Qwen/Qwen3-4B",
        "--dspark-name-or-path", ckpt,
        "--dataset", dataset,
        "--budgets", budgets,
        "--candidates", candidates,
        "--rounds", str(rounds),
        "--min-width", str(min_width),
        "--save-path", save_path,
    ], cwd="/root/ddtree")

    results = json.load(open(save_path))
    results["gpu"] = gpu_label
    results["dataset"] = dataset

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/treebuild_{gpu_label}_{int(time.time())}.json"
    json.dump(results, open(out, "w"), indent=2)
    vol.commit()
    print(f"\nsaved {out}\nTREE BUILD MICROBENCH COMPLETE", flush=True)
    return results


@app.function(image=image, gpu=BIG_GPU, timeout=7200, volumes={"/vol": vol, "/hfcache": hf_cache})
def bench_tree_build_h100(
    exp_suffix: str = "_bigdata",
    dataset: str = "gsm8k",
    budgets: str = "16,64,128,256",
    candidates: str = "256,512,2048",
    rounds: int = 60,
    min_width: int = 2,
):
    """Same microbenchmark on the reference GPU. §12's published numbers are H100."""
    return bench_tree_build.local(
        exp_suffix=exp_suffix, dataset=dataset, budgets=budgets, candidates=candidates,
        rounds=rounds, min_width=min_width, gpu_label="H100",
    )


@app.function(image=image, gpu="A10G", timeout=14400, volumes={"/vol": vol, "/hfcache": hf_cache})
def bench_depth_one(dataset: str, exp_suffix: str = "_bigdata", min_width: int = 2,
                    max_samples: int = 10, max_new_tokens: int = 512, tree_budget: str = "16,32,64,128,256"):
    """One dataset, one min_width - i.e. one depth/width split of the same budget.

    min_width caps effective depth at budget // min_width. At budget 64: 2 leaves
    the full 16 depths ([4]*16), 4 also gives 16, 5 truncates to 12 ([6,6,6,6,5...]),
    8 to 8. This is the experiment that decides whether the deep levels earn their
    slots or whether the freed budget is worth more spent shallow.
    """
    import os
    import torch

    ckpt = f"{CKPT_FINAL}{exp_suffix}"
    save_path = f"/root/depth_{dataset}_{min_width}.pt"
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
        "--beam-min-width", str(min_width),
        "--minimal",
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
    return agg


@app.function(image=image, timeout=28800, volumes={"/vol": vol, "/hfcache": hf_cache})
def bench_depth(datasets: str = "humaneval,gsm8k,math500,alpaca", exp_suffix: str = "_bigdata",
                min_widths: str = "2,5,8", max_samples: int = 10, max_new_tokens: int = 512,
                tree_budget: str = "16,32,64,128,256"):
    """Depth-vs-width sweep: is truncating the tree to 12 levels worth the freed budget?

    RESULTS.md §5 measured top-1 HIT RATE per depth, which is not the same question
    as how much acceptance MASS lives at depths 13-16. Mean acceptance is ~10.6 on
    gsm8k, so those depths are reached routinely on math and code; truncation is the
    one change here that can cost acceptance rather than just time, so it ships only
    if this sweep says it does not.
    """
    import json
    import os
    import time

    names = [d.strip() for d in datasets.split(",")]
    widths = [int(w) for w in min_widths.split(",")]
    budgets = [int(b) for b in tree_budget.split(",")]
    calls = {
        (n, w): bench_depth_one.spawn(dataset=n, exp_suffix=exp_suffix, min_width=w,
                                      max_samples=max_samples, max_new_tokens=max_new_tokens,
                                      tree_budget=tree_budget)
        for n in names for w in widths
    }
    results = {}
    for (n, w), call in calls.items():
        try:
            results.setdefault(n, {})[str(w)] = call.get()
            print(f"[done] {n} min_width={w}", flush=True)
        except Exception as exc:
            print(f"!! {n}/min_width={w} FAILED: {exc}", flush=True)

    print(f"\n{'dataset':11s} {'budget':>6s} {'min_w':>5s} {'depth':>5s} {'accept':>7s} {'speedup':>8s}")
    summary = {}
    for n, arms in results.items():
        for budget in budgets:
            key = f"dsparktree_markov_tb{budget}"
            for w in widths:
                arm = arms.get(str(w), {})
                row, base = arm.get(key), arm.get("baseline", {}).get("mean_tpot_ms")
                if not row or not base:
                    continue
                depth = min(16, budget // w)
                speedup = base / row["mean_tpot_ms"]
                summary.setdefault(f"tb{budget}_mw{w}", {"acc": [], "spd": []})
                summary[f"tb{budget}_mw{w}"]["acc"].append(row["mean_accept"])
                summary[f"tb{budget}_mw{w}"]["spd"].append(speedup)
                print(f"{n:11s} {budget:6d} {w:5d} {depth:5d} {row['mean_accept']:7.3f} {speedup:7.2f}x")

    print(f"\n{'config':14s} {'accept':>7s} {'speedup':>8s}   (mean over datasets)")
    for key, v in summary.items():
        v["mean_accept"] = sum(v["acc"]) / len(v["acc"])
        v["mean_speedup"] = sum(v["spd"]) / len(v["spd"])
        print(f"{key:14s} {v['mean_accept']:7.3f} {v['mean_speedup']:7.2f}x")

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/depth_{int(time.time())}.json"
    json.dump({"per_dataset": results, "summary": summary}, open(out, "w"), indent=2)
    vol.commit()
    print(f"\nsaved {out}\nDEPTH SWEEP COMPLETE", flush=True)
    return {"summary": summary, "per_dataset": results}


@app.function(image=image, timeout=28800, volumes={"/vol": vol, "/hfcache": hf_cache})
def bench_beam_candidates(datasets: str = "gsm8k,humaneval,alpaca", exp_suffix: str = "_bigdata",
                          candidate_sizes: str = "256,512,2048", max_samples: int = 10,
                          max_new_tokens: int = 512, tree_budget: str = "64,128"):
    """How much acceptance does the beam's candidate pool C actually buy?

    The microbenchmark (bench_tree_build) says the restricted pools are NOT close
    to the exact full-vocab beam - per-depth top-256 reproduces only ~72% of its
    nodes, top-2048 ~87%. That metric is harsh, because one swap high in the tree
    renames every descendant, and RESULTS.md §4 found restriction cost the
    best-first builder under 2% acceptance. But "cheap and slightly different" and
    "cheap and worse" are not the same claim, and C is now a QUADRATIC cost knob
    (the precomputed table is C x C), so the trade has to be measured rather than
    assumed from bias magnitudes.
    """
    import json
    import os
    import time

    names = [d.strip() for d in datasets.split(",")]
    sizes = [int(c) for c in candidate_sizes.split(",")]
    budgets = [int(b) for b in tree_budget.split(",")]
    calls = {
        (n, c): sweep_one.spawn(dataset=n, exp_suffix=exp_suffix, max_samples=max_samples,
                                max_new_tokens=max_new_tokens, tree_budget=tree_budget,
                                beam_candidates=c)
        for n in names for c in sizes
    }
    results = {}
    for (n, c), call in calls.items():
        try:
            results.setdefault(n, {})[str(c)] = call.get()
            print(f"[done] {n} C={c}", flush=True)
        except Exception as exc:
            print(f"!! {n}/C={c} FAILED: {exc}", flush=True)

    print(f"\n{'dataset':11s} {'budget':>6s} {'C':>6s} {'accept':>7s} {'speedup':>8s} {'build_s':>8s}")
    summary = {}
    for n, arms in results.items():
        for budget in budgets:
            key = f"dsparktree_markov_tb{budget}"
            for c in sizes:
                arm = arms.get(str(c), {})
                row, base = arm.get(key), arm.get("baseline", {}).get("mean_tpot_ms")
                if not row or not base:
                    continue
                speedup = base / row["mean_tpot_ms"]
                bucket = summary.setdefault(f"tb{budget}_C{c}", {"acc": [], "spd": [], "build": []})
                bucket["acc"].append(row["mean_accept"])
                bucket["spd"].append(speedup)
                bucket["build"].append(row["stage"].get("tree_build", 0.0))
                print(f"{n:11s} {budget:6d} {c:6d} {row['mean_accept']:7.3f} {speedup:7.2f}x "
                      f"{row['stage'].get('tree_build', 0.0):8.2f}")

    print(f"\n{'config':14s} {'accept':>7s} {'speedup':>8s} {'build_s':>8s}   (mean over datasets)")
    for key, v in summary.items():
        v["mean_accept"] = sum(v["acc"]) / len(v["acc"])
        v["mean_speedup"] = sum(v["spd"]) / len(v["spd"])
        v["mean_build"] = sum(v["build"]) / len(v["build"])
        print(f"{key:14s} {v['mean_accept']:7.3f} {v['mean_speedup']:7.2f}x {v['mean_build']:8.2f}")

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/candidates_{int(time.time())}.json"
    json.dump({"per_dataset": results, "summary": summary}, open(out, "w"), indent=2)
    vol.commit()
    print(f"\nsaved {out}\nCANDIDATE SWEEP COMPLETE", flush=True)
    return {"summary": summary, "per_dataset": results}


@app.function(image=image, timeout=28800, volumes={"/vol": vol, "/hfcache": hf_cache})
def bench_shape(datasets: str = "gsm8k,humaneval,alpaca", exp_suffix: str = "_bigdata",
                max_samples: int = 10, max_new_tokens: int = 512, tree_budget: str = "64,128",
                max_fanout: int = 0):
    """Adaptive best-first shape vs the beam's flat schedule, both over the table.

    The flat schedule was never argued to be a better tree - RESULTS.md §4 measured
    best-first at 11.45 acceptance against flat beam's 11.09, and the beam won only
    because best-first cost ~budget dependent GPU round-trips. build_markov_tree_-
    precomputed removes that cost, so the 3% is available again if the heap and the
    bigger transfer do not eat it. Arms are identical apart from tree shape.
    """
    import json
    import os
    import time

    names = [d.strip() for d in datasets.split(",")]
    modes = ("beam", "exact-precomputed")
    budgets = [int(b) for b in tree_budget.split(",")]
    calls = {
        (n, m): bench_mode_one.spawn(dataset=n, exp_suffix=exp_suffix, mode=m,
                                     max_samples=max_samples, max_new_tokens=max_new_tokens,
                                     tree_budget=tree_budget, max_fanout=max_fanout)
        for n in names for m in modes
    }
    results = {}
    for (n, m), call in calls.items():
        try:
            results.setdefault(n, {})[m] = call.get()
            print(f"[done] {n} {m}", flush=True)
        except Exception as exc:
            print(f"!! {n}/{m} FAILED: {exc}", flush=True)

    print(f"\n{'dataset':11s} {'budget':>6s} {'beam acc':>9s} {'bf acc':>9s} {'delta':>7s}  "
          f"{'beam spd':>9s} {'bf spd':>9s} {'delta':>7s} {'bf build':>9s}")
    deltas = {"acc": [], "spd": []}
    for n, arms in results.items():
        if not all(m in arms for m in modes):
            continue
        for budget in budgets:
            key = f"dsparktree_markov_tb{budget}"
            beam, best = arms["beam"].get(key), arms["exact-precomputed"].get(key)
            if not beam or not best:
                continue
            sb = arms["beam"]["baseline"]["mean_tpot_ms"] / beam["mean_tpot_ms"]
            sf = arms["exact-precomputed"]["baseline"]["mean_tpot_ms"] / best["mean_tpot_ms"]
            da = 100 * (best["mean_accept"] / beam["mean_accept"] - 1)
            ds = 100 * (sf / sb - 1)
            deltas["acc"].append(da)
            deltas["spd"].append(ds)
            print(f"{n:11s} {budget:6d} {beam['mean_accept']:9.3f} {best['mean_accept']:9.3f} {da:+6.1f}%  "
                  f"{sb:8.2f}x {sf:8.2f}x {ds:+6.1f}% {best['stage'].get('tree_build', 0):9.2f}")
    if deltas["acc"]:
        print(f"\nMEAN  accept {sum(deltas['acc'])/len(deltas['acc']):+.1f}%   "
              f"speed {sum(deltas['spd'])/len(deltas['spd']):+.1f}%")

    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/shape_{int(time.time())}.json"
    json.dump(results, open(out, "w"), indent=2)
    vol.commit()
    print(f"\nsaved {out}\nSHAPE EXPERIMENT COMPLETE", flush=True)
    return results
