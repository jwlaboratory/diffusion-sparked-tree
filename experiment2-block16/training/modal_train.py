"""Detached Modal pipeline that fine-tuned our block-16 DSpark drafter for
Qwen3-4B, warm-started from deepseek-ai/dspark_qwen3_4b_block7.

This is the recipe that produced shreybirmiwal/Qwen3-4B-DSpark-b16. Nothing here
scores the model - benchmarking lives in a separate harness (../modal_benchmark.py).
The job of this file is only: build chat data, precompute the target hidden-state
cache, warm-start fine-tune at block 16, and publish a from_pretrained checkpoint.

    # the shipped model: A10G, 2400 PerfectBlend rows -> the _best checkpoint
    modal run --detach modal_train.py::pipeline --exp-suffix _best

    # the larger arm -> the _bigdata checkpoint, which scored worse on all 6
    # datasets and so was not shipped. It moved five knobs at once (data, seq len,
    # anchors, steps, gamma), so that regression is unattributable -- see README.md.
    # The original invocation was not committed; ~9.5k conversations is what
    # RESULTS.md records, so this line is a reconstruction, not a verbatim replay.
    modal run --detach modal_train.py::pipeline_h100 --exp-suffix _bigdata --num-samples 10000 --cache-tag _bigdata

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
)

vol = modal.Volume.from_name("ddtree-train", create_if_missing=True)
hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)

# GPU tier for the large-memory pipeline. Modal validates EVERY function in the
# app at deploy time, so hardcoding H100 makes the whole file undeployable on a
# workspace without H100 access - even for CPU-only functions like publish().
BIG_GPU = os.environ.get("DDTREE_BIG_GPU", "H100")

# GPU for the data-scaling arms. Faster hardware only — the training config
# (anchors, seq len, batch, steps) is NOT scaled up with the bigger card, so
# arms stay comparable. Single GPU only: multi-GPU would change the DP world
# size and thus the effective batch semantics.
DATA_GPU = os.environ.get("DDTREE_DATA_GPU", "A10G")

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


def _patch_data_script_skip_invalid():
    """Make download_and_split.py skip rows that fail validation instead of
    aborting the whole download. PerfectBlend has occasional malformed rows
    (e.g. empty message content) that only surface at larger sample sizes —
    the 24k pull died on one such row. Dropping a handful of rows changes the
    effective train count negligibly; the skip count is printed."""
    path = "/root/DeepSpec/scripts/data/download_and_split.py"
    src = open(path).read()
    target = (
        "            converted = normalize_conversations(row)\n"
        "            validate_conversations(converted, row_number)\n"
    )
    replacement = (
        "            converted = normalize_conversations(row)\n"
        "            try:\n"
        "                validate_conversations(converted, row_number)\n"
        "            except ValueError as e:\n"
        "                print(f'[skip-invalid] {e}', flush=True)\n"
        "                continue\n"
    )
    assert src.count(target) == 2, f"expected 2 validate blocks, found {src.count(target)}"
    open(path, "w").write(src.replace(target, replacement))
    print("[patch] download_and_split.py now skips invalid rows", flush=True)


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
    gpu=DATA_GPU,
    timeout=86400,
    volumes={"/vol": vol, "/hfcache": hf_cache},
)
def pipeline_data(
    exp_suffix: str,
    num_samples: int,
    max_train_steps: int,
    cache_tag: str = "",
):
    """Data-scaling arms: the _best recipe exactly (A10G, seq 768, 32 anchors,
    gamma 4, global batch 32, lr 1e-4), varying ONLY the row count, with
    max_train_steps epoch-matched across arms (~5 epochs at global batch 32:
    2,400 rows -> 360 steps, 10,000 -> 1,500, 24,000 -> 3,600). This is the
    clean version of the confounded _bigdata run: one variable, everything else
    pinned. cache_tag="" reuses the untagged _best data + target cache.
    """
    import os
    import threading

    _patch_data_script_skip_invalid()
    _stage_files(
        gamma=4.0,
        exp_suffix=exp_suffix,
        overrides={
            "max_train_steps": max_train_steps,
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

    # "exists" is not enough: a crashed stage-1 leaves a PARTIAL jsonl on the
    # volume, and a naive skip then silently trains on a fraction of the data
    # (this happened: 5,085 of 22,800 rows). Require ~the expected row count.
    expected_rows = int(num_samples * 0.95 * 0.9)  # 5% eval split, 10% slack for skipped rows

    def _row_count(path):
        with open(path) as fh:
            return sum(1 for _ in fh)

    if os.path.exists(data_path) and _row_count(data_path) < expected_rows:
        n = _row_count(data_path)
        print(f"stage 1: {data_path} has {n} rows < expected {expected_rows} - partial from a crashed run, redownloading", flush=True)
        os.remove(data_path)
        vol.commit()

    if not os.path.exists(data_path):
        _sh(["python", "scripts/data/download_and_split.py",
             "--sample-size", str(num_samples),
             "--train-output-path", data_path,
             "--test-output-dir", "/vol/data",
             "--test-output-name", f"perfectblend{cache_tag}.jsonl",
             "--skip-existing"])
        assert _row_count(data_path) >= expected_rows, \
            f"stage 1 produced {_row_count(data_path)} rows, expected >= {expected_rows}"
        vol.commit()
    else:
        print(f"stage 1: {data_path} exists with {_row_count(data_path)} rows, skipping", flush=True)

    done_marker = f"{cache_dir}/.done"
    if not os.path.exists(done_marker):
        _sh(["python", "scripts/data/prepare_target_cache.py",
             "--config", "config_block16.py",
             "--train-data-path", data_path,
             "--output-dir", cache_dir, "--local-batch-size", "8"])
        open(done_marker, "w").write("ok")
        vol.commit()
    else:
        print(f"stage 2: {cache_dir} exists, skipping", flush=True)

    # Long train stage: commit the volume every 10 min so the step_* checkpoints
    # (written every checkpointing_steps=200) survive a container death - an
    # uncommitted volume loses everything since the last commit.
    stop = threading.Event()

    def _committer():
        while not stop.wait(600):
            try:
                vol.commit()
                print("[committer] volume committed", flush=True)
            except Exception as e:  # commit conflicts are retryable next tick
                print(f"[committer] commit failed: {e}", flush=True)

    committer = threading.Thread(target=_committer, daemon=True)
    committer.start()
    try:
        _sh(["python", "train_warmstart.py", "--config", "config_block16.py"])
    finally:
        stop.set()
        committer.join(timeout=30)
    vol.commit()

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

    This is the arm that produced the _bigdata checkpoint (num_samples 9600,
    cache_tag "_bigdata"). It scored worse than the smaller A10G _best run on all
    six datasets - the shipped model is _best. Kept here so that finding stays
    reproducible, not just asserted.
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
