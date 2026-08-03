"""Is the harness non-deterministic, or is speculative decoding actually lossy?

    modal run final_benchmark/diagnose_determinism.py::diagnose

verify_lossless.py found that EVERY method diverges from the autoregressive
baseline — including DFlash and DDTree, which contain none of our code. Two
explanations fit that:

  (a) bf16 non-determinism. The baseline decodes one position at a time; a
      speculative method pushes 17-129 positions through the same kernels. cuBLAS
      picks different reduction orders for different shapes, logits move at ~1e-3,
      and an argmax over two near-tied tokens flips. From there the sequences
      diverge for ordinary reasons, not because acceptance was wrong.
  (b) a genuine losslessness violation somewhere in the shared verify/commit path.

These have opposite consequences — under (a) the acceptance numbers stand and only
the "byte-identical" wording is wrong; under (b) every result in the repo is void.

The discriminator is simple: run the SAME configuration twice. If plain
autoregressive greedy decoding does not even reproduce ITSELF, the divergence
cannot be evidence of a bug in speculative decoding.
"""

import os
import sys
from pathlib import Path

import modal

for _candidate in (Path(__file__).resolve().parent, Path("/root/final_benchmark")):
    if (_candidate / "config.py").exists():
        sys.path.insert(0, str(_candidate))
        break
import config as cfg  # noqa: E402

app = modal.App("sparked-tree-determinism")

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


@app.function(image=image, gpu=os.environ.get("FINAL_GPU", "H100"), timeout=3600,
              volumes={"/vol": vol, "/hfcache": hf_cache})
def diagnose(dataset: str = "gsm8k", samples: int = 4, max_new_tokens: int = 256):
    """Run an identical config twice; compare every method against its own twin."""
    import subprocess

    import torch

    paths = []
    for repeat in (1, 2):
        save_path = f"/root/determinism_{repeat}.pt"
        cmd = [
            "python", "benchmark.py",
            "--model-name-or-path", cfg.TARGET,
            "--draft-name-or-path", cfg.DFLASH_DRAFT,
            "--dspark-name-or-path", cfg.CHECKPOINT,
            "--dataset", dataset,
            "--max-samples", str(samples),
            "--max-new-tokens", str(max_new_tokens),
            "--tree-budget", "64",
            "--tree-mode", "exact-precomputed",
            "--beam-widths", "flat",
            "--beam-candidates", str(cfg.SPARKED["beam_candidates"]),
            "--beam-min-width", str(cfg.SPARKED["beam_min_width"]),
            "--minimal",
            "--temperature", "0.0",
            "--disable-cpp-compact-cache",
            "--save-path", save_path,
        ]
        env = dict(os.environ, HF_HOME="/hfcache", TOKENIZERS_PARALLELISM="false")
        print(f"\n$ repeat {repeat} (identical arguments)", flush=True)
        subprocess.run(cmd, cwd="/root/ddtree", env=env, check=True)
        paths.append(save_path)

    first, second = (torch.load(p, weights_only=False) for p in paths)
    tally = {}
    for r1, r2 in zip(first["responses"], second["responses"]):
        for method in r1:
            a, b = r1[method].output_ids, r2[method].output_ids
            row = tally.setdefault(method, {"n": 0, "same": 0})
            row["n"] += 1
            row["same"] += int(a.shape == b.shape and bool((a == b).all()))

    print(f"\n{'method':34s} {'run1 == run2':>14s}")
    for method, row in sorted(tally.items()):
        print(f"{method:34s} {row['same']:6d}/{row['n']:<7d}")

    baseline = tally.get("baseline", {"same": 0, "n": 1})
    deterministic = baseline["same"] == baseline["n"]
    print()
    if deterministic:
        print("Plain autoregressive decoding IS reproducible run-to-run.")
        print("=> the divergences found by verify_lossless are NOT explained by")
        print("   non-determinism, and point at a real losslessness violation.")
    else:
        print("Plain autoregressive greedy decoding does NOT reproduce itself.")
        print("=> the harness is numerically non-deterministic at the argmax level,")
        print("   so 'byte-identical to autoregressive decoding' is not a property")
        print("   this setup can establish for ANY method, ours or the baselines'.")
        print("   Acceptance measurements remain valid; the losslessness WORDING does not.")
    return {"deterministic_baseline": deterministic, "tally": tally}
