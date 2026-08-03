"""The test that can invalidate everything else: is the output byte-identical to
plain autoregressive greedy decoding?

    modal run final_benchmark/verify_lossless.py::verify

Speculative decoding is only meaningful if it changes speed and nothing else. Every
acceptance number in this repo is reported under exact-match prefix acceptance,
which is *supposed* to guarantee the emitted text equals what the target model
would have produced on its own. If a tree builder has a bug — a node whose parent
link is wrong, a visibility mask that lets a node attend to a non-ancestor, a
token-id mapped back through the wrong candidate table — the walk can accept a
token the target never would have chosen. That would not crash, and it would not
look wrong: it would look like *higher acceptance*. Which is exactly the direction
this project has been optimising.

So this runs the real decode loop for every builder and diffs the token ids
against the autoregressive baseline. Cheap, and it is the only check that can
falsify the whole result set rather than one number in it.
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

app = modal.App("sparked-tree-lossless")

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
def verify(dataset: str = "gsm8k", samples: int = 4, max_new_tokens: int = 256,
           tree_budget: str = "16,64,128"):
    """Run every builder on the same prompts and diff token ids against baseline."""
    import subprocess

    import torch

    results = {}
    for mode in ("exact-precomputed", "beam", "exact"):
        save_path = f"/root/lossless_{mode}.pt"
        cmd = [
            "python", "benchmark.py",
            "--model-name-or-path", cfg.TARGET,
            "--draft-name-or-path", cfg.DFLASH_DRAFT,
            "--dspark-name-or-path", cfg.CHECKPOINT,
            "--dataset", dataset,
            "--max-samples", str(samples),
            "--max-new-tokens", str(max_new_tokens),
            "--tree-budget", tree_budget,
            "--tree-mode", mode,
            "--beam-widths", "flat",
            "--beam-candidates", str(cfg.SPARKED["beam_candidates"]),
            "--beam-min-width", str(cfg.SPARKED["beam_min_width"]),
            "--minimal",
            "--temperature", "0.0",
            "--disable-cpp-compact-cache",
            "--save-path", save_path,
        ]
        env = dict(os.environ, HF_HOME="/hfcache", TOKENIZERS_PARALLELISM="false")
        print(f"\n$ tree-mode={mode}", flush=True)
        subprocess.run(cmd, cwd="/root/ddtree", env=env, check=True)

        data = torch.load(save_path, weights_only=False)
        for prompt_index, response in enumerate(data["responses"]):
            reference = response["baseline"].output_ids
            for method, result in response.items():
                if method == "baseline":
                    continue
                key = f"{mode}::{method}"
                row = results.setdefault(key, {"checked": 0, "identical": 0, "diffs": []})
                row["checked"] += 1
                ours = result.output_ids
                same = ours.shape == reference.shape and bool((ours == reference).all())
                if same:
                    row["identical"] += 1
                else:
                    # locate the first divergence so a failure is actionable
                    n = min(ours.shape[1], reference.shape[1])
                    mismatch = (ours[0, :n] != reference[0, :n]).nonzero()
                    first = int(mismatch[0][0]) if mismatch.numel() else n
                    row["diffs"].append({
                        "prompt": prompt_index,
                        "first_divergence_token": first,
                        "our_len": int(ours.shape[1]),
                        "baseline_len": int(reference.shape[1]),
                    })

    print(f"\n{'builder :: method':46s} {'identical':>12s} {'verdict':>10s}")
    failures = 0
    for key, row in sorted(results.items()):
        ok = row["identical"] == row["checked"]
        failures += 0 if ok else 1
        print(f"{key:46s} {row['identical']:6d}/{row['checked']:<5d} {'PASS' if ok else 'FAIL':>10s}")
        for diff in row["diffs"][:3]:
            print(f"    prompt {diff['prompt']}: first divergence at token "
                  f"{diff['first_divergence_token']} (len {diff['our_len']} vs {diff['baseline_len']})")

    print(f"\n{'LOSSLESS: every builder matches autoregressive greedy' if not failures else f'{failures} BUILDER(S) NOT LOSSLESS — results invalid'}",
          flush=True)
    return {"failures": failures, "detail": results}
