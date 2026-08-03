"""Does the TARGET model's own greedy answer depend on how many positions it sees?

    modal run final_benchmark/diagnose_shape_numerics.py::diagnose

diagnose_determinism.py showed every method reproduces itself exactly run-to-run,
so the harness is deterministic. That does NOT explain the losslessness failures,
because both repeats used identical shapes. The remaining hypothesis is
shape-dependent numerics, and it is testable without any speculative decoding at
all:

  1. decode N tokens the slow way — one position at a time, KV cache, argmax.
  2. take that finished sequence and push it through the target in ONE forward pass.
  3. compare the argmax at each position with what step 1 emitted.

Step 2 is teacher-forcing the model on its own output, so in exact arithmetic the
two must agree everywhere. Any mismatch is bf16 accumulation differing between a
1-position forward and an L-position forward — no drafter, no tree, no acceptance
rule involved.

This matters because exact-match acceptance guarantees the emitted text equals the
target's argmax *as computed in the wide forward pass*. If that differs from the
target's argmax in the narrow pass, then "byte-identical to autoregressive
decoding" is unachievable by construction, for every speculative method, and the
acceptance numbers are still measuring exactly what they claim.
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

app = modal.App("sparked-tree-shape-numerics")

REPO_ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "datasets",
                 "safetensors", "loguru", "tqdm", "numpy")
    .add_local_dir(REPO_ROOT / "ddtree", remote_path="/root/ddtree")
    .add_local_dir(REPO_ROOT / "final_benchmark", remote_path="/root/final_benchmark")
)

hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)


@app.function(image=image, gpu=os.environ.get("FINAL_GPU", "H100"), timeout=3600,
              volumes={"/hfcache": hf_cache})
def diagnose(prompts: int = 4, new_tokens: int = 200):
    import sys

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    sys.path.insert(0, "/root/ddtree")
    from model import load_and_process_dataset

    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        cfg.TARGET, attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.TARGET)
    dataset = load_and_process_dataset("gsm8k").shuffle(seed=0).select(range(prompts))

    total_positions = total_mismatch = 0
    print(f"{'prompt':>7s} {'positions':>10s} {'mismatches':>11s} {'first at':>9s}")
    for index in range(prompts):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": dataset[index]["turns"][0]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]

        # 1. incremental greedy decode, one position at a time
        with torch.inference_mode():
            cache = DynamicCache()
            out = target(input_ids, past_key_values=cache, use_cache=True)
            token = out.logits[:, -1:].argmax(-1)
            generated = [int(token)]
            for _ in range(new_tokens - 1):
                out = target(token, past_key_values=cache, use_cache=True)
                token = out.logits[:, -1:].argmax(-1)
                generated.append(int(token))

            # 2. one wide forward over the finished sequence
            full = torch.cat([input_ids, torch.tensor([generated], device=device)], dim=1)
            wide = target(full[:, :-1], use_cache=False).logits.argmax(-1)

        # 3. compare at the positions that produced the generated tokens
        wide_preds = wide[0, prompt_len - 1:].tolist()
        mismatch = [i for i, (a, b) in enumerate(zip(generated, wide_preds)) if a != b]
        total_positions += len(generated)
        total_mismatch += len(mismatch)
        print(f"{index:7d} {len(generated):10d} {len(mismatch):11d} "
              f"{(mismatch[0] if mismatch else '-'):>9}")

    rate = total_mismatch / max(total_positions, 1)
    print(f"\nincremental vs single-pass argmax disagreement: "
          f"{total_mismatch}/{total_positions} = {rate:.2%}")
    print()
    if total_mismatch:
        print("The TARGET MODEL disagrees with itself depending on how many positions")
        print("it processes at once. No drafter or tree is involved here.")
        print("=> 'byte-identical to autoregressive decoding' is unachievable in bf16")
        print("   for ANY speculative method. Acceptance numbers remain valid: they")
        print("   measure agreement with the target's argmax in the pass that verifies.")
    else:
        print("The target is shape-stable, so numerics do NOT explain the losslessness")
        print("failures. Something in the shared verify/commit path is genuinely wrong.")
    return {"positions": total_positions, "mismatches": total_mismatch, "rate": rate}
