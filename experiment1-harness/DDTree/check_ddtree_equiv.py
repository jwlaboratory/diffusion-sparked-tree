"""One-off equivalence check (NOT part of the experiment run).

`dflash.tree` in run_experiment.py is `sparked_tree_generate(..., markov_head=None,
draft_mode="dflash")`. With no corrector this must reduce to the official
`ddtree_generate`: same per-depth top-k, same best-first order, same acceptance.

Run this manually after touching the tree code, to confirm we still reproduce the
official harness:

    python check_ddtree_equiv.py --dataset gsm8k --max-samples 4 --tree-budget 64

Exits non-zero if the two acceptance-length streams diverge beyond tolerance.
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, load_and_process_dataset
from ddtree import ddtree_generate, maybe_enable_cpp_compact
from sparked_tree import sparked_tree_generate


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Qwen/Qwen3-4B")
    p.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--max-samples", type=int, default=4)
    p.add_argument("--tree-budget", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--tol", type=float, default=1e-6)
    args = p.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0")
    maybe_enable_cpp_compact(True)

    target = AutoModelForCausalLM.from_pretrained(
        args.target, attn_implementation="sdpa", dtype=torch.bfloat16
    ).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(
        args.draft, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.target)

    dataset = load_and_process_dataset(args.dataset)
    if len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    common = dict(
        mask_token_id=draft.mask_token_id, block_size=draft.block_size,
        max_new_tokens=args.max_new_tokens, stop_token_ids=[tok.eos_token_id],
        temperature=0.0, tree_budget=args.tree_budget,
    )

    max_abs_diff = 0.0
    for row in dataset:
        text = tok.apply_chat_template(
            [{"role": "user", "content": row["turns"][0]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tok.encode(text, return_tensors="pt").to(device)
        official = ddtree_generate(model=draft, target=target, input_ids=ids, **common)
        ours = sparked_tree_generate(
            model=draft, target=target, input_ids=ids, markov_head=None, draft_mode="dflash", **common,
        )
        a, b = official.acceptance_lengths, ours.acceptance_lengths
        if len(a) != len(b):
            print(f"LENGTH MISMATCH: official {len(a)} rounds vs ours {len(b)}")
            return 1
        for x, y in zip(a, b):
            max_abs_diff = max(max_abs_diff, abs(x - y))

    print(f"max per-round acceptance-length diff: {max_abs_diff}")
    if max_abs_diff > args.tol:
        print("FAIL: dflash.tree diverges from official ddtree_generate")
        return 1
    print("PASS: dflash.tree == official ddtree_generate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
