"""Acceptance-length comparison across six drafter / corrector / verify configs.

Metric: mean acceptance length (accepted tokens per verify round). Timing ignored.

    0. dflash              DFlash-b16 drafter, chain verify            (dflash_generate)
    2. dspark              DSpark-b7  drafter + markov, chain verify   (dspark_generate)
    3. dflash+markov+tree  DFlash-b16 drafter, DSpark markov, tree
    4. dflash+tree         DFlash-b16 drafter, no corrector, tree      (== ddtree)
    5. dspark+tree         DSpark-b7  drafter, no corrector, tree
    6. dspark+markov+tree  DSpark-b7  drafter, DSpark markov, tree

Configs 3-6 share one tree path (sparked_tree_generate); the only variables are the
drafter (dflash/dspark) and whether the DSpark markov head is passed as corrector.

Note on comparability: block sizes differ (DFlash-b16 vs DSpark-b7), so the per-round
acceptance ceiling differs (>= up to 16 vs up to 8). Compare within a block size, or
read the tree configs (3 vs 4, 5 vs 6) where markov-on/off is the only change.

Example:
    python run_acceptance.py --dataset gsm8k --max-samples 50 --tree-budget 64
    python run_acceptance.py --tasks gsm8k:8,humaneval:8,mt-bench:8 \
        --tree-budget 64 --save-json /results/sparked.json
"""

import argparse
import json
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DFlashDraftModel, DSparkDraftModel, load_and_process_dataset
from dflash import dflash_generate
from dspark import dspark_generate
from ddtree import maybe_enable_cpp_compact
from sparked_tree import sparked_tree_generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dflash-draft", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--dspark-draft", type=str, default="deepseek-ai/dspark_qwen3_4b_block7")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        help="Single dataset (used when --tasks is not given).")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma list of 'name:max_samples' (e.g. gsm8k:8,humaneval:8). "
                             "Overrides --dataset/--max-samples; models load once for all tasks.")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--tree-budget", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--methods",
        type=str,
        default="0,2,3,4,5,6",
        help="Comma-separated subset of config ids to run.",
    )
    parser.add_argument("--save-json", type=str, default=None,
                        help="Write the full results (config + per-dataset means) to this path.")
    return parser.parse_args()


def resolve_tasks(args) -> list[tuple[str, int]]:
    """[(dataset_name, max_samples), ...] from --tasks, else the single --dataset."""
    if args.tasks:
        tasks = []
        for spec in args.tasks.split(","):
            spec = spec.strip()
            if not spec:
                continue
            if ":" in spec:
                name, n = spec.split(":", 1)
                tasks.append((name.strip(), int(n)))
            else:
                tasks.append((spec, args.max_samples))
        return tasks
    return [(args.dataset, args.max_samples)]


def load_models(args, device):
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        attn_implementation="sdpa",  # tree configs use a custom mask on the target
        dtype=torch.bfloat16,
    ).to(device).eval()

    dflash = DFlashDraftModel.from_pretrained(
        args.dflash_draft, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    ).to(device).eval()

    dspark = DSparkDraftModel.from_pretrained(
        args.dspark_draft, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    ).to(device).eval()

    assert dspark.markov_head is not None, "DSpark checkpoint has no markov head to use as corrector"
    return target, dflash, dspark


def build_methods(target, dflash, dspark, eos_id, args):
    """Return {config_id: (label, callable(input_ids) -> result)}."""
    markov = dspark.markov_head
    common = dict(
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[eos_id],
        temperature=args.temperature,
    )

    def config0(ids):  # dflash chain
        return dflash_generate(
            model=dflash, target=target, input_ids=ids,
            mask_token_id=dflash.mask_token_id, block_size=dflash.block_size, **common,
        )

    def config2(ids):  # dspark chain (markov serial sweep is intrinsic)
        return dspark_generate(
            model=dspark, target=target, input_ids=ids, block_size=dspark.block_size, **common,
        )

    def config3(ids):  # dflash drafter + dspark markov corrector + tree
        return sparked_tree_generate(
            model=dflash, target=target, input_ids=ids, mask_token_id=dflash.mask_token_id,
            block_size=dflash.block_size, tree_budget=args.tree_budget,
            markov_head=markov, draft_mode="dflash", **common,
        )

    def config4(ids):  # dflash drafter, no corrector, tree (== ddtree)
        return sparked_tree_generate(
            model=dflash, target=target, input_ids=ids, mask_token_id=dflash.mask_token_id,
            block_size=dflash.block_size, tree_budget=args.tree_budget,
            markov_head=None, draft_mode="dflash", **common,
        )

    def config5(ids):  # dspark drafter, no corrector, tree
        return sparked_tree_generate(
            model=dspark, target=target, input_ids=ids, mask_token_id=dspark.mask_token_id,
            block_size=dspark.block_size, tree_budget=args.tree_budget,
            markov_head=None, draft_mode="dspark", **common,
        )

    def config6(ids):  # dspark drafter + dspark markov corrector + tree
        return sparked_tree_generate(
            model=dspark, target=target, input_ids=ids, mask_token_id=dspark.mask_token_id,
            block_size=dspark.block_size, tree_budget=args.tree_budget,
            markov_head=markov, draft_mode="dspark", **common,
        )

    return {
        "0": ("dflash_b16", config0),
        "2": ("dspark_b7", config2),
        "3": ("dflash+markov+tree", config3),
        "4": ("dflash+tree(ddtree)", config4),
        "5": ("dspark+tree", config5),
        "6": ("dspark+markov+tree", config6),
    }


def run_dataset(dataset_name, max_samples, methods, tokenizer, device):
    """Return {config_id: [acceptance lengths across all rounds]} for one dataset."""
    dataset = load_and_process_dataset(dataset_name)
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=0).select(range(max_samples))

    per_method_lengths = {mid: [] for mid in methods}
    for row in dataset:
        prompt = row["turns"][0]
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        for mid, (_, fn) in methods.items():
            result = fn(input_ids)
            per_method_lengths[mid].extend(result.acceptance_lengths)
    return per_method_lengths, len(dataset)


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0")
    maybe_enable_cpp_compact(True)

    target, dflash, dspark = load_models(args, device)
    tokenizer = AutoTokenizer.from_pretrained(args.target)

    all_methods = build_methods(target, dflash, dspark, tokenizer.eos_token_id, args)
    selected = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = {mid: all_methods[mid] for mid in selected if mid in all_methods}

    # Warmup each method once (kernels, cache-compaction extension, etc.).
    warmup_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    warmup_ids = tokenizer.encode(warmup_text, return_tensors="pt").to(device)
    for _, fn in methods.values():
        _ = fn(warmup_ids)

    tasks = resolve_tasks(args)
    summary = {
        "config": {
            "target": args.target,
            "dflash_draft": args.dflash_draft,
            "dspark_draft": args.dspark_draft,
            "dflash_block_size": int(dflash.block_size),
            "dspark_block_size": int(dspark.block_size),
            "tree_budget": args.tree_budget,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "tasks": tasks,
            "metric": "mean_acceptance_length",
            "methods": {mid: all_methods[mid][0] for mid in methods},
        },
        "results": {},
    }

    for dataset_name, max_samples in tasks:
        per_method_lengths, n = run_dataset(dataset_name, max_samples, methods, tokenizer, device)
        summary["results"][dataset_name] = {
            mid: {
                "label": all_methods[mid][0],
                "mean_accept": (mean(v) if v else 0.0),
                "rounds": len(v),
            }
            for mid, v in per_method_lengths.items()
        }
        print(f"\nDataset: {dataset_name}  samples: {n}  tree_budget: {args.tree_budget}")
        print(f"{'cfg':<4}{'method':<24}{'mean_accept':>12}{'rounds':>10}")
        print("-" * 50)
        for mid in methods:
            r = summary["results"][dataset_name][mid]
            print(f"{mid:<4}{r['label']:<24}{r['mean_accept']:>12.3f}{r['rounds']:>10}")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved results to {args.save_json}")


if __name__ == "__main__":
    main()
