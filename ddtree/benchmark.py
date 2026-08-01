import argparse
import random
from itertools import chain
from pathlib import Path

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from model import DFlashDraftModel, DSparkDraftModel, load_and_process_dataset
from dflash import dflash_generate
from ddtree import ddtree_generate, maybe_enable_cpp_compact
from dspark import dspark_generate
from ddtree_markov import ddtree_markov_generate, ddtree_cross_markov_generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--dspark-name-or-path", type=str, default=None, help="e.g. deepseek-ai/dspark_qwen3_4b_block7; enables the dspark method")
    parser.add_argument("--minimal", action="store_true", help="only baseline/dflash/dspark/ddtree/sparked-tree (skip nomkv, wave, xmkv ablations)")
    parser.add_argument("--collect-tree-stats", action="store_true", help="record (depth, slot) of every accepted node for width-schedule tuning")
    parser.add_argument("--tree-mode", type=str, default="exact", choices=["exact", "beam"], help="markov tree builder: best-first (exact) or level-synchronous beam")
    parser.add_argument("--beam-widths", type=str, default=None, help="explicit beam width per depth, e.g. 2,3,7,7,5 (overrides --beam-decay)")
    parser.add_argument("--beam-decay", type=float, default=0.6, help="beam width decay per depth")
    parser.add_argument("--tree-candidates", type=int, default=2048, help="markov tree: restrict per-depth candidates before applying bias; 0 = full vocab")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="DSpark confidence head threshold; 0 disables proposal truncation")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--tree-budget", type=str, default="16,32,64,128,256,512,1024")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    maybe_enable_cpp_compact(not args.disable_cpp_compact_cache)

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    installed_flash_attn = has_flash_attn()
    target_attn_implementation = "flash_attention_2" if args.flash_attn else "sdpa"
    draft_attn_implementation = "flash_attention_2"
    if not installed_flash_attn:
        if args.flash_attn:
            raise RuntimeError("--flash-attn requested but flash_attn is not installed")
        logger.warning("flash_attn not installed; draft models fall back to sdpa (slower draft stage, same outputs).")
        draft_attn_implementation = "sdpa"

    if not args.flash_attn and installed_flash_attn:
        logger.warning("DDTree uses a custom tree attention mask on the target model. For compatibility, forcing the target verifier to torch.sdpa.")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=target_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=draft_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    dspark_model = None
    if args.dspark_name_or_path is not None:
        dspark_model = DSparkDraftModel.from_pretrained(
            args.dspark_name_or_path,
            attn_implementation=draft_attn_implementation,
            dtype=torch.bfloat16,
        ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tree_budgets = [int(tree_budget) for tree_budget in args.tree_budget.split(",")]
    methods_to_run = ["dflash"]
    method_key_to_tree_budget = {}
    if not args.flash_attn:
        ddtree_method_keys = [f"ddtree_tb{tree_budget}" for tree_budget in tree_budgets]
        methods_to_run.extend(ddtree_method_keys)
        method_key_to_tree_budget.update({f"ddtree_tb{tree_budget}": tree_budget for tree_budget in tree_budgets})
    if dspark_model is not None:
        methods_to_run.append("dspark")
        if not args.flash_attn:
            # dspark-drafter trees: independence-assumed ablation vs markov-guided
            for tree_budget in tree_budgets:
                methods_to_run.append(f"dsparktree_markov_tb{tree_budget}")
                method_key_to_tree_budget[f"dsparktree_markov_tb{tree_budget}"] = tree_budget
                if args.minimal:
                    continue
                methods_to_run.append(f"dsparktree_nomkv_tb{tree_budget}")
                method_key_to_tree_budget[f"dsparktree_nomkv_tb{tree_budget}"] = tree_budget
                # same tree scoring, batched-sync builder (~7x fewer GPU syncs)
                methods_to_run.append(f"dsparktree_wave_tb{tree_budget}")
                method_key_to_tree_budget[f"dsparktree_wave_tb{tree_budget}"] = tree_budget
                # DDTree proper (DFlash drafter, full horizon) + DSpark's markov head as guidance
                methods_to_run.append(f"ddtree_xmkv_tb{tree_budget}")
                method_key_to_tree_budget[f"ddtree_xmkv_tb{tree_budget}"] = tree_budget

    beam_flat = args.beam_widths == "flat"
    beam_widths = None if (not args.beam_widths or beam_flat) else [int(w) for w in args.beam_widths.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    warmup_input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_input_ids = tokenizer.encode(warmup_input_text, return_tensors="pt").to(target.device)
    warmup_max_new_tokens = min(args.max_new_tokens, 16)

    _ = dflash_generate(
        model=draft_model,
        target=target,
        input_ids=warmup_input_ids,
        mask_token_id=draft_model.mask_token_id,
        max_new_tokens=warmup_max_new_tokens,
        block_size=1,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
    )
    for method_key in methods_to_run:
        if method_key == "dflash":
            _ = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
        elif method_key == "dspark":
            _ = dspark_generate(
                model=dspark_model,
                target=target,
                input_ids=warmup_input_ids,
                max_new_tokens=warmup_max_new_tokens,
                block_size=dspark_model.block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                confidence_threshold=args.confidence_threshold,
            )
        elif method_key.startswith("ddtree_xmkv_"):
            _ = ddtree_cross_markov_generate(
                model=draft_model,
                markov_head=dspark_model.markov_head,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                tree_budget=method_key_to_tree_budget[method_key],
            )
        elif method_key.startswith("dsparktree_"):
            _ = ddtree_markov_generate(
                model=dspark_model,
                target=target,
                input_ids=warmup_input_ids,
                max_new_tokens=warmup_max_new_tokens,
                block_size=dspark_model.block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                tree_budget=method_key_to_tree_budget[method_key],
                use_markov=("markov" in method_key) or ("wave" in method_key),
                tree_exact="wave" not in method_key,
                tree_candidates=args.tree_candidates,
                tree_mode=args.tree_mode,
                beam_decay=args.beam_decay,
                beam_widths=beam_widths,
                beam_flat=beam_flat,
            )
        else:
            _ = ddtree_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                tree_budget=method_key_to_tree_budget[method_key],
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            response["baseline"] = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=1,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
            for method_key in methods_to_run:
                if method_key == "dflash":
                    response[method_key] = dflash_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )
                elif method_key == "dspark":
                    response[method_key] = dspark_generate(
                        model=dspark_model,
                        target=target,
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        block_size=dspark_model.block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                        confidence_threshold=args.confidence_threshold,
                    )
                elif method_key.startswith("ddtree_xmkv_"):
                    response[method_key] = ddtree_cross_markov_generate(
                        model=draft_model,
                        markov_head=dspark_model.markov_head,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                        tree_budget=method_key_to_tree_budget[method_key],
                    )
                elif method_key.startswith("dsparktree_"):
                    response[method_key] = ddtree_markov_generate(
                        model=dspark_model,
                        target=target,
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        block_size=dspark_model.block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                        tree_budget=method_key_to_tree_budget[method_key],
                        use_markov=("markov" in method_key) or ("wave" in method_key),
                        tree_exact="wave" not in method_key,
                        tree_candidates=args.tree_candidates,
                        tree_mode=args.tree_mode,
                        beam_decay=args.beam_decay,
                        beam_widths=beam_widths,
                        beam_flat=beam_flat,
                        collect_stats=args.collect_tree_stats,
                    )
                else:
                    response[method_key] = ddtree_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        tree_budget=method_key_to_tree_budget[method_key],
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )

            spec_response = response[methods_to_run[-1]]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    run_data = {
        "responses": responses,
        "block_size": block_size,
        "draft_attn_implementation": draft_attn_implementation,
        "target_attn_implementation": target_attn_implementation,
        "args": vars(args),
    }
    
    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run_data, save_path)


if __name__ == "__main__":
    main()
