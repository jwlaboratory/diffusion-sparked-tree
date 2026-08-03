"""Isolated tree-builder microbenchmark: ms/round and candidate-set fidelity.

RESULTS.md §12 established that the end-to-end harness cannot measure this stage.
The builder costs ~1-4 ms/round against a ~500 ms run; six independently-scheduled
containers carry ~±12% timing variance, which swamps the entire effect. Any claim
about builder cost has to come from here, not from a sweep.

Logits are real: a short generate is run first with a forward hook on the drafter's
lm_head, so every builder is timed on the same recorded distributions the decoder
actually produced. That matters for fidelity - candidate restriction is only safe
if the markov bias cannot lift a token from outside the raw top-C into the beam,
which is a property of real logit spreads, not random ones.

    python bench_tree_build.py --dspark-name-or-path <ckpt> --budgets 16,64,128,256
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM

from model import DSparkDraftModel, load_and_process_dataset
from ddtree_markov import (
    FLAT_MIN_WIDTH,
    build_beam_tree,
    build_beam_tree_graphed,
    build_beam_tree_precomputed,
    build_markov_tree,
    build_markov_tree_precomputed,
    ddtree_markov_generate,
    flat_width_schedule,
)


def record_base_logits(model, target, tokenizer, dataset, num_rounds, block_size):
    """Real per-round base_logits [block_size, V], captured from a live decode."""
    recorded = []

    def hook(module, inputs, output):
        if len(recorded) < num_rounds:
            recorded.append(output[0].detach().clone())

    handle = model.lm_head.register_forward_hook(hook)
    try:
        for instance in dataset:
            if len(recorded) >= num_rounds:
                break
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": instance["turns"][0]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            input_ids = tokenizer.encode(text, return_tensors="pt").to(target.device)
            ddtree_markov_generate(
                model=model, target=target, input_ids=input_ids,
                max_new_tokens=256, block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
                tree_budget=64, tree_mode="beam", beam_flat=True,
            )
    finally:
        handle.remove()
    return recorded[:num_rounds]


def time_builder(fn, logits_batch, warmup=5):
    """Mean ms per round, plus the tree each round produced.

    The root is taken as the drafter's own top depth-1 prediction. The decoder's
    real root is the last accepted token, which the lm_head hook does not see, but
    both are ordinary in-distribution tokens - and the root only ever enters as one
    row of markov_w1, so it changes which bias is read, not how much work is done.
    """
    for logits in logits_batch[:warmup]:
        fn(logits, int(logits[0].argmax()))

    torch.cuda.synchronize()
    per_round, trees = [], []
    for logits in logits_batch:
        root = int(logits[0].argmax())
        torch.cuda.synchronize()
        started = time.perf_counter()
        tree = fn(logits, root)
        torch.cuda.synchronize()
        per_round.append(1e3 * (time.perf_counter() - started))
        trees.append(tree)
    per_round.sort()
    return {
        "mean_ms": sum(per_round) / len(per_round),
        "p50_ms": per_round[len(per_round) // 2],
        "p90_ms": per_round[int(0.9 * len(per_round))],
    }, trees


def tree_key(tree):
    node_token_ids, node_depths, parents, _, _ = tree
    return (tuple(node_token_ids.tolist()), tuple(node_depths.tolist()), tuple(parents))


def node_overlap(reference, other):
    """Fraction of the reference tree's (depth, token) pairs the other tree also has.

    Compared as a set rather than positionally: a candidate cut that reorders two
    equally-scored siblings has not lost anything, one that drops a node has.
    """
    def pairs(tree):
        return {(int(d), int(t)) for d, t in zip(tree[1].tolist(), tree[0].tolist())}

    ref = pairs(reference)
    return len(ref & pairs(other)) / max(len(ref), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dspark-name-or-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--budgets", type=str, default="16,64,128,256")
    parser.add_argument("--candidates", type=str, default="256,512,1024,2048")
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--fidelity-rounds", type=int, default=25)
    parser.add_argument("--min-width", type=int, default=FLAT_MIN_WIDTH)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    model = DSparkDraftModel.from_pretrained(
        args.dspark_name_or_path, attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset).shuffle(seed=0).select(range(4))

    block_size = model.block_size
    print(f"recording {args.rounds} rounds of real base_logits (block {block_size})...", flush=True)
    logits_batch = record_base_logits(model, target, tokenizer, dataset, args.rounds, block_size)
    print(f"recorded {len(logits_batch)} rounds", flush=True)

    head = model.markov_head
    budgets = [int(b) for b in args.budgets.split(",")]
    candidate_sizes = [int(c) for c in args.candidates.split(",")]

    results = {"timing": {}, "fidelity": {}, "schedules": {}}

    for budget in budgets:
        widths = flat_width_schedule(budget, block_size, args.min_width)
        results["schedules"][budget] = widths
        print(f"\n===== budget {budget} · widths {widths} =====", flush=True)

        variants = {}
        for candidates in candidate_sizes:
            variants[f"loop_C{candidates}"] = (
                lambda lg, rt, c=candidates: build_beam_tree(
                    lg, budget, head, rt, candidates=c, flat=True,
                    min_width=args.min_width, per_depth=True,
                )
            )
            variants[f"precomputed_C{candidates}"] = (
                lambda lg, rt, c=candidates: build_beam_tree_precomputed(
                    lg, budget, head, rt, candidates=c, min_width=args.min_width,
                )
            )
            variants[f"graphed_C{candidates}"] = (
                lambda lg, rt, c=candidates: build_beam_tree_graphed(
                    lg, budget, head, rt, candidates=c, min_width=args.min_width,
                )
            )
            # best-first against the same precomputed table. Shares the beam's
            # [L-1, C, C] cost but keeps adaptive shape, so this row is the whole
            # question: does it land near the beam's ms/round or near the lazy
            # builder's? Acceptance is worth ~3% (11.45 vs 11.09) if it does.
            variants[f"bestfirst_precomputed_C{candidates}"] = (
                lambda lg, rt, c=candidates: build_markov_tree_precomputed(
                    lg, budget, head, rt, candidates=c,
                )
            )
        # the shipped pre-change path: in-loop matmul over a deduped union of
        # per-depth top-2048, which is what every number in RESULTS.md was measured on
        variants["baseline_union_C2048"] = (
            lambda lg, rt: build_beam_tree(
                lg, budget, head, rt, candidates=2048, flat=True, min_width=1,
            )
        )
        # the acceptance reference the precomputed best-first has to match: lazy
        # best-first at the candidate set RESULTS.md §4 measured (11.45 @ 2048)
        variants["bestfirst_lazy_C2048"] = (
            lambda lg, rt: build_markov_tree(
                lg, budget, head, rt, exact=True, candidates=2048,
            )
        )

        row, trees_by_variant = {}, {}
        for name, fn in variants.items():
            timing, trees = time_builder(fn, logits_batch)
            row[name] = timing
            trees_by_variant[name] = trees
            print(f"  {name:24s} {timing['mean_ms']:7.3f} ms/round  (p50 {timing['p50_ms']:.3f}, p90 {timing['p90_ms']:.3f})", flush=True)
        results["timing"][budget] = row

        # Does the precompute change the tree? Same candidate set, same schedule -
        # the only difference is matmul reduction order, so any mismatch is a tie.
        agreement = {}
        for candidates in candidate_sizes:
            loop = trees_by_variant[f"loop_C{candidates}"]
            precomputed = trees_by_variant[f"precomputed_C{candidates}"]
            graphed = trees_by_variant[f"graphed_C{candidates}"]
            agreement[f"precomputed_vs_loop_C{candidates}"] = sum(
                tree_key(a) == tree_key(b) for a, b in zip(loop, precomputed)
            ) / len(loop)
            agreement[f"graphed_vs_precomputed_C{candidates}"] = sum(
                tree_key(a) == tree_key(b) for a, b in zip(precomputed, graphed)
            ) / len(precomputed)
        results["fidelity"][budget] = agreement

        # How much tree does adaptive shape actually buy? Set overlap against the
        # lazy best-first reference, for the beam and for the precomputed
        # best-first that is supposed to reproduce it exactly.
        lazy = trees_by_variant["bestfirst_lazy_C2048"]
        shape_gap = {}
        for candidates in candidate_sizes:
            for kind, key in (("bestfirst", f"bestfirst_precomputed_C{candidates}"),
                              ("beam", f"precomputed_C{candidates}")):
                trees = trees_by_variant[key]
                shape_gap[f"{kind}_C{candidates}_vs_lazy"] = sum(
                    node_overlap(a, b) for a, b in zip(lazy, trees)
                ) / len(lazy)
        results["shape_gap"] = results.get("shape_gap", {})
        results["shape_gap"][budget] = shape_gap
        for name, value in shape_gap.items():
            print(f"  {name:36s} {value:6.1%} node overlap vs lazy best-first", flush=True)
        for name, value in agreement.items():
            print(f"  {name:36s} {value:6.1%} identical", flush=True)

    # Candidate-set fidelity against the exact full-vocab beam. Separate loop and
    # fewer rounds: the reference reads all of markov_w2 (~155MB fp32) per level.
    print(f"\n===== candidate fidelity vs full-vocab beam ({args.fidelity_rounds} rounds) =====", flush=True)
    fidelity_budget = budgets[-1]
    subset = logits_batch[: args.fidelity_rounds]
    exact_trees = [
        build_beam_tree(lg, fidelity_budget, head, int(lg[0].argmax()),
                        candidates=0, flat=True, min_width=args.min_width)
        for lg in subset
    ]
    results["candidate_fidelity"] = {"budget": fidelity_budget}
    for candidates in candidate_sizes:
        trees = [
            build_beam_tree_precomputed(lg, fidelity_budget, head, int(lg[0].argmax()),
                                        candidates=candidates, min_width=args.min_width)
            for lg in subset
        ]
        overlaps = [node_overlap(a, b) for a, b in zip(exact_trees, trees)]
        exact_match = sum(tree_key(a) == tree_key(b) for a, b in zip(exact_trees, trees)) / len(trees)
        mean_overlap = sum(overlaps) / len(overlaps)
        results["candidate_fidelity"][f"C{candidates}"] = {
            "node_overlap": mean_overlap,
            "exact_tree_match": exact_match,
            "worst_round_overlap": min(overlaps),
        }
        print(f"  C={candidates:5d}  node overlap {mean_overlap:6.2%}  "
              f"exact tree {exact_match:6.1%}  worst round {min(overlaps):6.2%}", flush=True)

    if args.save_path:
        with open(args.save_path, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nsaved {args.save_path}", flush=True)
    return results


if __name__ == "__main__":
    main()
