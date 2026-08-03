"""CPU checks for the precomputed-transition best-first builder.

The claim under test is narrow and load-bearing: build_markov_tree_precomputed
must produce the SAME tree as build_markov_tree, not a near one. That is the whole
difference between it and wave mode, which also cut syncs and lost 3.3 acceptance
points (FINDINGS §9) precisely because it perturbed pop order. A builder that is
"almost best-first" has no reason to beat the beam, so approximate agreement here
would falsify the idea rather than support it.

Two regimes, same split as test_beam_precompute.py:

  * integer weights - every matmul partial sum is exactly representable in float32,
    so reduction order cannot matter and the trees must be BITWISE identical. This
    is the algebraic proof that the [L-1, C, C] table is the same object the lazy
    builder computes one row at a time.
  * random float weights - the [C, C] batched matmul and the [M, 1, R] x [M, R, C]
    one it replaces reduce in different orders, so paths within float32 epsilon can
    swap. This measures how often that actually happens.

    python3 ddtree/test_markov_precompute.py
"""

import sys
import types
from types import SimpleNamespace

import torch


def _stub_heavy_imports() -> None:
    """ddtree_markov pulls in transformers and the sibling decoding loops at import
    time; none of it is reachable from the tree builders under test."""
    for name, attrs in {
        "transformers": ("AutoModelForCausalLM", "DynamicCache"),
        "model": ("DSparkDraftModel", "sample", "extract_context_feature"),
        "dflash": ("empty_stage_times",),
        "ddtree": ("build_ddtree_tree", "compile_ddtree_tree", "follow_verified_tree",
                   "compact_dynamic_cache"),
    }.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, object)
        sys.modules[name] = module


_stub_heavy_imports()

import ddtree_markov as mkv  # noqa: E402


VOCAB, RANK, DEPTH = 2000, 8, 16


def make_head(generator, integer: bool):
    """A vanilla markov head: bias(prev) = markov_w2(markov_w1[prev])."""
    if integer:
        w1 = torch.randint(-3, 4, (VOCAB, RANK), generator=generator).float()
        w2 = torch.randint(-3, 4, (VOCAB, RANK), generator=generator).float()
    else:
        w1 = torch.randn(VOCAB, RANK, generator=generator)
        w2 = torch.randn(VOCAB, RANK, generator=generator) * 0.3
    head = SimpleNamespace()
    head.markov_w1 = SimpleNamespace(weight=w1)
    head.markov_w2 = SimpleNamespace(weight=w2)
    return head


def make_logits(generator, integer: bool):
    if integer:
        return torch.randint(-40, 40, (DEPTH, VOCAB), generator=generator).float()
    return torch.randn(DEPTH, VOCAB, generator=generator) * 4.0


def tree_signature(tree):
    node_token_ids, node_depths, parents, _, visibility = tree
    return (
        node_token_ids.tolist(),
        node_depths.tolist(),
        list(parents),
        visibility.tolist(),
    )


def compare(budget, candidates, seeds, integer):
    """Reference build_markov_tree vs build_markov_tree_precomputed, same inputs.

    `per_depth=True` on the reference is the equivalence hook: without it the lazy
    builder restricts candidates to a deduped union across depths, which is a
    different candidate set and would make any mismatch uninterpretable.
    """
    agree = 0
    first_mismatch = None
    for seed in range(seeds):
        generator = torch.Generator().manual_seed(seed)
        head = make_head(generator, integer)
        logits = make_logits(generator, integer)
        root = int(torch.randint(0, VOCAB, (1,), generator=generator).item())

        reference = mkv.build_markov_tree(
            logits, budget, head, root, exact=True, candidates=candidates, per_depth=True,
        )
        new = mkv.build_markov_tree_precomputed(
            logits, budget, head, root, candidates=candidates,
        )
        if tree_signature(reference) == tree_signature(new):
            agree += 1
        elif first_mismatch is None:
            ref_tokens, new_tokens = reference[0].tolist(), new[0].tolist()
            differing = [i for i, (a, b) in enumerate(zip(ref_tokens, new_tokens)) if a != b]
            first_mismatch = (seed, len(differing), len(ref_tokens))
    return agree, first_mismatch


def test_bitwise_identical_on_exact_arithmetic():
    for budget in (16, 32, 64, 128):
        agree, mismatch = compare(budget, candidates=64, seeds=25, integer=True)
        assert agree == 25, f"budget {budget}: {25 - agree}/25 trees differ, first {mismatch}"
        print(f"  budget {budget:4d}: 25/25 trees bitwise identical")


def test_float_divergence_is_negligible():
    total_agree, total = 0, 0
    for budget in (16, 64, 256):
        agree, mismatch = compare(budget, candidates=128, seeds=40, integer=False)
        total_agree += agree
        total += 40
        note = "" if mismatch is None else f"  (worst: {mismatch[1]}/{mismatch[2]} nodes at seed {mismatch[0]})"
        print(f"  budget {budget:4d}: {agree}/40 trees identical{note}")
    rate = total_agree / total
    assert rate >= 0.90, f"only {rate:.0%} of float32 trees matched the reference"
    print(f"  overall float32 agreement: {rate:.1%}")


def test_budget_exceeding_candidate_set():
    """topk is capped by C, not by budget. The lazy builder's sibling guard used to
    read `rank + 1 < topk` against a table only `active_topk` long, which is
    unreachable at the shipped candidates=2048 but not once C is a few hundred."""
    for budget, candidates in ((128, 32), (256, 64), (64, 16)):
        agree, mismatch = compare(budget, candidates=candidates, seeds=10, integer=True)
        assert agree == 10, f"budget {budget}/C {candidates}: differ, first {mismatch}"
        print(f"  budget {budget:4d} vs C {candidates:4d}: 10/10 identical (no overrun)")


def test_lazy_builder_default_path_unchanged():
    """Blast-radius check: adding `per_depth` touched build_markov_tree's shared
    candidate setup and its sibling guard, so the default path has to still match
    build_markov_tree_sequential - the untouched reference this file's subject
    ultimately descends from."""
    for budget in (16, 64, 128):
        for seed in range(6):
            generator = torch.Generator().manual_seed(seed)
            head = make_head(generator, integer=True)
            logits = make_logits(generator, integer=True)
            root = int(torch.randint(0, VOCAB, (1,), generator=generator).item())

            sequential = mkv.build_markov_tree_sequential(logits, budget, head, root)
            batched = mkv.build_markov_tree(logits, budget, head, root, exact=True, candidates=0)
            assert tree_signature(sequential) == tree_signature(batched), (budget, seed)
        print(f"  budget {budget:4d}: 6/6 match build_markov_tree_sequential")


def test_tree_is_not_the_beam_tree():
    """Sanity: best-first and beam are different algorithms on the same inputs.

    If this ever passed, the precomputed builder would have silently become a beam
    and the acceptance case for it would evaporate."""
    generator = torch.Generator().manual_seed(3)
    head = make_head(generator, integer=False)
    logits = make_logits(generator, integer=False)
    root, budget, candidates = 41, 64, 128

    best_first = mkv.build_markov_tree_precomputed(logits, budget, head, root, candidates=candidates)
    beam = mkv.build_beam_tree_precomputed(logits, budget, head, root, candidates=candidates)

    assert tree_signature(best_first) != tree_signature(beam)
    depths = best_first[1].tolist()
    shape = [depths.count(d) for d in range(1, DEPTH + 1)]
    print(f"  best-first depth profile {shape} vs beam's flat {mkv.flat_width_schedule(budget, DEPTH)}")


def test_degenerate_inputs():
    generator = torch.Generator().manual_seed(5)
    head = make_head(generator, integer=True)
    logits = make_logits(generator, integer=True)

    empty = mkv.build_markov_tree_precomputed(logits, 0, head, 7)
    assert empty[0].numel() == 0 and list(empty[2]) == [-1]

    single = mkv.build_markov_tree_precomputed(logits, 1, head, 7, candidates=32)
    assert single[0].numel() == 1 and single[1].tolist() == [1]

    # every node accounted for, no depth beyond the drafter's horizon
    full = mkv.build_markov_tree_precomputed(logits, 128, head, 7, candidates=64)
    assert full[0].numel() == 128
    assert max(full[1].tolist()) <= DEPTH
    assert full[4].shape == (129, 129)

    # candidates=0 means full vocab, matching build_markov_tree's convention
    unrestricted = mkv.build_markov_tree_precomputed(logits, 32, head, 7, candidates=0)
    assert unrestricted[0].numel() == 32

    # depth_limit 1 leaves table_logp with an empty batch dim; the walk must still
    # emit a flat level rather than index into it
    single_depth = mkv.build_markov_tree_precomputed(logits[:1], 16, head, 7, candidates=32)
    assert set(single_depth[1].tolist()) == {1} and single_depth[0].numel() == 16

    print("  budget 0 / 1 / 128, full vocab, and depth_limit 1 all behave")


def main():
    torch.manual_seed(0)
    tests = [
        ("exact arithmetic -> bitwise identical trees", test_bitwise_identical_on_exact_arithmetic),
        ("float32 -> divergence only at score ties", test_float_divergence_is_negligible),
        ("budget larger than the candidate set", test_budget_exceeding_candidate_set),
        ("lazy builder's default path unchanged", test_lazy_builder_default_path_unchanged),
        ("best-first is still not a beam", test_tree_is_not_the_beam_tree),
        ("degenerate inputs", test_degenerate_inputs),
    ]
    for name, test in tests:
        print(f"\n{name}")
        test()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
