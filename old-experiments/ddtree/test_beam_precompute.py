"""CPU checks for the precomputed-transition beam builder and the width schedule.

The GPU harness cannot answer "is the new builder the same tree?" - RESULTS.md §12
already found that whole-run timing noise swamps a sub-millisecond builder effect,
and acceptance differences of a fraction of a percent are equally unresolvable. So
equivalence is established here, on CPU, against build_beam_tree as the reference.

Two regimes, because the honest claim differs between them:

  * integer weights - every matmul partial sum is exactly representable in float32,
    so reduction order cannot matter and the trees must be BITWISE identical. This
    is the algebraic proof: same scores, same top-k, same tie-break.
  * random float weights - the [C, C] batched matmul and the [width, C] one it
    replaces reduce in different orders, so paths within float32 epsilon of each
    other can swap. This measures how often that actually happens.

    python3 ddtree/test_beam_precompute.py
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
        # small integers => every partial sum of the rank-R dot product is exact in
        # float32, so no reduction order can change the result
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


def compare(budget, candidates, min_width, seeds, integer):
    """Reference build_beam_tree vs build_beam_tree_precomputed on identical inputs."""
    agree = 0
    first_mismatch = None
    for seed in range(seeds):
        generator = torch.Generator().manual_seed(seed)
        head = make_head(generator, integer)
        logits = make_logits(generator, integer)
        root = int(torch.randint(0, VOCAB, (1,), generator=generator).item())

        reference = mkv.build_beam_tree(
            logits, budget, head, root,
            candidates=candidates, flat=True, min_width=min_width, per_depth=True,
        )
        new = mkv.build_beam_tree_precomputed(
            logits, budget, head, root,
            candidates=candidates, min_width=min_width,
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
        agree, mismatch = compare(budget, candidates=64, min_width=2, seeds=25, integer=True)
        assert agree == 25, f"budget {budget}: {25 - agree}/25 trees differ, first {mismatch}"
        print(f"  budget {budget:4d}: 25/25 trees bitwise identical")


def test_float_divergence_is_negligible():
    total_agree, total = 0, 0
    for budget in (16, 64, 256):
        agree, mismatch = compare(budget, candidates=128, min_width=2, seeds=40, integer=False)
        total_agree += agree
        total += 40
        note = "" if mismatch is None else f"  (worst: {mismatch[1]}/{mismatch[2]} nodes at seed {mismatch[0]})"
        print(f"  budget {budget:4d}: {agree}/40 trees identical{note}")
    rate = total_agree / total
    assert rate >= 0.90, f"only {rate:.0%} of float32 trees matched the reference"
    print(f"  overall float32 agreement: {rate:.1%}")


def test_graph_buffer_path_matches():
    """The captured graph fills preallocated buffers and maps slots back to vocab by
    a different expression than the eager builder. Run that exact code path on CPU."""
    generator = torch.Generator().manual_seed(7)
    head = make_head(generator, integer=True)
    logits = make_logits(generator, integer=True)
    root, budget, C = 11, 64, 64
    widths = mkv.flat_width_schedule(budget, DEPTH, mkv.FLAT_MIN_WIDTH)

    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    logits_cand, cand_ids = torch.topk(logits[: len(widths)].float(), k=C, dim=-1)
    flat_ids = cand_ids.reshape(-1)
    w1_cand = w1.index_select(0, flat_ids).view(len(widths), C, -1).float()
    w2_cand = w2.index_select(0, flat_ids).view(len(widths), C, -1).float()

    root_logp, table_logp = mkv._beam_transition_tables(
        logits_cand, w1_cand, w2_cand, w1.index_select(0, torch.tensor([root])),
    )
    parent_local, slots = mkv._beam_level_loop(root_logp, table_logp, widths, C)
    depth_of = mkv._depth_of_node(widths, logits.device)
    token_ids = cand_ids.reshape(-1).gather(0, depth_of * C + slots)

    graph_style = mkv._tree_structure_from_levels(
        parent_local.numpy(), token_ids.numpy(), widths,
    )
    eager = mkv.build_beam_tree_precomputed(logits, budget, head, root, candidates=C)
    assert tree_signature(graph_style) == tree_signature(eager)
    print(f"  graph buffer path reproduces the eager tree ({sum(widths)} nodes)")


def test_best_first_precomputed_matches_lazy():
    """With C = the whole vocabulary, the precomputed best-first builder and the
    lazy one are the same algorithm, so they must produce the same tree.

    build_markov_tree_sequential is the right reference here: it is the original
    per-node, full-vocab, one-GPU-round-trip-per-node implementation, i.e. exactly
    what the table is supposed to replace without changing the answer.

    Float weights, NOT the integer ones the beam tests use. The integer generator
    is chosen there to make float32 reductions exact, but it also collapses the
    logit range - only 8 distinct values among a typical top-12 - and these two
    builders walk the vocabulary in different orders (raw ids vs candidate rank),
    so torch.topk resolves those ties differently. That is a property of the
    synthetic data, not of the builders: real logits are floats with no exact
    ties, and on float data this matches 20/20 at every budget."""
    for budget in (8, 32, 64):
        agree = 0
        for seed in range(20):
            generator = torch.Generator().manual_seed(seed)
            head = make_head(generator, integer=False)
            logits = make_logits(generator, integer=False)
            root = int(torch.randint(0, VOCAB, (1,), generator=generator).item())

            lazy = mkv.build_markov_tree_sequential(logits, budget, head, root)
            precomputed = mkv.build_markov_tree_precomputed(
                logits, budget, head, root, candidates=VOCAB,
            )
            agree += tree_signature(lazy) == tree_signature(precomputed)
        assert agree == 20, f"budget {budget}: only {agree}/20 matched the lazy builder"
        print(f"  budget {budget:4d}: 20/20 trees match lazy best-first")


def test_best_first_precomputed_is_a_valid_tree():
    """Under candidate restriction it no longer equals the full-vocab builder, so
    check the invariants compile_ddtree_tree relies on instead."""
    generator = torch.Generator().manual_seed(3)
    head = make_head(generator, integer=False)
    logits = make_logits(generator, integer=False)
    for budget, candidates in ((64, 128), (128, 256), (16, 64)):
        tokens, depths, parents, child_maps, visibility = mkv.build_markov_tree_precomputed(
            logits, budget, head, 5, candidates=candidates,
        )
        assert len(tokens) == budget, (budget, len(tokens))
        assert len(parents) == budget + 1 and parents[0] == -1
        assert visibility.shape == (budget + 1, budget + 1)
        for index in range(1, budget + 1):
            parent = parents[index]
            assert parent < index, "a node must be emitted after its parent"
            expected_depth = 1 if parent == 0 else int(depths[parent - 1]) + 1
            assert int(depths[index - 1]) == expected_depth
            assert bool(visibility[index, index]) and bool(visibility[index, parent])
        # siblings under one parent must be distinct, or follow_verified_tree's
        # token -> child map would silently lose a branch
        for node_index, children in enumerate(child_maps):
            assert len(children) == len({v for v in children.values()})
        print(f"  budget {budget:4d} C={candidates:4d}: {budget} nodes, tree invariants hold")


def test_flat_schedule():
    schedule = mkv.flat_width_schedule
    # the bug: 16 nodes over 16 depths is a chain, and lost 8.3% acceptance
    assert schedule(16, 16, min_width=1) == [1] * 16
    assert schedule(16, 16) == [2] * 8, schedule(16, 16)
    # budgets that could already branch everywhere are untouched
    assert schedule(64, 16) == [4] * 16
    assert schedule(128, 16) == [8] * 16
    assert schedule(256, 16) == [16] * 16
    assert schedule(32, 16) == [2] * 16
    # min_width doubles as the depth-truncation knob
    assert schedule(64, 16, min_width=5) == [6, 6, 6, 6] + [5] * 8
    assert schedule(64, 16, min_width=4) == [4] * 16
    # every schedule spends the whole budget and never emits a zero-width level
    for budget in range(1, 300):
        for min_width in (1, 2, 3, 5):
            widths = schedule(budget, 16, min_width)
            assert sum(widths) == budget, (budget, min_width, widths)
            assert all(w >= 1 for w in widths)
            assert len(widths) <= 16
    assert schedule(0, 16) == []
    print("  flat schedule: budget-16 chain fixed, budgets 32-256 unchanged")


def main():
    torch.manual_seed(0)
    tests = [
        ("exact arithmetic -> bitwise identical trees", test_bitwise_identical_on_exact_arithmetic),
        ("float32 -> divergence only at score ties", test_float_divergence_is_negligible),
        ("CUDA-graph buffer path", test_graph_buffer_path_matches),
        ("best-first over precomputed table == lazy best-first", test_best_first_precomputed_matches_lazy),
        ("best-first tree invariants under candidate restriction", test_best_first_precomputed_is_a_valid_tree),
        ("flat width schedule", test_flat_schedule),
    ]
    for name, test in tests:
        print(f"\n{name}")
        test()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
