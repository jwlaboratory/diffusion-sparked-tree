"""Tests for the tree -> SGLang verify-format bridge.

Two layers:

  * CPU (this file, runs anywhere) -- the mask really encodes the tree, over
    many topologies, including the shapes our builders actually emit.
  * GPU (test_kernel_agreement.py) -- SGLang's own
    `reconstruct_indices_from_tree_mask` recovers that same tree from our mask.

The CPU layer alone cannot catch a convention mismatch with SGLang; the GPU
layer is what closes that. Run both before trusting the plugin.

    python3 concurrency/test_sparked_bridge.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparked_bridge import (  # noqa: E402
    batch_full_mask,
    batch_qlen_mask,
    parents_from_retrieve,
    parents_from_visibility,
    positions_from_depths,
    tree_to_draft_tokens,
    tree_to_qlen_mask,
    visibility_from_parents,
)


# visibility_from_parents lives in sparked_bridge so production and tests share
# one definition of the convention; re-exported above.


def depths_from_parents(parents: list[int]) -> torch.Tensor:
    depths = [0] * len(parents)
    for i in range(1, len(parents)):
        depths[i] = depths[parents[i]] + 1
    return torch.tensor(depths[1:], dtype=torch.int64)


# --- topology generators: the shapes that actually occur -------------------

def chain(n):
    return [-1] + list(range(n - 1))


def star(n):
    return [-1] + [0] * (n - 1)


def balanced(n, fanout=2):
    return [-1] + [(i - 1) // fanout for i in range(1, n)]


def best_first_like(n, rng):
    """Irregular, parent always earlier than child -- the shape a best-first
    heap emits, and the case `tree_topk = -1` exists for."""
    return [-1] + [rng.randrange(0, i) for i in range(1, n)]


def deep_narrow(n):
    """min_width=2 schedule: pairs of siblings down a long spine."""
    parents = [-1]
    for i in range(1, n):
        parents.append(max(0, (i - 1) // 2))
    return parents


TOPOLOGIES = {
    "chain": chain, "star": star, "balanced": balanced,
    "deep_narrow": deep_narrow,
}


def check_tree(parents, draft_token_num=None, seq_len=137, label=""):
    n = len(parents)
    N = draft_token_num or n
    vis = visibility_from_parents(parents)

    # 1. the mask encodes the tree: parent recovered from the mask alone
    recovered = parents_from_visibility(vis)
    assert recovered == parents, f"{label}: mask lost the tree\n{recovered}\n{parents}"

    # 2. ancestor-path property: row i is exactly {i} u ancestors(i)
    for i in range(n):
        expected, j = {i}, parents[i]
        while j != -1:
            expected.add(j)
            j = parents[j]
        got = {j for j in range(n) if bool(vis[i, j])}
        assert got == expected, f"{label}: row {i} is {got}, want {expected}"

    # 3. strictly lower-triangular + diagonal (a child never precedes its parent)
    assert bool(vis.diagonal().all()), f"{label}: node not visible to itself"
    upper = torch.triu(vis, diagonal=1)
    assert not bool(upper.any()), f"{label}: node attends a later node"

    # 4. padding is isolated -- unreachable, so never acceptable
    mask = tree_to_qlen_mask(vis, N)
    assert mask.shape == (N, N)
    assert bool((mask[:n, :n] == vis).all()), f"{label}: real block corrupted by padding"
    for i in range(n, N):
        row = {j for j in range(N) if bool(mask[i, j])}
        assert row == {i}, f"{label}: pad row {i} attends {row}, want only itself"
        assert not bool(mask[:n, i].any()), f"{label}: real node attends pad column {i}"

    # 5. tokens and positions line up with the tree
    node_tokens = torch.arange(1000, 1000 + n - 1, dtype=torch.int64)
    tokens = tree_to_draft_tokens(7, node_tokens, N)
    assert int(tokens[0]) == 7 and tokens.shape == (N,)
    assert bool((tokens[1:n] == node_tokens).all()), f"{label}: token order changed"

    positions = positions_from_depths(depths_from_parents(parents), seq_len, N)
    assert int(positions[0]) == seq_len
    for i in range(1, n):
        depth = int(vis[i].sum()) - 1        # inclusive path length minus self
        assert int(positions[i]) == seq_len + depth, f"{label}: position {i} wrong"
    return mask


def test_topologies():
    rng = random.Random(0)
    count = 0
    for name, gen in TOPOLOGIES.items():
        for n in (1, 2, 3, 5, 17, 33, 65, 129):
            check_tree(gen(n), label=f"{name}/n={n}")
            count += 1
    # irregular trees, many seeds -- the best-first shape
    for seed in range(400):
        r = random.Random(seed)
        n = r.randrange(1, 130)
        check_tree(best_first_like(n, r), label=f"best_first/seed={seed}/n={n}")
        count += 1
    print(f"  topologies: {count} trees OK")


def test_padding():
    """Trees under budget, which is what a truncated heap produces."""
    for n, N in ((1, 65), (2, 65), (33, 65), (64, 65), (65, 65), (5, 129)):
        check_tree(best_first_like(n, random.Random(n)), draft_token_num=N,
                   label=f"pad/n={n}/N={N}")
    print("  padding: OK")


def test_batch_layouts():
    """Flat layouts must match what ngram_worker copies into its buffers."""
    N, seq_lens = 65, [10, 200, 3]
    masks = [tree_to_qlen_mask(
        visibility_from_parents(best_first_like(N, random.Random(i))), N)
        for i in range(len(seq_lens))]

    qlen = batch_qlen_mask(masks)
    assert qlen.shape == (len(masks) * N * N,), qlen.shape
    for i, mask in enumerate(masks):
        block = qlen[i * N * N : (i + 1) * N * N].reshape(N, N)
        assert bool((block == mask).all()), f"qlen block {i} mismatch"

    full = batch_full_mask(masks, seq_lens)
    expected = sum(N * (L + N) for L in seq_lens)
    assert full.shape == (expected,), (full.shape, expected)
    offset = 0
    for mask, seq_len in zip(masks, seq_lens):
        block = full[offset : offset + N * (seq_len + N)].reshape(N, seq_len + N)
        assert bool(block[:, :seq_len].all()), "prefix must be fully visible"
        assert bool((block[:, seq_len:] == mask).all()), "tree block mismatch"
        offset += N * (seq_len + N)
    print("  batch layouts: OK")


def test_retrieve_roundtrip():
    """parents_from_retrieve inverts a first-child/next-sibling encoding.

    Validates the checker itself against a hand-built encoding, so a GPU
    disagreement later points at the kernel or the mask -- not at this helper.
    """
    for seed in range(200):
        rng = random.Random(seed)
        n = rng.randrange(1, 80)
        parents = best_first_like(n, rng)

        children = [[] for _ in range(n)]
        for node in range(1, n):
            children[parents[node]].append(node)
        next_token = np.full(n, -1, dtype=np.int64)
        next_sibling = np.full(n, -1, dtype=np.int64)
        for node in range(n):
            if children[node]:
                next_token[node] = children[node][0]
                for a, b in zip(children[node], children[node][1:]):
                    next_sibling[a] = b

        assert parents_from_retrieve(next_token, next_sibling) == parents, seed
    print("  retrieve roundtrip: OK")


def test_degenerate():
    single = check_tree([-1], label="single")
    assert single.shape == (1, 1) and bool(single[0, 0])

    vis = visibility_from_parents([-1])
    padded = tree_to_qlen_mask(vis, 65)
    assert int(padded.sum()) == 65, "each pad row should attend only itself"
    assert int(padded[0].sum()) == 1
    print("  degenerate: OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("bridge tests")
    test_topologies()
    test_padding()
    test_batch_layouts()
    test_retrieve_roundtrip()
    test_degenerate()
    print("ALL CPU BRIDGE TESTS PASSED")
