"""Convert our tree-builder output into SGLang's verify format.

The design got much smaller once `ngram_worker.py` showed how NGRAM does it: it
does **not** hand-build `retrieve_index` / `retrieve_next_token` /
`retrieve_next_sibling`. It builds only the tree mask and calls

    reconstruct_indices_from_tree_mask(tree_mask, seq_lens, positions,
                                       retrieve_index, retrieve_next_token,
                                       retrieve_next_sibling, bs, draft_token_num)

which derives all four from the mask alone. So the first-child / next-sibling
encoding — the conversion flagged as the main correctness risk — is not ours to
write. We supply a mask and the kernel does the rest.

And the mask needs no conversion either. Ours is built as

    visibility[i, :i] = visibility[parent[i], :i]; visibility[i, i] = True

(ddtree_markov.py:632-637), i.e. row i attends exactly its inclusive ancestor
path — which is what SGLang's tree_mask means. The bridge is a reshape.

Two layouts, both of which NGRAM emits:

    QLEN_ONLY   flat [bs * N * N]         the tree block only
    FULL_MASK   flat [sum_i (N * (L_i + N))]  prefix ones ++ tree block per req

where N = draft_token_num = 1 + tree_budget.
"""

from __future__ import annotations

import numpy as np
import torch


def visibility_from_parents(parents: list[int]) -> torch.Tensor:
    """Ancestor mask from a parent array, by the recurrence our builders use
    (ddtree_markov.py:632-637). Tree sources that emit `parents` rather than a
    mask go through here, so there is exactly one definition of the convention.
    """
    n = len(parents)
    vis = np.zeros((n, n), dtype=np.bool_)
    vis[0, 0] = True
    for i in range(1, n):
        vis[i, :i] = vis[parents[i], :i]
        vis[i, i] = True
    return torch.from_numpy(vis)


def tree_to_draft_tokens(
    root_token_id: int,
    node_token_ids: torch.Tensor,
    draft_token_num: int,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """[root, node_0, ..., node_{n-1}] padded to draft_token_num.

    Verify consumes a fixed N per request, so a tree that came in under budget
    has to be topped up. Our builders fill the budget exactly on the main path
    (widths sum to budget), so padding is an edge case — a truncated heap, or a
    budget past what the draft horizon can supply.
    """
    tokens = torch.full((draft_token_num,), pad_token_id, dtype=torch.int64)
    tokens[0] = int(root_token_id)
    count = min(int(node_token_ids.numel()), draft_token_num - 1)
    if count > 0:
        tokens[1 : 1 + count] = node_token_ids[:count].to(torch.int64)
    return tokens


def tree_to_qlen_mask(visibility: torch.Tensor, draft_token_num: int) -> torch.Tensor:
    """[N, N] bool. Row i attends its inclusive ancestor path.

    Padded rows attend only themselves: an isolated row has no parent edge, so
    it cannot be reached by the tree walk and cannot be accepted. Padding that
    instead pointed at the root would make a dummy node a real candidate.
    """
    mask = torch.zeros((draft_token_num, draft_token_num), dtype=torch.bool)
    n = min(int(visibility.shape[0]), draft_token_num)
    mask[:n, :n] = visibility[:n, :n].to(torch.bool)
    for i in range(n, draft_token_num):
        mask[i, i] = True
    return mask


def batch_qlen_mask(masks: list[torch.Tensor]) -> torch.Tensor:
    """Flat [bs * N * N], the layout ngram_worker copies into its tree_mask buffer."""
    return torch.cat([m.reshape(-1) for m in masks], dim=0)


def batch_full_mask(masks: list[torch.Tensor], seq_lens: list[int]) -> torch.Tensor:
    """Flat [sum_i N * (L_i + N)] -- prefix ones ++ tree block, per request.

    Mirrors ngram_worker's USE_FULL_MASK branch: every draft row sees the whole
    committed prefix, then its own ancestor path inside the tree block.
    """
    if len(masks) != len(seq_lens):
        raise ValueError(f"{len(masks)} masks vs {len(seq_lens)} seq_lens")
    parts = []
    for mask, seq_len in zip(masks, seq_lens):
        prefix = torch.ones((mask.shape[0], int(seq_len)), dtype=torch.bool)
        parts.append(torch.cat([prefix, mask], dim=1).reshape(-1))
    return torch.cat(parts, dim=0)


def positions_from_depths(
    node_depths: torch.Tensor, seq_len: int, draft_token_num: int
) -> torch.Tensor:
    """Absolute position of each verify row: seq_len + depth, root at depth 0.

    Only used to cross-check the kernel — `reconstruct_indices_from_tree_mask`
    writes `positions` itself, from the mask.
    """
    positions = torch.full((draft_token_num,), int(seq_len), dtype=torch.int64)
    n = min(int(node_depths.numel()), draft_token_num - 1)
    if n > 0:
        positions[1 : 1 + n] = int(seq_len) + node_depths[:n].to(torch.int64)
    return positions


# ---------------------------------------------------------------------------
# Verification helpers. These exist to test the kernel's output against the tree
# we actually built, not to run in the hot path.
# ---------------------------------------------------------------------------


def parents_from_visibility(visibility: torch.Tensor) -> list[int]:
    """Recover the parent of each node from the mask alone.

    A node's ancestor set is its row; its parent is the deepest strict ancestor,
    i.e. the one whose own ancestor set is largest. Independent of `parents`, so
    comparing the two catches a mask built inconsistently with the tree.
    """
    mask = visibility.to(torch.bool).numpy()
    n = mask.shape[0]
    sizes = mask.sum(axis=1)
    parents = [-1] * n
    for i in range(1, n):
        ancestors = [j for j in range(n) if j != i and mask[i, j]]
        parents[i] = int(max(ancestors, key=lambda j: sizes[j])) if ancestors else -1
    return parents


def parents_from_retrieve(
    retrieve_next_token: np.ndarray, retrieve_next_sibling: np.ndarray
) -> list[int]:
    """Rebuild the parent relation from the kernel's first-child / next-sibling
    encoding: walk each node's child chain and attribute every link to it."""
    n = len(retrieve_next_token)
    parents = [-1] * n
    for node in range(n):
        child = int(retrieve_next_token[node])
        while child != -1:
            if not (0 <= child < n):
                raise ValueError(f"child index {child} out of range at node {node}")
            parents[child] = node
            child = int(retrieve_next_sibling[child])
    return parents
