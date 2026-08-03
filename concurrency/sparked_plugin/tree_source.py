"""Where the tree comes from.

The plugin is deliberately agnostic about this. A tree source takes the
committed context of each request and returns `(node_token_ids, parents)` in the
exact shape our batch-1 builders emit — node 0 implicit as the root, `parents[i]`
strictly less than `i` — and the worker turns that into SGLang's verify format
through `sparked_bridge`.

Two implementations:

  LookupTreeSource   no model. Branches on repeated context, the way
                     prompt-lookup decoding does. It exists so the whole
                     pipeline — registration, verify, accept, commit, and our
                     mask bridge — can be tested end to end with acceptance
                     provably above zero, without first porting a drafter.

  DSparkTreeSource   the real one, calling build_markov_tree_precomputed on
                     DSpark block logits. NOT IMPLEMENTED — see the note there.
"""

from __future__ import annotations

from typing import Sequence


class TreeSource:
    """Per-request tree proposal.

    Returns (node_token_ids, parents) where parents[0] == -1 for the root and
    len(parents) == len(node_token_ids) + 1. At most `budget` nodes beyond root.
    """

    def build(self, context: Sequence[int], budget: int) -> tuple[list[int], list[int]]:
        raise NotImplementedError


class LookupTreeSource(TreeSource):
    """Branch on repeated context. No model, no GPU, deterministic.

    For each n-gram length we look back for a match of the trailing context and
    follow the continuation that came after it. Different lengths often disagree,
    and each disagreement becomes a branch — so this produces genuinely irregular
    trees with varying per-level fanout, which is exactly the shape
    `tree_topk = -1` exists for and the shape our best-first builder emits.

    Not a research contribution. A test fixture that makes acceptance non-zero.
    """

    def __init__(self, ngram_sizes: Sequence[int] = (3, 2, 1), depth: int = 8):
        self.ngram_sizes = tuple(ngram_sizes)
        self.depth = depth

    def build(self, context: Sequence[int], budget: int) -> tuple[list[int], list[int]]:
        tokens = list(context)
        node_tokens: list[int] = []
        parents: list[int] = [-1]

        for n in self.ngram_sizes:
            if len(node_tokens) >= budget:
                break
            if len(tokens) <= n:
                continue
            pattern = tokens[-n:]
            # latest earlier occurrence, excluding the trailing copy itself
            start = None
            for i in range(len(tokens) - n - 1, -1, -1):
                if tokens[i : i + n] == pattern:
                    start = i + n
                    break
            if start is None:
                continue

            continuation = tokens[start : start + self.depth]
            if not continuation:
                continue

            # Hang this chain off the root; siblings across n give the branching.
            parent = 0
            for token in continuation:
                if len(node_tokens) >= budget:
                    break
                node_tokens.append(int(token))
                parents.append(parent)
                parent = len(node_tokens)   # 1-based: this node's index

        return node_tokens, parents


class DSparkTreeSource(TreeSource):
    """The real source: markov-guided best-first tree over DSpark block logits.

    Verified on GPU (`test_dspark_builder.py`, 18 trees, 0 failures) rather than
    assumed:

      * SGLang's drafter exposes `compute_base_logits(hidden)` returning
        `[bs, gamma, vocab]` — exactly the `[block_size, vocab]` slice
        `build_markov_tree_precomputed` takes.
      * SGLang's markov head is `srt/models/dspark.py::VanillaMarkov`, declaring
        `markov_w1 = nn.Embedding(vocab, rank)` and
        `markov_w2 = nn.Linear(rank, vocab, bias=False)` — structurally identical
        to ours (ddtree/model/dspark.py:68), so the builder's
        `.markov_w1.weight` / `.markov_w2.weight` access works **unmodified**.
      * The trees it emits round-trip through the bridge and SGLang's kernel,
        parents and positions both. They are irregular (max fanout 9-25 at
        budgets 16-64), which is the `tree_topk = -1` case.

    So the algorithm side is done. `build_from_logits` below is the real thing.

    What is NOT done is the worker wiring: reaching those logits requires
    subclassing `DSparkWorkerV2` so its drafter and `dspark_kv_inject` are reused
    as-is, and replacing only the chain verify it builds at
    `dspark_worker_v2.py:581` (`verify_ids_2d = cat([draft_block_ids[:, :1],
    draft_tokens])`, a linear chain) with a tree. That override lands in the
    middle of `_forward_decode`, alongside verify-window allocation and the
    planner, which is why it is not attempted blind.

    Note this source is logits-driven, so it does not implement `build(context,
    budget)` — the context-based signature belongs to lookup-style proposers.
    """

    def __init__(self, candidates: int = 512, max_fanout: int = 0):
        self.candidates = candidates
        self.max_fanout = max_fanout

    def build(self, context: Sequence[int], budget: int):
        raise NotImplementedError(
            "DSparkTreeSource is logits-driven; call build_from_logits(). The "
            "worker must supply DSpark's block logits and markov head."
        )

    def build_from_logits(self, base_logits, markov_head, root_token_id: int,
                          budget: int):
        """One request's tree. `base_logits` is [block_size, vocab] for that
        request -- i.e. `compute_base_logits(...).view(bs, gamma, -1)[i]`.

        Returns the builder's native tuple (node_token_ids, node_depths, parents,
        child_maps, visibility); the worker needs `node_token_ids` and
        `visibility` and can ignore the rest.
        """
        from ddtree_markov import build_markov_tree_precomputed

        return build_markov_tree_precomputed(
            base_logits, budget, markov_head, int(root_token_id),
            candidates=self.candidates, max_fanout=self.max_fanout,
        )


class DDTreeSource(DSparkTreeSource):
    """The DDTree arm: same plumbing, independence-scored builder.

    This is the whole difference between the two tree arms — `build_ddtree_tree`
    gives every sibling the same children table, while the markov builder scores
    each branch conditionally. One plugin, two sources.
    """

    def build_from_logits(self, base_logits, markov_head, root_token_id: int,
                          budget: int):
        from ddtree import build_ddtree_tree

        # build_ddtree_tree ignores the markov head by construction (that is the
        # ablation) and returns build subtimes as a sixth element.
        node_tokens, depths, parents, child_maps, visibility, _subtimes = (
            build_ddtree_tree(base_logits, budget))
        return node_tokens, depths, parents, child_maps, visibility
