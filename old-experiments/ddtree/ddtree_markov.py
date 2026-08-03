"""DDTree with markov-guided branches: the diffusion-sparked-tree experiment.

Vanilla DDTree builds its draft tree from per-position distributions that are
independent of which token any ancestor holds - build_ddtree_tree can precompute
one top-k table per depth because every node at a depth shares the same children
distribution. DSpark's markov head breaks exactly that assumption: the
distribution over depth d+1 tokens depends on the token drafted at depth d,
    children(node) ~ softmax(base_logits[depth] + markov_bias(node.token))

This module runs DDTree's draft-tree speculative decoding on the DSpark drafter
(Qwen3-4B checkpoint deepseek-ai/dspark_qwen3_4b_block7) with two tree builders:

  use_markov=False  -> build_ddtree_tree on the drafter's base logits (ablation:
                       same drafter, independence assumed, as in the DDTree paper)
  use_markov=True   -> build_markov_tree below: best-first heap where each
                       materialized node lazily computes its own branch-corrected
                       children top-k. Path score = sum of branch-conditional
                       per-step log-probs, so sibling/child expansion still works,
                       but the tables are per-node instead of per-depth.

Everything downstream of tree building (ancestor-mask compile, single verify
pass, tree walk acceptance, cache compaction) is reused from ddtree.py verbatim,
so any acceptance-length difference is attributable to tree construction alone.

Cost note: the markov builder does one bias matvec + top-k per materialized node
(lazy, sequential, one GPU sync each). This is the simple correct version for
measuring acceptance length; wall-clock could later be improved by batching
expansions per heap wave.
"""

import heapq
import time
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DSparkDraftModel, sample, extract_context_feature
from dflash import empty_stage_times
from ddtree import (
    build_ddtree_tree,
    compile_ddtree_tree,
    follow_verified_tree,
    compact_dynamic_cache,
)


DDTREE_MARKOV_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")


def cuda_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def build_markov_tree_sequential(
    base_logits: torch.Tensor,
    budget: int,
    markov_head,
    root_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor]:
    """Best-first draft tree with branch-conditional (markov-corrected) scores.

    base_logits: [L, V] backbone logits; row d-1 is the distribution over tokens
    at tree depth d (before correction). Returns the same tuple layout as
    build_ddtree_tree so compile_ddtree_tree / follow_verified_tree apply as-is.
    """
    depth_limit, vocab_size = base_logits.shape
    if budget <= 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
        )

    topk = min(budget, vocab_size)
    logits = base_logits.float()
    w1 = markov_head.markov_w1.weight
    w2 = markov_head.markov_w2.weight

    def children_topk(depth: int, prev_token_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k log-probs over depth-`depth` tokens given the parent's token."""
        bias = (w2 @ w1[prev_token_id]).float()
        corrected = logits[depth - 1] + bias
        log_probs = corrected - torch.logsumexp(corrected, dim=-1)
        top_vals, top_ids = torch.topk(log_probs, k=topk)
        stacked = torch.stack([top_vals, top_ids.float()]).cpu().numpy()  # one sync
        return stacked[0], stacked[1].astype(np.int64)

    # per-node children tables, keyed by node index (0 = root)
    table_vals: dict[int, np.ndarray] = {}
    table_ids: dict[int, np.ndarray] = {}
    table_vals[0], table_ids[0] = children_topk(1, int(root_token_id))

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0
    tiebreak = 0

    first_logw = float(table_vals[0][0])
    # entry: (-logw, tiebreak, parent_index, depth, rank, logw)
    heap: list[tuple[float, int, int, int, int, float]] = [(-first_logw, tiebreak, 0, 1, 0, first_logw)]

    while heap and node_count < budget:
        _, _, parent_index, depth, rank, logw = heapq.heappop(heap)
        token_id = int(table_ids[parent_index][rank])

        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_logw = logw - float(table_vals[parent_index][rank]) + float(table_vals[parent_index][rank + 1])
            tiebreak += 1
            heapq.heappush(heap, (-sibling_logw, tiebreak, parent_index, depth, rank + 1, sibling_logw))

        if depth < depth_limit:
            # this node's own branch-corrected children distribution
            table_vals[current_index], table_ids[current_index] = children_topk(depth + 1, token_id)
            child_logw = logw + float(table_vals[current_index][0])
            tiebreak += 1
            heapq.heappush(heap, (-child_logw, tiebreak, current_index, depth + 1, 0, child_logw))

    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True

    return (
        torch.from_numpy(node_token_ids_np[:node_count]),
        torch.from_numpy(node_depths_np[:node_count]),
        parents_np[:current_length].tolist(),
        child_maps,
        torch.from_numpy(visibility_np),
    )


@torch.inference_mode()
def build_markov_tree(
    base_logits: torch.Tensor,
    budget: int,
    markov_head,
    root_token_id: int,
    wave: int = 16,
    exact: bool = True,
    candidates: int = 2048,
    per_depth: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor]:
    """Batched-sync equivalent of build_markov_tree_sequential.

    The sequential builder does one bias-matvec + top-k + GPU sync per
    materialized node (~budget syncs/round), which dominates tree_build time.

    A heap entry's token is known *before* the node is materialized (it lives in
    the parent's table), so the children table it will need - keyed by
    (depth + 1, token) - is predictable. We peek the top-`wave` heap entries,
    compute all their tables in ONE batched matmul with ONE sync, then run those
    pops against the cache. Pop order is unchanged, tables are a pure function of
    (depth, token), so the resulting tree is IDENTICAL to the sequential builder
    (asserted in test_batched_builder.py) at ~budget/wave syncs instead of budget.

    Kept as the reference implementation for build_markov_tree_precomputed, which
    resolves every table before the walk starts; `per_depth=True` switches the
    candidate restriction to that builder's per-depth top-C so the two can be
    compared on identical candidate sets.
    """
    depth_limit, vocab_size = base_logits.shape
    if budget <= 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
        )

    topk = min(budget, vocab_size)
    logits = base_logits.float()
    device = logits.device
    w1 = markov_head.markov_w1.weight
    w2 = markov_head.markov_w2.weight

    # Candidate restriction. Computing a children table naively costs a full read
    # of markov_w2 (vocab x rank, ~78MB bf16) from HBM - and that read, repeated
    # per node, is what actually dominates tree_build time, not sync latency.
    #
    # We only ever need the top-k of (logits[d] + bias), and bias magnitudes are
    # small relative to the logit spread (mean ~0.4, p99 ~3 on the trained head),
    # so a token far outside the raw top-C cannot climb into the top-k. Take the
    # union of the per-depth top-C, gather those w2 rows ONCE per round, and every
    # subsequent table is a matmul against a ~17MB slice with a cheaper top-k.
    # Set candidates=0 for the exact full-vocab path.
    if per_depth:
        active_vocab = int(min(candidates or vocab_size, vocab_size))
        logits_active, cand_ids = torch.topk(logits, k=active_vocab, dim=-1)   # [L, C]
        w2_active = w2.index_select(0, cand_ids.reshape(-1)).view(depth_limit, active_vocab, -1).float()
    elif candidates and candidates < vocab_size:
        per_depth_k = min(int(candidates), vocab_size)
        cand_ids = torch.unique(torch.topk(logits, k=per_depth_k, dim=-1).indices.reshape(-1))
        w2_active = w2.index_select(0, cand_ids).float()      # [U, R] gathered once
        logits_active = logits.index_select(1, cand_ids)      # [L, U]
    else:
        cand_ids = None
        w2_active = w2.float()
        logits_active = logits
    active_topk = min(topk, logits_active.shape[-1])

    table_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    sync_count = 0

    def ensure_tables(pairs) -> None:
        """Compute children tables for all uncached (depth, prev_token) pairs at once."""
        nonlocal sync_count
        missing = sorted({p for p in pairs if p not in table_cache})
        if not missing:
            return
        depths = torch.tensor([d for d, _ in missing], dtype=torch.long, device=device)
        tokens = torch.tensor([t for _, t in missing], dtype=torch.long, device=device)
        rows = depths - 1
        if per_depth:
            # each depth owns its candidate set, so the matmul is batched per row
            bias = torch.bmm(
                w1[tokens].float().unsqueeze(1),                       # [M, 1, R]
                w2_active.index_select(0, rows).transpose(1, 2),       # [M, R, C]
            ).squeeze(1)                                              # [M, C]
        else:
            bias = w1[tokens].float() @ w2_active.T                   # [M, U] - one matmul
        corrected = logits_active.index_select(0, rows) + bias        # [M, U]
        log_probs = corrected - torch.logsumexp(corrected, dim=-1, keepdim=True)
        top_vals, top_idx = torch.topk(log_probs, k=active_topk, dim=-1)
        if per_depth:                                                 # map back to vocab
            top_ids = cand_ids.index_select(0, rows).gather(1, top_idx)
        else:
            top_ids = top_idx if cand_ids is None else cand_ids[top_idx]
        stacked = torch.stack([top_vals, top_ids.float()]).cpu().numpy()  # ONE sync
        sync_count += 1
        for i, pair in enumerate(missing):
            table_cache[pair] = (stacked[0, i], stacked[1, i].astype(np.int64))

    # Prefetch the greedy spine. Best-first descends greedily before it branches,
    # and each descent step needs the previous step's table - a true sequential
    # dependency that peek-ahead cannot cover. But the chain of argmaxes can be
    # walked entirely on GPU without ever syncing, so we resolve the whole spine
    # in two syncs (tokens, then their tables) instead of one per depth.
    spine_tokens = [torch.tensor(int(root_token_id), device=device)]
    for depth_idx in range(1, depth_limit):
        spine_bias = w1[spine_tokens[-1]].float() @ w2.T.float()
        spine_tokens.append(torch.argmax(logits[depth_idx - 1] + spine_bias))
    spine_cpu = torch.stack(spine_tokens).cpu().tolist()
    sync_count += 1
    ensure_tables([(depth_idx + 1, int(tok)) for depth_idx, tok in enumerate(spine_cpu)])

    table_vals: dict[int, np.ndarray] = {}
    table_ids: dict[int, np.ndarray] = {}
    table_vals[0], table_ids[0] = table_cache[(1, int(root_token_id))]

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0
    tiebreak = 0

    first_logw = float(table_vals[0][0])
    heap: list[tuple[float, int, int, int, int, float]] = [(-first_logw, tiebreak, 0, 1, 0, first_logw)]

    def materialize(entry):
        """Pop-and-record a heap entry; push its sibling (reuses the parent's
        table). Returns the child descriptor if this node can have children."""
        nonlocal node_count, tiebreak
        _, _, parent_index, depth, rank, logw = entry
        token_id = int(table_ids[parent_index][rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < active_topk:
            sibling_logw = logw - float(table_vals[parent_index][rank]) + float(table_vals[parent_index][rank + 1])
            tiebreak += 1
            heapq.heappush(heap, (-sibling_logw, tiebreak, parent_index, depth, rank + 1, sibling_logw))

        return None if depth >= depth_limit else (current_index, depth, token_id, logw)

    def push_child(child):
        nonlocal tiebreak
        child_index, depth, token_id, logw = child
        table_vals[child_index], table_ids[child_index] = table_cache[(depth + 1, token_id)]
        child_logw = logw + float(table_vals[child_index][0])
        tiebreak += 1
        heapq.heappush(heap, (-child_logw, tiebreak, child_index, depth + 1, 0, child_logw))

    while heap and node_count < budget:
        lookahead = min(wave, budget - node_count)

        if exact:
            # Strict best-first: pop ONE AT A TIME so every push is visible to the
            # next pop, but prefetch tables for the ENTIRE heap first.
            #
            # Each table costs a full read of markov_w2 (vocab x rank, ~78MB) from
            # HBM, which is what actually dominates - not sync latency. Batching
            # amortizes that read across rows, so computing tables for entries that
            # never get popped is nearly free, while computing them one at a time
            # re-reads w2 every time. Over-prefetch aggressively.
            ensure_tables([
                (depth + 1, int(table_ids[parent_index][rank]))
                for _, _, parent_index, depth, rank, _ in heap
                if depth < depth_limit
            ])
            for _ in range(lookahead):
                if not heap or node_count >= budget:
                    break
                child = materialize(heapq.heappop(heap))
                if child is not None:
                    key = (child[1] + 1, child[2])
                    if key not in table_cache:
                        ensure_tables([key])
                    push_child(child)
            continue

        # Wave mode: pop the whole wave, materialize it, then resolve every child
        # table in ONE batched sync. Pushes are deferred to the end of the wave, so
        # the tree can differ slightly from strict best-first (a successor pushed
        # mid-wave cannot outrank a later member of the same wave).
        popped = []
        for _ in range(lookahead):
            if not heap:
                break
            popped.append(heapq.heappop(heap))

        pending_children = [c for c in (materialize(e) for e in popped) if c is not None]
        if pending_children:
            ensure_tables([(depth + 1, token_id) for _, depth, token_id, _ in pending_children])
            for child in pending_children:
                push_child(child)

    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True

    return (
        torch.from_numpy(node_token_ids_np[:node_count]),
        torch.from_numpy(node_depths_np[:node_count]),
        parents_np[:current_length].tolist(),
        child_maps,
        torch.from_numpy(visibility_np),
    )



# A flat schedule is only meaningful while every depth it covers gets more than
# one slot. Below that it silently degenerates into a chain (see the docstring).
FLAT_MIN_WIDTH = 2


def flat_width_schedule(budget: int, depth_limit: int, min_width: int = FLAT_MIN_WIDTH) -> list[int]:
    """Uniform width per depth, over as many depths as the budget can afford to branch.

    Measured hit data (experiments/analyze_tree_slots.py) shows the drafter is
    right 87% of the time at depth 1 but only 30% at depth 16, so budget belongs
    spread across depths rather than front-loaded. Flat matched or beat every
    geometric schedule tested.

    Spreading has a floor, though. Dividing a budget of 16 over 16 depths returns
    [1,1,...,1] - a chain with zero branching, which is strictly worse than the
    same 16 nodes spent on a shallower tree, and was the cause of the budget-16
    regression in RESULTS.md §2. So depth is capped at budget // min_width rather
    than fixed at the drafter's horizon: trading depth for width is the correct
    response to a tight budget, and thinning every level to 1 is not.

    min_width doubles as the depth-truncation knob. At budget 64 the default
    min_width=2 leaves the full 16 depths untouched ([4]*16); min_width=5 caps the
    tree at 12 levels ([6,6,6,6,5,...]), shortening the builder's dependency chain
    and moving the freed slots to depths that are actually reached.
    """
    budget, depth_limit = int(budget), int(depth_limit)
    if budget <= 0 or depth_limit <= 0:
        return []
    depth = max(1, min(depth_limit, budget // max(int(min_width), 1)))
    base, extra = divmod(budget, depth)
    return [base + (1 if i < extra else 0) for i in range(depth)]


def confidence_width_schedule(accept_probs, budget: int, depth_limit: int) -> list[int]:
    """Per-round widths from DSpark's confidence head.

    The flat schedule is the *average* optimal allocation, measured over many
    rounds. The confidence head predicts P(accept) per position for THIS round, so
    the allocation can adapt: an easy continuation needs one candidate at every
    depth, a hard one needs hedging exactly where it is uncertain.

    Marginal value of a slot at depth d ~ P(reach d) x P(top pick is wrong at d):

        reach[d] = prod_{j<d} p_j      (no point widening a depth we never reach)
        value[d] = reach[d] * (1 - p_d) (no point widening where we are certain)

    Budget is split proportional to value, everything at least 1 slot while any
    value remains.
    """
    probs = [float(min(max(x, 1e-4), 1.0 - 1e-4)) for x in accept_probs[:depth_limit]]
    if not probs:
        return flat_width_schedule(budget, depth_limit)

    reach, running = [], 1.0
    for p in probs:
        reach.append(running)
        running *= p
    value = [r * (1.0 - p) for r, p in zip(reach, probs)]
    total = sum(value)
    if total <= 0:
        return flat_width_schedule(budget, depth_limit)

    widths, remaining = [], int(budget)
    for v in value:
        if remaining <= 0:
            break
        w = max(1, min(remaining, int(round(budget * v / total))))
        widths.append(w)
        remaining -= w
    if remaining > 0 and widths:
        # spare budget goes to the highest-value depth
        widths[max(range(len(widths)), key=lambda i: value[i])] += remaining
    return widths


def beam_width_schedule(budget: int, depth_limit: int, decay: float = 0.6) -> list[int]:
    """Nodes per depth for the beam builder: wide near the root, narrowing with
    depth (deep positions are less likely to be reached at all). Sums to <= budget."""
    widths: list[int] = []
    remaining = int(budget)
    width = max(1.0, budget * (1.0 - decay))
    for _ in range(depth_limit):
        if remaining <= 0:
            break
        w = max(1, min(remaining, int(round(width))))
        widths.append(w)
        remaining -= w
        width *= decay
    return widths


_BEAM_GRAPHS: dict = {}


def _beam_transition_tables(logits_cand, w1_cand, w2_cand, w1_root):
    """Every branch-conditional transition the beam can ever take, computed at once.

    build_beam_tree recomputes `w1[surviving_tokens] @ w2_cand[d].T` inside the
    level loop, so the matmul at depth d cannot start until the top-k at depth
    d-1 has landed: 16 serially dependent (gather -> matmul -> logsumexp -> top-k)
    chains per round, which is what the CUDA-graph profile in RESULTS.md §12 found
    the builder bound by.

    That dependency is avoidable. A parent at depth d is *always* an element of
    the depth d-1 candidate set - the beam has nothing else to select from - so
    the transition is a finite C x C table per depth, and the whole [L-1, C, C]
    stack is one batched matmul with no dependency on the beam at all. The
    logsumexp normaliser is a function of (depth, parent slot) only, so it folds
    into the same precompute. What remains in the loop is a gather, an add and a
    top-k over width x C - no matmul, no softmax.

    The cost is that this computes transitions for all C possible parents when
    only `width` of them are used, so the precompute is O(L C^2 R) against the
    loop's O(L width C R). That trade only pays because C is small (256) and
    because the C^2 work is one independent kernel while the width work is a
    16-long dependency chain. Keep C small: this table is the reason C is now a
    quadratic knob rather than a linear one.

    Returns (root_logp [C], table_logp [L-1, C, C]), both log-normalised over the
    candidate set exactly as build_beam_tree normalises over its active set.
    """
    root = logits_cand[0] + (w1_root.float() @ w2_cand[0].T).reshape(-1)
    root_logp = root - torch.logsumexp(root, dim=-1)

    # [L-1, C, R] @ [L-1, R, C] -> [L-1, C, C], plus the per-depth base logits
    table = torch.baddbmm(logits_cand[1:].unsqueeze(1), w1_cand[:-1], w2_cand[1:].transpose(1, 2))
    table_logp = table - torch.logsumexp(table, dim=-1, keepdim=True)
    return root_logp, table_logp


def _beam_level_loop(root_logp, table_logp, widths, candidate_count):
    """Level-synchronous expansion against precomputed tables.

    Carries *candidate slots* rather than vocab ids, since that is what indexes
    the transition tables; the caller maps slots back to vocab once at the end.
    """
    parent_locals, slots = [], []
    previous_slots, scores = None, None
    for depth_idx, width in enumerate(widths):
        if depth_idx == 0:
            path_scores = root_logp                                    # [C]
        else:
            path_scores = scores.unsqueeze(1) + table_logp[depth_idx - 1].index_select(0, previous_slots)
        top_vals, top_flat = torch.topk(path_scores.reshape(-1), width)
        parent_locals.append(torch.div(top_flat, candidate_count, rounding_mode="floor"))
        previous_slots = top_flat % candidate_count
        slots.append(previous_slots)
        scores = top_vals
    return torch.cat(parent_locals), torch.cat(slots)


def _depth_of_node(widths, device):
    """Depth index of every emitted node, so slots can be mapped back to vocab."""
    return torch.repeat_interleave(
        torch.arange(len(widths), device=device),
        torch.tensor(widths, device=device),
    )


def _resolve_beam_widths(widths, budget, depth_limit, min_width, candidate_count=None):
    """Schedule, truncated to the drafter's horizon and to what the level can hold.

    A level can emit at most parents x C nodes, so a width past that would ask
    top-k for more elements than exist. Only reachable at budgets far above 256,
    but the clamp has to happen before the CUDA-graph key is built, since it is
    part of the captured shape.
    """
    widths = list(widths) if widths else flat_width_schedule(budget, depth_limit, min_width)
    widths = [w for w in widths[:depth_limit] if w > 0]
    if candidate_count is None:
        return widths
    resolved, available = [], 1
    for width in widths:
        width = min(width, available * candidate_count)
        resolved.append(width)
        available = width
    return resolved


@torch.inference_mode()
def build_beam_tree_precomputed(
    base_logits: torch.Tensor,
    budget: int,
    markov_head,
    root_token_id: int,
    candidates: int = 512,
    widths: list[int] | None = None,
    min_width: int = FLAT_MIN_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor]:
    """build_beam_tree with the branch transitions hoisted out of the level loop.

    Algebraically identical to build_beam_tree at the same per-depth candidate
    set - same scores, same top-k, same tie-break - but NOT bitwise identical:
    the [C, C] batched matmul and the [width, C] one it replaces reduce in
    different orders, so trees can diverge where two paths score within float32
    epsilon of each other. test_beam_precompute.py measures that divergence rate.

    Candidates are per-depth ([L, C]) rather than build_beam_tree's deduped union,
    both because the table needs a compile-time shape and because each depth
    getting its own top-C is slightly tighter than sharing one pool.
    """
    depth_limit, vocab_size = base_logits.shape
    candidate_count = int(min(candidates or vocab_size, vocab_size))
    widths = _resolve_beam_widths(widths, budget, depth_limit, min_width, candidate_count)
    if budget <= 0 or not widths:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                [-1], [dict()], visibility)

    logits = base_logits.float()
    device = logits.device
    w1, w2 = markov_head.markov_w1.weight, markov_head.markov_w2.weight

    logits_cand, cand_ids = torch.topk(logits[: len(widths)], k=candidate_count, dim=-1)
    flat_ids = cand_ids.reshape(-1)
    w1_cand = w1.index_select(0, flat_ids).view(len(widths), candidate_count, -1).float()
    w2_cand = w2.index_select(0, flat_ids).view(len(widths), candidate_count, -1).float()
    w1_root = w1.index_select(0, torch.tensor([int(root_token_id)], device=device))

    root_logp, table_logp = _beam_transition_tables(logits_cand, w1_cand, w2_cand, w1_root)
    parent_local, slots = _beam_level_loop(root_logp, table_logp, widths, candidate_count)
    token_ids = flat_ids.gather(0, _depth_of_node(widths, device) * candidate_count + slots)

    packed = torch.stack([parent_local, token_ids]).cpu().numpy()   # ONE transfer
    return _tree_structure_from_levels(packed[0], packed[1], widths)



def _tree_structure_from_levels(parent_local_np, token_np, widths):
    """Shared CPU tail: level-local (parent, token) pairs -> tree arrays."""
    total = int(sum(widths))
    node_token_ids_np = np.empty(total, dtype=np.int64)
    node_depths_np = np.empty(total, dtype=np.int64)
    parents_np = np.empty(total + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]

    node_count, prev_global, offset = 0, [0], 0
    for depth_idx, width in enumerate(widths):
        for i in range(width):
            parent_index = prev_global[int(parent_local_np[offset + i])]
            token_id = int(token_np[offset + i])
            current_index = node_count + 1
            node_token_ids_np[node_count] = token_id
            node_depths_np[node_count] = depth_idx + 1
            parents_np[current_index] = parent_index
            child_maps.append(dict())
            child_maps[parent_index][token_id] = current_index
            node_count += 1
        prev_global = list(range(node_count - width + 1, node_count + 1))
        offset += width

    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True

    return (
        torch.from_numpy(node_token_ids_np[:node_count]),
        torch.from_numpy(node_depths_np[:node_count]),
        parents_np[:current_length].tolist(),
        child_maps,
        torch.from_numpy(visibility_np),
    )


@torch.inference_mode()
def build_markov_tree_precomputed(
    base_logits: torch.Tensor,
    budget: int,
    markov_head,
    root_token_id: int,
    candidates: int = 512,
    max_fanout: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor]:
    """Best-first markov tree at ONE sync per round - batched *before* the walk.

    build_markov_tree cannot know node n+1's table until node n is popped, so it
    interleaves GPU work with the heap: a big prefetch per outer iteration plus a
    single-pair `ensure_tables` for every child whose table the prefetch missed.
    That is what leaves it at ~48 syncs/round (RESULTS.md §4).

    `wave` mode attacked this by batching *around* the walk - pop K, materialize
    all K, resolve their children together - and destroyed acceptance (11.45 ->
    8.13, FINDINGS §9), because deferring pushes means a successor cannot outrank
    later members of its own wave. That is precisely the ordering best-first is.

    The fix is to batch *before* the walk instead. A node at depth d always holds
    an element of the depth d-1 candidate set - the builder has nothing else to
    select from - so the complete set of transitions best-first could ever ask for
    is the same finite [L-1, C, C] tensor `_beam_transition_tables` already builds
    for the beam, and it depends on the candidate sets alone, not on which nodes
    get expanded. Resolve it, take the per-row top-k once, ship it to CPU, and the
    heap walk becomes pure CPU work against resident arrays: `ensure_tables` is not
    batched but *deleted*, since a cache miss is no longer possible.

    Pop order is therefore untouched - this is strict best-first, not an
    approximation of it - so acceptance should match build_markov_tree at the same
    candidate set rather than beam's (11.09 vs 11.45 at budget 64, gsm8k).

    The cost is the one flagged in _beam_transition_tables: O(L C^2 R) to build
    transitions for all C possible parents when at most `budget` are used, plus a
    [1 + (L-1)C, k] top-k slice to transfer. Both are quadratic in C, so this
    builder wants C in the hundreds - unlike build_markov_tree, which defaults to
    2048 because its per-node matmul is only linear in C. §4 measured that
    shrinking C is nearly free for acceptance (11.45 @ 2048 -> 11.35 @ 512), which
    is what makes the trade available at all.

    Unlike the beam builders this cannot be CUDA-graphed: the walk is a Python
    heap with data-dependent control flow. It has no GPU work left to capture,
    though, so that is a forfeit rather than a regression.

    max_fanout bounds how many siblings ONE node may contribute, which is what
    sets the transferred slice width. Defaulting it to `budget` is the safe bound
    (a single parent could in principle own the whole tree) but it makes transfer
    volume scale with budget, and that is what turned the measured +2.9% acceptance
    at budget 64 into -13.8% speed at budget 128 on gsm8k. Real best-first trees
    are nowhere near that concentrated - the measured depth profile at budget 64 is
    [16, 18, 15, 12, 3] - so a cap well below budget is normally free. Set 0 for
    the safe bound.
    """
    depth_limit, vocab_size = base_logits.shape
    if budget <= 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                [-1], [dict()], visibility)

    logits = base_logits.float()
    device = logits.device
    w1, w2 = markov_head.markov_w1.weight, markov_head.markov_w2.weight
    candidate_count = int(min(candidates or vocab_size, vocab_size))
    topk = min(int(max_fanout) or budget, budget, candidate_count)

    logits_cand, cand_ids = torch.topk(logits, k=candidate_count, dim=-1)   # [L, C]
    flat_ids = cand_ids.reshape(-1)
    w1_cand = w1.index_select(0, flat_ids).view(depth_limit, candidate_count, -1).float()
    w2_cand = w2.index_select(0, flat_ids).view(depth_limit, candidate_count, -1).float()
    w1_root = w1.index_select(0, torch.tensor([int(root_token_id)], device=device))

    root_logp, table_logp = _beam_transition_tables(logits_cand, w1_cand, w2_cand, w1_root)
    root_vals, root_slots = torch.topk(root_logp, k=topk, dim=-1)          # [k]
    table_vals, table_slots = torch.topk(table_logp, k=topk, dim=-1)       # [L-1, C, k]

    # ONE transfer for every table the walk can reach. Slots and token ids ride
    # through float32 exactly (vocab << 2^24), the same trick build_markov_tree
    # uses to pack its own top-k into a single stacked tensor.
    flat = torch.cat([
        root_vals, table_vals.reshape(-1),
        root_slots.float(), table_slots.reshape(-1).float(),
        flat_ids.float(),
    ]).cpu().numpy()

    table_count = 1 + (depth_limit - 1) * candidate_count
    span = table_count * topk
    vals = flat[:span].reshape(table_count, topk)
    slots = flat[span:2 * span].reshape(table_count, topk).astype(np.int64)
    cand = flat[2 * span:].reshape(depth_limit, candidate_count).astype(np.int64)

    # Row of the children table for a node at `depth` holding candidate `slot`.
    # Root is row 0; a node at depth d indexes table_logp[d-1], which exists only
    # while d <= L-1 - exactly the depth_limit guard below.
    def child_row(depth, slot):
        return 1 + (depth - 1) * candidate_count + slot

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_rows = np.zeros(budget + 1, dtype=np.int64)   # node 0 is the root -> row 0
    node_count = 0
    tiebreak = 0

    first_logw = float(vals[0, 0])
    heap: list[tuple[float, int, int, int, int, float]] = [(-first_logw, tiebreak, 0, 1, 0, first_logw)]

    while heap and node_count < budget:
        _, _, parent_index, depth, rank, logw = heapq.heappop(heap)
        parent_row = int(node_rows[parent_index])
        slot = int(slots[parent_row, rank])
        token_id = int(cand[depth - 1, slot])

        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_logw = logw - float(vals[parent_row, rank]) + float(vals[parent_row, rank + 1])
            tiebreak += 1
            heapq.heappush(heap, (-sibling_logw, tiebreak, parent_index, depth, rank + 1, sibling_logw))

        if depth < depth_limit:
            row = child_row(depth, slot)
            node_rows[current_index] = row
            child_logw = logw + float(vals[row, 0])
            tiebreak += 1
            heapq.heappush(heap, (-child_logw, tiebreak, current_index, depth + 1, 0, child_logw))

    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True

    return (
        torch.from_numpy(node_token_ids_np[:node_count]),
        torch.from_numpy(node_depths_np[:node_count]),
        parents_np[:current_length].tolist(),
        child_maps,
        torch.from_numpy(visibility_np),
    )


@torch.inference_mode()
def build_beam_tree_graphed(
    base_logits: torch.Tensor,
    budget: int,
    markov_head,
    root_token_id: int,
    candidates: int = 512,
    widths: list[int] | None = None,
    min_width: int = FLAT_MIN_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor]:
    """CUDA-graph replay of the level-synchronous beam expansion.

    Profiling (H100, budget 64): the level loop is ~100% of tree_build, and a
    captured graph replays it in 1.61ms vs 3.98ms eager - 60% saved. Graph capture
    needs STATIC shapes, which only became possible once the width schedule went
    flat; the adaptive schedules could not be captured.

    The captured body is _beam_transition_tables + _beam_level_loop, so the graph
    replays one batched matmul followed by a chain of gather/add/top-k rather than
    a chain of matmuls - the graph removes launch overhead, the precompute removes
    the work each link was waiting on.
    """
    depth_limit, vocab_size = base_logits.shape
    C = int(min(candidates or vocab_size, vocab_size))
    widths = _resolve_beam_widths(widths, budget, depth_limit, min_width, C)
    if budget <= 0 or not widths:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                [-1], [dict()], visibility)

    device = base_logits.device
    w1, w2 = markov_head.markov_w1.weight, markov_head.markov_w2.weight
    key = (tuple(widths), C, int(w1.shape[1]), device, id(markov_head))

    entry = _BEAM_GRAPHS.get(key)
    if entry is None:
        entry = _capture_beam_graph(w1, w2, widths, C, device)
        _BEAM_GRAPHS[key] = entry
    if entry is False:                       # capture failed once; stay on eager
        return build_beam_tree_precomputed(base_logits, budget, markov_head, root_token_id,
                                           candidates=candidates, widths=widths)

    logits = base_logits.float()
    top_vals, top_ids = torch.topk(logits[: len(widths)], k=C, dim=-1)
    flat_ids = top_ids.reshape(-1)
    entry["cand_ids"].copy_(top_ids)
    entry["logits_cand"].copy_(top_vals)
    entry["w1_cand"].copy_(w1.index_select(0, flat_ids).view(len(widths), C, -1).float())
    entry["w2_cand"].copy_(w2.index_select(0, flat_ids).view(len(widths), C, -1).float())
    entry["root"].fill_(int(root_token_id))
    entry["graph"].replay()

    packed = torch.stack([entry["out_parent"], entry["out_token"]]).cpu().numpy()
    return _tree_structure_from_levels(packed[0], packed[1], widths)


def _capture_beam_graph(w1, w2, widths, C, device):
    """Capture one replayable graph for this (widths, C) shape. Returns False on
    failure so the caller can fall back permanently instead of retrying."""
    rank = int(w1.shape[1])
    total = int(sum(widths))
    buf = dict(
        cand_ids=torch.zeros(len(widths), C, dtype=torch.long, device=device),
        logits_cand=torch.zeros(len(widths), C, dtype=torch.float32, device=device),
        w1_cand=torch.zeros(len(widths), C, rank, dtype=torch.float32, device=device),
        w2_cand=torch.zeros(len(widths), C, rank, dtype=torch.float32, device=device),
        root=torch.zeros(1, dtype=torch.long, device=device),
        out_parent=torch.zeros(total, dtype=torch.long, device=device),
        out_token=torch.zeros(total, dtype=torch.long, device=device),
        depth_of=_depth_of_node(widths, device),
        widths=widths,
    )

    def body():
        root_logp, table_logp = _beam_transition_tables(
            buf["logits_cand"], buf["w1_cand"], buf["w2_cand"], w1.index_select(0, buf["root"]),
        )
        parent_local, slots = _beam_level_loop(root_logp, table_logp, widths, C)
        buf["out_parent"].copy_(parent_local)
        buf["out_token"].copy_(buf["cand_ids"].reshape(-1).gather(0, buf["depth_of"] * C + slots))

    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        buf["graph"] = graph
        return buf
    except Exception as exc:  # noqa: BLE001 - any capture failure means fall back
        import logging
        logging.getLogger(__name__).warning("CUDA graph capture failed (%s); using eager beam", exc)
        return False


@torch.inference_mode()
def build_beam_tree(
    base_logits: torch.Tensor,
    budget: int,
    markov_head,
    root_token_id: int,
    decay: float = 0.6,
    candidates: int = 2048,
    widths: list[int] | None = None,
    flat: bool = False,
    min_width: int = FLAT_MIN_WIDTH,
    per_depth: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor]:
    """Level-synchronous (beam) markov tree - one batched expansion per DEPTH.

    build_markov_tree expands one node at a time, so the table for node n+1 is not
    known until node n has been popped: ~budget GPU round-trips per round, and the
    ~0.26ms fixed cost of each is what dominates tree_build (measured 2.15s floor,
    vs 0.10s for DDTree's single per-depth call).

    Fixing the tree SHAPE up front removes that dependency. Each level expands all
    its parents in one batched matmul, and - crucially - the surviving tokens stay
    on the GPU as a tensor, so no level needs a sync. Every transfer is deferred to
    a single .cpu() at the end: 1 sync per round instead of ~budget.

    The cost is that budget is allocated by a fixed schedule rather than
    adaptively by path score (beam search, not best-first).

    Kept as the reference implementation for build_beam_tree_precomputed, which
    hoists the per-level matmul out of the loop; `per_depth=True` switches the
    candidate restriction to the precomputed builder's per-depth top-C so the two
    can be compared on identical candidate sets.
    """
    depth_limit, vocab_size = base_logits.shape
    if budget <= 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                [-1], [dict()], visibility)

    logits = base_logits.float()
    device = logits.device
    w1 = markov_head.markov_w1.weight
    w2 = markov_head.markov_w2.weight

    # Measured hit data (experiments/analyze_tree_slots.py) shows the drafter is
    # confident near the root (87% of depth-1 acceptances are its top pick) and
    # uncertain deep (30% at depth 16), so a decaying schedule is backwards - an
    # explicit measured schedule beats any geometric curve.
    if widths:
        widths = list(widths)[:depth_limit]
    elif flat:
        widths = flat_width_schedule(budget, depth_limit, min_width)
    else:
        widths = beam_width_schedule(budget, depth_limit, decay)[:depth_limit]
    widths = [w for w in widths if w > 0]
    if not widths:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                [-1], [dict()], visibility)

    if per_depth:
        active_vocab = int(min(candidates or vocab_size, vocab_size))
        logits_active, cand_ids = torch.topk(logits[: len(widths)], k=active_vocab, dim=-1)
        w2_active = w2.index_select(0, cand_ids.reshape(-1)).view(len(widths), active_vocab, -1).float()
    elif candidates and candidates < vocab_size:
        cand_ids = torch.unique(torch.topk(logits, k=min(int(candidates), vocab_size), dim=-1).indices.reshape(-1))
        w2_active = w2.index_select(0, cand_ids).float()
        logits_active = logits.index_select(1, cand_ids)
        active_vocab = logits_active.shape[1]
    else:
        cand_ids = None
        w2_active = w2.float()
        logits_active = logits
        active_vocab = logits_active.shape[1]

    parent_tokens = torch.tensor([int(root_token_id)], dtype=torch.long, device=device)
    parent_scores = torch.zeros(1, dtype=torch.float32, device=device)
    levels = []  # (parent_local_idx, token_id) per level, all still on GPU

    for depth_idx, width in enumerate(widths):
        level_w2 = w2_active[depth_idx] if per_depth else w2_active
        bias = w1[parent_tokens].float() @ level_w2.T                  # [P, U]
        corrected = logits_active[depth_idx].unsqueeze(0) + bias       # [P, U]
        log_probs = corrected - torch.logsumexp(corrected, dim=-1, keepdim=True)
        path_scores = parent_scores.unsqueeze(1) + log_probs           # [P, U]

        flat_scores = path_scores.reshape(-1)   # not `flat` - that is the schedule flag
        k = min(width, flat_scores.numel())
        top_vals, top_flat = torch.topk(flat_scores, k)
        parent_local = torch.div(top_flat, active_vocab, rounding_mode="floor")
        token_local = top_flat % active_vocab
        if cand_ids is None:
            token_ids = token_local
        else:
            token_ids = cand_ids[depth_idx][token_local] if per_depth else cand_ids[token_local]

        levels.append((parent_local, token_ids))
        parent_tokens = token_ids            # stays on GPU - no sync
        parent_scores = top_vals

    # ONE transfer for the whole tree
    packed = torch.cat([torch.stack([p, t]) for p, t in levels], dim=1).cpu().numpy()
    level_sizes = [int(p.numel()) for p, _ in levels]

    node_token_ids_np = np.empty(sum(level_sizes), dtype=np.int64)
    node_depths_np = np.empty(sum(level_sizes), dtype=np.int64)
    parents_np = np.empty(sum(level_sizes) + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]

    node_count = 0
    prev_global = [0]                      # root
    offset = 0
    for depth_idx, size in enumerate(level_sizes):
        parent_local = packed[0, offset:offset + size]
        token_ids = packed[1, offset:offset + size]
        offset += size
        current_global = []
        for i in range(size):
            parent_index = prev_global[int(parent_local[i])]
            token_id = int(token_ids[i])
            current_index = node_count + 1
            node_token_ids_np[node_count] = token_id
            node_depths_np[node_count] = depth_idx + 1
            parents_np[current_index] = parent_index
            child_maps.append(dict())
            child_maps[parent_index][token_id] = current_index
            current_global.append(current_index)
            node_count += 1
        prev_global = current_global

    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True

    return (
        torch.from_numpy(node_token_ids_np[:node_count]),
        torch.from_numpy(node_depths_np[:node_count]),
        parents_np[:current_length].tolist(),
        child_maps,
        torch.from_numpy(visibility_np),
    )


@torch.inference_mode()
def ddtree_markov_generate(
    model: DSparkDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int = 64,
    use_markov: bool = True,
    tree_exact: bool = True,
    tree_candidates: int = 2048,
    tree_mode: str = "exact",
    beam_decay: float = 0.6,
    beam_widths: list[int] | None = None,
    beam_flat: bool = False,
    beam_cuda_graph: bool = False,
    beam_precompute: bool = True,
    beam_candidates: int = 512,
    beam_max_fanout: int = 0,
    beam_min_width: int = FLAT_MIN_WIDTH,
    collect_stats: bool = False,
) -> SimpleNamespace:
    """ddtree_generate adapted to the DSpark drafter, with selectable tree builder.

    Differences from ddtree_generate mirror dspark.py vs dflash.py: the drafter
    owns embed_tokens/lm_head, and its logits are next-token (all block_size rows
    are draft distributions, so tree depth limit = block_size, not block_size-1).
    """
    mask_token_id = model.mask_token_id
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size  # dspark: row i predicts depth i+1 -> block_size depths

    tree_budget = max(int(tree_budget), 1)
    max_tree_nodes = 1 + tree_budget

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    verify_input_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    verify_position_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=model.device,
    )
    tree_visibility_buffer = torch.empty((max_tree_nodes, max_tree_nodes), dtype=torch.bool, device=model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DDTREE_MARKOV_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = num_input_tokens
    acceptance_lengths = []
    round_timestamps = []
    accepted_slots: list[tuple[int, int]] = []   # (depth, slot-within-depth) of accepted nodes
    depth_reached: list[int] = []
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        root_token = output_ids[:, start : start + 1]

        draft_stage_start = cuda_time()
        draft_input_ids = torch.full(
            (1, block_size), mask_token_id, dtype=torch.long, device=output_ids.device
        )
        draft_input_ids[:, 0] = root_token[0]
        block_hidden = model(
            target_hidden=target_hidden,
            noise_embedding=model.embed_tokens(draft_input_ids),
            position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        past_key_values_draft.crop(start)
        base_logits = model.lm_head(block_hidden)  # [1, block_size, V], all rows kept
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        tree_build_start = cuda_time()
        if use_markov:
            if tree_mode == "beam" and beam_cuda_graph and base_logits.is_cuda:
                node_token_ids, node_depths, parents, child_maps, visibility_cpu = build_beam_tree_graphed(
                    base_logits[0], tree_budget, model.markov_head, int(root_token[0, 0]),
                    candidates=beam_candidates, widths=beam_widths, min_width=beam_min_width,
                )
            elif tree_mode == "beam" and beam_precompute:
                node_token_ids, node_depths, parents, child_maps, visibility_cpu = build_beam_tree_precomputed(
                    base_logits[0], tree_budget, model.markov_head, int(root_token[0, 0]),
                    candidates=beam_candidates, widths=beam_widths, min_width=beam_min_width,
                )
            elif tree_mode == "exact-precomputed":
                # best-first shape, but every children table already resolved, so
                # the heap costs one transfer instead of ~budget round-trips
                node_token_ids, node_depths, parents, child_maps, visibility_cpu = build_markov_tree_precomputed(
                    base_logits[0], tree_budget, model.markov_head, int(root_token[0, 0]),
                    candidates=beam_candidates, max_fanout=beam_max_fanout,
                )
            elif tree_mode == "confidence" and model.confidence_head is not None:
                # one cheap chain sample to get prev-token context for the head
                chain_tokens, _ = model.sample_draft_tokens(
                    base_logits, first_prev_token_ids=draft_input_ids[:, 0],
                    hidden_states=block_hidden, temperature=0.0,
                )
                prev_ids = torch.cat([draft_input_ids[:, :1], chain_tokens[:, :-1]], dim=1)
                conf = model.predict_confidence(block_hidden, prev_token_ids=prev_ids)
                probs = conf.sigmoid()[0].tolist()
                node_token_ids, node_depths, parents, child_maps, visibility_cpu = build_beam_tree(
                    base_logits[0], tree_budget, model.markov_head, int(root_token[0, 0]),
                    candidates=tree_candidates,
                    widths=confidence_width_schedule(probs, tree_budget, block_size),
                )
            elif tree_mode in ("beam", "confidence"):
                node_token_ids, node_depths, parents, child_maps, visibility_cpu = build_beam_tree(
                    base_logits[0], tree_budget, model.markov_head, int(root_token[0, 0]),
                    decay=beam_decay, candidates=tree_candidates, widths=beam_widths,
                    flat=beam_flat, min_width=beam_min_width,
                )
            else:
                node_token_ids, node_depths, parents, child_maps, visibility_cpu = build_markov_tree(
                    base_logits[0], tree_budget, model.markov_head, int(root_token[0, 0]),
                    exact=tree_exact, candidates=tree_candidates
                )
        else:
            node_token_ids, node_depths, parents, child_maps, visibility_cpu, _ = build_ddtree_tree(
                base_logits[0], tree_budget
            )
        stage_times["tree_build"] += cuda_time() - tree_build_start

        tree_compile_start = cuda_time()
        verify_input_ids, verify_position_ids, verify_attention_mask, previous_tree_start, previous_tree_length = compile_ddtree_tree(
            root_token_id=root_token[0, 0],
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            past_length=start,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        stage_times["tree_compile"] += cuda_time() - tree_compile_start

        verify_stage_start = cuda_time()
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=verify_input_ids.device)
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids).index_select(1, accepted_index_tensor)

        if collect_stats and node_depths.numel():
            # Slot = how many nodes at the same depth the builder emitted first.
            # For beam that is literally the level's top-k position; for best-first
            # it is the order best-first chose to materialize them. Either way it
            # answers "how wide did this depth need to be?".
            depths_list = node_depths.tolist()
            seen: dict[int, int] = {}
            slot_of = []
            for node_depth in depths_list:
                slot = seen.get(node_depth, 0)
                seen[node_depth] = slot + 1
                slot_of.append(slot)
            for accepted_index in accepted_indices[1:]:
                accepted_slots.append((int(depths_list[accepted_index - 1]), int(slot_of[accepted_index - 1])))
            depth_reached.append(len(accepted_indices) - 1)

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - len(accepted_indices) : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        accepted_slots=accepted_slots,
        depth_reached=depth_reached,
    )


@torch.inference_mode()
def ddtree_cross_markov_generate(
    model,  # DFlashDraftModel
    markov_head,  # DSpark's markov head (cross-applied guidance)
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int = 64,
) -> SimpleNamespace:
    """DDTree proper (DFlash drafter, horizon block_size-1), with the tree built
    by the markov-conditioned heap using DSpark's bigram head as guidance.

    Cross-model caveat: the head was trained to correct DSpark's backbone, not
    DFlash's. This tests whether it encodes transferable target-bigram structure.
    """
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1  # dflash: in-place denoise, row d-1 -> depth d

    tree_budget = max(int(tree_budget), 1)
    max_tree_nodes = 1 + tree_budget

    output_ids = torch.full(
        (1, max_length + max_tree_nodes), mask_token_id, dtype=torch.long, device=model.device
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    verify_input_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    verify_position_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes), dtype=target.dtype, device=model.device
    )
    tree_visibility_buffer = torch.empty((max_tree_nodes, max_tree_nodes), dtype=torch.bool, device=model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DDTREE_MARKOV_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = num_input_tokens
    acceptance_lengths = []
    round_timestamps = []
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

        draft_stage_start = cuda_time()
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, -draft_horizon:, :])
        past_key_values_draft.crop(start)
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        tree_build_start = cuda_time()
        node_token_ids, node_depths, parents, child_maps, visibility_cpu = build_markov_tree(
            draft_logits[0], tree_budget, markov_head, int(root_token[0, 0])
        )
        stage_times["tree_build"] += cuda_time() - tree_build_start

        tree_compile_start = cuda_time()
        verify_input_ids, verify_position_ids, verify_attention_mask, previous_tree_start, previous_tree_length = compile_ddtree_tree(
            root_token_id=root_token[0, 0],
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            past_length=start,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        stage_times["tree_compile"] += cuda_time() - tree_compile_start

        verify_stage_start = cuda_time()
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=verify_input_ids.device)
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids).index_select(1, accepted_index_tensor)

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - len(accepted_indices) : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
    )
