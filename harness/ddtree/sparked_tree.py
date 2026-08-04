"""Naive sparked tree decoding loop.

This is DDTree's best-first draft tree, but conditioned on the DSpark markov head.

DDTree builds its tree from a *single* parallel backbone pass: it takes the per-
position draft logits, computes one top-k table per depth, and every node at a
given depth draws its children from that same fixed table. That is the diffusion
independence assumption -- a node's continuation does not depend on which token its
parent actually took.

The naive sparked tree drops that assumption. The backbone still runs exactly once
(same cost as DDTree), producing base logits per position. But whenever the heap
expands a node, we *rerun the markov head* conditioned on that node's token to get
its children's distribution:

    child_logits = base_logits[parent_depth] + markov_head.bias(parent_token)

So each node's children are top-k of a distribution that knows what the parent
drafted, exactly like DSpark's serial sweep -- only now branched into a tree. The
top-k / log-softmax that DDTree does once up front therefore moves *inside* the
heap loop (one rerun per created node). It is "naive" because that rerun is done on
CPU per node with no batching; correctness and clarity over speed.

Everything downstream of the tree -- compile, tree-masked verify, follow-the-tree
commit, and KV compaction -- is reused unchanged from ddtree.py. With no markov
head this degenerates to a per-depth-independent tree (like DDTree, but with
DSpark's block_size drafts and next-token indexing).
"""

import heapq
import time
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import sample, extract_context_feature
from timing import cuda_time, sync_time, empty_stage_times
from ddtree import (
    DDTREE_TREE_BUILD_STAGE_ORDER,
    compile_ddtree_tree,
    follow_verified_tree,
    compact_dynamic_cache,
)


SPARKED_TREE_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "walk_accept", "kv_update", "state_carry")


# --------------------------------------------------------------------------- #
# Width schedules (beam mode). A schedule decides, IN ADVANCE, how many nodes  #
# each depth gets -- that is what makes level-synchronous expansion possible.  #
# --------------------------------------------------------------------------- #

# A flat schedule is only meaningful while every depth it covers gets more than
# one slot; below that it silently degenerates into a chain (see the docstring).
FLAT_MIN_WIDTH = 2


def flat_width_schedule(budget: int, depth_limit: int, min_width: int = FLAT_MIN_WIDTH) -> list[int]:
    """Uniform width per depth, over as many depths as the budget can afford to branch.

    Spreading has a floor: dividing a budget of 16 over 16 depths returns
    [1,1,...,1] -- a chain with zero branching, strictly worse than the same 16
    nodes spent on a shallower tree. Depth is therefore capped at
    budget // min_width rather than fixed at the drafter's horizon."""
    budget, depth_limit = int(budget), int(depth_limit)
    if budget <= 0 or depth_limit <= 0:
        return []
    depth = max(1, min(depth_limit, budget // max(int(min_width), 1)))
    base, extra = divmod(budget, depth)
    return [base + (1 if i < extra else 0) for i in range(depth)]


def flat_depth_schedule(budget: int, depth: int, depth_limit: int) -> list[int]:
    """Uniform width over a FIXED number of depths: trades depth for width while
    always spending the full budget (unlike fixing the width, which strands
    budget once depth_limit clamps)."""
    budget = int(budget)
    if budget <= 0 or depth_limit <= 0:
        return []
    depth = max(1, min(int(depth), int(depth_limit), budget))
    base, extra = divmod(budget, depth)
    return [base + (1 if i < extra else 0) for i in range(depth)]


def geometric_width_schedule(budget: int, depth_limit: int, decay: float = 0.6) -> list[int]:
    """Front-loaded: wide near the root, narrowing by `decay` per depth. This is
    the intuitive allocation (hedge early, deep nodes are rarely reached) and the
    one the old experiments measured to be backwards. Sums to <= budget."""
    widths: list[int] = []
    remaining = int(budget)
    width = max(1.0, budget * (1.0 - decay))
    for _ in range(int(depth_limit)):
        if remaining <= 0:
            break
        w = max(1, min(remaining, int(round(width))))
        widths.append(w)
        remaining -= w
        width *= decay
    return widths


def resolve_width_schedule(spec: dict, budget: int, depth_limit: int) -> list[int]:
    """Schedule spec -> per-depth widths. Kinds:

      {"kind": "flat", "min_width": 2}         uniform over affordable depths
      {"kind": "flat_depth", "depth": 8}       uniform over exactly `depth` levels
      {"kind": "geometric", "decay": 0.6}      front-loaded (decaying)
      {"kind": "inv_geometric", "decay": 0.6}  back-loaded: the geometric widths
                                               reversed -- same multiset of widths,
                                               same depth extent, opposite
                                               orientation (the clean mirror control)
      {"kind": "explicit", "widths": [...]}    verbatim
    """
    kind = spec.get("kind", "flat")
    if kind == "flat":
        return flat_width_schedule(budget, depth_limit, spec.get("min_width", FLAT_MIN_WIDTH))
    if kind == "flat_depth":
        return flat_depth_schedule(budget, spec["depth"], depth_limit)
    if kind == "geometric":
        return geometric_width_schedule(budget, depth_limit, spec.get("decay", 0.6))
    if kind == "inv_geometric":
        return list(reversed(geometric_width_schedule(budget, depth_limit, spec.get("decay", 0.6))))
    if kind == "explicit":
        return [int(w) for w in spec["widths"]]
    raise ValueError(f"unknown width schedule kind {kind!r}")


def build_sparked_tree(
    base_logits: torch.Tensor,
    markov_head,
    root_token_id: int,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor, dict[str, float]]:
    """Best-first tree whose branching is conditioned on the markov head.

    base_logits: [depth_limit, vocab] backbone next-token logits (one row per block
        position; row d feeds depth-(d+1) nodes).
    markov_head: DSpark VanillaMarkovHead, or None for a per-depth-independent tree.
    root_token_id: the committed anchor token; conditions the depth-1 distribution.
    """
    build_subtimes = empty_stage_times(DDTREE_TREE_BUILD_STAGE_ORDER)

    if budget <= 0 or base_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            build_subtimes,
        )

    depth_limit = int(base_logits.shape[0])
    topk = min(budget, base_logits.shape[-1])

    # Copy the base logits (and the two tiny markov matrices) to CPU once. Unlike
    # DDTree we cannot pre-topk here: the distribution is not known until we know
    # which token each parent took, so the topk happens per node inside the loop.
    copy_start = cuda_time()
    base_cpu = base_logits.float().to(device="cpu")
    markov_w1 = markov_w2 = None
    if markov_head is not None:
        markov_w1 = markov_head.markov_w1.weight.detach().float().to(device="cpu")  # [vocab, rank]
        markov_w2 = markov_head.markov_w2.weight.detach().float().to(device="cpu")  # [vocab, rank]
    build_subtimes["tree_build_copy"] = cuda_time() - copy_start

    def conditioned_topk(parent_depth: int, prev_token_id: int):
        """Top-k (token_ids, log_probs) for the children of a node.

        This is the per-step markov rerun: base logits for this depth plus the
        additive bias markov_w2(markov_w1[prev_token]) = W2 @ W1[prev_token]."""
        logits = base_cpu[parent_depth]
        if markov_head is not None:
            bias = markov_w2 @ markov_w1[prev_token_id]  # [vocab]
            logits = logits + bias
        log_probs = torch.log_softmax(logits, dim=-1)
        top_lp, top_ids = torch.topk(log_probs, k=topk)
        return top_ids.tolist(), top_lp.tolist()

    heap_start = time.perf_counter()

    # Per-node conditioned distributions, keyed by node index (0 == root/anchor).
    dist_tokens: dict[int, list[int]] = {}
    dist_logprobs: dict[int, list[float]] = {}
    root_tokens, root_logprobs = conditioned_topk(0, int(root_token_id))
    dist_tokens[0] = root_tokens
    dist_logprobs[0] = root_logprobs

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    # Heap entry: (-logw, parent_index, depth, rank, logw). A node's token is the
    # rank-th entry of its parent's conditioned distribution.
    first_logw = float(root_logprobs[0])
    heap: list[tuple[float, int, int, int, float]] = [(-first_logw, 0, 1, 0, first_logw)]

    while heap and node_count < budget:
        _, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(dist_tokens[parent_index][rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        # Sibling: the next-best child of the SAME parent, so it reuses the parent's
        # already-computed distribution (no rerun).
        parent_logprobs = dist_logprobs[parent_index]
        if rank + 1 < len(parent_logprobs):
            sibling_logw = logw - parent_logprobs[rank] + parent_logprobs[rank + 1]
            heapq.heappush(heap, (-sibling_logw, parent_index, depth, rank + 1, sibling_logw))

        # Child: rerun the markov head conditioned on THIS node's token to get its
        # children's distribution, then push its best child.
        if depth < depth_limit:
            child_tokens, child_logprobs = conditioned_topk(depth, token_id)
            dist_tokens[current_index] = child_tokens
            dist_logprobs[current_index] = child_logprobs
            child_logw = logw + float(child_logprobs[0])
            heapq.heappush(heap, (-child_logw, current_index, depth + 1, 0, child_logw))

    build_subtimes["tree_build_heap"] = time.perf_counter() - heap_start

    visibility_start = time.perf_counter()
    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True
    build_subtimes["tree_build_visibility"] = time.perf_counter() - visibility_start

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()

    return node_token_ids, node_depths, parents, child_maps, visibility, build_subtimes


def build_beam_tree(
    base_logits: torch.Tensor,
    markov_head,
    root_token_id: int,
    widths: list[int],
    candidates: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor, dict[str, float]]:
    """Level-synchronous (beam) markov tree -- one batched expansion per DEPTH.

    build_sparked_tree expands one node at a time, and each expansion needs the
    markov rerun conditioned on that node's token: the table for node n+1 is not
    known until node n is popped, so the builder pays ~budget dependent
    round-trips per round. That serial dependency -- not arithmetic -- is what
    made the naive tree ~20x more expensive than DDTree's.

    Fixing the tree SHAPE up front removes it. `widths[d]` nodes at depth d+1 are
    decided in advance; each level expands ALL its surviving parents in one
    batched matmul (bias = W1[parents] @ W2_active^T), takes a single top-k over
    [parents x candidates], and the winners stay on GPU as the next level's
    parents. Every transfer is deferred to a single .cpu() at the end: 1 sync per
    round instead of ~budget.

    The cost is that budget is allocated by the fixed schedule rather than
    adaptively by path score (beam search, not best-first) -- which is exactly
    the knob the width-schedule experiment sweeps.

    Candidate restriction: only the union of the per-depth top-`candidates`
    tokens can ever win a slot (bias magnitudes are small relative to the logit
    spread), so W2 is gathered once against that union and each level's matmul
    runs on the ~C-column slice instead of the full vocab. candidates=0 disables.
    """
    build_subtimes = empty_stage_times(DDTREE_TREE_BUILD_STAGE_ORDER)

    depth_limit = int(base_logits.shape[0])
    widths = [int(w) for w in list(widths)[:depth_limit] if int(w) > 0]
    if not widths or depth_limit == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            build_subtimes,
        )

    prep_start = cuda_time()
    logits = base_logits.float()
    device = logits.device
    vocab_size = int(logits.shape[-1])
    w1 = w2_active = None
    if markov_head is not None:
        w1 = markov_head.markov_w1.weight.detach().float()   # [vocab, rank]
        w2 = markov_head.markov_w2.weight.detach().float()   # [vocab, rank]
    if candidates and candidates < vocab_size:
        cand_ids = torch.unique(
            torch.topk(logits, k=min(int(candidates), vocab_size), dim=-1).indices.reshape(-1)
        )
        logits_active = logits.index_select(1, cand_ids)     # [L, U]
        if markov_head is not None:
            w2_active = w2.index_select(0, cand_ids)         # [U, rank]
    else:
        cand_ids = None
        logits_active = logits
        if markov_head is not None:
            w2_active = w2
    active_vocab = int(logits_active.shape[-1])
    build_subtimes["tree_build_copy"] = cuda_time() - prep_start

    loop_start = cuda_time()
    parent_tokens = torch.tensor([int(root_token_id)], dtype=torch.long, device=device)
    parent_scores = torch.zeros(1, dtype=torch.float32, device=device)
    level_parents, level_tokens, emitted = [], [], []
    for depth_idx, width in enumerate(widths):
        if w2_active is not None:
            bias = w1.index_select(0, parent_tokens) @ w2_active.T           # [P, U]
            corrected = logits_active[depth_idx].unsqueeze(0) + bias
        else:
            # No corrector: every parent shares the depth's base distribution.
            corrected = logits_active[depth_idx].unsqueeze(0).expand(parent_tokens.shape[0], -1)
        log_probs = corrected - torch.logsumexp(corrected, dim=-1, keepdim=True)
        path_scores = parent_scores.unsqueeze(1) + log_probs                 # [P, U]

        flat_scores = path_scores.reshape(-1)
        k = min(int(width), int(flat_scores.numel()))
        top_vals, top_flat = torch.topk(flat_scores, k)
        parent_local = torch.div(top_flat, active_vocab, rounding_mode="floor")
        token_local = top_flat % active_vocab
        token_ids = token_local if cand_ids is None else cand_ids.index_select(0, token_local)

        level_parents.append(parent_local)
        level_tokens.append(token_ids)
        emitted.append(k)
        parent_tokens = token_ids            # stays on GPU -- no sync
        parent_scores = top_vals
    # ONE transfer for the whole tree (this is the builder's only sync point).
    packed = torch.stack([torch.cat(level_parents), torch.cat(level_tokens)]).cpu().numpy()
    build_subtimes["tree_build_heap"] = cuda_time() - loop_start

    tail_start = time.perf_counter()
    parent_local_np, token_np = packed[0], packed[1]
    total = int(parent_local_np.shape[0])
    node_token_ids_np = np.empty(total, dtype=np.int64)
    node_depths_np = np.empty(total, dtype=np.int64)
    parents_np = np.empty(total + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]

    node_count, prev_global, offset = 0, [0], 0
    for depth_idx, width in enumerate(emitted):
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
    build_subtimes["tree_build_visibility"] = time.perf_counter() - tail_start

    return (
        torch.from_numpy(node_token_ids_np[:node_count]),
        torch.from_numpy(node_depths_np[:node_count]),
        parents_np[:current_length].tolist(),
        child_maps,
        torch.from_numpy(visibility_np),
        build_subtimes,
    )


@torch.inference_mode()
def sparked_tree_generate(
    model,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
    markov_head=None,
    draft_mode: str = "dspark",
    tree_mode: str = "best-first",
    beam_schedule: dict | None = None,
    beam_candidates: int = 2048,
    probe_markov_head=None,
    save_tree_traces: bool = False,
) -> SimpleNamespace:
    """Tree decoding with a pluggable drafter and corrector.

    draft_mode:
      "dspark" - model is a DSparkDraftModel; next-token indexing, block_size
                 drafts, model owns embed_tokens/lm_head.
      "dflash" - model is a DFlashDraftModel; in-place indexing, block_size - 1
                 drafts, embeddings/lm_head borrowed from the target.
    markov_head: the DSpark markov head to use as the per-step corrector, or None
      for a per-depth-independent tree. It is token-only, so it can be spliced onto
      either backbone.
    tree_mode:
      "best-first" - the naive per-node builder (build_sparked_tree), adaptive
                     budget allocation, ~budget markov reruns per round.
      "beam"       - level-synchronous builder (build_beam_tree): tree shape fixed
                     up front by `beam_schedule`, one batched expansion per depth.
    beam_schedule: width-schedule spec for tree_mode="beam"; see
      resolve_width_schedule. Defaults to {"kind": "flat"}. Resolved once per
      generate call from (tree_budget, depth_limit).
    beam_candidates: candidate-union restriction for the beam builder (0 = full
      vocab).
    probe_markov_head: measurement-only. When set (typically on a *no-corrector*
      run), we record, at each committed position along the accepted path, whether
      argmax(base_logits) and argmax(base_logits + bias(prev)) match the target's
      token, plus their cross-entropies. This is the confound-free "does this head
      fit this backbone?" probe: real positions, real previous tokens, no tree/depth
      extrapolation. It never affects the tree that is actually built. Returned as
      `probe_by_depth`. Ignored if markov_head is also set (the corrector is live).
    """
    if draft_mode not in ("dspark", "dflash"):
        raise ValueError(f"draft_mode must be 'dspark' or 'dflash', got {draft_mode!r}")
    if tree_mode not in ("best-first", "beam"):
        raise ValueError(f"tree_mode must be 'best-first' or 'beam', got {tree_mode!r}")
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    # DSpark is next-token (block_size drafts, tree up to block_size deep); DFlash is
    # in-place (block_size - 1 drafts). The corrector, if any, is applied identically
    # in both: bias(parent_token) added to each child's base logits.
    depth_limit = block_size if draft_mode == "dspark" else block_size - 1
    tree_budget = depth_limit if tree_budget is None else max(tree_budget, 0)
    max_tree_nodes = 1 + tree_budget

    # Beam widths are a pure function of (schedule, budget, depth_limit); resolve
    # once, reuse every round. The emitted node count is sum(widths) <= budget.
    beam_widths = None
    if tree_mode == "beam":
        beam_widths = resolve_width_schedule(
            beam_schedule or {"kind": "flat"}, tree_budget, depth_limit)
        if sum(beam_widths) > tree_budget:
            raise ValueError(
                f"beam schedule {beam_schedule!r} spends {sum(beam_widths)} nodes "
                f"on a budget of {tree_budget}")

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
    stage_times = empty_stage_times(SPARKED_TREE_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

    prefill_start = sync_time()
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

    time_to_first_token = sync_time() - prefill_start

    # Decode window = everything after prefill, cold round included; the first loop
    # iteration (draft-KV prefill, lazy kernel init) stays out of every stage bucket
    # and is reported as cold_round_time.
    decode_start = sync_time()
    round_clock_start = decode_start
    start = num_input_tokens
    acceptance_lengths = []
    round_timestamps = []
    round_trees = [] if save_tree_traces else None
    cold_round_time = None
    previous_tree_start = 0
    previous_tree_length = 0

    # Corrector-fit probe (measurement-only; never touches the live tree). Active
    # when a probe head is supplied and the run itself is not already corrected.
    probe_active = probe_markov_head is not None and markov_head is None
    probe_by_depth: dict[int, dict[str, float]] = {}
    if probe_active:
        probe_w1 = probe_markov_head.markov_w1.weight.detach().float()  # [vocab, rank]
        probe_w2 = probe_markov_head.markov_w2.weight.detach().float()  # [vocab, rank]

    while start < max_length:
        is_cold = cold_round_time is None
        # Draft stage: one parallel DSpark backbone pass over [anchor, MASK, ...].
        draft_input_ids = torch.full(
            (1, block_size),
            mask_token_id,
            dtype=torch.long,
            device=output_ids.device,
        )
        draft_input_ids[:, 0] = output_ids[:, start]
        root_token = draft_input_ids[:, :1]

        draft_stage_start = cuda_time()
        # DSpark owns embed/lm_head; DFlash borrows the target's.
        if draft_mode == "dspark":
            noise_embedding = model.embed_tokens(draft_input_ids)
        else:
            noise_embedding = target.model.embed_tokens(draft_input_ids)
        block_hidden = model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        past_key_values_draft.crop(start)
        if draft_mode == "dspark":
            # Next-token: every position kept, hidden i predicts block token i + 1.
            base_draft_logits = model.lm_head(block_hidden)
        else:
            # In-place: positions 1: predict their own token -> block_size - 1 rows.
            base_draft_logits = target.lm_head(block_hidden[:, -depth_limit:, :])
        if not is_cold:
            stage_times["draft"] += cuda_time() - draft_stage_start

        # Tree build: markov-conditioned expansion (the rerun happens here).
        tree_build_start = cuda_time()
        if tree_mode == "beam":
            node_token_ids, node_depths, parents, child_maps, visibility_cpu, tree_build_subtimes = build_beam_tree(
                base_draft_logits[0],
                markov_head,
                int(root_token[0, 0]),
                beam_widths,
                beam_candidates,
            )
        else:
            node_token_ids, node_depths, parents, child_maps, visibility_cpu, tree_build_subtimes = build_sparked_tree(
                base_draft_logits[0],
                markov_head,
                int(root_token[0, 0]),
                tree_budget,
            )
        if not is_cold:
            stage_times["tree_build"] += cuda_time() - tree_build_start
            for stage_name, stage_elapsed in tree_build_subtimes.items():
                stage_times[stage_name] += stage_elapsed

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
        if not is_cold:
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
        if not is_cold:
            stage_times["verify"] += cuda_time() - verify_stage_start

        walk_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=verify_input_ids.device)
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token
        if not is_cold:
            stage_times["walk_accept"] += cuda_time() - walk_stage_start

        # Corrector-fit probe: along the committed path, compare base vs
        # base+bias(prev) against the target's actual token. Depth d uses base row
        # d-1; prev token is the committed token at depth d-1 (anchor at d=1). Depth
        # L is the rejection/bonus point (target token = next_token).
        if probe_active:
            accepted_len = len(accepted_indices)
            n_rows = base_draft_logits.shape[1]
            for depth in range(1, accepted_len + 1):
                row_idx = depth - 1
                if row_idx >= n_rows:
                    break
                prev_tok = int(accepted_tokens[0, depth - 1])
                true_tok = int(accepted_tokens[0, depth]) if depth < accepted_len else int(next_token)
                base_row = base_draft_logits[0, row_idx].float()
                corr_row = base_row + (probe_w2 @ probe_w1[prev_tok])
                base_hit = int(int(base_row.argmax()) == true_tok)
                corr_hit = int(int(corr_row.argmax()) == true_tok)
                ce_base = float(-torch.log_softmax(base_row, dim=-1)[true_tok])
                ce_corr = float(-torch.log_softmax(corr_row, dim=-1)[true_tok])
                bucket = probe_by_depth.setdefault(
                    depth, {"n": 0, "base_hit": 0, "corr_hit": 0, "ce_base": 0.0, "ce_corr": 0.0}
                )
                bucket["n"] += 1
                bucket["base_hit"] += base_hit
                bucket["corr_hit"] += corr_hit
                bucket["ce_base"] += ce_base
                bucket["ce_corr"] += ce_corr

        kv_stage_start = cuda_time()
        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        if not is_cold:
            stage_times["kv_update"] += cuda_time() - kv_stage_start

        carry_stage_start = cuda_time()
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids).index_select(1, accepted_index_tensor)
        if not is_cold:
            stage_times["state_carry"] += cuda_time() - carry_stage_start

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        if is_cold:
            cold_round_time = cuda_time() - decode_start
        round_timestamps.append(cuda_time() - round_clock_start)
        if save_tree_traces:
            round_trees.append({
                "accepted_indices": [int(index) for index in accepted_indices],
                "tree": {
                    "node_token_ids": [int(token_id) for token_id in node_token_ids.tolist()],
                    "node_depths": [int(depth) for depth in node_depths.tolist()],
                    "parents": [int(parent) for parent in parents],
                },
            })

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
    total_decode_time = sync_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        total_decode_time=total_decode_time,
        cold_round_time=cold_round_time or 0.0,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        round_trees=round_trees,
        probe_by_depth=probe_by_depth,
    )
