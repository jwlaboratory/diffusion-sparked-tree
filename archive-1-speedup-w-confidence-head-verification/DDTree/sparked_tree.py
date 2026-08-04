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
from dflash import cuda_time, empty_stage_times
from confidence import round_confidence, stack_rounds, PRED_CHAIN_LEN
from ddtree import (
    DDTREE_TREE_BUILD_STAGE_ORDER,
    compile_ddtree_tree,
    follow_verified_tree,
    compact_dynamic_cache,
)


SPARKED_TREE_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")


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
    probe_markov_head=None,
    save_tree_traces: bool = False,
    measure_confidence: bool = False,
    confidence_gate: tuple[float, int] | None = None,
    tree_acceptance_head=None,
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
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    # DSpark is next-token (block_size drafts, tree up to block_size deep); DFlash is
    # in-place (block_size - 1 drafts). The corrector, if any, is applied identically
    # in both: bias(parent_token) added to each child's base logits.
    depth_limit = block_size if draft_mode == "dspark" else block_size - 1
    tree_budget = depth_limit if tree_budget is None else max(tree_budget, 0)
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
    stage_times = empty_stage_times(SPARKED_TREE_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

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
    round_trees = [] if save_tree_traces else None
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0
    # One tensor per round, stacked and transferred once at the end. Paired
    # positionally with acceptance_lengths, so a single run yields
    # (confidence, accepted) for every round -- no cross-run alignment, which is
    # what broke the offline analysis in investigate/FINDINGS.md finding 1.
    confidence_rounds: list[torch.Tensor] = []
    gate_budgets: list[int] = []
    # The gate needs a host-side scalar to size the tree, so it costs one sync per
    # round. Negligible here: build_sparked_tree is a CPU heap doing a full-vocab
    # matmul per node, which dominates by orders of magnitude.
    gate_active = confidence_gate is not None

    # Corrector-fit probe (measurement-only; never touches the live tree). Active
    # when a probe head is supplied and the run itself is not already corrected.
    probe_active = probe_markov_head is not None and markov_head is None
    probe_by_depth: dict[int, dict[str, float]] = {}
    if probe_active:
        probe_w1 = probe_markov_head.markov_w1.weight.detach().float()  # [vocab, rank]
        probe_w2 = probe_markov_head.markov_w2.weight.detach().float()  # [vocab, rank]

    while start < max_length:
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
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        # Confidence. Measurement sits between the two stage timers so it lands in
        # no bucket; the gate is charged to tree_build, since sizing the tree is
        # genuinely part of building it.
        round_conf = None
        if measure_confidence or gate_active:
            round_conf = round_confidence(base_draft_logits[0])
        if measure_confidence:
            confidence_rounds.append(round_conf)

        # Tree build: markov-conditioned best-first expansion (the rerun happens here).
        tree_build_start = cuda_time()
        round_budget = tree_budget
        if gate_active:
            threshold, small_budget = confidence_gate
            # Predicted acceptance for THIS round: the trained tree-aware head when
            # supplied, else the free chain estimate. Shrink only when the drafter is
            # CONFIDENT -- finding 5: rounds where the chain accepts <=3 are 64% of
            # rounds and carry 71% of everything tree width produces.
            if tree_acceptance_head is not None:
                with torch.no_grad():
                    predicted = float(tree_acceptance_head(round_conf))
            else:
                predicted = float(round_conf[PRED_CHAIN_LEN])
            if predicted >= threshold:
                round_budget = small_budget
            gate_budgets.append(round_budget)

        node_token_ids, node_depths, parents, child_maps, visibility_cpu, tree_build_subtimes = build_sparked_tree(
            base_draft_logits[0],
            markov_head,
            int(root_token[0, 0]),
            round_budget,
        )
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

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids).index_select(1, accepted_index_tensor)

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        stage_times["commit"] += cuda_time() - commit_stage_start
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
        round_trees=round_trees,
        probe_by_depth=probe_by_depth,
        confidence_by_round=stack_rounds(confidence_rounds),
        gate_budgets=gate_budgets,
    )
