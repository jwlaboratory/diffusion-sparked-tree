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
def build_markov_tree(
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
            node_token_ids, node_depths, parents, child_maps, visibility_cpu = build_markov_tree(
                base_logits[0], tree_budget, model.markov_head, int(root_token[0, 0])
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
