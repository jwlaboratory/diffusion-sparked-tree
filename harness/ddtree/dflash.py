from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature
from timing import cuda_time, sync_time, empty_stage_times


# Exp3 canonical split: "commit" is gone, replaced by walk_accept / kv_update /
# state_carry, and the parallel argmax is broken out of "draft" as candidate_build
# so chain arms have a candidate-generation segment comparable to the tree arms'.
DFLASH_STAGE_ORDER = ("draft", "candidate_build", "verify", "walk_accept", "kv_update", "state_carry")


@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DFLASH_STAGE_ORDER)

    prefill_start = sync_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = sync_time() - prefill_start

    # Decode window = everything after prefill, cold round included. The first loop
    # iteration carries the draft-KV prefill and any lazy kernel init, so it is kept
    # out of every stage bucket and reported as cold_round_time instead; the runner
    # computes unaccounted = total - cold_round - sum(stages).
    decode_start = sync_time()
    round_clock_start = decode_start
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    cold_round_time = None

    while start < max_length:
        is_cold = cold_round_time is None
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            draft_stage_start = cuda_time()
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size + 1 :, :])
            past_key_values_draft.crop(start)
            if not is_cold:
                stage_times["draft"] += cuda_time() - draft_stage_start

            # Candidate build: for a chain drafter, just the parallel argmax.
            candidate_stage_start = cuda_time()
            block_output_ids[:, 1:] = sample(draft_logits)
            if not is_cold:
                stage_times["candidate_build"] += cuda_time() - candidate_stage_start

        verify_stage_start = cuda_time()
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )
        if not is_cold:
            stage_times["verify"] += cuda_time() - verify_stage_start

        walk_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        if not is_cold:
            stage_times["walk_accept"] += cuda_time() - walk_stage_start

        kv_stage_start = cuda_time()
        past_key_values_target.crop(start)
        if not is_cold:
            stage_times["kv_update"] += cuda_time() - kv_stage_start

        carry_stage_start = cuda_time()
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, : acceptance_length + 1, :]
        if not is_cold:
            stage_times["state_carry"] += cuda_time() - carry_stage_start

        if is_cold:
            cold_round_time = cuda_time() - decode_start
        round_timestamps.append(cuda_time() - round_clock_start)

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - acceptance_length - 1 : start + 1]
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
    )
