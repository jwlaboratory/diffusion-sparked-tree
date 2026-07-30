import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature


DFLASH_STAGE_ORDER = ("draft", "verify", "commit")


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

    ## figure out how many total tokens need to be generated (input tokens (all tokens) + output tokens length))
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    # this max length is geniunly the MAX. not per round (each round, drafter drafts 1 block which is 1 denoising step). the query might require multiple rounds, bcz the max new tokens is geniungly can be like "create 100 new tokens", so it needs ot do 100

    output_ids = torch.full(
        (1, max_length + block_size), # extra padding , one buffer to write into with all set to MASK TOKENS
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    # creates new EMPTY Caches that we can use. 
    past_key_values_target = DynamicCache() #useful for the targets croppe daccepted kv
    past_key_values_draft = DynamicCache() # useful for the  gets cropped back to start every single round (dflash.py:78), because the drafter's KV for the masked block is throwaway
    stage_times = empty_stage_times(DFLASH_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target( #doesnt generate just creates the stuff
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True, # we build cache as a side effect 
        logits_to_keep=1, # means apply the FINAL LM head to the LAST one only -- the LM head for actual prediciton is useless for the first few tokens 
        output_hidden_states=True if block_size > 1 else False, # get hidden states 
    )
    # get the target to do the prefil, ()

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature) #gets the first token
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
        # is this getting the hidedn states of the target model

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    draft_prefill = True

    while start < max_length: 
        # max is geniuny the max amount
        # max is different from block size / per round # of tokens generated.

        block_output_ids = output_ids[:, start : start + block_size].clone() 
        block_position_ids = position_ids[:, start : start + block_size]
        # this does this [LAST_KNOWN_TOKEN, mask, mask, mask ..... [block_size]]
        

        if block_size > 1: # if we want to even run the frafter
            draft_stage_start = cuda_time()
            noise_embedding = target.model.embed_tokens(block_output_ids) #use the target model token embeddings for the first LAST_KNOWN_TOKEN
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden, # use target hidden states
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False, #non casual bc diffusion
            )[:, -block_size + 1 :, :])
            past_key_values_draft.crop(start) #crop out the GUESSES . GO back to teh START (KNOWN VERIFIED ) 
            block_output_ids[:, 1:] = sample(draft_logits) #sample the outputs
            draft_stage_elapsed = cuda_time() - draft_stage_start
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()
            else:
                stage_times["draft"] += draft_stage_elapsed


        # verify start

        # The targets KV is only used for verify. its never used for drafting

        verify_stage_start = cuda_time()
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, : acceptance_length + 1, :]
        stage_times["commit"] += cuda_time() - commit_stage_start
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


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def empty_stage_times(stage_names: tuple[str, ...]) -> dict[str, float]:
    return {stage_name: 0.0 for stage_name in stage_names}
