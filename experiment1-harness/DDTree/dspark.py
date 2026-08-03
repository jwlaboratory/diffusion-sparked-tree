"""DSpark decoding loop.

Same skeleton as dflash_generate. Three differences, all inside the draft stage:

  1. The backbone's logits are read at *every* block position and interpreted as
     next-token predictions, so a block_size-wide block yields block_size drafts
     (DFlash reads positions 1: and yields block_size - 1).
  2. Draft tokens are produced by a serial sweep of the markov head instead of one
     parallel argmax, so draft k+1 conditions on draft k. The backbone still runs
     exactly once per round; only a rank-r head is autoregressive.
  3. An optional confidence head truncates the proposal to the prefix the drafter
     believes will be accepted, trading draft length for a cheaper verify.

Verification and commit are unchanged from DFlash: exact-match prefix acceptance
plus one free token from the target.

Note on acceptance: DeepSpec verifies with rejection sampling (distribution
preserving at temperature > 0). This file uses exact-match acceptance to stay
consistent with dflash_generate and ddtree_generate, so the three methods are
compared under identical rules. At temperature 0 the two are equivalent; above it
this loop is greedy-lossless rather than sampling-lossless.
"""

from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DSparkDraftModel, sample, extract_context_feature
from dflash import cuda_time, empty_stage_times


DSPARK_STAGE_ORDER = (
    "draft_backbone",
    "draft_markov",
    "draft_confidence",
    "verify",
    "commit",
)


@torch.inference_mode()
def dspark_generate(
    model: DSparkDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    confidence_threshold: float = 0.0,
) -> SimpleNamespace:
    mask_token_id = model.mask_token_id
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    # One extra slot versus dflash: DSpark drafts block_size tokens rather than
    # block_size - 1, so the bonus token can land one position further out.
    output_ids = torch.full(
        (1, max_length + block_size + 1),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DSPARK_STAGE_ORDER)

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
    proposal_lengths = []
    round_timestamps = []
    draft_prefill = True

    while start < max_length:
        # Draft stage 1: one parallel backbone pass over [anchor, MASK, ..., MASK].
        backbone_stage_start = cuda_time()
        draft_input_ids = torch.full(
            (1, block_size),
            mask_token_id,
            dtype=torch.long,
            device=output_ids.device,
        )
        draft_input_ids[:, 0] = output_ids[:, start]
        block_hidden = model(
            target_hidden=target_hidden,
            noise_embedding=model.embed_tokens(draft_input_ids),
            position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        past_key_values_draft.crop(start)
        # Every position is kept: hidden state i predicts block token i + 1.
        base_draft_logits = model.lm_head(block_hidden)
        backbone_stage_elapsed = cuda_time() - backbone_stage_start

        # Draft stage 2: serial markov sweep. block_size tiny steps, no backbone work.
        markov_stage_start = cuda_time()
        draft_token_ids, _ = model.sample_draft_tokens(
            base_draft_logits,
            first_prev_token_ids=draft_input_ids[:, 0],
            temperature=0.0,
        )
        markov_stage_elapsed = cuda_time() - markov_stage_start

        # Draft stage 3: keep only the prefix the drafter is confident about.
        confidence_stage_start = cuda_time()
        num_draft_tokens = block_size
        if model.confidence_head is not None and confidence_threshold > 0.0:
            prev_token_ids = torch.cat([draft_input_ids[:, :1], draft_token_ids[:, :-1]], dim=1)
            confidence_logits = model.predict_confidence(block_hidden, prev_token_ids=prev_token_ids)
            if confidence_logits is not None:
                below_threshold = confidence_logits.sigmoid() < confidence_threshold
                if bool(below_threshold[0].any().item()):
                    num_draft_tokens = int(torch.nonzero(below_threshold[0], as_tuple=False)[0].item())
        confidence_stage_elapsed = cuda_time() - confidence_stage_start

        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft_backbone"] += backbone_stage_elapsed
            stage_times["draft_markov"] += markov_stage_elapsed
            stage_times["draft_confidence"] += confidence_stage_elapsed

        # Verify stage: [anchor, draft_1, ..., draft_n] is a plain chain, so the
        # target's default causal mask is correct (no tree mask, unlike ddtree).
        verify_input_ids = torch.cat([draft_input_ids[:, :1], draft_token_ids[:, :num_draft_tokens]], dim=1)
        verify_length = verify_input_ids.shape[1]
        proposal_lengths.append(num_draft_tokens)

        verify_stage_start = cuda_time()
        output = target(
            verify_input_ids,
            position_ids=position_ids[:, start : start + verify_length],
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        if num_draft_tokens == 0:
            acceptance_length = 0
        else:
            acceptance_length = (
                (verify_input_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            )
        output_ids[:, start : start + acceptance_length + 1] = verify_input_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[
            :, : acceptance_length + 1, :
        ]
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
        proposal_lengths=proposal_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
    )
