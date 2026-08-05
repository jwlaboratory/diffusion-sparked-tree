"""Plain autoregressive decode of the target model -- the 1x speedup baseline.

No drafter, no tree: one target forward per output token. This is the denominator
every speculative method's "N x speedup" is measured against (the DDTree/DFlash
papers report speedup vs exactly this). It emits the same SimpleNamespace contract
as the speculative generators so it slots straight into the driver's two-pass /
rollup machinery. Every round accepts exactly one token, so acceptance_lengths is
all 1s and mean_accept == 1.0 by construction.
"""

from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import sample
from timing import cuda_time, sync_time, empty_stage_times

# AR has one real stage: the target forward ("verify", to align with the canonical
# phase set). draft/tree phases are structurally absent -> reported as None.
AUTOREGRESSIVE_STAGE_ORDER = ("verify",)


@torch.inference_mode()
def autoregressive_generate(
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> SimpleNamespace:
    device = target.device
    num_input_tokens = input_ids.shape[1]
    stop = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=device)
    stage_times = empty_stage_times(AUTOREGRESSIVE_STAGE_ORDER)

    past = DynamicCache()

    # Prefill -> first token (charged to TTFT, exactly like the spec generators).
    prefill_start = sync_time()
    out = target(input_ids, past_key_values=past, use_cache=True, logits_to_keep=1)
    next_tok = sample(out.logits, temperature)            # [1, 1]
    time_to_first_token = sync_time() - prefill_start

    generated = [int(next_tok[0, 0])]
    decode_start = sync_time()
    round_clock_start = decode_start
    acceptance_lengths: list[int] = []
    round_timestamps: list[float] = []
    cold_round_time = None
    cur = num_input_tokens                                # position of next_tok

    while len(generated) < max_new_tokens:
        if stop is not None and torch.isin(next_tok[0], stop).any():
            break
        is_cold = cold_round_time is None
        verify_start = cuda_time()
        pos = torch.tensor([[cur]], device=device)
        out = target(next_tok, position_ids=pos, past_key_values=past,
                     use_cache=True, logits_to_keep=1)
        next_tok = sample(out.logits, temperature)
        if not is_cold:
            stage_times["verify"] += cuda_time() - verify_start
        else:
            cold_round_time = cuda_time() - decode_start
        generated.append(int(next_tok[0, 0]))
        acceptance_lengths.append(1)                      # AR: exactly one token/round
        round_timestamps.append(cuda_time() - round_clock_start)
        cur += 1

    # Trim at the first stop token (inclusive), matching the spec generators.
    if stop_token_ids is not None:
        for i, tok in enumerate(generated):
            if tok in stop_token_ids:
                generated = generated[: i + 1]
                break

    num_output_tokens = len(generated)
    total_decode_time = sync_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=torch.tensor([list(input_ids[0].tolist()) + generated]).cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        total_decode_time=total_decode_time,
        cold_round_time=cold_round_time or 0.0,
        acceptance_lengths=acceptance_lengths,
        proposal_lengths=[1] * len(acceptance_lengths),
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
    )
