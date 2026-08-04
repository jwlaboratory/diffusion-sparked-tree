"""DSpark draft model (semi-autoregressive block drafter).

DSpark keeps DFlash's parallel block-diffusion backbone and adds two small heads:

  * markov_head     - a low-rank additive logit bias conditioned on the *previous*
                      drafted token. The backbone still runs once in parallel, but
                      the head is applied in a short serial loop over the block, so
                      draft token k+1 is aware of draft token k. This is what
                      recovers the intra-block dependency a pure parallel drafter
                      throws away.
  * confidence_head - predicts per-position acceptance probability, so the caller
                      can truncate the proposal to a prefix it actually believes in
                      instead of always paying to verify the full block.

Indexing differs from DFlash. DFlash denoises in place: hidden state at block
position i predicts the token at position i. DSpark is next-token: hidden state at
block position i predicts the token at position i+1. So a block of `block_size`
inputs yields `block_size` drafts (DFlash yields `block_size - 1`).

Unlike DFlashDraftModel, this model owns `embed_tokens` and `lm_head` (copied from
the target at training time and frozen) rather than borrowing the target's.

The backbone is *not* reimplemented here. DSpark's attention, decoder layer, and
rotary embedding are byte-for-byte identical to DFlash's, so we reuse
``Qwen3DFlashDecoderLayer`` directly and inherit ``DFlashDraftModel.forward``. The
class name of the layer does not affect checkpoint loading -- ``from_pretrained``
matches on submodule attribute names (``layers.N.self_attn.q_proj``...), which are
identical -- so released DSpark checkpoints load unchanged.

Ported from deepseek-ai/DeepSpec for inference only; training paths (anchor
sampling, losses, flex_attention masks) are omitted. Module names match the
released checkpoints so `from_pretrained` loads them unchanged.
"""

from typing import Optional

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
)

from .dflash import DFlashDraftModel, Qwen3DFlashDecoderLayer
from .utils import sample


class VanillaMarkovHead(nn.Module):
    """Rank-r additive logit bias conditioned on the previous drafted token.

    bias(x) = markov_w2(markov_w1[x]), so the whole head is two matrices of shape
    [vocab, r] and [r, vocab]. Cheap enough to run serially inside a block.
    """

    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.get_prev_embeddings(token_ids))

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Serial sweep over the block. The backbone already ran once in parallel;
        only this tiny head is autoregressive."""
        bsz, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(bsz, 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            step_logits = base_logits[:, step_idx, :] + self.compute_step_bias(prev_token_ids)
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = sample(step_logits.unsqueeze(1), temperature).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


class AcceptRatePredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features):
        return self.proj(features).squeeze(-1)


def build_markov_head(config) -> Optional[nn.Module]:
    markov_rank = int(getattr(config, "markov_rank", 0))
    if markov_rank == 0:
        return None
    head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
    if head_type != "vanilla":
        raise NotImplementedError(
            f"markov_head_type={head_type!r} is not ported here; see deepspec/modeling/dspark/markov_head.py"
        )
    return VanillaMarkovHead(vocab_size=config.vocab_size, markov_rank=markov_rank)


class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone + markov/confidence heads.

    Subclasses ``DFlashDraftModel`` purely to inherit its ``forward`` (identical).
    The backbone stack is built from the imported ``Qwen3DFlashDecoderLayer``, so
    no attention/layer/rotary code is duplicated here.
    """

    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        # DFlashDraftModel.__init__ reads config.dflash_config and borrows the
        # target's embed/lm_head. DSpark uses a flat config and owns those modules,
        # so skip DFlash's __init__ and build the stack against the PreTrainedModel
        # base directly. forward() is still inherited from DFlashDraftModel.
        Qwen3PreTrainedModel.__init__(self, config)
        self.config = config
        self.target_layer_ids = list(config.target_layer_ids)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=getattr(config, "pad_token_id", None)
        )
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = int(config.mask_token_id)

        self.markov_head = build_markov_head(config)

        self.confidence_head = None
        self.confidence_head_with_markov = False
        if bool(getattr(config, "enable_confidence_head", False)):
            self.confidence_head_with_markov = bool(getattr(config, "confidence_head_with_markov", False))
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                assert self.markov_head is not None
                input_dim += int(config.markov_rank)
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)
        self.post_init()

    # forward() is inherited unchanged from DFlashDraftModel: it runs the
    # Qwen3DFlashDecoderLayer stack over [target_hidden ; noise_embedding] and
    # returns self.norm(hidden_states). DSpark reads logits off every position via
    # self.lm_head in the generate loop.

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn per-position backbone logits into a block of draft tokens.

        Without a markov head this is DFlash: one parallel argmax, positions
        independent. With one, it becomes a serial sweep where each step's logits
        are biased by the token sampled at the previous step.
        """
        if base_logits.shape[1] == 0:
            empty = torch.empty(base_logits.shape[0], 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits
        if self.markov_head is None:
            return sample(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            temperature=temperature,
        )

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Per-position acceptance logits. Apply sigmoid for a probability."""
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None and prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(dtype=hidden_states.dtype)
            return self.confidence_head(torch.cat([hidden_states, prev_embeddings], dim=-1)).float()
        return self.confidence_head(hidden_states).float()
