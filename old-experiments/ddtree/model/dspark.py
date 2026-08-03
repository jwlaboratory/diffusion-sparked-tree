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

Ported from deepseek-ai/DeepSpec for inference only; training paths (anchor
sampling, losses, flex_attention masks) are omitted. Module names match the
released checkpoints so `from_pretrained` loads them unchanged.
"""

from typing import Optional, Callable
from typing_extensions import Unpack, Tuple
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers.cache_utils import Cache
from .utils import sample


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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

    def compute_step_bias(self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]) -> torch.Tensor:
        return self.markov_w2(self.get_prev_embeddings(token_ids))

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
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
            step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
            step_logits = base_logits[:, step_idx, :] + self.compute_step_bias(prev_token_ids, step_hidden)
            corrected_logits.append(step_logits.unsqueeze(1))
            next_token_ids = sample(step_logits.unsqueeze(1), temperature).squeeze(1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


class GatedMarkovHead(VanillaMarkovHead):
    """Markov head whose bias is gated by the backbone hidden state."""

    def __init__(self, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_step_bias(self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]) -> torch.Tensor:
        assert hidden_states is not None, "GatedMarkovHead requires backbone hidden states"
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = torch.sigmoid(self.gate_proj(torch.cat([hidden_states, prev_embeddings], dim=-1)))
        return self.markov_w2(gate.to(dtype=prev_embeddings.dtype) * prev_embeddings)


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
    if head_type == "vanilla":
        return VanillaMarkovHead(vocab_size=config.vocab_size, markov_rank=markov_rank)
    if head_type == "gated":
        return GatedMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    raise NotImplementedError(
        f"markov_head_type={head_type!r} is not ported here; see deepspec/modeling/dspark/markov_head.py"
    )


class Qwen3DSparkAttention(nn.Module):
    """Identical to Qwen3DFlashAttention: queries come from the noise block, keys and
    values from [target context ; noise block]."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_is_causal = bool(kwargs.get("is_causal", False))
        self.is_causal = attn_is_causal
        kwargs["is_causal"] = attn_is_causal
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DSparkDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DSparkAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DSparkDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DSparkDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.target_layer_ids = list(config.target_layer_ids)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=getattr(config, "pad_token_id", None)
        )
        self.layers = nn.ModuleList(
            [Qwen3DSparkDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
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
            hidden_states=hidden_states,
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
