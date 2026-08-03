"""The SpeculativeAlgorithm descriptor for our tree plugin.

Most of this file exists to work around a real gap in SGLang 0.5.16's plugin
base class. `CustomSpecAlgo` is documented as duck-typing `SpeculativeAlgorithm`,
and `_assert_custom_spec_algo_conforms` enforces that — but the guard only checks
names starting with `is_` / `supports_`, so three methods the scheduler calls
unconditionally are missing from the base:

    create_future_map          scheduler.init_overlap
    need_topk                  overlap_utils.FutureMap.__init__
    carries_draft_hidden_states  scheduler (PD-disaggregation branch)

Verified against the shipped image, not inferred: all three are absent from
`CustomSpecAlgo`, present on the enum, and called on `self.spec_algorithm`.
And `init_overlap` runs at scheduler.py:525 unconditionally — its own comment
says "FutureMap is always-on: input_ids relay used in both modes" — so this
bites with `--disable-overlap-schedule` too. A plugin that does not supply them
raises AttributeError at startup, before any of our code runs.

The implementations below are copied from the enum so behaviour matches exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.speculative.spec_registry import CustomSpecAlgo

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.server_args import ServerArgs


class SparkedSpecAlgo(CustomSpecAlgo):
    """Tree-shaped, drafter-backed, no separate draft KV pool.

    `is_ngram()` returns True deliberately. It is not a claim to be NGRAM — it is
    SGLang's de-facto "tree-only algorithm whose draft input is not
    EAGLE-shaped" flag, and every one of its runtime call sites is behaviour we
    need:

      * overlap_utils.stash / _resolve_spec_extras early-return. Without it,
        RelayPayload.from_draft_input reaches for .bonus_tokens / .topk_p /
        .topk_index / .hidden_states and raises on our verify input.
      * scheduler._relay_forward_payload — same.
      * kv_cache_builder.get_draft_kv_pool returns None (we have no draft KV).
      * decode_cuda_graph_runner.get_spec_info — the only branch that builds a
        dummy verify SpecInput for a treeless algorithm. An unrecognised plugin
        falls through to None and TARGET_VERIFY graph capture breaks.

    The cost is that the scheduler also constructs an ExternalCorpusManager over
    this worker. That constructor only stores references and its poll no-ops
    while idle, so it is inert unless someone issues a corpus RPC — see the
    stubs on the worker.
    """

    # --- the three the base class is missing -------------------------------

    def create_future_map(
        self,
        device: torch.device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ) -> "FutureMap":
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(
            device, self, req_to_token_pool, needs_cpu_seq_lens,
            needs_confidence_relay,
        )

    def need_topk(self) -> bool:
        # enum: is_eagle() or is_standalone(). We are neither.
        return False

    def carries_draft_hidden_states(self) -> bool:
        # enum: is_eagle(). Disagg prefill->decode carries no draft hidden state.
        return False

    # --- behaviour flags ---------------------------------------------------

    def is_ngram(self) -> bool:
        return True

    def has_draft_kv(self) -> bool:
        # The base default is True (the larger reserve). We inject target hidden
        # states rather than keeping a draft KV pool, so False — this only
        # changes allocation when page_size > 1 and topk > 1.
        return False

    def supports_ragged_verify(self) -> bool:
        # False keeps one uniform verify width per request, which is what a
        # fixed-budget tree wants; it only gates token-bucket compact CUDA graphs.
        return False

    def handle_server_args(self, server_args: "ServerArgs") -> None:
        # Verify width is 1 + tree_budget: the root plus every tree node.
        budget = int(getattr(server_args, "speculative_num_draft_tokens", None) or 65)
        server_args.speculative_num_draft_tokens = budget
        # Irregular tree: no fixed per-level branching, so the EAGLE-shaped
        # step/topk knobs are meaningless here. Pin them to the chain-degenerate
        # values so nothing downstream computes an EAGLE geometry from them.
        server_args.speculative_num_steps = budget - 1
        server_args.speculative_eagle_topk = 1
        # Asserted whenever speculative_algorithm is set.
        server_args.enable_mixed_chunk = False
