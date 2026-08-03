"""The real sparked-tree worker: DSpark drafter + our markov tree + tree verify.

Subclasses `DSparkWorkerV2` so the expensive, fiddly half is reused verbatim --
draft model loading, the block proposer, and `TargetHiddenKvInjector`. Only the
decode step is replaced: DSpark verifies a linear chain
(`dspark_worker_v2.py:581`, `cat([draft_block_ids[:, :1], draft_tokens])`), we
verify a tree.

I previously called this multi-day, on the belief that `commit_hidden` was
chain-shaped and a tree would need its own commit path. That was wrong, and it
was wrong because I had flagged `dspark_kv_inject.py` as unread and then reasoned
around it instead of opening it. `TargetHiddenKvInjector.inject_target_hidden`
is flat and index-driven:

    inject_target_hidden(target_hidden, cache_loc, positions, ...)

It writes whatever hidden states you hand it at whatever slots you name. Prefill
calls it with `(logits_output.hidden_states, batch.out_cache_loc, positions)`.
Nothing about it assumes a chain. So the tree version is the same call with the
accepted *path* gathered out of the tree:

    target_hidden = hidden_states[accept_index]     # committed path, contiguous
    cache_loc     = slots for [seq_len, seq_len + commit_len)
    positions     = seq_len + arange(commit_len)

which is exactly the shape prefill already passes.

STATUS: written against the real signatures, NOT yet validated on a GPU. The
oracle is `test_plugin_e2e.py`'s losslessness check -- greedy output must be
byte-identical to no-speculation decoding. Do not trust acceptance numbers from
this until that passes.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dspark_components.dspark_draft import (
    make_next_draft_input,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    alloc_verify_window,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_utils import (
    move_accept_tokens_to_target_kvcache,
    prepare_mamba_track_for_verify,
    record_stream_for_v2_verify,
)

from .tree_source import DDTreeSource, DSparkTreeSource

logger = logging.getLogger(__name__)


class SparkedDSparkWorker(DSparkWorkerV2):
    """DSpark drafter, markov-guided best-first tree, tree verify."""

    # Set by the plugin registration; "ddtree" swaps the builder for the
    # independence-scored ablation. One worker, both arms.
    TREE_ARM = "sparked"

    def __init__(self, server_args, gpu_id, ps, nccl_port, target_worker):
        super().__init__(server_args, gpu_id, ps, nccl_port, target_worker)
        self.tree_width = int(server_args.speculative_num_draft_tokens)
        self.tree_budget = self.tree_width - 1
        self.tree_source = (
            DDTreeSource() if self.TREE_ARM == "ddtree" else DSparkTreeSource()
        )
        self._tree_buffers_for: Optional[int] = None

    # --- buffers -----------------------------------------------------------

    def _ensure_tree_buffers(self, bs: int):
        if self._tree_buffers_for is not None and self._tree_buffers_for >= bs:
            return
        N, device = self.tree_width, self.device
        self._t_draft_tokens = torch.empty((bs * N,), dtype=torch.int64, device=device)
        self._t_mask = torch.empty((bs * N * N,), dtype=torch.bool, device=device)
        self._t_positions = torch.empty((bs * N,), dtype=torch.int64, device=device)
        self._t_ri = torch.empty((bs, N), dtype=torch.int64, device=device)
        self._t_nt = torch.empty((bs, N), dtype=torch.int64, device=device)
        self._t_ns = torch.empty((bs, N), dtype=torch.int64, device=device)
        self._tree_buffers_for = bs

    # --- the tree ----------------------------------------------------------

    def _build_trees(self, base_logits: torch.Tensor, root_tokens: list[int]):
        """base_logits: [bs, gamma, vocab] from the DSpark drafter.

        Host-side and serial by construction -- this is the cost that does not
        amortise across a batch, and the reason this worker registers
        supports_overlap=False.
        """
        from sgl_kernel.speculative import reconstruct_indices_from_tree_mask  # noqa
        from sparked_bridge import (
            batch_qlen_mask, tree_to_draft_tokens, tree_to_qlen_mask,
        )

        N = self.tree_width
        markov_head = self._draft_model_for_tree().markov_head
        token_rows, mask_rows = [], []
        for i, root in enumerate(root_tokens):
            node_tokens, _depths, _parents, _cmaps, visibility = (
                self.tree_source.build_from_logits(
                    base_logits[i], markov_head, root, self.tree_budget))
            token_rows.append(tree_to_draft_tokens(root, node_tokens, N))
            mask_rows.append(tree_to_qlen_mask(visibility, N))
        return torch.cat(token_rows), batch_qlen_mask(mask_rows)

    def _draft_model_for_tree(self):
        # The proposer owns the draft model; name differs across versions, so
        # fail loudly rather than silently picking the wrong module.
        for attr in ("draft_model", "model", "_draft_model"):
            m = getattr(self._proposer, attr, None)
            if m is not None and hasattr(m, "markov_head"):
                return m
        raise RuntimeError(
            "could not locate the DSpark draft model with a markov_head on the "
            "proposer; inspect DraftBlockProposer's attributes")

    # --- decode ------------------------------------------------------------

    def _forward_decode(self, batch: ScheduleBatch, on_publish=None):
        if batch.forward_mode.is_idle():
            return super()._forward_decode(batch, on_publish=on_publish)

        fwd_stream = torch.get_device_module(self.device).current_stream()
        record_stream_for_v2_verify(batch, None, fwd_stream)

        draft_input = batch.spec_info
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens
        N = self.tree_width
        self._ensure_tree_buffers(bs)

        verify_window = alloc_verify_window(
            batch=batch, bs=bs, device=device,
            verify_num_draft_tokens=N,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
        )

        # --- 1. drafter: reused verbatim -----------------------------------
        with self._draft_context():
            proposal = self._proposer.propose(
                batch=batch, draft_input=draft_input, verify_window=verify_window,
                bs=bs, device=device,
                target_model=self.target_worker.model_runner.model,
                sampling_info=batch.sampling_info,
            )

        # --- 2. block logits -> our tree ------------------------------------
        draft_model = self._draft_model_for_tree()
        base_logits, _tap = draft_model.compute_base_logits(proposal.draft_hidden)
        base_logits = base_logits.view(bs, -1, base_logits.shape[-1])
        root_tokens = proposal.draft_block_ids[:, 0].tolist()   # host sync, see docstring

        tokens_cpu, mask_cpu = self._build_trees(base_logits, root_tokens)
        draft_tokens = self._t_draft_tokens[: bs * N]
        tree_mask = self._t_mask[: bs * N * N]
        positions = self._t_positions[: bs * N]
        ri, nt, ns = self._t_ri[:bs], self._t_nt[:bs], self._t_ns[:bs]
        draft_tokens.copy_(tokens_cpu, non_blocking=True)
        tree_mask.copy_(mask_cpu, non_blocking=True)

        from sgl_kernel.speculative import reconstruct_indices_from_tree_mask
        reconstruct_indices_from_tree_mask(
            tree_mask, batch.seq_lens, positions, ri, nt, ns, bs, N)

        # FULL_MASK: every draft row sees the whole prefix, then its ancestors.
        mask_3d = mask_cpu.reshape(bs, N, N)
        full_rows = []
        for i in range(bs):
            seq_len = int(batch.seq_lens_cpu[i])
            full_rows.append(torch.cat([
                torch.ones((N, seq_len), device=device, dtype=torch.bool),
                mask_3d[i].to(device, non_blocking=True)], dim=1).flatten())
        custom_mask = torch.cat(full_rows, dim=0)

        # --- 3. target verify ------------------------------------------------
        from sglang.kernels.ops.speculative.cache_locs import (
            assign_extend_cache_locs_func,
        )
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = draft_tokens
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens, end_offset=batch.seq_lens + N,
            batch_size=bs, draft_token_num=N, device=device,
        )
        prepare_mamba_track_for_verify(batch)
        batch.spec_info = NgramVerifyInput(
            draft_token=draft_tokens, custom_mask=custom_mask, positions=positions,
            retrieve_index=ri, retrieve_next_token=nt, retrieve_next_sibling=ns,
            draft_token_num=N,
        )

        target_out = self.target_worker.forward_batch_generation(batch, is_verify=True)
        logits_output = target_out.logits_output

        # --- 4. accept -------------------------------------------------------
        predict, accept_lens, accept_index = eagle_sample(
            batch.spec_info, batch, logits_output, None)
        new_seq_lens = batch.seq_lens + accept_lens
        accept_tokens = predict[accept_index].flatten()

        move_accept_tokens_to_target_kvcache(
            batch, accept_index, accept_lens - 1, self.token_to_kv_pool_allocator)

        # --- 5. feed the drafter: inject the ACCEPTED PATH's hidden states ---
        # The only real difference from the chain worker. accept_index names the
        # tree slots that were committed; gathering there turns the accepted path
        # into exactly the contiguous [n_committed, hidden] block prefill passes.
        hidden = logits_output.hidden_states
        if hidden is not None:
            flat_idx = accept_index.reshape(-1)
            valid = flat_idx >= 0
            committed_hidden = hidden[flat_idx[valid]]
            commit_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=batch.req_to_token_pool.req_to_token,
                start_offset=prefix_lens, end_offset=new_seq_lens,
                batch_size=bs, draft_token_num=N, device=device,
            )[: committed_hidden.shape[0]]
            commit_positions = torch.cat([
                prefix_lens[i] + torch.arange(int(accept_lens[i]), device=device)
                for i in range(bs)])
            self._kv_injector.inject_target_hidden(
                target_hidden=committed_hidden,
                cache_loc=commit_cache_loc,
                positions=commit_positions,
            )
        logits_output.hidden_states = None

        if on_publish is not None:
            on_publish(new_seq_lens)
        batch.forward_mode = ForwardMode.DECODE

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accept_tokens,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            next_draft_input=make_next_draft_input(
                bonus_tokens=predict[accept_index[:, 0]],
                new_seq_lens=new_seq_lens,
            ),
            speculative_num_draft_tokens=N,
        )


class SparkedDDTreeWorker(SparkedDSparkWorker):
    """Same worker, independence-scored builder. The DDTree arm."""
    TREE_ARM = "ddtree"
