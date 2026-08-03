"""The V2 speculative worker for tree-shaped drafts.

Structurally a close sibling of `ngram_worker.py`, because NGRAM is the only
upstream algorithm with our shape: a tree it builds itself, no draft KV pool, no
EAGLE-geometry draft loop. The differences are confined to two places — where the
tree comes from (`tree_source`) and how it becomes a mask (`sparked_bridge`).

Everything downstream of the mask is upstream code we reuse unchanged:

    reconstruct_indices_from_tree_mask   mask -> positions + retrieve arrays
    eagle_sample                         acceptance decision
    move_accept_tokens_to_target_kvcache commit accepted branch, drop the rest
    assign_extend_cache_locs_func        gather the pre-reserved verify slots

`supports_overlap=False` (see __init__.py). That is not a shortcut: the V1 worker
path was removed in 0.5.16, so this is the V2 schema run synchronously. It costs
~29% throughput against the DSpark chain — measured, see FINDINGS.md — and the
cause is that our tree builder is host-resident.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    move_accept_tokens_to_target_kvcache,
    prepare_mamba_track_for_verify,
    record_stream_for_v2_verify,
)

from .tree_source import LookupTreeSource, TreeSource

logger = logging.getLogger(__name__)

# Reuse NgramVerifyInput rather than subclassing SpecInput. Its defaults are
# already what an irregular budgeted tree needs -- max_tree_depth ==
# draft_token_num (no depth cap) and tree_topk == -1 (no fixed branching) -- and
# SpecInputType is a closed IntEnum whose membership SpecInput.is_verify_input()
# hardcodes, so a new type is not addable from a plugin anyway.
SparkedVerifyInput = NgramVerifyInput


class SparkedWorker(BaseSpecWorker):
    def __init__(self, server_args, gpu_id, ps, nccl_port, target_worker):
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.device = server_args.device
        self.server_args = server_args
        self.gpu_id = gpu_id

        self.draft_token_num = int(server_args.speculative_num_draft_tokens)
        self.speculative_num_draft_tokens = self.draft_token_num
        self.tree_budget = self.draft_token_num - 1
        self.enable_overlap = not server_args.disable_overlap_schedule

        self.tree_source: TreeSource = LookupTreeSource()
        self.max_batch_size = int(getattr(server_args, "max_running_requests", None) or 256)
        self._buffers_for: Optional[int] = None

    # --- plumbing the scheduler expects ------------------------------------

    @property
    def draft_worker(self):
        # No separate draft TpModelWorker. The base alloc/init helpers and the
        # weight-update paths all branch on this being None.
        return None

    def alloc_memory_pool(self, **kwargs):
        pools = self._target_worker.get_memory_pool()
        self.req_to_token_pool, self.token_to_kv_pool_allocator = pools[0], pools[1]
        self._ensure_buffers(self.max_batch_size)

    def init_attention_backends(self):
        pass    # target-only; the base default would deref draft_worker

    def init_cuda_graphs(self):
        pass

    def clear_cache_pool(self):
        pass

    def update_weights_from_tensor(self, recv_req):
        # The base update_weights_from_* iterate draft_worker.draft_runners and
        # would raise with draft_worker None.
        return self._target_worker.update_weights_from_tensor(recv_req)

    # is_ngram() is True, so the scheduler builds an ExternalCorpusManager over
    # this worker. It is inert unless a corpus RPC arrives; these keep that from
    # being an AttributeError.
    def add_external_corpus(self, *args, **kwargs):
        raise NotImplementedError("SPARKED has no external corpus")

    def commit_corpus_load(self, *args, **kwargs):
        return None

    def remove_external_corpus(self, *args, **kwargs):
        return None

    def list_external_corpora(self, *args, **kwargs):
        return []

    # --- tree construction --------------------------------------------------

    def _ensure_buffers(self, bs: int):
        if self._buffers_for is not None and self._buffers_for >= bs:
            return
        N, device = self.draft_token_num, self.device
        self._draft_tokens = torch.empty((bs * N,), dtype=torch.int64, device=device)
        self._tree_mask = torch.empty((bs * N * N,), dtype=torch.bool, device=device)
        self._positions = torch.empty((bs * N,), dtype=torch.int64, device=device)
        self._retrieve_index = torch.empty((bs, N), dtype=torch.int64, device=device)
        self._retrieve_next_token = torch.empty((bs, N), dtype=torch.int64, device=device)
        self._retrieve_next_sibling = torch.empty((bs, N), dtype=torch.int64, device=device)
        self._buffers_for = bs

    def _build_trees(self, batch: ScheduleBatch):
        """Per request: propose a tree, convert to (tokens, mask) via the bridge."""
        from sparked_bridge import (
            batch_qlen_mask,
            tree_to_draft_tokens,
            tree_to_qlen_mask,
            visibility_from_parents,
        )

        N = self.draft_token_num
        token_rows, mask_rows = [], []
        for req in batch.reqs:
            context = list(req.origin_input_ids) + list(req.output_ids)
            root = context[-1] if context else 0
            node_tokens, parents = self.tree_source.build(context, self.tree_budget)
            visibility = visibility_from_parents(parents)
            token_rows.append(tree_to_draft_tokens(
                root, torch.tensor(node_tokens, dtype=torch.int64), N))
            mask_rows.append(tree_to_qlen_mask(visibility, N))
        return torch.cat(token_rows), batch_qlen_mask(mask_rows)

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        if not batch.forward_mode.is_decode():
            return
        bs = len(batch.reqs)
        self._ensure_buffers(bs)
        N = self.draft_token_num

        draft_tokens = self._draft_tokens[: bs * N]
        tree_mask = self._tree_mask[: bs * N * N]
        positions = self._positions[: bs * N]
        retrieve_index = self._retrieve_index[:bs]
        retrieve_next_token = self._retrieve_next_token[:bs]
        retrieve_next_sibling = self._retrieve_next_sibling[:bs]

        tokens_cpu, mask_cpu = self._build_trees(batch)
        draft_tokens.copy_(tokens_cpu, non_blocking=True)
        tree_mask.copy_(mask_cpu, non_blocking=True)

        # Derives positions + the whole first-child/next-sibling encoding from
        # the mask. Verified against 494 of our real trees: see
        # test_kernel_agreement.py.
        reconstruct_indices_from_tree_mask(
            tree_mask, batch.seq_lens, positions, retrieve_index,
            retrieve_next_token, retrieve_next_sibling, bs, N,
        )

        # FULL_MASK layout: every draft row sees the committed prefix, then its
        # own ancestor path. NGRAM notes QLEN is faster but needs matching
        # flashinfer changes, so full mask is the safe default.
        full_rows = []
        mask_3d = mask_cpu.reshape(bs, N, N)
        for i in range(bs):
            seq_len = int(batch.seq_lens_cpu[i])
            prefix = torch.ones((N, seq_len), device=self.device, dtype=torch.bool)
            full_rows.append(torch.cat(
                [prefix, mask_3d[i].to(self.device, non_blocking=True)], dim=1).flatten())
        custom_mask = torch.cat(full_rows, dim=0)

        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.input_ids = draft_tokens
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=batch.seq_lens + N,
            batch_size=bs, draft_token_num=N, device=self.device,
        )
        prepare_mamba_track_for_verify(batch)
        batch.spec_info = SparkedVerifyInput(
            draft_token=draft_tokens, custom_mask=custom_mask, positions=positions,
            retrieve_index=retrieve_index, retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling, draft_token_num=N,
        )

    # --- the step ----------------------------------------------------------

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        fwd_stream = torch.get_device_module(self.device).current_stream()
        record_stream_for_v2_verify(batch, None, fwd_stream)
        bs = len(batch.reqs)

        self._prepare_for_speculative_decoding(batch)
        accept_lens = torch.ones(bs, dtype=torch.int32, device=self.device)

        if batch.forward_mode.is_target_verify():
            batch_result = self._target_worker.forward_batch_generation(
                batch, is_verify=True)
            logits_output = batch_result.logits_output
            can_run_cuda_graph = batch_result.can_run_cuda_graph

            verify_input = batch.spec_info
            predict, accept_lens, accept_index = eagle_sample(
                verify_input, batch, logits_output, None)
            new_seq_lens = batch.seq_lens + accept_lens
            commit_mamba_states_after_verify(
                self._target_worker, batch, accept_lens, accept_index,
                self.draft_token_num)

            accept_tokens = predict[accept_index].flatten()
            next_token_ids = accept_tokens

            # accept_lens includes the bonus token; the KV mover wants
            # drafts-only, hence the -1.
            move_accept_tokens_to_target_kvcache(
                batch, accept_index, accept_lens - 1,
                self.token_to_kv_pool_allocator)

            if on_publish is not None:
                on_publish(new_seq_lens)
            batch.forward_mode = ForwardMode.DECODE
        else:
            batch_result = self._target_worker.forward_batch_generation(batch)
            logits_output = batch_result.logits_output
            predict = batch_result.next_token_ids
            can_run_cuda_graph = batch_result.can_run_cuda_graph
            new_seq_lens = batch.seq_lens.clone()

            accept_tokens = torch.zeros(
                bs, self.draft_token_num, dtype=torch.int32, device=self.device)
            accept_tokens[:, 0] = predict
            accept_tokens = accept_tokens.flatten()
            next_token_ids = predict
            if on_publish is not None:
                on_publish(new_seq_lens)

        next_draft_input = SparkedVerifyInput(
            draft_token_num=self.draft_token_num,
            new_seq_lens=new_seq_lens,
            accept_tokens=accept_tokens,
            accept_lens=accept_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
        )
