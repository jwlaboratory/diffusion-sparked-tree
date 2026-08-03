"""Registers SPARKED as a speculative algorithm with SGLang.

    python -m sglang.launch_server --model-path Qwen/Qwen3-4B \
        --speculative-algorithm SPARKED \
        --speculative-num-draft-tokens 65 \
        --disable-overlap-schedule

Registration has to happen before ServerArgs post-init resolves the algorithm
name. The supported hook is a setuptools entry point in group
`sglang.srt.plugins`, which `launch_server` / `cli.serve` / `entrypoints.engine`
load; importing this package directly also works when the import precedes
server construction.

`supports_overlap=False` is required, not chosen: the tree builder is
host-resident. It logs a deprecation warning on every create_worker and runs the
V2 schema synchronously. Cost measured at ~29% mean throughput against the
DSpark chain (concurrency/FINDINGS.md).
"""

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from .algo import SparkedSpecAlgo
from .tree_source import (
    DDTreeSource, DSparkTreeSource, LookupTreeSource, TreeSource,
)
from .worker import SparkedVerifyInput, SparkedWorker

__all__ = [
    "SparkedSpecAlgo", "SparkedWorker", "SparkedVerifyInput",
    "TreeSource", "LookupTreeSource", "DSparkTreeSource", "DDTreeSource",
    "register",
]

_REGISTERED = False


# Tree width is NOT speculative_num_draft_tokens. That knob drives DSpark's
# drafter geometry: the draft sampler does `input_ids.view(bs, gamma)`, so
# forcing it to 65 makes the draft CUDA graph capture 65 tokens per request while
# the block-7 checkpoint only produces ~8, and capture dies with
#   shape '[8, 64]' is invalid for input of size 520
#
# The two are genuinely different quantities. Our builder turns an 8-deep block
# of logits into a 64-node tree -- depth is bounded by the drafter's block size,
# width is not. DSpark conflates them only because its verify is a chain of
# exactly gamma tokens.
#
# SGLang already anticipates this. `get_num_tokens_per_req_for_target_verify`
# takes an `is_draft_worker` flag and carries a FIXME saying it exists so target
# verify can use a different width "for other use cases which is not target
# verify". That is precisely this case: the draft side keeps its natural gamma+1,
# the target side verifies the whole tree.
SPARKED_TREE_WIDTH_ENV = "SPARKED_TREE_WIDTH"


def tree_width() -> int:
    import os
    return int(os.environ.get(SPARKED_TREE_WIDTH_ENV, "65"))


class _DSparkTreeAlgo(SparkedSpecAlgo):
    """The real arms subclass DSparkWorkerV2, so DSpark-specific branches in the
    scheduler and model loader must take the DSpark path. Kept separate from
    SparkedSpecAlgo (which backs the standalone stand-in worker) so a failure in
    one does not silently change the other."""

    def get_num_tokens_per_req_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        # Draft worker keeps DSpark's geometry; target verifies the tree.
        return num_draft_tokens if is_draft_worker else tree_width()

    def handle_server_args(self, server_args) -> None:
        # The KV reserve per decode is computed from these
        # (allocation_sizing.get_alloc_len_per_decode); leaving them None raises
        # TypeError on the first decode. The reserve must cover the TREE, so
        # num_draft_tokens is the tree width here -- that is what gets allocated.
        #
        # The tension: this knob ALSO feeds the draft sampler's gamma
        # (`input_ids.view(bs, gamma)`). With draft CUDA graphs captured it
        # forces 65 tokens per request through a block-7 drafter and dies. With
        # --disable-cuda-graph that capture never happens, which is why the two
        # can coexist for now. Making them coexist WITH graphs is the open item.
        w = tree_width()
        server_args.speculative_num_draft_tokens = w
        server_args.speculative_num_steps = w - 1
        server_args.speculative_eagle_topk = 1
        server_args.enable_mixed_chunk = False

    def is_dspark(self) -> bool:
        return True

    def is_dflash_family(self) -> bool:
        # DSPARK is in the dflash family upstream; the draft-KV reserve and
        # several loader branches key off this.
        return True

    def is_ngram(self) -> bool:
        # False here: with overlap OFF the ngram-only relay early-returns never
        # run, and claiming ngram would divert the DSpark loader paths we need.
        return False

    def has_draft_kv(self) -> bool:
        return True

    def supports_target_verify_for_draft(self) -> bool:
        # Required. The draft worker's decode CUDA graph runner raises a bare
        # RuntimeError("This should not happen") without it
        # (decode_cuda_graph_runner.py:271) -- the base default is False.
        return True

    def supports_ragged_verify(self) -> bool:
        # Matches DSPARK. Our tree is uniform-width, so we do not need ragged
        # verify ourselves, but DSparkWorkerV2's planner and verify window are
        # built assuming it; diverging here diverges from the machinery we reuse.
        return True


def register() -> None:
    """Idempotent: register_algorithm raises on a duplicate name."""
    global _REGISTERED
    if _REGISTERED:
        return

    @SpeculativeAlgorithm.register(
        "SPARKED", supports_overlap=False, spec_class=SparkedSpecAlgo
    )
    def _factory(server_args):
        return SparkedWorker

    # The real arms. Imported lazily: they pull in DSparkWorkerV2, and a failure
    # there must not take down the validated stand-in worker above.
    try:
        from .dspark_tree_worker import SparkedDDTreeWorker, SparkedDSparkWorker

        @SpeculativeAlgorithm.register(
            "SPARKED_DSPARK", supports_overlap=False, spec_class=_DSparkTreeAlgo
        )
        def _factory_dspark(server_args):
            return SparkedDSparkWorker

        @SpeculativeAlgorithm.register(
            "DDTREE_DSPARK", supports_overlap=False, spec_class=_DSparkTreeAlgo
        )
        def _factory_ddtree(server_args):
            return SparkedDDTreeWorker
    except Exception as exc:   # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "DSpark-backed tree arms unavailable: %s", exc)

    _REGISTERED = True


register()
