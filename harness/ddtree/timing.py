"""Toggleable stage timing for the decode loops.

Every stage boundary in the generators calls `cuda_time()`, which is a full-device
barrier (`torch.cuda.synchronize()`) followed by `perf_counter()`. That gives honest
per-stage attribution, but the barriers themselves destroy CPU/GPU overlap -- and
tree arms hit more boundaries per round than chain arms, so an instrumented run is
not just slower, it is *unevenly* slower.

Experiment 3 therefore runs every unit twice:

  * clean pass        -- set_timing(False): `cuda_time()` skips the barrier, so the
                         decode loop runs at full overlap. Stage deltas are garbage
                         (kernels are still queued when the clock is read) and are
                         ignored; only the totals matter, and those are taken with
                         `sync_time()`, which ALWAYS syncs.
  * instrumented pass -- set_timing(True): per-stage numbers are real.

Do not use `cuda_time()` for TTFT or the decode total: those must be honest in both
modes, which is what `sync_time()` is for.
"""

import time

import torch

_ENABLED = True


def set_timing(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def timing_enabled() -> bool:
    return _ENABLED


def cuda_time() -> float:
    """Stage-boundary clock. Barrier + clock when timing is on; bare clock when off."""
    if _ENABLED:
        torch.cuda.synchronize()
    return time.perf_counter()


def sync_time() -> float:
    """Always-honest clock for totals (TTFT, decode wall). Syncs in both modes."""
    torch.cuda.synchronize()
    return time.perf_counter()


def empty_stage_times(stage_names: tuple[str, ...]) -> dict[str, float]:
    return {stage_name: 0.0 for stage_name in stage_names}
