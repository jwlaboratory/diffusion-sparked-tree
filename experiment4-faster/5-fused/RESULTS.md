# Experiment 4-faster / 5 — fused table kernel (negative result, kept as prototype)

One CUDA kernel (CUB block radix sort, one block per (depth, parent) row) fusing the
union table's add + logsumexp + top-k, replacing ~3 global-memory passes over the
[n,U,U] slab with one. Opt-in only (`tree_kwargs: {"fused_table": true}`), default
OFF, CODE_VERSION unchanged.

## Validation (H100, `test_fused_modal.py`; report in `results/report.json`)

- **Tree-level: 100.00% node agreement** with the torch path (strong bias, C=128,
  budgets 64/256). The trees are unaffected.
- Op-level slots match to 99.994%; the residual is fp32 epsilon exactly at the
  top-k boundary (torch top-ks the normalized slab, the kernel the raw one — same
  epsilon class the repo already accepts between fast/precompute).
- Tie-breaking reproduces `torch.topk` (composite u64 key: flipped value bits high,
  column index low, radix ascending).

## Microbench (L=16, U=2048, 500 iters, CUDA events, µs/call)

| k | torch path | fused | ratio |
|---|---|---|---|
| 64 | 2415 | 2549 | **0.95× (fused slower)** |
| 128 | 2422 | 2554 | 0.95× |
| 256 | 2942 | 2545 | 1.16× |

Fused is k-independent (~2550 µs — the per-row full SORT costs what skipping the
slab materialization saves); the torch path grows with k. So fused only wins at
k=256.

## Verdict

**Not enabled.** The same day, the `max_fanout=64` lever (2-precompute lever test)
made **k=64 the production operating point at every budget** — zero measured
acceptance cost, 4× smaller transfer — which is exactly where this kernel is 5%
*slower*. The config lever beat the kernel lever.

Future work if ever revisited: replace the per-row full radix SORT with a radix
SELECT (partial top-k). That is what would make k=64/128 a clear win; meaningfully
larger kernel, not currently justified.

Code: `harness/ddtree/fused_table.py` (graceful-fallback load_inline, same pattern
as the C++ compaction module); integration behind `use_fused` in
`harness/ddtree/sparked_tree.py::_union_transition_topk`.
