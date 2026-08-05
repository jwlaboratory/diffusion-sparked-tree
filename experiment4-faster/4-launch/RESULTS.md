# Experiment 4-faster / 4 — launch-batched transition table

Close the last per-round cost gap between SparklingTree and DDTree: kernel-launch
overhead in the union transition-table precompute. One code move, tree unchanged
bit-for-bit.

All numbers here follow **DDTree benchmarking practices**: sync ON (instrumented
pass, upstream's per-stage `cuda_time` barriers), C++ KV compaction ON. One GPU,
one job. Protocol: 6 datasets (humaneval, mbpp, gsm8k, math500, mt-bench, alpaca)
× 3 samples (first discarded → 2 measured) × 512 tokens, budget 64, temp 0,
DDTree vs SparklingTree (union builder, C=128).

## Diagnosis (before, CODE_VERSION `harness-6-union`)

Instrumented phases showed the ENTIRE ST-vs-DDTree round gap in one bucket
(ms/round, aggregate over 6 datasets):

| phase | DDTree | SparklingTree |
|---|---|---|
| verify | 23.66 | 23.62 (same) |
| draft_forward | 5.09 | 5.10 (same) |
| kv_update / pack / walk_accept | ~1.5 | ~1.5 (same) |
| **candidate_build** | **0.49** | **2.99** |
| — `.prep` (GPU table + one transfer) | 0.26 | **2.18** |
| — `.expand` (CPU heap walk) | 0.12 | 0.14 |

The heap walk is already trivial; `.prep` dominates. Cause: the per-depth loop in
`_union_transition_topk` fired ~5 kernels × 16 depths ≈ **80 launches/round** plus
Python loop overhead — fully exposed under sync-on timing. DDTree's build is one
cheap top-k.

Head-to-head at this state: ST **+1.9%** aggregate TPS (228.2 vs 223.9), acceptance
+10.7% (7.88 vs 7.12) — wins alpaca +17.6% / mt-bench +11.7% / gsm8k +3.5%,
loses humaneval −4% / mbpp −10% / math500 −13.5%.
Raw: `results/summary_before.json`.

## The one move (`harness-7-batched`)

`harness/ddtree/sparked_tree.py::_union_transition_topk`: batch depths adaptively —
as many `[U, U]` slabs as fit under 512 MB fp32 go into ONE `[n, U, U]`
add + logsumexp + topk. At C=128 (U≈2k, 16·U² ≈ 256 MB) all 16 depths batch into a
single launch set (~5 kernels/round instead of ~80). At C=512 (U≈8k) the chunk
degrades gracefully to one depth at a time — never worse than the old loop.

Why not a custom CUDA kernel: the gap is launch count, not FLOPs. Batching existing
torch ops removes it with zero equivalence risk; a fused add+lse+topk kernel would
buy ~0.3 ms more for far more effort/risk. Not taken.

**Equivalence gate:** same math bit-for-bit — `2-precompute/test_precompute_builder.py`
still 100% on every check (fast == precompute == reference, strong-bias included).
`CODE_VERSION` bumped so no stale timing units are reused.

## Results (after, measured)

`.prep` **2.18 → 1.21 ms/round (−45%)**; `.expand` 0.15, visibility 0.04 —
ST candidate_build now ~1.4 ms vs DDTree's ~0.5. Identical protocol, same GPU
class, spawn `fc-01KZ8P6CAMD7FRS523CPZPVN96`:

| dataset | DDTree acc/tps | SparklingTree acc/tps | ST vs DDTree (before → after) |
|---|---|---|---|
| alpaca | 4.71 / 140.8 | 5.97 / 170.8 | +17.6% → **+21.3%** |
| mt-bench | 3.10 / 94.4 | 3.70 / 107.1 | +11.7% → **+13.5%** |
| gsm8k | 9.94 / 298.1 | 11.20 / 318.3 | +3.5% → **+6.8%** |
| humaneval | 9.24 / 276.6 | 9.80 / 275.2 | −4.0% → **−0.5%** (even) |
| mbpp | 8.88 / 263.4 | 8.84 / 248.6 | −10.0% → −5.6% |
| math500 | 10.57 / 314.2 | 9.84 / 280.0 | −13.5% → −10.9% |
| **AGGREGATE** | **7.12 / 213.3** | **7.88 / 224.0** | **+1.9% → +5.0%** |

Acceptance identical before/after (7.88 / 7.12) — real-model confirmation the
batched builder produces the same trees. Every dataset moved in ST's favor.
Absolute DDTree TPS drifted ~5% between the two runs (GPU variance across
containers) — the within-run ST-vs-DDTree ratio is the number that means anything.

Remaining gap to DDTree's build (~0.9 ms) is now real memory bandwidth
(materialising the 256 MB [16,U,U] slab + top-k), not launch overhead. A fused
kernel or fp16 table could shave ~half of it; diminishing returns — not taken.

Raw: `results/summary_after.json`.

## Reproduce

```bash
# equivalence gate (local, CPU)
python experiment4-faster/2-precompute/test_precompute_builder.py

# benchmark (Modal, one H100, detached)
cd experiment4-faster/2-precompute && modal run --detach modal_benchmark.py --spawn
modal volume get ddtree-results precompute/summary.json
```
The benchmark script is shared with 2-precompute (same protocol; only
CODE_VERSION distinguishes before/after cache namespaces).
