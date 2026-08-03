# Final benchmark

One locked configuration, one comparison: **DSpark** (chain) vs **DDTree** (tree,
independence-scored) vs **sparked-tree** (tree, branch-conditional). Nothing here
is swept — the exploration lives in `experiments/`, and this folder exists only to
produce the number you would quote.

```bash
modal run --detach final_benchmark/run_final.py::final          # H100 (default)
FINAL_GPU=A10G modal run --detach final_benchmark/run_final.py::final

modal volume get ddtree-train results/FINAL_H100_<ts>.json .
python3 final_benchmark/report.py FINAL_H100_<ts>.json --out final_benchmark/REPORT.md
```

## Files

| file | what it is |
|---|---|
| `config.py` | every setting, each with the experiment that chose it |
| `run_final.py` | the runner; one container per (dataset, arm) |
| `report.py` | turns the result JSON into `REPORT.md` |
| `REPORT.md` | generated output |

## Two arms

`sparked` is the shipped method: best-first expansion over a precomputed
`[L-1, C, C]` transition table. `beam_graphed` is the level-synchronous beam
replayed from a CUDA graph — a *faster builder* (0.94 ms/round vs 3.82 on H100)
that produces a *worse tree*. Both are measured rather than one being asserted,
because the trade between them is not obvious: builder time is ~5% of a round,
while acceptance determines how many rounds are needed.

## Reading the output

Two runs of an identical configuration in this harness reproduce acceptance to
~0.5% (often exactly — decoding is greedy and seeded) but differ on speed by ~5%
on average and up to 16% on a single dataset. `report.py` labels any delta inside
those bands as "within noise" instead of printing it as a result.

Acceptance is the axis that transfers to batched serving. Wall-clock speedup at
batch 1 does not, and every number here is batch 1.

## Why these settings

Short version, with the full reasoning in `config.py`:

- **`_best` checkpoint, not `_bigdata`.** `_best` reproduces the published headline
  to +0.1%; `_bigdata` changed four training variables at once and is worse on all
  six datasets.
- **Best-first, not beam.** +6.2% acceptance at budget 64 (6/6 datasets).
- **C = 512.** 256 costs 3.6% acceptance; 2048 buys ~4% more at 4x the build time.
  The precomputed table is `C x C`, so this knob is quadratic — and the penalty for
  a large C is far worse on A10G than H100, which have different compute headroom.
- **`min_width = 2`.** Prevents `flat(16)` at budget 16 from degenerating into
  `[1]*16`, a chain with no branching. Worth +1.5% there and inert above budget 32.

## Known limitation

Batch 1 only. The transformers harness asserts `batch_size == 1`, so none of this
speaks to serving concurrency, where a chain's ~5x better compute efficiency per
accepted token is expected to dominate.
