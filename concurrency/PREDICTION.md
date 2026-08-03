# Predicted sparked-tree throughput at concurrency

`H100` · `lmsysorg/sglang:v0.5.16-cu129` · `Qwen/Qwen3-4B` · `sharegpt`

Derived, not measured end to end. See the module docstring of
`predict.py` for the identity and its two assumptions. Every number here
is an **upper bound**: the host-resident builder's own per-round cost is
not included, and it does not amortise across a batch.

## Measured: round time by verify width (ms/round)

| arm | width | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|---|
| dspark_capped | 17 | 9.82 | 10.17 | 11.01 | 12.40 | 15.63 | 23.87 |
| tree_w17_noov | 17 | 7.67 | 8.56 | 9.45 | 13.27 | 17.29 | 30.31 |
| tree_w65_noov | 65 | 8.54 | 11.38 | 13.56 | 20.98 | 35.22 | 65.98 |
| tree_w129_noov | 129 | 11.08 | 12.89 | 19.48 | 34.36 | 59.06 | 114.01 |

## Measured: cost of width, relative to the width-17 chain

| width | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| 17 | 0.78x | 0.84x | 0.86x | 1.07x | 1.11x | 1.27x |
| 65 | 0.87x | 1.12x | 1.23x | 1.69x | 2.25x | 2.76x |
| 129 | 1.13x | 1.27x | 1.77x | 2.77x | 3.78x | 4.78x |

## Predicted: sparked-tree vs the DSpark chain

`(acceptance ratio) / (round-time ratio)`, using acceptance ratios
**re-measured on the same block-7 drafter and chat workload as the
cost sweep**: 1.36x at budget 64, 1.402x at 128. (The batch-1 splice assumed 1.291 and 1.389 — it understated the tree.)

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| sparked_tb64 (w=65) | **1.25x** | **1.06x** | 0.99x | 0.84x | 0.63x | 0.55x |
| sparked_tb128 (w=129) | **1.04x** | 0.98x | 0.73x | 0.52x | 0.38x | 0.31x |

Width increment isolated from the two arms that differ only in width (`tree_w65`/`tree_w129` minus `tree_w17`), then added to the measured DSpark chain round time — so the drafter forward is counted once, not zero times or twice. See the code comment for why the direct ratio would be wrong.

## Model check: is round time linear in width?

The additive model assumes a constant per-verify-token cost. Fit the
slope on widths 17->65, extrapolate to 129, compare to measured.

| c | slope (ms/token) | predicted w=129 | measured w=129 | error |
|---|---|---|---|---|
| 1 | 0.018 | 9.7 | 11.1 | -12% |
| 2 | 0.059 | 15.1 | 12.9 | +17% |
| 4 | 0.086 | 19.0 | 19.5 | -2% |
| 8 | 0.161 | 31.3 | 34.4 | -9% |
| 16 | 0.373 | 59.1 | 59.1 | +0% |
| 32 | 0.743 | 113.5 | 114.0 | -0% |

Worst extrapolation error **17%** — linear enough to trust the additive model.

## Crossover

- **sparked_tb64**: falls below 1.0x at **c=4** (0.99x — a tie rather than a loss at that rung). Above it the DSpark chain wins on throughput, even granting the tree its full measured acceptance advantage and charging it nothing for building the tree.
- **sparked_tb128**: falls below 1.0x at **c=2** (0.98x — a tie rather than a loss at that rung). Above it the DSpark chain wins on throughput, even granting the tree its full measured acceptance advantage and charging it nothing for building the tree.

Both lines are upper bounds. Adding the builder's ~3.8 ms/round of host-serial work — which does not amortise — moves every crossover left, and moves it further left the larger the batch.
