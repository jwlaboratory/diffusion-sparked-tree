# Final benchmark — DSpark vs DDTree vs sparked-tree

**H100** · `dspark_block16_best` · block 16 · greedy (temperature 0) · exact-match prefix acceptance · 12 prompts x 512 tokens x 6 datasets

Every method verified byte-identical to plain autoregressive decoding.
`acc` is accepted tokens per round; `speedup` is against no drafter.

## Configuration

| setting | value |
|---|---|
| tree_mode | `exact-precomputed` |
| beam_candidates | `512` |
| beam_min_width | `2` |
| max_fanout | `0` |

See `config.py` for why each value was chosen.


## Arm: `sparked` — best-first over precomputed table

### Budget 64

| dataset | DSpark acc / speedup | DDTree acc / speedup | sparked-tree acc / speedup |
|---|---|---|---|
| humaneval | 6.365 / 3.09x | 8.564 / 3.24x | 8.578 / 3.59x |
| mbpp | 6.557 / 3.11x | 8.394 / 3.19x | 8.453 / 3.67x |
| gsm8k | 7.803 / 3.38x | 8.519 / 2.74x | 9.587 / 3.53x |
| math500 | 7.833 / 3.25x | 9.962 / 3.25x | 9.654 / 3.40x |
| mt-bench | 3.832 / 2.17x | 4.692 / 1.97x | 5.303 / 2.70x |
| alpaca | 4.003 / 1.81x | 4.198 / 1.42x | 5.418 / 2.36x |
| **MEAN** | **6.065 / 2.80x** | **7.388 / 2.64x** | **7.832 / 3.21x** |

| sparked-tree vs | acceptance | speed | acc wins |
|---|---|---|---|
| DSpark | **+29.1%** | **+14.7%** | 6/6 |
| DDTree | **+6.0%** | **+21.8%** | 5/6 |

### Budget 128

| dataset | DSpark acc / speedup | DDTree acc / speedup | sparked-tree acc / speedup |
|---|---|---|---|
| humaneval | 6.365 / 3.09x | 8.933 / 3.37x | 8.877 / 3.81x |
| mbpp | 6.557 / 3.11x | 8.878 / 3.54x | 8.678 / 3.77x |
| gsm8k | 7.803 / 3.38x | 9.110 / 3.05x | 9.992 / 3.68x |
| math500 | 7.833 / 3.25x | 10.410 / 3.47x | 10.171 / 3.60x |
| mt-bench | 3.832 / 2.17x | 4.894 / 2.06x | 5.566 / 2.80x |
| alpaca | 4.003 / 1.81x | 4.402 / 1.55x | 6.066 / 2.44x |
| **MEAN** | **6.065 / 2.80x** | **7.771 / 2.84x** | **8.225 / 3.35x** |

| sparked-tree vs | acceptance | speed | acc wins |
|---|---|---|---|
| DSpark | **+35.6%** | **+19.6%** | 6/6 |
| DDTree | **+5.8%** | **+17.8%** | 3/6 |

## Arm: `beam_graphed` — CUDA-graphed beam

### Budget 64

| dataset | DSpark acc / speedup | DDTree acc / speedup | sparked-tree acc / speedup |
|---|---|---|---|
| humaneval | 6.365 / 3.04x | 8.564 / 3.10x | 8.179 / 3.40x |
| mbpp | 6.557 / 2.85x | 8.394 / 2.81x | 8.088 / 3.18x |
| gsm8k | 7.803 / 3.49x | 8.519 / 2.90x | 9.330 / 3.60x |
| math500 | 7.833 / 3.46x | 9.962 / 3.41x | 9.735 / 3.71x |
| mt-bench | 3.787 / 2.43x | 4.674 / 2.19x | 4.822 / 2.27x |
| alpaca | 4.003 / 1.88x | 4.198 / 1.45x | 5.013 / 2.42x |
| **MEAN** | **6.058 / 2.86x** | **7.385 / 2.64x** | **7.528 / 3.10x** |

| sparked-tree vs | acceptance | speed | acc wins |
|---|---|---|---|
| DSpark | **+24.3%** | **+8.3%** | 6/6 |
| DDTree | **+1.9%** | **+17.2%** | 3/6 |

### Budget 128

| dataset | DSpark acc / speedup | DDTree acc / speedup | sparked-tree acc / speedup |
|---|---|---|---|
| humaneval | 6.365 / 3.04x | 8.933 / 3.26x | 8.924 / 3.75x |
| mbpp | 6.557 / 2.85x | 8.878 / 3.10x | 8.937 / 3.50x |
| gsm8k | 7.803 / 3.49x | 9.110 / 3.20x | 9.755 / 3.95x |
| math500 | 7.833 / 3.46x | 10.410 / 3.72x | 10.129 / 3.97x |
| mt-bench | 3.787 / 2.43x | 4.933 / 2.38x | 5.154 / 2.98x |
| alpaca | 4.003 / 1.88x | 4.402 / 1.59x | 5.673 / 2.50x |
| **MEAN** | **6.058 / 2.86x** | **7.778 / 2.88x** | **8.095 / 3.44x** |

| sparked-tree vs | acceptance | speed | acc wins |
|---|---|---|---|
| DSpark | **+33.6%** | **+20.4%** | 6/6 |
| DDTree | **+4.1%** | **+19.6%** | 4/6 |

---

## Reading these numbers

Two runs of an identical configuration in this harness differ by ~1% on acceptance (often exactly reproducing, since decoding is greedy and seeded) and ~5% on speed, with single cells up to 16%. Deltas inside those bands are labelled as such above and should not be read as effects.

Acceptance is the axis that transfers to batched serving; wall-clock speedup at batch 1 does not. All numbers here are batch 1.
