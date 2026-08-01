# Every benchmark run

All methods greedy-lossless (temperature 0), verified byte-identical to plain
autoregressive decoding. `efficiency` = accepted tokens per token the target must
score — the metric that predicts behaviour at serving concurrency.


## wide-block7

**GPU** A10G · **drafter** block-7 (released) · **builder** best-first (exact) · **prompts** 16 x 512  
first broad sweep; markov-vs-independence ablation

| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |
|---|---|---|---|---|---|---|---|---|
| alpaca | - | sparked-tree tb64 | 5.393 | 14.93 | 2.60x | 65 | 0.083 | 13.65 |
| alpaca | - | sparked-tree tb16 | 4.534 | 14.19 | 2.73x | 17 | 0.267 | 4.77 |
| alpaca | - | DDTree tb64 | 4.132 | 13.77 | 2.82x | 65 | 0.064 | 0.68 |
| alpaca | - | DDTree+foreign head tb64 | 3.848 | 21.04 | 1.84x | 65 | 0.059 | 24.57 |
| alpaca | - | DSpark chain | 3.747 | 15.68 | 2.47x | 17 | 0.220 | - |
| alpaca | - | DDTree tb16 | 3.558 | 15.72 | 2.47x | 17 | 0.209 | 0.60 |
| alpaca | - | DDTree+foreign head tb16 | 3.261 | 19.63 | 1.98x | 17 | 0.192 | 7.29 |
| alpaca | - | tree, independence tb64 | 3.170 | 17.38 | 2.23x | 65 | 0.049 | 0.78 |
| alpaca | - | DFlash chain | 2.850 | 19.12 | 2.03x | 16 | 0.178 | - |
| alpaca | - | tree, independence tb16 | 2.679 | 19.97 | 1.94x | 17 | 0.158 | 0.70 |
| alpaca | - | no drafter | 1.000 | 38.79 | 1.00x | 1 | 1.000 | - |
| gsm8k | - | DDTree tb64 | 8.481 | 5.59 | 5.91x | 65 | 0.130 | 0.44 |
| gsm8k | - | sparked-tree tb64 | 7.647 | 8.14 | 4.06x | 65 | 0.118 | 10.83 |
| gsm8k | - | DDTree+foreign head tb64 | 7.577 | 9.14 | 3.61x | 65 | 0.117 | 14.87 |
| gsm8k | - | DDTree tb16 | 7.416 | 6.26 | 5.27x | 17 | 0.436 | 0.40 |
| gsm8k | - | sparked-tree tb16 | 7.213 | 7.00 | 4.71x | 17 | 0.424 | 3.49 |
| gsm8k | - | DFlash chain | 6.545 | 6.88 | 4.80x | 16 | 0.409 | - |
| gsm8k | - | DSpark chain | 6.442 | 7.11 | 4.64x | 17 | 0.379 | - |
| gsm8k | - | DDTree+foreign head tb16 | 6.296 | 8.30 | 3.98x | 17 | 0.370 | 4.95 |
| gsm8k | - | tree, independence tb64 | 4.585 | 10.25 | 3.22x | 65 | 0.071 | 0.74 |
| gsm8k | - | tree, independence tb16 | 3.826 | 12.04 | 2.74x | 17 | 0.225 | 0.72 |
| gsm8k | - | no drafter | 1.000 | 33.00 | 1.00x | 1 | 1.000 | - |
| humaneval | - | DDTree tb64 | 8.426 | 6.64 | 6.08x | 65 | 0.130 | 0.59 |
| humaneval | - | DDTree+foreign head tb64 | 7.667 | 10.28 | 3.93x | 65 | 0.118 | 20.15 |
| humaneval | - | DDTree tb16 | 7.292 | 7.71 | 5.24x | 17 | 0.429 | 0.54 |
| humaneval | - | sparked-tree tb64 | 6.990 | 10.20 | 3.96x | 65 | 0.108 | 16.17 |
| humaneval | - | sparked-tree tb16 | 6.449 | 9.26 | 4.36x | 17 | 0.379 | 5.48 |
| humaneval | - | DFlash chain | 6.320 | 8.53 | 4.73x | 16 | 0.395 | - |
| humaneval | - | DDTree+foreign head tb16 | 6.219 | 9.90 | 4.08x | 17 | 0.366 | 6.92 |
| humaneval | - | DSpark chain | 5.559 | 9.76 | 4.14x | 17 | 0.327 | - |
| humaneval | - | tree, independence tb64 | 4.477 | 12.37 | 3.26x | 65 | 0.069 | 1.04 |
| humaneval | - | tree, independence tb16 | 3.788 | 14.44 | 2.80x | 17 | 0.223 | 0.96 |
| humaneval | - | no drafter | 1.000 | 40.38 | 1.00x | 1 | 1.000 | - |
| math500 | - | DDTree tb64 | 9.670 | 5.73 | 6.98x | 65 | 0.149 | 0.55 |
| math500 | - | DDTree tb16 | 8.443 | 6.60 | 6.06x | 17 | 0.497 | 0.49 |
| math500 | - | DDTree+foreign head tb64 | 8.060 | 9.69 | 4.13x | 65 | 0.124 | 20.88 |
| math500 | - | DFlash chain | 7.670 | 7.04 | 5.68x | 16 | 0.479 | - |
| math500 | - | sparked-tree tb64 | 7.495 | 9.38 | 4.26x | 65 | 0.115 | 16.13 |
| math500 | - | sparked-tree tb16 | 7.090 | 8.34 | 4.80x | 17 | 0.417 | 5.58 |
| math500 | - | DDTree+foreign head tb16 | 6.396 | 9.45 | 4.23x | 17 | 0.376 | 7.32 |
| math500 | - | DSpark chain | 6.315 | 8.45 | 4.73x | 17 | 0.371 | - |
| math500 | - | tree, independence tb64 | 4.316 | 12.60 | 3.17x | 65 | 0.066 | 1.13 |
| math500 | - | tree, independence tb16 | 3.587 | 15.07 | 2.65x | 17 | 0.211 | 1.06 |
| math500 | - | no drafter | 1.000 | 39.99 | 1.00x | 1 | 1.000 | - |
| mbpp | - | DDTree tb64 | 8.403 | 5.61 | 5.80x | 65 | 0.129 | 0.40 |
| mbpp | - | sparked-tree tb64 | 7.184 | 8.51 | 3.82x | 65 | 0.111 | 10.13 |
| mbpp | - | DDTree+foreign head tb64 | 7.110 | 9.61 | 3.39x | 65 | 0.109 | 15.49 |
| mbpp | - | DDTree tb16 | 7.012 | 6.55 | 4.96x | 17 | 0.412 | 0.38 |
| mbpp | - | sparked-tree tb16 | 6.635 | 7.49 | 4.34x | 17 | 0.390 | 3.47 |
| mbpp | - | DFlash chain | 6.197 | 7.29 | 4.46x | 16 | 0.387 | - |
| mbpp | - | DDTree+foreign head tb16 | 5.984 | 8.54 | 3.81x | 17 | 0.352 | 4.84 |
| mbpp | - | DSpark chain | 5.689 | 7.97 | 4.08x | 17 | 0.335 | - |
| mbpp | - | tree, independence tb64 | 4.028 | 11.19 | 2.91x | 65 | 0.062 | 0.77 |
| mbpp | - | tree, independence tb16 | 3.423 | 12.83 | 2.54x | 17 | 0.201 | 0.68 |
| mbpp | - | no drafter | 1.000 | 32.54 | 1.00x | 1 | 1.000 | - |
| mt-bench | - | sparked-tree tb64 | 5.204 | 13.88 | 2.92x | 65 | 0.080 | 37.61 |
| mt-bench | - | DDTree tb64 | 4.622 | 11.55 | 3.51x | 65 | 0.071 | 1.69 |
| mt-bench | - | sparked-tree tb16 | 4.531 | 12.91 | 3.14x | 17 | 0.267 | 13.22 |
| mt-bench | - | DDTree+foreign head tb64 | 4.025 | 18.52 | 2.19x | 65 | 0.062 | 61.11 |
| mt-bench | - | DDTree tb16 | 3.997 | 13.12 | 3.09x | 17 | 0.235 | 1.55 |
| mt-bench | - | DSpark chain | 3.654 | 14.66 | 2.77x | 17 | 0.215 | - |
| mt-bench | - | DDTree+foreign head tb16 | 3.471 | 17.25 | 2.35x | 17 | 0.204 | 19.50 |
| mt-bench | - | DFlash chain | 3.159 | 16.13 | 2.51x | 16 | 0.197 | - |
| mt-bench | - | tree, independence tb64 | 3.071 | 16.97 | 2.39x | 65 | 0.047 | 2.53 |
| mt-bench | - | tree, independence tb16 | 2.651 | 19.62 | 2.07x | 17 | 0.156 | 2.35 |
| mt-bench | - | no drafter | 1.000 | 40.56 | 1.00x | 1 | 1.000 | - |

## block16-g4

**GPU** A10G · **drafter** block-16 gamma=4 · **builder** best-first (exact) · **prompts** 6 x 384  
first block-16 model; best acceptance of the session

| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |
|---|---|---|---|---|---|---|---|---|
| alpaca | - | sparked-tree tb64 | 5.529 | 13.23 | 2.43x | 65 | 0.085 | - |
| alpaca | - | sparked-tree wave tb64 | 5.316 | 11.90 | 2.70x | 65 | 0.082 | - |
| alpaca | - | sparked-tree tb16 | 4.658 | 13.69 | 2.35x | 17 | 0.274 | - |
| alpaca | - | DDTree tb64 | 4.029 | 12.17 | 2.64x | 65 | 0.062 | - |
| alpaca | - | sparked-tree wave tb16 | 3.897 | 16.04 | 2.00x | 17 | 0.229 | - |
| alpaca | - | DSpark chain | 3.870 | 12.29 | 2.61x | 17 | 0.228 | - |
| alpaca | - | DDTree+foreign head tb64 | 3.767 | 18.08 | 1.78x | 65 | 0.058 | - |
| alpaca | - | DDTree tb16 | 3.485 | 13.92 | 2.31x | 17 | 0.205 | - |
| alpaca | - | DDTree+foreign head tb16 | 3.109 | 20.95 | 1.53x | 17 | 0.183 | - |
| alpaca | - | tree, independence tb64 | 2.766 | 15.34 | 2.09x | 65 | 0.043 | - |
| alpaca | - | DFlash chain | 2.755 | 16.79 | 1.91x | 16 | 0.172 | - |
| alpaca | - | tree, independence tb16 | 2.356 | 19.20 | 1.67x | 17 | 0.139 | - |
| alpaca | - | no drafter | 1.000 | 32.13 | 1.00x | 1 | 1.000 | - |
| gsm8k | - | sparked-tree tb64 | 10.520 | 6.30 | 5.22x | 65 | 0.162 | - |
| gsm8k | - | DDTree tb64 | 9.014 | 5.12 | 6.42x | 65 | 0.139 | - |
| gsm8k | - | sparked-tree tb16 | 8.622 | 7.01 | 4.69x | 17 | 0.507 | - |
| gsm8k | - | DDTree+foreign head tb64 | 7.917 | 8.12 | 4.05x | 65 | 0.122 | - |
| gsm8k | - | DSpark chain | 7.893 | 6.03 | 5.45x | 17 | 0.464 | - |
| gsm8k | - | DDTree tb16 | 7.676 | 5.91 | 5.56x | 17 | 0.452 | - |
| gsm8k | - | sparked-tree wave tb64 | 7.565 | 8.35 | 3.94x | 65 | 0.116 | - |
| gsm8k | - | DFlash chain | 6.905 | 6.34 | 5.18x | 16 | 0.432 | - |
| gsm8k | - | DDTree+foreign head tb16 | 6.427 | 9.42 | 3.49x | 17 | 0.378 | - |
| gsm8k | - | sparked-tree wave tb16 | 5.374 | 11.30 | 2.91x | 17 | 0.316 | - |
| gsm8k | - | tree, independence tb64 | 3.404 | 13.63 | 2.41x | 65 | 0.052 | - |
| gsm8k | - | tree, independence tb16 | 2.809 | 16.18 | 2.03x | 17 | 0.165 | - |
| gsm8k | - | no drafter | 1.000 | 32.86 | 1.00x | 1 | 1.000 | - |
| humaneval | - | DDTree tb64 | 9.341 | 5.06 | 6.38x | 65 | 0.144 | - |
| humaneval | - | sparked-tree tb64 | 8.942 | 7.52 | 4.30x | 65 | 0.138 | - |
| humaneval | - | DDTree+foreign head tb64 | 8.304 | 7.78 | 4.15x | 65 | 0.128 | - |
| humaneval | - | DDTree tb16 | 8.000 | 5.51 | 5.86x | 17 | 0.471 | - |
| humaneval | - | sparked-tree tb16 | 7.478 | 8.08 | 4.00x | 17 | 0.440 | - |
| humaneval | - | DFlash chain | 7.116 | 6.13 | 5.27x | 16 | 0.445 | - |
| humaneval | - | sparked-tree wave tb64 | 7.088 | 9.02 | 3.58x | 65 | 0.109 | - |
| humaneval | - | DDTree+foreign head tb16 | 6.867 | 8.60 | 3.76x | 17 | 0.404 | - |
| humaneval | - | DSpark chain | 6.850 | 6.89 | 4.69x | 17 | 0.403 | - |
| humaneval | - | sparked-tree wave tb16 | 5.094 | 11.81 | 2.74x | 17 | 0.300 | - |
| humaneval | - | tree, independence tb64 | 3.636 | 13.05 | 2.48x | 65 | 0.056 | - |
| humaneval | - | tree, independence tb16 | 3.174 | 14.02 | 2.30x | 17 | 0.187 | - |
| humaneval | - | no drafter | 1.000 | 32.31 | 1.00x | 1 | 1.000 | - |

## block16-g8

**GPU** A10G · **drafter** block-16 gamma=8 · **builder** best-first (exact) · **prompts** 6 x 384  
loss_decay_gamma ablation - null result

| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |
|---|---|---|---|---|---|---|---|---|
| alpaca | - | sparked-tree tb64 | 5.697 | 12.60 | 2.45x | 65 | 0.088 | - |
| alpaca | - | sparked-tree wave tb64 | 5.183 | 12.31 | 2.51x | 65 | 0.080 | - |
| alpaca | - | sparked-tree tb16 | 4.749 | 13.33 | 2.32x | 17 | 0.279 | - |
| alpaca | - | DDTree tb64 | 4.029 | 12.00 | 2.57x | 65 | 0.062 | - |
| alpaca | - | sparked-tree wave tb16 | 3.956 | 16.04 | 1.92x | 17 | 0.233 | - |
| alpaca | - | DSpark chain | 3.858 | 12.41 | 2.49x | 17 | 0.227 | - |
| alpaca | - | DDTree+foreign head tb64 | 3.773 | 17.85 | 1.73x | 65 | 0.058 | - |
| alpaca | - | DDTree tb16 | 3.485 | 13.39 | 2.30x | 17 | 0.205 | - |
| alpaca | - | DDTree+foreign head tb16 | 3.097 | 20.18 | 1.53x | 17 | 0.182 | - |
| alpaca | - | tree, independence tb64 | 2.757 | 15.79 | 1.95x | 65 | 0.042 | - |
| alpaca | - | DFlash chain | 2.755 | 16.39 | 1.88x | 16 | 0.172 | - |
| alpaca | - | tree, independence tb16 | 2.355 | 18.60 | 1.66x | 17 | 0.139 | - |
| alpaca | - | no drafter | 1.000 | 30.86 | 1.00x | 1 | 1.000 | - |
| gsm8k | - | sparked-tree tb64 | 10.181 | 6.59 | 5.07x | 65 | 0.157 | - |
| gsm8k | - | DDTree tb64 | 9.014 | 5.14 | 6.50x | 65 | 0.139 | - |
| gsm8k | - | sparked-tree tb16 | 8.514 | 7.13 | 4.70x | 17 | 0.501 | - |
| gsm8k | - | DDTree+foreign head tb64 | 7.983 | 8.16 | 4.10x | 65 | 0.123 | - |
| gsm8k | - | DSpark chain | 7.797 | 6.17 | 5.42x | 17 | 0.459 | - |
| gsm8k | - | DDTree tb16 | 7.676 | 5.94 | 5.63x | 17 | 0.452 | - |
| gsm8k | - | sparked-tree wave tb64 | 7.554 | 8.38 | 3.99x | 65 | 0.116 | - |
| gsm8k | - | DFlash chain | 6.905 | 6.44 | 5.20x | 16 | 0.432 | - |
| gsm8k | - | DDTree+foreign head tb16 | 6.423 | 9.44 | 3.55x | 17 | 0.378 | - |
| gsm8k | - | sparked-tree wave tb16 | 5.357 | 11.46 | 2.92x | 17 | 0.315 | - |
| gsm8k | - | tree, independence tb64 | 3.403 | 13.63 | 2.45x | 65 | 0.052 | - |
| gsm8k | - | tree, independence tb16 | 2.803 | 16.40 | 2.04x | 17 | 0.165 | - |
| gsm8k | - | no drafter | 1.000 | 33.45 | 1.00x | 1 | 1.000 | - |
| humaneval | - | DDTree tb64 | 9.341 | 5.03 | 6.46x | 65 | 0.144 | - |
| humaneval | - | sparked-tree tb64 | 8.946 | 7.45 | 4.36x | 65 | 0.138 | - |
| humaneval | - | DDTree+foreign head tb64 | 8.354 | 7.73 | 4.20x | 65 | 0.129 | - |
| humaneval | - | DDTree tb16 | 8.000 | 5.60 | 5.80x | 17 | 0.471 | - |
| humaneval | - | sparked-tree tb16 | 7.415 | 8.17 | 3.98x | 17 | 0.436 | - |
| humaneval | - | DFlash chain | 7.116 | 6.17 | 5.27x | 16 | 0.445 | - |
| humaneval | - | sparked-tree wave tb64 | 7.036 | 8.95 | 3.63x | 65 | 0.108 | - |
| humaneval | - | DDTree+foreign head tb16 | 6.873 | 8.72 | 3.73x | 17 | 0.404 | - |
| humaneval | - | DSpark chain | 6.750 | 7.07 | 4.60x | 17 | 0.397 | - |
| humaneval | - | sparked-tree wave tb16 | 5.131 | 11.72 | 2.77x | 17 | 0.302 | - |
| humaneval | - | tree, independence tb64 | 3.781 | 12.43 | 2.61x | 65 | 0.058 | - |
| humaneval | - | tree, independence tb16 | 3.135 | 14.48 | 2.24x | 17 | 0.184 | - |
| humaneval | - | no drafter | 1.000 | 32.49 | 1.00x | 1 | 1.000 | - |

## block16-h100-overfit

**GPU** H100 · **drafter** block-16 H100 (31 epochs) · **builder** best-first (exact) · **prompts** 6 x 384  
lower train loss (0.522) but WORSE acceptance - overfit

| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |
|---|---|---|---|---|---|---|---|---|
| alpaca | - | sparked-tree tb64 | 5.243 | 7.57 | 3.37x | 65 | 0.081 | - |
| alpaca | - | sparked-tree wave tb64 | 5.008 | 7.16 | 3.56x | 65 | 0.077 | - |
| alpaca | - | sparked-tree tb16 | 4.562 | 8.26 | 3.08x | 17 | 0.268 | - |
| alpaca | - | DDTree tb64 | 4.006 | 14.27 | 1.78x | 65 | 0.062 | - |
| alpaca | - | sparked-tree wave tb16 | 3.860 | 10.31 | 2.47x | 17 | 0.227 | - |
| alpaca | - | DDTree+foreign head tb64 | 3.751 | 9.37 | 2.72x | 65 | 0.058 | - |
| alpaca | - | DSpark chain | 3.663 | 13.01 | 1.96x | 17 | 0.215 | - |
| alpaca | - | DDTree tb16 | 3.420 | 16.26 | 1.57x | 17 | 0.201 | - |
| alpaca | - | DDTree+foreign head tb16 | 3.154 | 11.06 | 2.30x | 17 | 0.186 | - |
| alpaca | - | DFlash chain | 2.754 | 19.25 | 1.32x | 16 | 0.172 | - |
| alpaca | - | tree, independence tb64 | 2.675 | 14.69 | 1.73x | 65 | 0.041 | - |
| alpaca | - | tree, independence tb16 | 2.350 | 17.20 | 1.48x | 17 | 0.138 | - |
| alpaca | - | no drafter | 1.000 | 25.47 | 1.00x | 1 | 1.000 | - |
| gsm8k | - | sparked-tree tb64 | 9.926 | 4.78 | 4.84x | 65 | 0.153 | - |
| gsm8k | - | DDTree tb64 | 8.820 | 6.30 | 3.68x | 65 | 0.136 | - |
| gsm8k | - | sparked-tree tb16 | 8.198 | 5.18 | 4.47x | 17 | 0.482 | - |
| gsm8k | - | DDTree tb16 | 7.888 | 6.88 | 3.37x | 17 | 0.464 | - |
| gsm8k | - | DDTree+foreign head tb64 | 7.834 | 5.37 | 4.32x | 65 | 0.121 | - |
| gsm8k | - | DSpark chain | 7.610 | 6.83 | 3.39x | 17 | 0.448 | - |
| gsm8k | - | sparked-tree wave tb64 | 7.457 | 5.60 | 4.14x | 65 | 0.115 | - |
| gsm8k | - | DFlash chain | 6.989 | 7.53 | 3.08x | 16 | 0.437 | - |
| gsm8k | - | DDTree+foreign head tb16 | 6.488 | 6.23 | 3.72x | 17 | 0.382 | - |
| gsm8k | - | sparked-tree wave tb16 | 5.285 | 8.60 | 2.70x | 17 | 0.311 | - |
| gsm8k | - | tree, independence tb64 | 3.540 | 13.40 | 1.73x | 65 | 0.054 | - |
| gsm8k | - | tree, independence tb16 | 2.915 | 17.37 | 1.33x | 17 | 0.171 | - |
| gsm8k | - | no drafter | 1.000 | 23.17 | 1.00x | 1 | 1.000 | - |
| humaneval | - | DDTree tb64 | 9.117 | 6.30 | 3.81x | 65 | 0.140 | - |
| humaneval | - | DDTree+foreign head tb64 | 8.601 | 4.94 | 4.86x | 65 | 0.132 | - |
| humaneval | - | sparked-tree tb64 | 8.339 | 5.85 | 4.10x | 65 | 0.128 | - |
| humaneval | - | DDTree tb16 | 8.093 | 7.16 | 3.35x | 17 | 0.476 | - |
| humaneval | - | sparked-tree tb16 | 7.079 | 6.29 | 3.81x | 17 | 0.416 | - |
| humaneval | - | DFlash chain | 7.030 | 7.81 | 3.07x | 16 | 0.439 | - |
| humaneval | - | DDTree+foreign head tb16 | 6.888 | 5.52 | 4.35x | 17 | 0.405 | - |
| humaneval | - | sparked-tree wave tb64 | 6.730 | 6.53 | 3.67x | 65 | 0.104 | - |
| humaneval | - | DSpark chain | 6.148 | 8.51 | 2.82x | 17 | 0.362 | - |
| humaneval | - | sparked-tree wave tb16 | 4.991 | 9.10 | 2.63x | 17 | 0.294 | - |
| humaneval | - | tree, independence tb64 | 3.706 | 13.45 | 1.78x | 65 | 0.057 | - |
| humaneval | - | tree, independence tb16 | 3.123 | 17.05 | 1.41x | 17 | 0.184 | - |
| humaneval | - | no drafter | 1.000 | 23.98 | 1.00x | 1 | 1.000 | - |

## beam-sweep

**GPU** A10G · **drafter** block-16 gamma=4 · **builder** exact vs beam (decay sweep) · **prompts** 4 x 384  
gsm8k only; first beam builder measurements

| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |
|---|---|---|---|---|---|---|---|---|
| beam_d0.5 | - | DDTree tb64 | 10.024 | 4.85 | 6.87x | 65 | 0.154 | 0.10 |
| beam_d0.5 | - | DDTree+foreign head tb64 | 8.657 | 7.60 | 4.39x | 65 | 0.133 | 2.63 |
| beam_d0.5 | - | DSpark chain | 8.419 | 5.82 | 5.73x | 17 | 0.495 | - |
| beam_d0.5 | - | DFlash chain | 7.792 | 5.85 | 5.70x | 16 | 0.487 | - |
| beam_d0.5 | - | sparked-tree tb64 | 7.080 | 7.01 | 4.75x | 65 | 0.109 | 0.57 |
| beam_d0.5 | - | sparked-tree wave tb64 | 7.080 | 7.15 | 4.66x | 65 | 0.109 | 0.55 |
| beam_d0.5 | - | tree, independence tb64 | 3.437 | 13.60 | 2.45x | 65 | 0.053 | 0.28 |
| beam_d0.5 | - | no drafter | 1.000 | 33.35 | 1.00x | 1 | 1.000 | - |
| beam_d0.6 | - | DDTree tb64 | 10.024 | 4.85 | 7.03x | 65 | 0.154 | 0.10 |
| beam_d0.6 | - | DDTree+foreign head tb64 | 8.657 | 7.68 | 4.44x | 65 | 0.133 | 2.64 |
| beam_d0.6 | - | DSpark chain | 8.419 | 5.95 | 5.73x | 17 | 0.495 | - |
| beam_d0.6 | - | sparked-tree tb64 | 8.129 | 6.35 | 5.37x | 65 | 0.125 | 0.59 |
| beam_d0.6 | - | sparked-tree wave tb64 | 8.129 | 6.35 | 5.37x | 65 | 0.125 | 0.53 |
| beam_d0.6 | - | DFlash chain | 7.792 | 5.96 | 5.72x | 16 | 0.487 | - |
| beam_d0.6 | - | tree, independence tb64 | 3.437 | 13.82 | 2.47x | 65 | 0.053 | 0.27 |
| beam_d0.6 | - | no drafter | 1.000 | 34.10 | 1.00x | 1 | 1.000 | - |
| beam_d0.75 | - | DDTree tb64 | 10.024 | 4.76 | 6.99x | 65 | 0.154 | 0.09 |
| beam_d0.75 | - | sparked-tree tb64 | 9.653 | 5.31 | 6.26x | 65 | 0.149 | 0.61 |
| beam_d0.75 | - | sparked-tree wave tb64 | 9.653 | 5.28 | 6.29x | 65 | 0.149 | 0.56 |
| beam_d0.75 | - | DDTree+foreign head tb64 | 8.657 | 7.61 | 4.37x | 65 | 0.133 | 2.64 |
| beam_d0.75 | - | DSpark chain | 8.419 | 5.75 | 5.79x | 17 | 0.495 | - |
| beam_d0.75 | - | DFlash chain | 7.792 | 5.74 | 5.80x | 16 | 0.487 | - |
| beam_d0.75 | - | tree, independence tb64 | 3.437 | 13.42 | 2.48x | 65 | 0.053 | 0.27 |
| beam_d0.75 | - | no drafter | 1.000 | 33.25 | 1.00x | 1 | 1.000 | - |
| exact_d0.6 | - | sparked-tree tb64 | 11.453 | 5.96 | 5.53x | 65 | 0.176 | 2.14 |
| exact_d0.6 | - | DDTree tb64 | 10.024 | 4.71 | 6.99x | 65 | 0.154 | 0.10 |
| exact_d0.6 | - | DDTree+foreign head tb64 | 8.657 | 7.60 | 4.34x | 65 | 0.133 | 2.63 |
| exact_d0.6 | - | DSpark chain | 8.419 | 5.78 | 5.71x | 17 | 0.495 | - |
| exact_d0.6 | - | sparked-tree wave tb64 | 7.855 | 8.12 | 4.06x | 65 | 0.121 | 2.69 |
| exact_d0.6 | - | DFlash chain | 7.792 | 5.89 | 5.60x | 16 | 0.487 | - |
| exact_d0.6 | - | tree, independence tb64 | 3.437 | 13.56 | 2.43x | 65 | 0.053 | 0.28 |
| exact_d0.6 | - | no drafter | 1.000 | 32.97 | 1.00x | 1 | 1.000 | - |

## final-a10g

**GPU** A10G · **drafter** block-16 gamma=4 · **builder** beam + flat · **prompts** 12 x 512  
6-dataset validation; corrected the earlier gsm8k-only claim

| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |
|---|---|---|---|---|---|---|---|---|
| alpaca | - | sparked-tree tb64 | 5.314 | 12.96 | 2.96x | 65 | 0.082 | 2.89 |
| alpaca | - | sparked-tree wave tb64 | 5.314 | 12.88 | 2.98x | 65 | 0.082 | 2.86 |
| alpaca | - | DDTree tb64 | 4.173 | 13.82 | 2.77x | 65 | 0.064 | 0.42 |
| alpaca | - | DSpark chain | 4.124 | 16.19 | 2.37x | 17 | 0.243 | - |
| alpaca | - | DDTree+foreign head tb64 | 3.981 | 20.28 | 1.89x | 65 | 0.061 | 12.39 |
| alpaca | - | tree, independence tb64 | 2.829 | 18.48 | 2.07x | 65 | 0.044 | 0.62 |
| alpaca | - | DFlash chain | 2.825 | 19.26 | 1.99x | 16 | 0.177 | - |
| alpaca | - | no drafter | 1.000 | 38.34 | 1.00x | 1 | 1.000 | - |
| gsm8k | - | sparked-tree tb64 | 9.912 | 6.82 | 6.34x | 65 | 0.152 | 2.55 |
| gsm8k | - | sparked-tree wave tb64 | 9.912 | 6.75 | 6.40x | 65 | 0.152 | 2.51 |
| gsm8k | - | DDTree tb64 | 8.856 | 6.74 | 6.41x | 65 | 0.136 | 0.34 |
| gsm8k | - | DSpark chain | 7.701 | 8.02 | 5.39x | 17 | 0.453 | - |
| gsm8k | - | DDTree+foreign head tb64 | 7.681 | 10.34 | 4.18x | 65 | 0.118 | 9.44 |
| gsm8k | - | DFlash chain | 6.685 | 8.77 | 4.93x | 16 | 0.418 | - |
| gsm8k | - | tree, independence tb64 | 3.393 | 17.21 | 2.51x | 65 | 0.052 | 0.86 |
| gsm8k | - | no drafter | 1.000 | 43.21 | 1.00x | 1 | 1.000 | - |
| humaneval | - | DDTree tb64 | 8.633 | 5.49 | 5.92x | 65 | 0.133 | 0.48 |
| humaneval | - | sparked-tree tb64 | 8.327 | 6.24 | 5.20x | 65 | 0.128 | 3.63 |
| humaneval | - | sparked-tree wave tb64 | 8.327 | 6.17 | 5.26x | 65 | 0.128 | 3.40 |
| humaneval | - | DDTree+foreign head tb64 | 7.926 | 8.19 | 3.97x | 65 | 0.122 | 12.51 |
| humaneval | - | DFlash chain | 6.517 | 6.68 | 4.86x | 16 | 0.407 | - |
| humaneval | - | DSpark chain | 6.423 | 7.40 | 4.39x | 17 | 0.378 | - |
| humaneval | - | tree, independence tb64 | 3.574 | 13.20 | 2.46x | 65 | 0.055 | 1.16 |
| humaneval | - | no drafter | 1.000 | 32.49 | 1.00x | 1 | 1.000 | - |
| math500 | - | DDTree tb64 | 10.095 | 4.78 | 6.77x | 65 | 0.155 | 0.40 |
| math500 | - | sparked-tree tb64 | 10.075 | 5.19 | 6.24x | 65 | 0.155 | 3.00 |
| math500 | - | sparked-tree wave tb64 | 10.075 | 5.18 | 6.25x | 65 | 0.155 | 2.91 |
| math500 | - | DDTree+foreign head tb64 | 8.078 | 8.02 | 4.04x | 65 | 0.124 | 12.94 |
| math500 | - | DFlash chain | 7.897 | 5.64 | 5.73x | 16 | 0.494 | - |
| math500 | - | DSpark chain | 7.751 | 6.17 | 5.25x | 17 | 0.456 | - |
| math500 | - | tree, independence tb64 | 3.224 | 14.47 | 2.24x | 65 | 0.050 | 1.26 |
| math500 | - | no drafter | 1.000 | 32.37 | 1.00x | 1 | 1.000 | - |
| mbpp | - | DDTree tb64 | 8.623 | 5.41 | 6.10x | 65 | 0.133 | 0.29 |
| mbpp | - | sparked-tree tb64 | 8.233 | 6.08 | 5.44x | 65 | 0.127 | 2.46 |
| mbpp | - | sparked-tree wave tb64 | 8.233 | 6.16 | 5.36x | 65 | 0.127 | 2.46 |
| mbpp | - | DDTree+foreign head tb64 | 7.435 | 8.84 | 3.74x | 65 | 0.114 | 8.78 |
| mbpp | - | DSpark chain | 6.417 | 7.44 | 4.44x | 17 | 0.377 | - |
| mbpp | - | DFlash chain | 6.309 | 7.12 | 4.64x | 16 | 0.394 | - |
| mbpp | - | tree, independence tb64 | 3.269 | 13.79 | 2.39x | 65 | 0.050 | 0.77 |
| mbpp | - | no drafter | 1.000 | 33.03 | 1.00x | 1 | 1.000 | - |
| mt-bench | - | sparked-tree tb64 | 4.940 | 10.09 | 3.27x | 65 | 0.076 | 8.15 |
| mt-bench | - | sparked-tree wave tb64 | 4.940 | 10.10 | 3.26x | 65 | 0.076 | 7.96 |
| mt-bench | - | DDTree tb64 | 4.606 | 9.64 | 3.42x | 65 | 0.071 | 1.15 |
| mt-bench | - | DDTree+foreign head tb64 | 4.050 | 15.29 | 2.16x | 65 | 0.062 | 35.69 |
| mt-bench | - | DSpark chain | 3.762 | 11.93 | 2.76x | 17 | 0.221 | - |
| mt-bench | - | DFlash chain | 3.098 | 13.38 | 2.46x | 16 | 0.194 | - |
| mt-bench | - | tree, independence tb64 | 2.805 | 15.91 | 2.07x | 65 | 0.043 | 1.95 |
| mt-bench | - | no drafter | 1.000 | 32.95 | 1.00x | 1 | 1.000 | - |

## gpu-ctrl-A10G

**GPU** A10G · **drafter** block-7 (released) · **builder** best-first (exact) · **prompts** 8 x 384  
GPU control arm - identical config to gpu-ctrl-H100

| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |
|---|---|---|---|---|---|---|---|---|
| alpaca | - | sparked-tree wave tb64 | 5.480 | 10.42 | 3.05x | 65 | 0.084 | 2.19 |
| alpaca | - | sparked-tree tb64 | 5.430 | 10.90 | 2.92x | 65 | 0.084 | 2.97 |
| alpaca | - | DDTree tb64 | 4.236 | 11.28 | 2.82x | 65 | 0.065 | 0.23 |
| alpaca | - | DSpark chain | 4.026 | 11.68 | 2.72x | 17 | 0.237 | - |
| alpaca | - | DDTree+foreign head tb64 | 3.871 | 16.94 | 1.88x | 65 | 0.060 | 7.11 |
| alpaca | - | tree, independence tb64 | 3.256 | 13.44 | 2.37x | 65 | 0.050 | 0.26 |
| alpaca | - | DFlash chain | 2.911 | 15.37 | 2.07x | 16 | 0.182 | - |
| alpaca | - | no drafter | 1.000 | 31.79 | 1.00x | 1 | 1.000 | - |
| gsm8k | - | DDTree tb64 | 8.778 | 5.30 | 6.17x | 65 | 0.135 | 0.19 |
| gsm8k | - | DDTree+foreign head tb64 | 7.933 | 8.02 | 4.07x | 65 | 0.122 | 5.49 |
| gsm8k | - | sparked-tree tb64 | 7.676 | 7.51 | 4.35x | 65 | 0.118 | 3.45 |
| gsm8k | - | sparked-tree wave tb64 | 7.443 | 7.28 | 4.49x | 65 | 0.115 | 2.71 |
| gsm8k | - | DFlash chain | 6.657 | 6.68 | 4.90x | 16 | 0.416 | - |
| gsm8k | - | DSpark chain | 6.591 | 6.88 | 4.75x | 17 | 0.388 | - |
| gsm8k | - | tree, independence tb64 | 4.598 | 10.21 | 3.20x | 65 | 0.071 | 0.35 |
| gsm8k | - | no drafter | 1.000 | 32.68 | 1.00x | 1 | 1.000 | - |
| humaneval | - | DDTree tb64 | 8.914 | 6.38 | 6.54x | 65 | 0.137 | 0.25 |
| humaneval | - | DDTree+foreign head tb64 | 8.128 | 9.33 | 4.47x | 65 | 0.125 | 7.30 |
| humaneval | - | sparked-tree tb64 | 7.147 | 9.49 | 4.40x | 65 | 0.110 | 5.02 |
| humaneval | - | sparked-tree wave tb64 | 6.975 | 9.28 | 4.50x | 65 | 0.107 | 3.96 |
| humaneval | - | DFlash chain | 6.798 | 8.05 | 5.18x | 16 | 0.425 | - |
| humaneval | - | DSpark chain | 5.744 | 9.70 | 4.30x | 17 | 0.338 | - |
| humaneval | - | tree, independence tb64 | 4.645 | 12.19 | 3.42x | 65 | 0.071 | 0.45 |
| humaneval | - | no drafter | 1.000 | 41.73 | 1.00x | 1 | 1.000 | - |

## gpu-ctrl-H100

**GPU** H100 · **drafter** block-7 (released) · **builder** best-first (exact) · **prompts** 8 x 384  
GPU control arm - isolates hardware from checkpoint

| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |
|---|---|---|---|---|---|---|---|---|
| alpaca | - | sparked-tree tb64 | 5.563 | 13.84 | 2.79x | 65 | 0.086 | 1.84 |
| alpaca | - | sparked-tree wave tb64 | 5.465 | 11.53 | 3.35x | 65 | 0.084 | 1.19 |
| alpaca | - | DDTree tb64 | 4.203 | 23.25 | 1.66x | 65 | 0.065 | 0.24 |
| alpaca | - | DSpark chain | 4.009 | 22.08 | 1.75x | 17 | 0.236 | - |
| alpaca | - | DDTree+foreign head tb64 | 3.936 | 17.48 | 2.21x | 65 | 0.061 | 3.82 |
| alpaca | - | tree, independence tb64 | 3.310 | 18.33 | 2.10x | 65 | 0.051 | 0.28 |
| alpaca | - | DFlash chain | 2.909 | 27.00 | 1.43x | 16 | 0.182 | - |
| alpaca | - | no drafter | 1.000 | 38.57 | 1.00x | 1 | 1.000 | - |
| gsm8k | - | DDTree tb64 | 8.583 | 9.71 | 3.16x | 65 | 0.132 | 0.18 |
| gsm8k | - | DDTree+foreign head tb64 | 7.727 | 8.85 | 3.47x | 65 | 0.119 | 2.58 |
| gsm8k | - | sparked-tree tb64 | 7.702 | 10.23 | 3.00x | 65 | 0.118 | 2.01 |
| gsm8k | - | sparked-tree wave tb64 | 7.434 | 8.60 | 3.57x | 65 | 0.114 | 1.41 |
| gsm8k | - | DSpark chain | 6.701 | 10.31 | 2.98x | 17 | 0.394 | - |
| gsm8k | - | DFlash chain | 6.664 | 10.76 | 2.85x | 16 | 0.416 | - |
| gsm8k | - | tree, independence tb64 | 4.725 | 14.78 | 2.08x | 65 | 0.073 | 0.34 |
| gsm8k | - | no drafter | 1.000 | 30.68 | 1.00x | 1 | 1.000 | - |
| humaneval | - | DDTree tb64 | 8.900 | 11.13 | 3.11x | 65 | 0.137 | 0.26 |
| humaneval | - | DDTree+foreign head tb64 | 8.167 | 9.60 | 3.61x | 65 | 0.126 | 3.24 |
| humaneval | - | sparked-tree tb64 | 7.174 | 12.15 | 2.85x | 65 | 0.110 | 2.76 |
| humaneval | - | sparked-tree wave tb64 | 6.978 | 9.42 | 3.68x | 65 | 0.107 | 2.00 |
| humaneval | - | DFlash chain | 6.791 | 12.71 | 2.73x | 16 | 0.424 | - |
| humaneval | - | DSpark chain | 5.682 | 14.46 | 2.40x | 17 | 0.334 | - |
| humaneval | - | tree, independence tb64 | 4.727 | 16.78 | 2.07x | 65 | 0.073 | 0.44 |
| humaneval | - | no drafter | 1.000 | 34.68 | 1.00x | 1 | 1.000 | - |

## h100-bestcfg

**GPU** H100 · **drafter** block-16 H100 (overfit) · **builder** beam + flat · **prompts** 8 x 384  
3 datasets; first H100 run with the good builder

| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |
|---|---|---|---|---|---|---|---|---|
| alpaca | - | sparked-tree tb128 | 5.345 | 11.05 | 2.88x | 129 | 0.041 | 1.45 |
| alpaca | - | sparked-tree tb64 | 5.021 | 10.30 | 3.10x | 65 | 0.077 | 1.54 |
| alpaca | - | DDTree tb128 | 4.328 | 12.38 | 2.57x | 129 | 0.034 | 0.31 |
| alpaca | - | DDTree tb64 | 4.236 | 11.30 | 2.82x | 65 | 0.065 | 0.25 |
| alpaca | - | DSpark chain | 3.747 | 12.55 | 2.54x | 17 | 0.220 | - |
| alpaca | - | DFlash chain | 2.911 | 15.48 | 2.06x | 16 | 0.182 | - |
| alpaca | - | no drafter | 1.000 | 31.87 | 1.00x | 1 | 1.000 | - |
| gsm8k | - | sparked-tree tb128 | 10.214 | 5.69 | 5.79x | 129 | 0.079 | 1.36 |
| gsm8k | - | sparked-tree tb64 | 9.498 | 5.46 | 6.03x | 65 | 0.146 | 1.37 |
| gsm8k | - | DDTree tb128 | 9.259 | 5.88 | 5.60x | 129 | 0.072 | 0.25 |
| gsm8k | - | DDTree tb64 | 8.778 | 5.32 | 6.18x | 65 | 0.135 | 0.19 |
| gsm8k | - | DSpark chain | 7.242 | 6.63 | 4.96x | 17 | 0.426 | - |
| gsm8k | - | DFlash chain | 6.657 | 6.69 | 4.92x | 16 | 0.416 | - |
| gsm8k | - | no drafter | 1.000 | 32.91 | 1.00x | 1 | 1.000 | - |
| humaneval | - | DDTree tb128 | 9.375 | 5.83 | 5.49x | 129 | 0.073 | 0.31 |
| humaneval | - | DDTree tb64 | 9.147 | 5.08 | 6.29x | 65 | 0.141 | 0.24 |
| humaneval | - | sparked-tree tb128 | 8.673 | 6.80 | 4.70x | 129 | 0.067 | 2.02 |
| humaneval | - | sparked-tree tb64 | 8.052 | 6.40 | 5.00x | 65 | 0.124 | 2.11 |
| humaneval | - | DFlash chain | 7.000 | 6.23 | 5.13x | 16 | 0.438 | - |
| humaneval | - | DSpark chain | 6.188 | 7.59 | 4.21x | 17 | 0.364 | - |
| humaneval | - | no drafter | 1.000 | 31.96 | 1.00x | 1 | 1.000 | - |

## FINAL

**GPU** H100 · **drafter** block-16 gamma=4 (best) · **builder** beam + flat · **prompts** 12 x 512  
**headline result**

| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |
|---|---|---|---|---|---|---|---|---|
| alpaca | - | sparked-tree tb128 | 5.709 | 11.66 | 2.75x | 129 | 0.044 | 2.33 |
| alpaca | - | sparked-tree tb64 | 5.135 | 11.57 | 2.77x | 65 | 0.079 | 2.62 |
| alpaca | - | DDTree tb128 | 4.300 | 13.13 | 2.44x | 129 | 0.033 | 0.54 |
| alpaca | - | DDTree tb64 | 4.176 | 11.86 | 2.70x | 65 | 0.064 | 0.42 |
| alpaca | - | DSpark chain | 4.110 | 14.19 | 2.26x | 17 | 0.242 | - |
| alpaca | - | DFlash chain | 2.901 | 16.20 | 1.98x | 16 | 0.181 | - |
| alpaca | - | no drafter | 1.000 | 32.02 | 1.00x | 1 | 1.000 | - |
| gsm8k | - | sparked-tree tb128 | 10.630 | 5.26 | 6.01x | 129 | 0.082 | 1.88 |
| gsm8k | - | sparked-tree tb64 | 9.928 | 5.05 | 6.27x | 65 | 0.153 | 1.96 |
| gsm8k | - | DDTree tb128 | 9.154 | 5.77 | 5.48x | 129 | 0.071 | 0.38 |
| gsm8k | - | DDTree tb64 | 8.686 | 5.16 | 6.13x | 65 | 0.134 | 0.30 |
| gsm8k | - | DSpark chain | 7.901 | 5.81 | 5.45x | 17 | 0.465 | - |
| gsm8k | - | DFlash chain | 6.646 | 6.51 | 4.86x | 16 | 0.415 | - |
| gsm8k | - | no drafter | 1.000 | 31.66 | 1.00x | 1 | 1.000 | - |
| humaneval | - | sparked-tree tb128 | 8.899 | 7.46 | 5.60x | 129 | 0.069 | 4.10 |
| humaneval | - | DDTree tb128 | 8.765 | 6.88 | 6.07x | 129 | 0.068 | 0.56 |
| humaneval | - | DDTree tb64 | 8.603 | 6.61 | 6.32x | 65 | 0.132 | 0.45 |
| humaneval | - | sparked-tree tb64 | 8.226 | 7.62 | 5.49x | 65 | 0.127 | 4.35 |
| humaneval | - | DSpark chain | 6.465 | 9.05 | 4.62x | 17 | 0.380 | - |
| humaneval | - | DFlash chain | 6.452 | 8.51 | 4.91x | 16 | 0.403 | - |
| humaneval | - | no drafter | 1.000 | 41.78 | 1.00x | 1 | 1.000 | - |
| math500 | - | sparked-tree tb128 | 10.590 | 6.21 | 6.40x | 129 | 0.082 | 3.41 |
| math500 | - | DDTree tb128 | 10.418 | 5.85 | 6.80x | 129 | 0.081 | 0.48 |
| math500 | - | sparked-tree tb64 | 10.179 | 6.07 | 6.54x | 65 | 0.157 | 3.42 |
| math500 | - | DDTree tb64 | 9.959 | 5.58 | 7.12x | 65 | 0.153 | 0.39 |
| math500 | - | DFlash chain | 8.110 | 6.70 | 5.93x | 16 | 0.507 | - |
| math500 | - | DSpark chain | 7.604 | 7.37 | 5.39x | 17 | 0.447 | - |
| math500 | - | no drafter | 1.000 | 39.73 | 1.00x | 1 | 1.000 | - |
| mbpp | - | sparked-tree tb128 | 9.278 | 6.18 | 5.37x | 129 | 0.072 | 2.13 |
| mbpp | - | DDTree tb128 | 8.802 | 6.10 | 5.43x | 129 | 0.068 | 0.39 |
| mbpp | - | DDTree tb64 | 8.623 | 5.38 | 6.17x | 65 | 0.133 | 0.29 |
| mbpp | - | sparked-tree tb64 | 8.524 | 5.91 | 5.61x | 65 | 0.131 | 2.21 |
| mbpp | - | DSpark chain | 6.417 | 7.58 | 4.38x | 17 | 0.377 | - |
| mbpp | - | DFlash chain | 6.309 | 7.17 | 4.62x | 16 | 0.394 | - |
| mbpp | - | no drafter | 1.000 | 33.16 | 1.00x | 1 | 1.000 | - |
| mt-bench | - | sparked-tree tb128 | 5.513 | 10.19 | 3.10x | 129 | 0.043 | 7.01 |
| mt-bench | - | sparked-tree tb64 | 5.062 | 9.64 | 3.28x | 65 | 0.078 | 7.25 |
| mt-bench | - | DDTree tb128 | 5.013 | 10.27 | 3.07x | 129 | 0.039 | 1.36 |
| mt-bench | - | DDTree tb64 | 4.698 | 9.37 | 3.37x | 65 | 0.072 | 1.12 |
| mt-bench | - | DSpark chain | 3.951 | 11.02 | 2.86x | 17 | 0.232 | - |
| mt-bench | - | DFlash chain | 3.197 | 12.56 | 2.51x | 16 | 0.200 | - |
| mt-bench | - | no drafter | 1.000 | 31.56 | 1.00x | 1 | 1.000 | - |

## final-bigdata

**GPU** H100 · **drafter** block-16 bigdata (9.5k convs) · **builder** beam + flat · **prompts** 12 x 512  
4x data BUT also seq 768->512, anchors 32->96 - confounded, worse

| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |
|---|---|---|---|---|---|---|---|---|
| alpaca | - | sparked-tree tb128 | 5.680 | 14.28 | 2.85x | 129 | 0.044 | 2.89 |
| alpaca | - | sparked-tree tb64 | 5.078 | 14.85 | 2.74x | 65 | 0.078 | 3.19 |
| alpaca | - | DDTree tb128 | 4.447 | 13.97 | 2.91x | 129 | 0.034 | 0.46 |
| alpaca | - | DDTree tb64 | 4.173 | 14.59 | 2.79x | 65 | 0.064 | 0.41 |
| alpaca | - | DSpark chain | 3.906 | 17.58 | 2.31x | 17 | 0.230 | - |
| alpaca | - | DFlash chain | 2.825 | 20.35 | 2.00x | 16 | 0.177 | - |
| alpaca | - | no drafter | 1.000 | 40.66 | 1.00x | 1 | 1.000 | - |
| gsm8k | - | sparked-tree tb128 | 10.131 | 6.40 | 6.34x | 129 | 0.079 | 2.40 |
| gsm8k | - | sparked-tree tb64 | 9.620 | 6.51 | 6.23x | 65 | 0.148 | 2.49 |
| gsm8k | - | DDTree tb128 | 9.079 | 6.52 | 6.22x | 129 | 0.070 | 0.38 |
| gsm8k | - | DDTree tb64 | 8.856 | 6.43 | 6.31x | 65 | 0.136 | 0.30 |
| gsm8k | - | DSpark chain | 7.349 | 7.93 | 5.11x | 17 | 0.432 | - |
| gsm8k | - | DFlash chain | 6.685 | 8.22 | 4.93x | 16 | 0.418 | - |
| gsm8k | - | no drafter | 1.000 | 40.57 | 1.00x | 1 | 1.000 | - |
| humaneval | - | DDTree tb128 | 8.860 | 6.24 | 5.20x | 129 | 0.069 | 0.58 |
| humaneval | - | DDTree tb64 | 8.633 | 5.45 | 5.96x | 65 | 0.133 | 0.46 |
| humaneval | - | sparked-tree tb128 | 8.511 | 6.94 | 4.68x | 129 | 0.066 | 3.51 |
| humaneval | - | sparked-tree tb64 | 7.923 | 6.51 | 4.99x | 65 | 0.122 | 3.79 |
| humaneval | - | DFlash chain | 6.517 | 6.71 | 4.83x | 16 | 0.407 | - |
| humaneval | - | DSpark chain | 6.169 | 7.73 | 4.20x | 17 | 0.363 | - |
| humaneval | - | no drafter | 1.000 | 32.46 | 1.00x | 1 | 1.000 | - |
| math500 | - | sparked-tree tb128 | 10.334 | 5.81 | 5.72x | 129 | 0.080 | 3.14 |
| math500 | - | DDTree tb128 | 10.300 | 5.53 | 6.01x | 129 | 0.080 | 0.52 |
| math500 | - | DDTree tb64 | 10.095 | 4.81 | 6.91x | 65 | 0.155 | 0.40 |
| math500 | - | sparked-tree tb64 | 9.641 | 5.45 | 6.09x | 65 | 0.148 | 3.28 |
| math500 | - | DFlash chain | 7.897 | 5.78 | 5.75x | 16 | 0.494 | - |
| math500 | - | DSpark chain | 7.457 | 6.55 | 5.08x | 17 | 0.439 | - |
| math500 | - | no drafter | 1.000 | 33.23 | 1.00x | 1 | 1.000 | - |
| mbpp | - | DDTree tb128 | 8.802 | 6.01 | 5.46x | 129 | 0.068 | 0.38 |
| mbpp | - | sparked-tree tb128 | 8.653 | 6.57 | 5.00x | 129 | 0.067 | 2.32 |
| mbpp | - | DDTree tb64 | 8.623 | 5.35 | 6.15x | 65 | 0.133 | 0.29 |
| mbpp | - | sparked-tree tb64 | 8.059 | 6.24 | 5.27x | 65 | 0.124 | 2.34 |
| mbpp | - | DFlash chain | 6.309 | 7.11 | 4.62x | 16 | 0.394 | - |
| mbpp | - | DSpark chain | 5.988 | 7.97 | 4.12x | 17 | 0.352 | - |
| mbpp | - | no drafter | 1.000 | 32.86 | 1.00x | 1 | 1.000 | - |
| mt-bench | - | sparked-tree tb128 | 5.259 | 10.91 | 3.03x | 129 | 0.041 | 7.71 |
| mt-bench | - | DDTree tb128 | 4.895 | 10.75 | 3.07x | 129 | 0.038 | 1.39 |
| mt-bench | - | sparked-tree tb64 | 4.832 | 10.29 | 3.21x | 65 | 0.074 | 8.29 |
| mt-bench | - | DDTree tb64 | 4.651 | 9.62 | 3.44x | 65 | 0.072 | 1.12 |
| mt-bench | - | DSpark chain | 3.691 | 12.35 | 2.68x | 17 | 0.217 | - |
| mt-bench | - | DFlash chain | 3.072 | 13.58 | 2.44x | 16 | 0.192 | - |
| mt-bench | - | no drafter | 1.000 | 33.06 | 1.00x | 1 | 1.000 | - |

## confidence

**GPU** A10G · **drafter** block-16 bigdata · **builder** beam flat vs confidence-adaptive · **prompts** 10 x 512  
confidence-head widths: -2.7% accept, -6.6% speed

| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |
|---|---|---|---|---|---|---|---|---|
| alpaca | beam | sparked-tree tb128 | 5.733 | 11.97 | 2.73x | 129 | 0.044 | 2.37 |
| alpaca | beam | sparked-tree tb64 | 5.134 | 11.96 | 2.73x | 65 | 0.079 | 2.57 |
| alpaca | beam | DDTree tb128 | 4.311 | 13.39 | 2.44x | 129 | 0.033 | 0.54 |
| alpaca | beam | DDTree tb64 | 4.194 | 11.90 | 2.74x | 65 | 0.065 | 0.41 |
| alpaca | beam | DSpark chain | 3.939 | 13.59 | 2.40x | 17 | 0.232 | - |
| alpaca | beam | DFlash chain | 2.907 | 16.49 | 1.98x | 16 | 0.182 | - |
| alpaca | beam | no drafter | 1.000 | 32.64 | 1.00x | 1 | 1.000 | - |
| alpaca | confidence | sparked-tree tb128 | 5.693 | 12.45 | 2.63x | 129 | 0.044 | 3.21 |
| alpaca | confidence | sparked-tree tb64 | 5.316 | 12.28 | 2.67x | 65 | 0.082 | 3.04 |
| alpaca | confidence | DDTree tb128 | 4.311 | 13.37 | 2.45x | 129 | 0.033 | 0.54 |
| alpaca | confidence | DDTree tb64 | 4.194 | 11.86 | 2.76x | 65 | 0.065 | 0.42 |
| alpaca | confidence | DSpark chain | 3.939 | 13.70 | 2.39x | 17 | 0.232 | - |
| alpaca | confidence | DFlash chain | 2.907 | 16.49 | 1.99x | 16 | 0.182 | - |
| alpaca | confidence | no drafter | 1.000 | 32.74 | 1.00x | 1 | 1.000 | - |
| gsm8k | beam | sparked-tree tb128 | 10.361 | 5.58 | 5.93x | 129 | 0.080 | 1.74 |
| gsm8k | beam | sparked-tree tb64 | 9.842 | 5.29 | 6.26x | 65 | 0.151 | 1.84 |
| gsm8k | beam | DDTree tb128 | 9.228 | 5.89 | 5.63x | 129 | 0.072 | 0.34 |
| gsm8k | beam | DDTree tb64 | 8.741 | 5.34 | 6.21x | 65 | 0.134 | 0.26 |
| gsm8k | beam | DSpark chain | 7.484 | 6.52 | 5.08x | 17 | 0.440 | - |
| gsm8k | beam | DFlash chain | 6.628 | 6.79 | 4.88x | 16 | 0.414 | - |
| gsm8k | beam | no drafter | 1.000 | 33.12 | 1.00x | 1 | 1.000 | - |
| gsm8k | confidence | sparked-tree tb128 | 9.746 | 6.34 | 5.26x | 129 | 0.076 | 2.78 |
| gsm8k | confidence | sparked-tree tb64 | 9.486 | 5.71 | 5.85x | 65 | 0.146 | 2.70 |
| gsm8k | confidence | DDTree tb128 | 9.228 | 5.96 | 5.60x | 129 | 0.072 | 0.35 |
| gsm8k | confidence | DDTree tb64 | 8.741 | 5.42 | 6.15x | 65 | 0.134 | 0.29 |
| gsm8k | confidence | DSpark chain | 7.484 | 6.50 | 5.14x | 17 | 0.440 | - |
| gsm8k | confidence | DFlash chain | 6.628 | 6.80 | 4.90x | 16 | 0.414 | - |
| gsm8k | confidence | no drafter | 1.000 | 33.37 | 1.00x | 1 | 1.000 | - |
| humaneval | beam | DDTree tb128 | 8.964 | 6.34 | 5.24x | 129 | 0.069 | 0.50 |
| humaneval | beam | DDTree tb64 | 8.769 | 5.47 | 6.07x | 65 | 0.135 | 0.40 |
| humaneval | beam | sparked-tree tb128 | 8.675 | 6.98 | 4.76x | 129 | 0.067 | 2.97 |
| humaneval | beam | sparked-tree tb64 | 8.037 | 6.56 | 5.06x | 65 | 0.124 | 3.26 |
| humaneval | beam | DFlash chain | 6.635 | 6.71 | 4.95x | 16 | 0.415 | - |
| humaneval | beam | DSpark chain | 6.271 | 7.75 | 4.28x | 17 | 0.369 | - |
| humaneval | beam | no drafter | 1.000 | 33.20 | 1.00x | 1 | 1.000 | - |
| humaneval | confidence | DDTree tb128 | 8.964 | 6.22 | 5.21x | 129 | 0.069 | 0.47 |
| humaneval | confidence | DDTree tb64 | 8.769 | 5.40 | 6.00x | 65 | 0.135 | 0.39 |
| humaneval | confidence | sparked-tree tb128 | 8.401 | 7.38 | 4.39x | 129 | 0.065 | 4.35 |
| humaneval | confidence | sparked-tree tb64 | 7.926 | 6.80 | 4.77x | 65 | 0.122 | 4.45 |
| humaneval | confidence | DFlash chain | 6.635 | 6.64 | 4.88x | 16 | 0.415 | - |
| humaneval | confidence | DSpark chain | 6.271 | 7.46 | 4.34x | 17 | 0.369 | - |
| humaneval | confidence | no drafter | 1.000 | 32.41 | 1.00x | 1 | 1.000 | - |
| math500 | beam | DDTree tb128 | 11.023 | 5.13 | 6.34x | 129 | 0.085 | 0.43 |
| math500 | beam | DDTree tb64 | 10.903 | 4.35 | 7.49x | 65 | 0.168 | 0.33 |
| math500 | beam | sparked-tree tb128 | 10.853 | 5.53 | 5.89x | 129 | 0.084 | 2.58 |
| math500 | beam | sparked-tree tb64 | 10.110 | 5.19 | 6.27x | 65 | 0.156 | 2.69 |
| math500 | beam | DFlash chain | 8.716 | 5.04 | 6.46x | 16 | 0.545 | - |
| math500 | beam | DSpark chain | 7.932 | 6.03 | 5.40x | 17 | 0.467 | - |
| math500 | beam | no drafter | 1.000 | 32.54 | 1.00x | 1 | 1.000 | - |
| math500 | confidence | DDTree tb128 | 11.023 | 5.12 | 6.41x | 129 | 0.085 | 0.42 |
| math500 | confidence | DDTree tb64 | 10.903 | 4.35 | 7.55x | 65 | 0.168 | 0.33 |
| math500 | confidence | sparked-tree tb128 | 10.312 | 6.04 | 5.44x | 129 | 0.080 | 3.92 |
| math500 | confidence | sparked-tree tb64 | 9.579 | 5.68 | 5.78x | 65 | 0.147 | 4.04 |
| math500 | confidence | DFlash chain | 8.716 | 5.06 | 6.49x | 16 | 0.545 | - |
| math500 | confidence | DSpark chain | 7.932 | 6.07 | 5.41x | 17 | 0.467 | - |
| math500 | confidence | no drafter | 1.000 | 32.83 | 1.00x | 1 | 1.000 | - |
