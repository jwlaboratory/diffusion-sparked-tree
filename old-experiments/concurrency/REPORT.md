# Concurrency benchmark

`H100` · `lmsysorg/sglang:v0.5.16-cu129` · target `Qwen/Qwen3-4B` · dataset `sharegpt`

## Output throughput (tok/s)

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| baseline_ov | 240.0 | 441.8 | 778.4 | 1456.3 | 2573.5 | 4490.7 |
| baseline_noov | 204.4 | 374.8 | 658.7 | 1221.7 | 2147.9 | 3700.2 |
| dflash_ov | 565.0 | 952.9 | 1585.3 | 2626.2 | 3926.8 | 6205.3 |
| dflash_noov | 481.8 | 843.8 | 1350.3 | 2227.0 | 3526.9 | 5453.6 |
| dspark_ov | 610.3 | 1064.8 | 1760.1 | 3201.5 | 4627.9 | 8069.0 |
| dspark_noov | 489.9 | 855.2 | 1392.9 | 2343.3 | 3565.6 | 6029.7 |

## Mean TPOT (ms)

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| baseline_ov | 4.09 | 4.25 | 4.44 | 4.76 | 5.43 | 6.34 |
| baseline_noov | 4.87 | 5.08 | 5.33 | 5.74 | 6.55 | 7.90 |
| dflash_ov | 2.16 | 2.40 | 2.57 | 3.15 | 4.47 | 6.49 |
| dflash_noov | 2.72 | 2.98 | 3.67 | 3.82 | 5.54 | 7.52 |
| dspark_ov | 2.02 | 2.17 | 2.32 | 2.50 | 3.85 | 5.17 |
| dspark_noov | 2.71 | 2.93 | 4.48 | 3.82 | 7.41 | 7.62 |

## Acceptance length

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| baseline_ov | — | — | — | — | — | — |
| baseline_noov | — | — | — | — | — | — |
| dflash_ov | 3.380 | 3.328 | 3.337 | 3.308 | 3.293 | 3.296 |
| dflash_noov | 3.352 | 3.339 | 3.330 | 3.240 | 3.266 | 3.273 |
| dspark_ov | 3.822 | 3.794 | 3.798 | 3.761 | 3.730 | 3.752 |
| dspark_noov | 3.837 | 3.827 | 3.821 | 3.787 | 3.738 | 3.727 |

## What overlap is worth

Ratio of output throughput. The chain rows are the price a host-resident
tree builder pays for being unable to register `supports_overlap=True`.

| ratio | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| overlap value, baseline | 1.174x | 1.179x | 1.182x | 1.192x | 1.198x | 1.214x |
| overlap value, DFlash | 1.173x | 1.129x | 1.174x | 1.179x | 1.113x | 1.138x |
| overlap value, DSpark | 1.246x | 1.245x | 1.264x | 1.366x | 1.298x | 1.338x |
| DSpark vs DFlash (off) | 1.017x | 1.013x | 1.032x | 1.052x | 1.011x | 1.106x |

## Not measured

| arm | blocked by |
|---|---|
| ddtree_noov | no DDTREE builtin; needs a plugin via SpeculativeAlgorithm.register(). Same plugin as sparked_noov with build_ddtree_tree swapped in for the markov builder — one plugin unblocks both arms. |
| sparked_noov | plugin not yet written. Verify path is reusable (see above); the work is a V2 worker plus a SpecInput subclass. Overlap stays off regardless: the builder syncs at int(root_token_id) in and .cpu().numpy() out (ddtree_markov.py:851,855), plus the numpy heapq walk in the shipped exact-precomputed arm. |

Both tree arms are absent, so this run does **not** compare trees against
chains at concurrency. What it establishes is the handicap those arms will
carry when they land, and the chain baseline they must beat.
