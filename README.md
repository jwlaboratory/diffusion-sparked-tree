# SparklingTree

Frontier speculative decoding. Read more here: https://jwlabs.vercel.app/post/sparklingtree

## Folders

- `harness/` — shared decode core (DDTree, DFlash, DSpark, SparklingTree) + the benchmark runner
- `experiment1-harness/` — first end-to-end harness benchmark
- `experiment2-block16/` — fine-tuning DSpark to draft 16 tokens/block + training recipe
- `experiment3-timings/` — per-phase wall-clock timing instrumentation
- `experiment4-faster/` — faster tree builder (the precompute best-first builder)
- `experiment5-final-results/` — head-to-head: DFlash vs DSpark vs DDTree vs SparklingTree
- `archive-1-speedup-w-confidence-head-verification/` — earlier confidence-head verification attempt
- `archive-2-finding_best-beam/` — earlier beam-search tuning
- `old-experiments/` — scratch / superseded runs
- `reference-papers/` — DDTree, DFlash, DSpark PDFs
- `assets/` — figures and images



EXACT FULL RESULTS — citable run (seed 1, 6ds x 12, 512 tok, temp 0, sync ON, compaction ON, C=128, fanout 64)

===== BUDGET 64 =====
dataset           Autoregressiv              DFlash              DSpark              DDTree       SparklingTree
alpaca       1.00a    51.0t       3.07a    95.4t       3.66a   109.9t       4.46a   132.7t       5.10a   149.3t     
gsm8k        1.00a    52.2t       6.71a   218.0t       7.88a   242.6t       8.65a   268.7t       9.73a   298.6t     
humaneval    1.00a    52.7t       6.32a   207.5t       6.41a   200.2t       8.19a   267.7t       8.78a   273.1t     
math500      1.00a    52.0t       7.80a   253.6t       7.37a   230.5t       9.70a   309.7t       9.25a   287.2t     
mbpp         1.00a    52.4t       5.47a   176.6t       5.77a   179.8t       7.60a   231.6t       7.53a   224.5t     
mt-bench     1.00a    51.5t       3.31a   105.8t       3.76a   114.7t       4.64a   145.3t       5.15a   157.0t     
AGGREGATE    1.00a    52.0t       5.10a   163.9t       5.53a   170.3t       6.86a   214.4t       7.34a   223.4t     
SPEEDUP                 1.00x                3.15x                3.27x                4.12x                4.29x   

===== BUDGET 128 =====
dataset           Autoregressiv              DFlash              DSpark              DDTree       SparklingTree
alpaca       1.00a    51.0t       3.07a    95.4t       3.66a   109.9t       4.61a   147.8t       5.40a   169.0t     
gsm8k        1.00a    52.2t       6.71a   218.0t       7.88a   242.6t       8.99a   280.9t      10.30a   311.9t     
humaneval    1.00a    52.7t       6.32a   207.5t       6.41a   200.2t       8.97a   284.7t       9.13a   279.8t     
math500      1.00a    52.0t       7.80a   253.6t       7.37a   230.5t      10.14a   323.9t       9.71a   302.0t     
mbpp         1.00a    52.4t       5.47a   176.6t       5.77a   179.8t       7.80a   252.9t       7.88a   244.7t     
mt-bench     1.00a    51.5t       3.31a   105.8t       3.76a   114.7t       5.02a   158.0t       5.44a   167.1t     
AGGREGATE    1.00a    52.0t       5.10a   163.9t       5.53a   170.3t       7.28a   231.5t       7.73a   238.3t     
SPEEDUP                 1.00x                3.15x                3.27x                4.45x                4.58x   

===== BUDGET 256 =====
dataset           Autoregressiv              DFlash              DSpark              DDTree       SparklingTree
alpaca       1.00a    51.0t       3.07a    95.4t       3.66a   109.9t       4.95a   152.1t       5.53a   168.1t     
gsm8k        1.00a    52.2t       6.71a   218.0t       7.88a   242.6t       9.53a   300.8t      10.72a   327.5t     
humaneval    1.00a    52.7t       6.32a   207.5t       6.41a   200.2t       9.31a   286.2t       9.52a   283.7t     
math500      1.00a    52.0t       7.80a   253.6t       7.37a   230.5t      10.39a   331.3t      10.22a   316.6t     
mbpp         1.00a    52.4t       5.47a   176.6t       5.77a   179.8t       8.21a   252.5t       8.38a   248.8t     
mt-bench     1.00a    51.5t       3.31a   105.8t       3.76a   114.7t       5.24a   158.8t       5.87a   172.9t     
AGGREGATE    1.00a    52.0t       5.10a   163.9t       5.53a   170.3t       7.67a   237.1t       8.15a   245.0t     
SPEEDUP                 1.00x                3.15x                3.27x                4.55x                4.71x   
