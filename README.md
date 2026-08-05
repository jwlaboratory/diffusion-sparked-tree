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
