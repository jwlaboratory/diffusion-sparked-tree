# Exp4-faster / 2-precompute / csweep: candidate-size (C) sweep, both builders

**Question.** What is the smallest candidate size C at which each restricted best-first
builder saturates acceptance against the exact best-first ceiling — and how does that C
trade off against speed? The two builders restrict differently (`fast` = deduped UNION
of per-depth top-C; `precompute` = PER-DEPTH top-C), so their knees can differ. We sweep
C on each directly and read acceptance / TPS / candidate_build cost off the curve.

**Arms** (DSpark-b16 + `dspark_b16_markov`, verify=tree, one job):
`bestfirst.ref` (exact, C=∞ ceiling); `fast.c{128,256,512,1024,2048}`;
`precompute.c{128,256,512,1024}` (precompute is O(L·C²·R), so stopped at 1024).
Budget 64; gsm8k/humaneval/mt-bench ×8; temp 0, max_new_tokens 512, seed 0, warmup 256,
discard-first, passes clean+instrumented; H100 + 8 CPU.

## Local correctness gate (CPU, no GPU) — run first

```
python ../test_precompute_builder.py
```
Must print `All precompute builder checks passed.` The load-bearing check is
`test_reference_equivalence`: at C ≥ vocab, `build_sparked_tree_precompute` reproduces
the exact `build_sparked_tree` tree (>99%, observed 100%) up to float32 reduction-order
epsilon — with and without a corrector.

## Benchmark (Modal, H100, checkpointed)

```
# 1. end-to-end GPU validation (validates all 10 arms incl. precompute)
modal run modal_benchmark.py --smoke

# 2. full run, DETACHED (a bare --spawn dies with the CLI)
modal run --detach modal_benchmark.py --spawn

# 3. progress (units checkpointed per budget,dataset) and fetch
modal volume ls  ddtree-results csweep/cache
modal volume get ddtree-results csweep/summary.json results/summary.json
```
App `ddtree-exp4-csweep`; cache `/results/csweep/cache`; summary `/results/csweep/summary.json`.
Driver `CODE_VERSION=harness-5-csweep` so no fastbf/precompute unit resumes here.

## Analysis

```
python analyze.py               # acceptance & TPS vs C per builder; .prep/.expand vs C; knee
python make_charts.py           # pareto_accept_tps.png, accept_vs_c.png, prep_expand_vs_c.png
```
Caveat: per-arm acceptance deltas are ~1-2%, near the n=8 noise floor; analyze.py prints
the per-dataset spread and flags the knee only when the gap to ceiling clears that spread.

## Note on directory layout

This C-sweep lives in the `csweep/` subdir to avoid clobbering the finished
`2-precompute/` head-to-head experiment (fast@512 vs precompute@512), whose untracked
`modal_benchmark.py`, `RESULTS.md`, and `results/summary.json` remain intact one level up.
