# Exp4b (1-transfer-less): fast best-first markov tree

**Question.** Exp3 showed `build_sparked_tree`'s `candidate_build` is 89-96% of decode
wall-clock, split between `.prep` (copying the static full-vocab markov matrices
GPU->CPU every round, ~120 ms budget-invariant) and `.expand` (a full-vocab
log_softmax + top-k + bias per popped node). Does porting the beam builder's
candidate-union restriction into the best-first heap kill both costs **without**
losing acceptance?

**The change.** New `build_sparked_tree_fast` in `harness/ddtree/sparked_tree.py`:
top-k on GPU -> gather only the ~U active columns of `base_logits` and the ~U active
rows of `W1`/`W2` to CPU **once**; every per-pop compute runs on that length-U slice.
The heap's adaptive (best-first) node allocation is unchanged. `beam_candidates=0`
makes it byte-for-byte identical to `build_sparked_tree` (the correctness gate).
Wired as `tree_mode="best-first-fast"`. Driver `CODE_VERSION` bumped to
`harness-4-fastbf` so no stale exp3/exp4 cache unit resumes against the new code.

## Local correctness gate (CPU, no GPU)

```
python test_fast_builder.py
```
Asserts exact tree equivalence (`candidates=0` and `candidates>=vocab`) vs
`build_sparked_tree` across seeds/budgets, the `markov_head=None` path, and a
subset/near-match under a moderate restriction. Must print
`ALL EXACT-EQUIVALENCE GATES PASSED`.

## Benchmark (Modal, H100, checkpointed)

Two arms on the same backbone+corrector (DSpark-b16 + `dspark_b16_markov`), one job:
`bestfirst.ref` (slow builder) and `bestfirst.fast` (new). Matched to exp3/exp4:
budgets {64,256}, gsm8k/humaneval/mt-bench x4, temp 0, max_new_tokens 512, seed 0,
warmup 256, discard-first, passes clean+instrumented. Fresh cache `/results/fastbf/cache`.

```
# 1. end-to-end validation on GPU (minutes)
modal run modal_benchmark.py --smoke

# 2. full run, DETACHED (a bare --spawn dies with the CLI; --detach keeps it alive)
modal run --detach modal_benchmark.py --spawn

# 3. progress (units checkpointed per budget,dataset) and fetch
modal volume ls  ddtree-results fastbf/cache
modal volume get ddtree-results fastbf/summary.json results/summary.json
```

## Analysis

```
python analyze.py            # reads results/summary.json, prints the three deltas
```
Prints, per budget: (a) candidate_build ms/round & ms/commit-token ref vs fast with
`.prep`/`.expand` broken out and total ms/round; (b) mean acceptance ref vs fast per
dataset + round-weighted aggregate and the delta; (c) `tps_clean` fast/ref per budget
and dataset as a multiple. See `RESULTS.md` for the computed numbers.
