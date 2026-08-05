# Reproduce — Exp 4-faster / 2 (precompute, stacked on transfer-less)

The precompute builder is built **on top of** the 1-transfer-less fixes, not instead
of them: it GPU-top-k's to `C` candidates and ships ONE small top-k slice to CPU (no
311 MB static-weight resend — the transfer-less win), then adds the batched
`[L-1, C, C]` transition table so the heap walk is pure CPU. So `bestfirst.precompute`
is the **combined stack** (transfer-less + precompute); `bestfirst.fast` is
transfer-less alone, kept as the baseline so the marginal precompute gain is visible.

Both arms run at **C=512** so the candidate set (hence the tree, hence acceptance) is
controlled — any speed delta is purely the precompute mechanism.

## Correctness gate (local, CPU, no GPU)

```bash
# needs torch + numpy only; stubs out transformers/model at import
python experiment4-faster/2-precompute/test_precompute_builder.py
```

Asserts `build_sparked_tree_precompute` builds the SAME best-first tree as
`build_sparked_tree_fast` at C=512 (measured 100% node agreement, seeds 0–5,
budgets 64/256), plus the `markov_head=None` and empty-budget paths.

## Smoke (Modal, minutes)

```bash
cd experiment4-faster/2-precompute
modal run modal_benchmark.py --smoke      # gsm8k n1, 64 tokens, budget 64
```

## Full run (Modal, detached — survives CLI disconnect)

```bash
cd experiment4-faster/2-precompute
modal run --detach modal_benchmark.py --spawn
# progress:  modal volume ls ddtree-results precompute/cache/<fingerprint>
# fetch:     modal volume get ddtree-results precompute/summary.json results/summary.json
```

Per-unit checkpointed + resumable (config+CODE_VERSION fingerprinted); a re-run
skips completed `both__b{budget}__{dataset}__n{samples}.json` units.

Config: H100 + 8 CPU, DSpark-b16 + its markov head, budgets {64, 256},
gsm8k/humaneval/mt-bench × 4 samples × 512 tokens, temp 0, two-pass
(clean TPS / instrumented phases). CODE_VERSION `harness-5-precompute`.

## Analyze

```bash
python analyze.py results/summary.json      # (a) time (b) acceptance (c) speedup
python make_charts.py results/summary.json  # speedup_acceptance.png, phase_collapse.png
```
