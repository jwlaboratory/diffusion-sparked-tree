# Shared harness — migration plan

**Status: plan, not yet built.** Nothing under `harness/` is wired up yet; exp1 and
exp2 still run from their own `DDTree/` copies.

## Why

`experiment1-harness/DDTree/` and `experiment2-block16/DDTree/` are **byte-identical**
for the entire decode core:

```
ddtree.py  dflash.py  dspark.py  sparked_tree.py
model/__init__.py  model/dflash.py  model/dspark.py  model/utils.py
```

Only `run_experiment.py` and `aggregate.py` diverge, and they diverge in ways that are
complementary rather than conflicting:

| | exp1 driver | exp2 driver |
|---|---|---|
| backbone kinds | dflash **and** dspark | dspark only (raises otherwise) |
| RoPE normalization + `inv_freq` guard | no | **yes** |
| block-size guard | no | yes (keyed by backbone name) |
| tree budget | scalar | **swept list** |
| cache unit | per-dataset, fingerprinted with `code_version` | per-`(budget, dataset)`, **no fingerprint** |
| parallelism | one container per dataset (`.map`) | one container, sequential |
| timing captured | **yes** (`stage_times`, `tpot`, `ttft`) — but never surfaced into `summary.json` | **no** — only `acceptance_lengths` |

Experiment 3 needs dflash *and* the rope guard *and* the budget sweep *and* timing
capture — i.e. the union. Copying a third time is the wrong move, and exp3's
instrumentation work (splitting `commit` into `walk_accept` / `kv_update` /
`state_carry`) touches the decode core, which is exactly the part that is currently
duplicated. Instrument it three times and the copies stop being identical, silently.

**Also:** exp1's per-dataset fan-out puts different datasets on different containers.
That is fine for acceptance (deterministic) and wrong for timing (different hardware
states). The shared driver keeps both execution modes and exp3 selects the sequential
one.

## Target layout

```
harness/
  PLAN.md                  this file
  README.md                usage, once built
  pyproject.toml           installed editable into the Modal image
  ddtree/                  the decode core — ONE copy
    __init__.py
    ddtree.py              ddtree_generate + build_ddtree_tree + KV compaction
    dflash.py              dflash_generate + cuda_time/empty_stage_times
    dspark.py              dspark_generate
    sparked_tree.py        sparked_tree_generate + build_sparked_tree
    timing.py              NEW — timer registry, on/off switch, canonical phase map
    model/{__init__,dflash,dspark,utils}.py
  runner/
    backbones.py           Backbone dataclass, load_config (rope), load_backbones (both kinds), guards
    methods.py             Method dataclass, build_method_callable incl. verify="ddtree"
    metrics.py             acceptance, per-depth, NEW phase/TPS aggregation
    driver.py              run(cfg, on_checkpoint); sequential and fan-out modes
    cache.py               unit cache + config fingerprint
  modalkit/
    image.py               the pinned image, volumes, secrets — one definition
  charts/
    palette.py             the exp1/exp2 palette + phase ramp
    common.py              faceting, null-segment handling, derived titles

experiment1-harness/   modal_benchmark.py + reproduce.md + Results/
experiment2-block16/   modal_benchmark.py + reproduce.md + results/ + training/
experiment3-timings/   modal_benchmark.py + reproduce.md + results/
```

Each experiment keeps **only** its constants, its rollup, its charts and its results.
`DDTree/` disappears from all three.

## What is genuinely new code

1. **`ddtree/timing.py`** — a timer registry with a global on/off switch, so the
   `clean` pass runs with `cuda_time()` as a no-op (no `torch.cuda.synchronize()` at
   all) and the `instrumented` pass runs with full stage capture. Plus the canonical
   phase map and the native→canonical translation.
2. **The `commit` split** in all four generators → `walk_accept` / `kv_update` /
   `state_carry`. Boundaries: `dflash.py:97-108`, `dspark.py:164-181`,
   `ddtree.py:~446`, `sparked_tree.py:356-398`.
3. **The argmax split** out of `dflash.py`'s `draft` stage, so chain arms have a
   `candidate_build` segment.
4. **`verify="ddtree"`** dispatch calling `ddtree_generate` — exists in neither driver.
5. **`runner/metrics.py` phase aggregation** — parent/sub namespace separation (kills
   the `tree_build` double-count), round-0 exclusion into `cold_round`, `unaccounted`
   residual, per-output-token normalization, TPS.
6. **`"dflash_b16": 16`** added to the block-size guard, which is keyed by backbone
   name and therefore silently skips unknown backbones today.

Everything else is a move plus a merge of the two drivers.

## Migration steps

Each step ends green. Do not batch them.

1. **Move the core.** `git mv` exp2's `DDTree/*.py` + `model/` into `harness/ddtree/`.
   Verify byte-identity against exp1's copy first (`diff -r`) and abort if anything
   has drifted since this plan was written. Delete both `DDTree/` copies and both
   `__pycache__/` dirs (the latter are already gitignored per commit `c37d723`).
2. **Merge the drivers** into `runner/`. Base = exp2's (rope guard, budget sweep,
   sequential cache); graft exp1's dflash branches (`load_backbones`, the
   `build_method_callable` dflash chain case with its corrector-forbidden raise) and
   exp1's `run_one_dataset_raw` timing capture. Keep both cache strategies behind a
   `cfg["execution"]` switch.
3. **Fingerprint the cache.** Exp2's cache path is `b{budget}__{dataset}__n{n}.json`
   with no config hash — adding timing fields to the unit schema would silently resume
   from stale timing-free units. Adopt exp1's `fingerprint(cfg)` including a
   `CODE_VERSION`, as a directory level. Bump `CODE_VERSION` in the same commit.
4. **Equivalence gate — the safety net.** Re-run exp1 and exp2 through the shared
   harness and byte-diff the resulting `summary.json` against the committed ones.
   Acceptance at temperature 0 is deterministic, so this must match **exactly**, not
   approximately. Any diff is a refactor bug, not noise. Run from the existing volume
   caches so the gate costs minutes, not GPU-hours. Do not proceed past this step on a
   red diff.
5. **Instrument.** Items 1–3 and 5–6 from "new code" above. Re-run the gate: the
   commit split and the timing dict must not move a single acceptance number.
6. **Fix `verify_equiv`.** ~~Diagnose~~ **Diagnosed (2026-08-03): false alarm.** The
   check compares `acceptance_lengths` round-for-round, an invariant greedy
   speculative decoding does not provide — heap tie-break keys differ
   (`ddtree.py:131` ranks-tuple vs `sparked_tree.py:124` parent_index) and
   GPU-vs-CPU logsumexp noise (~1e-6) flips pops at the budget cutoff. Committed
   `output_ids` are provably identical at temperature 0. Remaining work: change
   exp1's check (`modal_benchmark.py:222-228`) to compare `output_ids`, re-run,
   commit a green `verify_equiv.json`.
7. **Build exp3** on the finished harness, per
   [`experiment3-timings/reproduce.md`](../experiment3-timings/reproduce.md).

## Risks

| risk | mitigation |
|---|---|
| Refactor silently changes exp1/exp2 numbers | step 4's exact-match gate, run twice (after the move, after instrumenting) |
| Stale caches resume with the wrong schema | step 3's fingerprint + `CODE_VERSION` bump |
| `harness/` import path differs local vs Modal | one `modalkit/image.py` that installs the package editable; no `sys.path.insert` shims |
| The two `aggregate.py` signatures are incompatible | exp2's is already dead from its driver's view (only the offline `results/build_summary.py:78` calls it) — port that one caller, keep exp1's shape |
| `git mv` loses per-experiment history | the core files are identical, so history lives in one place afterwards by design; note the pre-merge SHAs in the migration commit message |
