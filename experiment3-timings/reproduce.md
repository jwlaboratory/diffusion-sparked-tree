# Experiment 3 — where does the time actually go?

Experiments 1 and 2 measured **acceptance length**: how many tokens the target
commits per verifier call. Acceptance is a quality metric — at temperature 0 it is
deterministic and hardware-independent, which is why those experiments reproduce
exactly on any GPU.

Acceptance is not speed. A method can accept more tokens per round and still be
slower per token, if the extra tokens cost more to produce than they save. This
experiment measures the other half: **wall-clock, decomposed into phases**, for four
decoding methods under one identical harness.

Two numbers come out of it:

1. **Net decode throughput** — output tokens per second, measured with the stage
   timers *off* (see [Two passes](#two-passes-why-there-are-two-throughput-numbers)).
2. **A standardized per-phase breakdown** — the same phase vocabulary for every arm,
   so "time spent building the tree" and "time spent walking the tree" are readable
   side by side against arms that have no tree at all.

**This experiment's numbers are hardware-bound.** Unlike exp1/exp2, the results depend
on the GPU, the container's CPU, and the driver version. Reproduce on the pinned
`H100` + `cpu=8` configuration or the absolute seconds will not match — the
*shape* (phase shares, relative ordering) is the portable part.

---

## The arms

Four methods. Every one is a real decoding path someone would actually run, not an
ablation cell.

| # | arm | backbone | drafts | candidate structure | generator |
|---|---|---|---|---|---|
| 1 | `dflash.chain` | DFlash-b16 | 15 | linear chain, parallel argmax | `dflash_generate` |
| 2 | `dspark_b7.chain` | DSpark-b7 | 7 | linear chain, serial markov sweep | `dspark_generate` |
| 3 | `ddtree` | DFlash-b16 | 15 | tree, depth-independent top-k | `ddtree_generate` |
| 4 | `dspark_b16.markov.tree` | DSpark-b16 | 16 | tree, per-node markov rerun | `sparked_tree_generate` |

(Note when reading arm 4 vs arm 2: they differ in *two* ways at once — block size
7→16 **and** chain→tree — so their delta is a bundle, not a single-axis effect. If a
future cut needs those separated, a `dspark_b16.chain` arm is one `METHODS` line and
zero new code.)

### Exact settings

| arm | model_id | kind | eff. `block_size` | tree depth limit | markov head | corrector slot | verify | budget |
|---|---|---|---|---|---|---|---|---|
| `dflash.chain` | `z-lab/Qwen3-4B-DFlash-b16` | dflash | 16 | n/a | **none** (checkpoint has no head) | forbidden — raises | `chain` | n/a |
| `dspark_b7.chain` | `deepseek-ai/dspark_qwen3_4b_block7` | dspark | 7 | n/a | **ON, intrinsic** | ignored | `chain` | n/a |
| `ddtree` | `z-lab/Qwen3-4B-DFlash-b16` | dflash | 16 | **15** (`bs-1`) | **none** | n/a | `ddtree` | {64, 256} |
| `dspark_b16.markov.tree` | `shreybirmiwal/Qwen3-4B-DSpark-b16` | dspark | 16 | **16** (`bs`) | **ON, native** | `dspark_b16_markov` | `tree` | {64, 256} |

Shared by all: target `Qwen/Qwen3-4B` (sdpa, bf16); drafts `flash_attention_2`, bf16;
`temperature=0.0`; `max_new_tokens=512`; `seed=0`; `confidence_threshold=0.0`.

**Three things about markov that are easy to get wrong:**

- A `.chain` arm on a DSpark backbone is **markov-ON with no off switch**.
  `dspark_generate` → `model.sample_draft_tokens` → `self.markov_head`, applied
  unconditionally. `corrector=None` on arm 2 does *not* mean "no markov"; the
  field is simply not read on the chain path. This is why the arm list calls it
  "dspark normal" — markov is what DSpark *is*.
- Arm 4 uses the **naive per-node markov tree** exactly as exp1/exp2 ran it — no
  optimization pass, no rewrite. That is deliberate: exp3 is measuring the thing we
  actually built and reported acceptance numbers for. See
  [the tree-builder caveat](#the-caveat-that-decides-how-to-read-this).
- Arm 3 is the **official DDTree reference implementation** (`ddtree_generate`), not
  `sparked_tree_generate(draft_mode="dflash", markov_head=None)`. Those two are
  supposed to be equivalent, and they produce the same accepted tokens — but they are
  *not* equivalent in cost, and running DDTree through the sparked-tree builder would
  charge it ~20 s/dataset of overhead it does not have. See
  [equivalence](#known-issue-the-ddtree-equivalence-check-currently-fails).

### `confidence_threshold = 0.0` — and why that is a real choice here

DSpark's confidence head can truncate a draft block early. At threshold 0 it never
fires, so arms 2 and 4 always propose their full block. Exp1/exp2 used 0.0 and this
experiment keeps it, so `draft_confidence` shows up as a small constant cost with no
behavioral effect. That is the honest baseline: turning it on would change acceptance
too, and this experiment holds acceptance-affecting knobs fixed at the exp1/exp2
values so the timings are attributable to the *arms*, not to a new setting.

---

## The phase taxonomy (design this first — everything else follows)

The three generator families report **three different stage vocabularies** today:

```
dflash.py:11        ("draft", "verify", "commit")
dspark.py:33-39     ("draft_backbone", "draft_markov", "draft_confidence", "verify", "commit")
sparked_tree.py:48  ("draft", "tree_build", "tree_compile", "verify", "commit")
ddtree.py:15-16       + ("tree_build_copy", "tree_build_heap", "tree_build_visibility")
```

They are not comparable. `draft` means different things; `commit` bundles a tree walk,
a KV rollback and a hidden-state carry into one number on one arm and two of those
three on another. So exp3 defines **one canonical phase set** that every arm reports,
and the harness maps each generator's native stages into it.

### Canonical phases

Prefill is measured separately (it is the TTFT, not part of decode throughput). These
eight cover the decode loop and, by construction, **sum to the decode wall clock**:

| # | phase | what it is | the question it answers |
|---|---|---|---|
| 1 | `draft_forward` | draft backbone forward (embed → layers → lm_head) + draft-KV crop | how expensive is the drafter itself |
| 2 | `candidate_build` | turning draft logits into the candidate set | **time spent creating the tree** |
| 3 | `candidate_pack` | building what the target consumes: tree attention mask + position ids, H2D | the price of tree-shaped input |
| 4 | `verify` | the target forward | the cost the speculation is trying to amortize |
| 5 | `walk_accept` | deciding the accepted prefix and gathering it | **time spent walking the tree** |
| 6 | `kv_update` | KV rollback — compaction (tree) or crop (chain) | the price of speculating and being wrong |
| 7 | `state_carry` | hidden-state extraction + buffer writes for the next round | loop bookkeeping |
| 8 | `unaccounted` | `decode_total − Σ(1..7)` | keeps the shares honest (see below) |

Sub-phases are reported in a **separate namespace** and never summed with the above:

| parent | sub-phase | native key |
|---|---|---|
| `candidate_build` | `.prep` | `tree_build_copy` — D2H of base logits (+ markov matrices) |
| `candidate_build` | `.expand` | `tree_build_heap` — the heap loop **and every per-node top-k** |
| `candidate_build` | `.visibility` | `tree_build_visibility` — O(budget²) ancestor mask |
| `candidate_build` | `.markov` | `draft_markov` — DSpark chain's serial block sweep |
| `candidate_build` | `.confidence` | `draft_confidence` — DSpark chain's truncation head |

### Per-arm mapping

| canonical phase | `dflash.chain` | `dspark_b7.chain` | `ddtree` | `dspark_b16.markov.tree` |
|---|---|---|---|---|
| `draft_forward` | `draft` less the argmax | `draft_backbone` | `draft` | `draft` |
| `candidate_build` | parallel argmax | `draft_markov` + `draft_confidence` | `tree_build` | `tree_build` |
| `candidate_pack` | **null** (n/a) | **null** (n/a) | `tree_compile` | `tree_compile` |
| `verify` | `verify` | `verify` | `verify` | `verify` |
| `walk_accept` | `commit` → cumprod accept | `commit` → accept | `commit` → `follow_verified_tree` + gather | same |
| `kv_update` | `commit` → `crop` | `commit` → `crop` | `commit` → `compact_dynamic_cache` | same |
| `state_carry` | `commit` → hidden extract | `commit` → hidden extract | `commit` → hidden extract + gather | same |

A phase a given arm structurally does not have is reported as **`null`, not `0`** —
`0` reads as "measured, took no time", which is a different claim. Charts render null
as an absent segment with the arm labeled n/a in the legend.

### Three instrumentation changes this requires

1. **Split `commit`.** Today it is one timer around sample → accept → KV → hidden. It
   must become three (`walk_accept`, `kv_update`, `state_carry`) in all four
   generators. "Time per walking tree" is the headline the current instrumentation
   cannot produce. Boundaries are unambiguous — `dflash.py:97-108`,
   `dspark.py:164-181`, `ddtree.py:~446`, `sparked_tree.py:356-398`.
2. **Split the argmax out of dflash's `draft`.** Otherwise arm 1 reports `null` for
   `candidate_build` and its bar has no candidate-generation segment at all, which
   makes the tree arms' `candidate_build` look like a cost with no counterpart.
3. **Add `unaccounted`.** Not cosmetic — it closes three real accounting holes:
   - **Round 0 is counted inconsistently.** Every generator skips round 0's `draft`
     (draft-KV prefill happens there and `decode_start` resets after) but *does*
     accumulate round 0's `tree_build` / `verify` / `commit`. So today
     `Σ stage_times ≠ total`. Exp3 excludes **round 0 entirely** from all phases and
     reports it in its own `cold_round` field.
   - **`tree_build` double-counts its sub-times.** `sparked_tree.py:322-324` records
     the total *and* the three sub-keys. Summing all eight native keys inflates the
     total ~2×. The parent/sub namespace split prevents this structurally.
   - Untimed residue inside the loop (`draft_input_ids` construction, stop checks,
     trace saving) now shows up rather than silently vanishing.

### The corrector-fit probe must be OFF

`sparked_tree.py:369-391` runs a per-depth Python loop with two full-vocab
`log_softmax`+`argmax` per accepted token — **inside the commit window**. Exp1 enables
it on the markov-*off* tree arms, which in a timing experiment would make exactly the
cheap arms look slow. Exp3 sets `measure_corrector_fit: False` and asserts
`probe_markov_head is None` at dispatch.

---

## Two passes (why there are two throughput numbers)

`cuda_time()` is `torch.cuda.synchronize()` + `perf_counter()` (`dflash.py:140-142`).
It fires 8–12× per round. Every call is a full-device barrier that destroys CPU/GPU
overlap, and **tree arms pay more barriers per round than chain arms** — so the
instrumented run is not just slower, it is *unevenly* slower in a way that flatters
the chain arms.

So every `(arm, budget, dataset)` unit runs **twice**:

| pass | timers | what it yields |
|---|---|---|
| `clean` | off (`cuda_time` → no-op; only TTFT + decode total) | **the headline `tps_decode`** |
| `instrumented` | on | the per-phase breakdown, and `tps_decode_instrumented` |

The two passes are **interleaved per sample** — each prompt runs clean and
instrumented back-to-back, alternating which goes first on even/odd samples. This is
load-bearing, learned the hard way: running the passes sequentially produced a
consistently *negative* tax of −11 % to −36 %, because the second pass saw a warmer
machine (sustained boost clocks, warm allocator, datasets already in memory) — it
measured machine state, not timer overhead. Back-to-back pairs cancel the drift; the
even/odd alternation cancels the residual within-pair order effect.

Reported together with `instrumentation_tax = 1 − tps_instrumented / tps_clean` per
arm. If the tax differs a lot across arms, the phase-share chart is still valid (it is
internally normalized) but cross-arm *absolute* phase seconds must be read against the
clean total, not the instrumented one. The charts do this rescaling and say so.

Acceptance length is identical in both passes (temperature 0, timers do not touch the
math); the harness asserts this as a cheap correctness check on the whole two-pass
setup.

---

## The caveat that decides how to read this

`build_sparked_tree` (`sparked_tree.py:92-103`) does, **per created node**, a CPU
`log_softmax` + `topk` over the full 151,936-token vocabulary. Its own docstring says
it favors "correctness and clarity over speed". `build_ddtree_tree`
(`ddtree.py:107-114`) instead does **one** GPU top-k for all depths and ships only the
top-k table to the CPU.

From exp1's measured `Results/results_detailed.json` (gsm8k, n=8, budget 64, seconds):

```
dflash.chain        draft 4.3   verify 23.5   commit 0.3                    tpot 0.0104
dflash.tree         draft 3.3   tree_build  19.4   verify 17.4  commit 2.0  tpot 0.0157
dflash.markov.tree  draft 3.7   tree_build 437.6   verify 19.5  commit 0.9  tpot 0.1593
dspark.chain        backbone 4.4  markov 0.45  conf 0.006  verify 23.4      tpot 0.0117
dspark.markov.tree  draft 3.9   tree_build 341.6   verify 20.2  commit 0.9  tpot 0.1504
```

Expect arm 4 to land **~10–15× slower per token than the chain arms, with ~95 % of it
in `candidate_build.expand`** — and expect that to get roughly 4× worse again at
`tree_budget=256`, since `topk = min(budget, vocab)` and the node count scales with the
budget.

**That number is a property of our research implementation, not of the algorithm.**
The experiment is worth running precisely because it localizes the cost to one
function: if `candidate_build.expand` is 95 % of the runtime, then "is the sparked tree
viable?" is a question about a CPU top-k loop, not about tree speculation. Every chart
title and the summary rollup state this explicitly, so no reader walks away thinking
the algorithm is inherently 10× slow.

Arm 3 is the honest counterexample in the same chart: a tree method with a competent
builder, ~1.5× the chain `tpot` rather than ~15×.

---

## Why the budget is swept, and what that does to the chart

`TREE_BUDGETS = [64, 256]`. `candidate_build` is roughly linear in budget (more nodes,
more per-node top-k) while `verify` grows sublinearly (one wider forward) — so the
sweep shows whether tree cost is dominated by the drafting side or the verification
side, which is the whole design question.

**Arms 1 and 2 have no budget knob.** Their bars are *identical* across the two
facets. That is not a bug and not padding: they are the constant reference line
against which the tree arms' budget scaling is read. Chart panels label them
`(budget-invariant)` so nobody reads the repetition as two measurements.

---

## Where the code lives — the shared harness

Exp1 and exp2 each carry a full copy of `DDTree/`. Those copies are **byte-identical**
for `ddtree.py`, `dflash.py`, `dspark.py`, `sparked_tree.py` and all of `model/`; only
`run_experiment.py` and `aggregate.py` diverge. Exp3 needs pieces of both (dflash from
exp1, the b16 checkpoint + rope guard + unit cache from exp2), and instrumenting the
commit split in three copies of the same file is how the copies stop being identical.

So the decode core and the driver move **out** of the experiment directories into a
single shared package, and each experiment keeps only its configuration, its results
and its `reproduce.md`. Layout, migration steps and the equivalence gate that protects
exp1/exp2's published numbers: **[`harness/PLAN.md`](../harness/PLAN.md)**.

Once migrated, this directory contains:

```
experiment3-timings/
  modal_benchmark.py     constants + the timing rollup print
  reproduce.md           this file
  results/
    summary.json         downloaded from the volume
    make_charts.py       the four charts below
    cache/               per-(pass, budget, dataset) resume units
```

## Tunables (constants at the top of `modal_benchmark.py`)

| knob | default | notes |
|---|---|---|
| `TARGET` | `Qwen/Qwen3-4B` | verifier, sdpa bf16 |
| `BACKBONES` | dflash_b16, dspark_b7, dspark_b16 | all three co-resident (~17 GB bf16 + KV), so no per-arm reload pollutes the timings |
| `METHODS` | the four arms above | |
| `TASKS` | gsm8k:4, humaneval:4, mt-bench:4 | smoke-sized; timing needs fewer samples than acceptance does, but see [statistics](#statistics) |
| `TREE_BUDGETS` | `[64, 256]` | tree arms only |
| `PASSES` | `["clean", "instrumented"]` | both required for the two headline numbers |
| `TEMPERATURE` / `MAX_NEW_TOKENS` / `SEED` | 0.0 / 512 / 0 | matched to exp1+exp2 |
| `CONFIDENCE_THRESHOLD` | 0.0 | dspark chain arms; inert at 0 |
| `MEASURE_CORRECTOR_FIT` | **False** | load-bearing — the probe sits inside the commit window |
| `WARMUP_TOKENS` | **256** | not 32: a 32-token warmup never reaches deep tree positions, so the first measured sample eats the deep-path kernel compile |
| `DISCARD_FIRST_SAMPLE` | True | belt-and-braces on top of warmup |
| `CPU` | **8** | pinned. A large share of tree cost is single-threaded CPU; an unpinned container makes runs incomparable |
| `GPU` / `TIMEOUT_SECONDS` | **H100** / 6 h | units checkpoint, so a long timeout is cheap |
| `CACHE_DIR` | `/results/timings/cache` | namespaced away from exp1 (`/results/`) and exp2 (`/results/block16/`) |

## How to run

```bash
pip install modal
modal setup                        # one-time auth
cd experiment3-timings
modal run modal_benchmark.py       # --detach to survive a dropped connection
```

```bash
cd experiment3-timings/results
python make_charts.py [path/to/summary.json]
```

## Charts

| file | what it shows |
|---|---|
| `tps.png` | the headline: net decode tokens/sec per arm, clean pass, one panel per budget. Instrumented TPS overlaid as a hollow marker so the instrumentation tax is visible, not hidden |
| `phase_breakdown.png` | stacked horizontal bars, **seconds per output token** (not per round — round counts differ 38/30/41/51 across arms for the same token count, so per-round is not comparable). One bar per arm, faceted by budget, segments = the 8 canonical phases |
| `phase_share.png` | the same data as 100 %-normalized shares — this is the chart that survives a hardware change, and the one that shows `candidate_build.expand` swallowing arm 4 |
| `tree_cost_scaling.png` | `candidate_build` vs `verify` per output token at budget 64 → 256, tree arms only. Separates "the tree got expensive to build" from "the tree got expensive to verify" |

Palette follows exp1/exp2: cool hues for DFlash-family arms, warm for DSpark-family;
chain arms hatched, tree arms solid, so arm identity never rests on color alone. Phase
segments use a single sequential ramp ordered by the phase table, so the same phase is
the same shade in every chart.

## `summary.json` shape

Keyed **pass → budget → dataset → arm**. Budget-invariant arms are written under every
budget key with an `budget_invariant: true` flag rather than being special-cased.

```json
{
  "config": { "target": "...", "tree_budgets": [64, 256], "passes": ["clean", "instrumented"],
              "cpu": 8, "gpu": "H100", "measure_corrector_fit": false,
              "backbones": { "dflash_b16": {"model_id": "...", "kind": "dflash", "block_size": 16}, "...": "..." },
              "methods":   { "ddtree": {"backbone": "dflash_b16", "corrector": null, "verify": "ddtree"}, "...": "..." },
              "phase_order": ["draft_forward", "candidate_build", "candidate_pack", "verify",
                              "walk_accept", "kv_update", "state_carry", "unaccounted"] },

  "results": { "clean": { "64": { "gsm8k": { "ddtree": {
                   "tps_decode": 96.1, "ttft": 0.041, "output_tokens": 2048, "rounds": 164,
                   "mean_accept": 3.12, "budget_invariant": false } } } },
               "instrumented": { "64": { "gsm8k": { "ddtree": {
                   "tps_decode": 71.3, "output_tokens": 2048, "rounds": 164,
                   "phases":    { "draft_forward": {"sec": 3.3, "sec_per_token": 0.0016, "share": 0.11},
                                  "candidate_pack": null, "...": "..." },
                   "subphases": { "candidate_build.prep": {"sec": 0.79}, "candidate_build.expand": {"sec": 18.5}, "...": "..." },
                   "cold_round": {"sec": 0.21} } } } } },

  "timing": { "64": { "ddtree": { "tps_clean": 96.1, "tps_instrumented": 71.3,
                                  "instrumentation_tax": 0.258,
                                  "dominant_phase": "verify", "dominant_share": 0.44,
                                  "per_dataset": {"gsm8k": {"tps_clean": 96.1}, "...": "..."} } } }
}
```

`timing` is exp3's rollup — the analogue of exp1's `transfer` and exp2's `block_size`.
`dominant_phase` / `dominant_share` are derived from the measurement, never hardcoded,
so a chart title cannot assert a conclusion the data does not support.

## Statistics

n=4 per dataset × 3 datasets is small. For **acceptance** that meant only large effects
resolve; for **timing** it is different in both directions:

- Timing has far more samples than it looks — each generation is 30–50 rounds, and each
  round is an independent phase measurement. Phase *shares* stabilize quickly.
- But timing has variance acceptance does not: thermal drift, host CPU contention, and
  cache growth over a generation. `round_timestamps` (already returned by every
  generator) is retained per unit so drift is auditable after the fact, and the rollup
  reports the median across rounds alongside the mean.

Bump `TASKS` to `n=8` before quoting a cross-arm speedup ratio as a result rather than
as a direction.

## Known issue, resolved: the DDTree equivalence check "failure" is a false alarm

`experiment1-harness/Results/verify_equiv.json` reports
`{"max_abs_diff": 0, "length_mismatch": true, "pass": false}`. Diagnosed by code
inspection (2026-08-03): **the check asserts an invariant greedy speculative decoding
does not provide** — round-for-round identical trees. It compares only
`acceptance_lengths` (`experiment1-harness/modal_benchmark.py:222-228`), never
`output_ids`.

The two builders differ in two places that reorder best-first heap pops right at the
`tree_budget` cutoff: (1) the heap tie-break key — DDTree breaks ties by the `ranks`
path tuple (`ddtree.py:131`), sparked by `parent_index` (`sparked_tree.py:124`); and
(2) log-prob normalization on different devices — DDTree logsumexps on GPU
(`ddtree.py:108-114`), sparked log_softmaxes on CPU (`sparked_tree.py:97-103`) —
whose ~1e-6 float noise doesn't cancel across depths and can flip a pop near the
budget boundary. Slightly different node set → different per-round acceptance
partition → different round count.

**The committed token sequence is provably identical** under temperature 0 (the
accepted path is always the longest prefix of the target's greedy continuation present
in the tree, and the bonus token is the target's argmax), so this affects round
accounting only. For exp3's timings that means: arm 3 vs arm 4 round counts are not
comparable round-for-round anyway — which is exactly why every timing number here is
normalized per output token, never per round. The proper fix, when exp1's check is
next touched, is to compare `output_ids` instead of `acceptance_lengths`.

## Known gotcha: RoPE base across transformers majors

Both DSpark configs declare RoPE the transformers-v5 way — nested
`rope_parameters: {rope_theta: 1000000}`, **no top-level `rope_theta`**. Pinned
transformers 4.57.1 does not know that field and silently falls back to
`rope_theta=10000.0`. Nothing raises; the drafts just get quietly worse.

**DFlash has a top-level `rope_theta: 1000000` and needs no fix** — which is exactly
what makes a mixed dflash+dspark experiment dangerous: it would run the DFlash arms
correctly and the DSpark arms degraded, and the resulting "DFlash is faster" would look
entirely plausible. The `load_config` normalization plus the `inv_freq`-derived
assertion (both carried over from exp2) are mandatory here, and the block-size guard
gains a `"dflash_b16": 16` entry — exp2's guard is keyed by backbone *name*, so a new
backbone silently gets no guard at all.

## Pinned environment

| component | version / source |
|---|---|
| base image | `nvidia/cuda:12.4.1-devel-ubuntu22.04`, Python 3.11 |
| torch | `2.5.1` (cu124) |
| flash-attn | `2.7.4.post1` prebuilt wheel (cu12 / torch2.5 / cxx11abiFALSE / cp311) |
| transformers | `4.57.1` |
| datasets | `3.6.0` |
| GPU / CPU | H100 / 8 vCPU — **both pinned**, both affect the numbers |

`flash_attn` is mandatory (draft models use `flash_attention_2`). `maybe_enable_cpp_compact(True)`
is called once before the measurement loop so the inline C++ KV-compaction extension is
compiled outside the timed region; the harness logs whether the C++ or the Python
fallback path is live, because that choice changes `kv_update`. The `huggingface` Modal
secret is attached; all models and datasets are public. HF downloads cache in the
`ddtree-hf-cache` volume.
