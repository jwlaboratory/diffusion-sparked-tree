# Concurrency findings

H100 · sglang 0.5.16 · Qwen3-4B · sharegpt · 6 arms × 6 concurrency levels.
Raw numbers in [REPORT.md](REPORT.md), raw JSON in `results_H100.json`.

## 1. What the overlap scheduler is worth

Ratio of output throughput, overlap on ÷ overlap off:

| | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | mean |
|---|---|---|---|---|---|---|---|
| baseline | 1.174 | 1.179 | 1.182 | 1.192 | 1.198 | 1.214 | **1.190** |
| DFlash | 1.173 | 1.129 | 1.174 | 1.179 | 1.113 | 1.138 | **1.151** |
| DSpark | 1.246 | 1.245 | 1.264 | 1.366 | 1.298 | 1.338 | **1.293** |

**This is the number the tree arms need.** A plugin registered
`supports_overlap=False` runs only under `--disable-overlap-schedule`. Sparked-tree
is built on the DSpark drafter, so it inherits the DSpark row: **~29% forfeited on
average, up to 37% at c=8.**

For that to be recoverable, sparked-tree would have to beat DSpark by ~29% in
throughput at matched concurrency. At batch 1 it beats DSpark by 18.2% in speed
(RESULTS.md §1) — and that margin is measured where trees are cheapest, before
the ~5× FLOP-efficiency penalty of §9 has bitten. The gap does not close on its
own.

**So the device-resident builder is not optional.** That was the open question
going in; it is now answered with a measurement rather than an argument — and §5
reaches the same conclusion by an independent route.

The baseline row is why the control was worth running: overlap is worth 19% to
plain decoding, so most of DFlash's 15% is not about speculation at all. Only
DSpark clears the baseline meaningfully — its planner does more host-side work
per step (`compute_verify_token_budget`, the block-accept estimator), so it has
more to overlap. That the *most* host-active chain gains *most* from overlap is
the mechanism, stated plainly, and it is exactly why a host-side heap walk is the
wrong shape for this scheduler.

## 2. Speculative speedup decays with concurrency, as §9 predicted

Speedup vs baseline at the same overlap setting:

| | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| DFlash (ov) | 2.35× | 2.16× | 2.04× | 1.80× | 1.53× | 1.38× |
| DSpark (ov) | 2.54× | 2.41× | 2.26× | 2.20× | 1.80× | 1.80× |
| DFlash (noov) | 2.36× | 2.25× | 2.05× | 1.82× | 1.64× | 1.47× |
| DSpark (noov) | 2.40× | 2.28× | 2.11× | 1.92× | 1.66× | 1.63× |

DFlash loses 41% of its speedup from c=1 to c=32; DSpark 29%. Neither inverts by
c=32 — chains still beat baseline 1.4–1.8×.

These are **chains**, at 0.35 accepted-per-scored. Sparked-tree runs at 0.121
(§9), ~3× less efficient per scored token. The decay curve is set by how much
wasted compute the batch can absorb, so the tree's curve is steeper by
construction. Where it crosses is not inferable from these rows — **§5 measures
it: c≈4 at budget 64, c≈2 at budget 128.**

## 3. Acceptance is flat across concurrency

| | c=1 | c=8 | c=32 |
|---|---|---|---|
| DFlash | 3.38 | 3.31 | 3.30 |
| DSpark | 3.82 | 3.76 | 3.75 |

Under 3% drift across a 32× batch range (2.6% DFlash, 2.5% DSpark, max/min over
all six levels). Acceptance is a property of the drafter
and the tree, not of the scheduler — so the batch-1 acceptance results transfer
to serving unchanged, and every concurrency effect is on the **cost** side.

That is what makes the §9 accounting the right model, and it means the tree's
acceptance advantage (+40.3% over DSpark) is real at concurrency too. The
question was never whether the tree accepts more; it is whether accepting more is
worth what it costs to verify. These numbers confirm the numerator holds, and
they are what license §5 to carry batch-1 acceptance into a serving prediction:
if acceptance moved with batch size, that decomposition would be invalid.

Absolute acceptance is lower here than in RESULTS.md (3.8 vs 6.1 for DSpark)
because this is sharegpt chat against `dspark_qwen3_4b_block7`, not our block-16
checkpoint on task datasets. Cross-harness absolute values are not comparable;
the within-harness ratios are.

## 4. Status of the tree arms

Not yet measured, but no longer blocked on unknowns. The plugin exists and runs.

- **Verify path: confirmed reusable.** `reconstruct_indices_from_tree_mask`
  derives `retrieve_index` / `retrieve_next_token` / `retrieve_next_sibling` from
  the tree mask alone, so the first-child/next-sibling conversion flagged as the
  main risk is not ours to write.
- **Mask convention: confirmed identical.** 494 real trees — chains, stars,
  balanced, irregular best-first, batched, under-budget — round-tripped through
  SGLang's own kernel on GPU. Zero mismatches on parents *and* positions.
  Negative control confirms the test detects transposition and ancestor loss.
- **Plugin: written and validated end to end.** `SPARKED` registers, the server
  boots, TARGET_VERIFY CUDA-graph capture works, and generation is **byte-identical
  to greedy no-speculation decoding** across three prompts — the check that
  matters, since a subtly wrong mask degrades quality without crashing.
  Acceptance 1.375 (1.00–1.85 per prompt) with a deliberately weak stand-in
  proposer, which proves the tree path is live rather than degenerating to
  single-token decoding.
- **Both builders drive SGLang's own markov head, unmodified.** SGLang's head is
  `srt/models/dspark.py::VanillaMarkov`, declaring
  `markov_w1 = nn.Embedding(vocab, rank)` / `markov_w2 = nn.Linear(rank, vocab)`
  — structurally identical to ours (`ddtree/model/dspark.py:68`), so the
  builder's `.markov_w1.weight` access needs no adapter. And
  `compute_base_logits()` returns `[bs, gamma, vocab]`, exactly the slice
  `build_markov_tree_precomputed` takes. 36 trees from the **real** builders over
  real logits — `build_markov_tree_precomputed` (sparked) and `build_ddtree_tree`
  (ddtree), budgets 16/32/64 — round-tripped through the bridge and SGLang's
  kernel. Zero failures. Emitted shapes are irregular, max fanout 9–41, which is
  the `tree_topk = -1` case.

- **Remaining: worker wiring only.** The algorithm side is done; what is left is
  reaching those logits from inside a worker. DSpark builds a linear chain at
  `dspark_worker_v2.py:581` (`verify_ids_2d = cat([draft_block_ids[:, :1],
  draft_tokens])`), and replacing it means subclassing `DSparkWorkerV2` — keeping
  its drafter and `dspark_kv_inject` — and overriding inside `_forward_decode`,
  next to verify-window allocation and the planner. Not attempted blind.
  The validated plugin run therefore still uses `LookupTreeSource`, a fixture,
  so it exercises the machinery and the bridge, not the sparked tree.

One upstream gap worth knowing about: `CustomSpecAlgo` is missing
`create_future_map`, `need_topk` and `carries_draft_hidden_states`, all of which
the scheduler calls unconditionally. The conformance guard
(`_assert_custom_spec_algo_conforms`) only checks `is_*` / `supports_*` names, so
it does not catch them. `init_overlap` runs at scheduler.py:525 regardless of the
overlap flag — its comment says "FutureMap is always-on" — so **any** plugin dies
at startup without supplying them. `sparked_plugin/algo.py` copies the enum
implementations.

## 5. The crossover: trees lose above c≈4

Full tables in [PREDICTION.md](PREDICTION.md). Measured round time by verify
width (median of 3 repeats, fixed 96-prompt set, radix cache off, all arms capped
identically), then combined with batch-1 acceptance.

**Measured — cost of width, relative to the width-17 chain:**

| width | c=1 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|
| 65 | 0.87× | 1.23× | 1.69× | 2.25× | 2.76× |
| 129 | 1.13× | 1.77× | 2.77× | 3.78× | 4.78× |

At batch 1 a width-65 tree is nearly free — 0.87×, *cheaper* than the chain.
By c=32 it costs 2.76×. That is §9's argument, measured.

**Predicted — sparked-tree vs the DSpark chain:**

| | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|---|
| sparked tb64 | **1.25×** | **1.06×** | 0.99× | 0.84× | 0.63× | 0.55× |
| sparked tb128 | **1.04×** | 0.98× | 0.73× | 0.52× | 0.38× | 0.31× |

**tb64 crosses below 1.0 at c≈4; tb128 at c≈2.** Above that the chain wins, even
granting the tree its full acceptance advantage and charging it nothing for
building the tree.

The acceptance ratios here are **not** the batch-1 splice. They were re-measured
on the same block-7 drafter and chat workload the cost sweep ran on
(`validate_acceptance.py`, via `ddtree/benchmark.py` unchanged):

| | alpaca | mt-bench | mean | batch-1 splice |
|---|---|---|---|---|
| tb64 | 1.386 | 1.334 | **1.360** | 1.291 |
| tb128 | 1.407 | 1.396 | **1.402** | 1.389 |

The measured ratios are **higher** — the original prediction understated the
tree, i.e. it erred toward the chain rather than toward our own hypothesis.
Correcting it moves tb64's c=4 point from 0.94 to **0.99**. The crossover rung
does not move, but at budget 64 it is now *marginal* rather than clear: c=4 is
effectively a tie, and the honest read is that trees stop paying somewhere
between c=4 and c=8. tb128 is unaffected and unambiguous.

Absolute acceptance is far below RESULTS.md (5.4 vs 7.8) exactly as expected for
block-7 on chat rather than block-16 on task data — only the ratio feeds the
prediction.

The additive model was checked, not assumed: fitting per-token cost on widths
17→65 and extrapolating to 129 gives ≤17% error, and **≈0% at c=16 and c=32** —
the model is most accurate exactly where the conclusion is strongest. The two
larger errors (−12% at c=1, +17% at c=2) sit where absolute times are smallest
and where the prediction is closest to 1.0 anyway, so they move the crossover by
at most one rung.

### An unplanned confirmation of the builder problem

`tree_w17` and `dspark_capped` verify the **same width (17)** and differ only in
proposer: DSpark runs a real drafter on the GPU, our plugin runs a nearly-free
lookup — but through a **host-side Python loop**.

| | c=1 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|
| tree_w17 ÷ dspark | 0.78× | 0.86× | 1.07× | 1.11× | 1.27× |

At batch 1 the cheap proposer wins by 22% — no drafter forward to pay for. By
c=32 it *loses* by 27%, a **1.62× relative degradation**, despite doing strictly
less arithmetic. Nothing about the tree explains that; the only thing that grew
is per-request host work that does not amortise across a batch.

That is the host-resident builder problem showing up in an experiment not
designed to look for it, at a proposer far cheaper than
`build_markov_tree_precomputed` (~3.8 ms/round, RESULTS.md §12). It is
independent evidence for the same conclusion §1 reached from the overlap delta.

## Caveats

- One run per cell, no repeats. The batch-1 harness puts run-to-run speed
  variance at ~5% mean / 16% worst-case on a single cell (`final_benchmark/report.py`)
  and ~±12% across independently-scheduled H100 containers (RESULTS.md §12).
  This harness differs — one warm server per arm, all six rungs inside it, so
  container scheduling cannot skew rungs against each other — but it has not been
  characterised. Taking the batch-1 figures as the best available prior, DSpark's
  overlap delta (25–37%) clears them and DFlash's (11–18%) does not. **The DFlash
  row should not be quoted to two digits without repeats.**
- sharegpt only. The batch-1 work used six task datasets; workload mix moves
  acceptance and therefore speedup.
- DSpark arm uses published `deepseek-ai/dspark_qwen3_4b_block7`, not our
  block-16 checkpoint — a like-for-like cross-harness comparison needs that
  checkpoint loaded into SGLang's DSPARK worker, which is unverified.
