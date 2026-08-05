# Optimal SparklingTree config — DSpark-b16 / Qwen3-4B / H100, batch-1, temp-0

> **⚠️ SUPERSEDED (2026-08-05, harness-6-union).** Everything below was measured on
> the per-depth-top-C precompute builder, which leaked ~2-6% acceptance (RESULTS.md
> finding #4 in 2-precompute/). The builder now pools the deduped union: acceptance
> equals `fast` at any C (the "acceptance climbs with C" premise is gone), and
> precompute's build is O(L·U²) — *more* expensive than `fast` at large C. The
> builder/C/budget recommendation below is therefore VOID. The replacement
> measurement (fast vs precompute vs DDTree, one GPU, old-BLOG scale) runs in
> `2-precompute/`; its result picks the final builder + C for experiment 5.

**Bottom line (STALE — see banner):**

> **builder = `best-first-precompute`, C (`beam_candidates`) = 256, tree_budget = 128**

Runs ~300 TPS on gsm8k/humaneval, **~10–15× over the original naive sparked tree**,
with **byte-identical output** (greedy speculative decoding is exact at temp 0).

## Tunables

| tunable | optimal | why |
|---|---|---|
| **builder** (`tree_mode`) | **`best-first-precompute`** | weakly dominant: ties `fast` at low budget, wins 1.15× @128, 1.43× @256 |
| **C** = **K** (`beam_candidates`) | **256** | one shortlist knob (K *is* C). Acceptance-vs-C is flat 256–1024; O(C²) says take the low end |
| **tree_budget** | **128** (256 for hardest) | measured plateau peak; see below |
| temperature / markov head / confidence_threshold | 0 / on / 0 | not speed knobs — changing them changes output |

There is no second "K": the per-node branching width is derived from the budget,
not a free knob. The only real dial is **budget**.

## Budget — the one dial, and its plateau (measured, n=4 × 6 datasets)

Clean TPS of `precompute.c256` per dataset (◯ = peak):

| dataset | b16 | b32 | b64 | b128 | b256 | peak |
|---|---|---|---|---|---|---|
| humaneval | 260 | 282 | 287 | **314** | 306 | 128 |
| gsm8k | 264 | 283 | 299 | **307** | 293 | 128 |
| livecodebench | 219 | 231 | 241 | **247** | 240 | 128 |
| mt-bench | 116 | 119 | 125 | **133** | 132 | 128 |
| alpaca | 175 | 187 | 203 | 208 | **213** | 256 |
| aime24 | 138 | 153 | 160 | 198 | **212** | 256 |

**Every dataset peaks at 128 or 256; 64 is below peak everywhere.** Why: at batch-1
you are memory-bandwidth-bound (loading the 4B weights dominates), so adding tree
nodes is nearly free until the roofline knee (~128). Push budget *up* to the knee —
this holds even for chat (alpaca/mt-bench want high budget too, not low).

**Heuristic for a new dataset:** default to **128**; bump to **256** for very hard
math or chat that is still climbing. The plateau is broad (b64–b256 within ~10% of
peak), so you never need the exact argmax — 128 is safe.

## Why precompute (not fast)

Both apply the top-256 candidate restriction (the transfer-less trick, ~8× over
naive on its own). They differ only in *where* the markov arithmetic runs:

- `fast`: per-pop CPU matmul (`.expand`), serial, grows with budget → **peaks at b64
  then declines** (b64 194 → b128 193 → b256 157 aggregate).
- `precompute`: one upfront `[L,C,C]` GPU matmul, then pure table lookup — `.expand`
  is flat ~0.1 ms → **keeps climbing** with budget.

Since the optimum is at *high* budget (128), precompute wins there by 1.15×, widening
to 1.43× at 256. At temp-0 the lower acceptance it shows (6.65 vs 7.09) is not a
cost — output is identical, and the cheaper rounds already net higher TPS.

## Caveats

- **Batch > 1 shifts the roofline.** More concurrent sequences → compute-bound sooner
  → optimal budget *drops*. Same heuristic, tighter ceiling. Re-measure the plateau
  if you serve batched.
- Numbers are H100 + cpu=8, DSpark-b16 + its markov head. The *shape* (precompute
  dominant, budget plateau ~128) is portable; absolute TPS is hardware-bound.

## Provenance

- Builder/C: `2-precompute/csweep/` (b64 C-sweep, n=8).
- Budget plateau: `3-budget-dataset/` (5 budgets × 6 datasets, n=4, parallelized).
- Charts: `3-budget-dataset/results/budget_plateau.png`,
  `2-precompute/results/speedup_acceptance.png`.
- Builder code: `harness/ddtree/sparked_tree.py::build_sparked_tree_precompute`
  (`tree_mode="best-first-precompute"`), equivalence-gated in
  `2-precompute/test_precompute_builder.py`.
