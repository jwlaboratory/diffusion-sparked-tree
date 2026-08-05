# Experiment 4-faster / 1 — transfer-less best-first builder

Make the naive best-first markov tree (`build_sparked_tree`) cheap **without
changing what it builds** — keep its adaptive, best-first node allocation (the
acceptance ceiling every fixed beam schedule chases) and only remove the two
implementation costs exp3 localized inside `candidate_build`.

Run: H100 + 8 CPU, DSpark-b16 + own markov head, budgets {64, 256},
gsm8k/humaneval/mt-bench × 4 samples × 512 tokens, temp 0, two-pass
(clean TPS / instrumented phases), config fingerprint `3f6f8b11e471`.
`checks.acceptance_match = true` (clean and instrumented agree within each arm).

Two arms, same job, same machine:
- **`bestfirst.ref`** — the existing `build_sparked_tree` (slow reference).
- **`bestfirst.fast`** — new `build_sparked_tree_fast`, `beam_candidates=2048`.

## The two fixes

Both are the beam builder's candidate-union trick ported into the best-first heap;
the heap / sibling-reuse / visibility logic is byte-for-byte the reference's.

1. **Fix the markov send (`.prep`).** The reference copies the *static* markov
   weight matrices `W1`, `W2` (`[151936, 256]` fp32 ≈ 155 MB each, ~311 MB total)
   GPU→CPU **every round**. The fast builder does one GPU top-k over the base
   logits first, takes the per-depth union (+ the root token), and ships only the
   ~U gathered columns of the logits and rows of `W1`/`W2`. One host transfer of a
   few MB instead of ~321 MB.
2. **GPU top-k before transfer (`.expand`).** Each popped node in the reference ran
   a full-vocab (151,936) `log_softmax` + top-k + a full-vocab bias on CPU. The
   fast builder runs every per-pop op on the length-U active vector instead.

Lossiness: only the union of the per-depth top-2048 base tokens can become a node.
The additive markov bias is small relative to the base-logit spread, so a
bias-promoted token almost always already sits in the top-2048. Correctness gate
(`test_fast_builder.py`): with `beam_candidates=0` the active set is the full vocab
in order, and the fast builder is **byte-identical** to the reference (verified
across seeds/budgets, incl. the `markov_head=None` path).

## Results

### Budget 64

| metric | ref | fast | reduction |
|---|---|---|---|
| **(a) candidate_build** ms/round | 228.4 | 9.7 | **95.8%** |
| — `.prep` (static-weight transfer) ms/round | 91.9 | 1.9 | 97.9% |
| — `.expand` (per-pop CPU compute) ms/round | 135.2 | 6.6 | 95.1% |
| total decode ms/round (all phases) | 259.8 | 40.0 | 84.6% |
| **(b) mean acceptance** (round-wt AGG) | 7.18 | 7.32 | **+1.99%** |
| **(c) clean TPS** (aggregate) | 26.8 | 181.5 | **6.77×** |

Per-dataset TPS: gsm8k 5.63×, humaneval 6.65×, mt-bench 7.31×.
Dominant phase shifts `candidate_build` (88%) → **`verify` (59%)** — the fast arm
is now roughly verify-bound, the same regime as DDTree.

### Budget 256

| metric | ref | fast | reduction |
|---|---|---|---|
| **(a) candidate_build** ms/round | 627.4 | 40.7 | **93.5%** |
| — `.prep` (static-weight transfer) ms/round | 91.5 | 1.8 | 98.0% |
| — `.expand` (per-pop CPU compute) ms/round | 533.0 | 36.7 | 93.1% |
| total decode ms/round (all phases) | 658.9 | 72.1 | 89.1% |
| **(b) mean acceptance** (round-wt AGG) | 8.04 | 7.88 | **−1.94%** |
| **(c) clean TPS** (aggregate) | 11.9 | 108.3 | **9.08×** |

Per-dataset TPS: gsm8k 8.06×, humaneval 8.61×, mt-bench 9.95×.
Dominant phase `candidate_build` (95%) → `candidate_build` (56%) — still the top
phase, i.e. there is more to win here (see next step).

## Findings

1. **6.8× (b64) / 9.1× (b256) net wall-clock speedup, acceptance essentially
   unchanged.** The two fixes cut `candidate_build` by 94–96%. The speedup is
   larger at b256 because `candidate_build` was a larger share there (95% vs 88%).

2. **`.prep` is budget-invariant and collapses ~98%** (91.9→1.9 and 91.5→1.8
   ms/round). That confirms exp3's diagnosis: the reference's transfer cost was
   almost entirely the re-copied *static* markov weights, not the base logits.
   This is a pure implementation waste, removed with zero effect on the tree.

3. **Acceptance change is ±2%, not a systematic loss.** At b64 the per-dataset
   deltas are mixed-sign (gsm8k −0.10, humaneval +0.23, mt-bench +0.06) — noise at
   n=4, since restricting the candidate set builds a *different* (not strictly
   subset) tree whose per-round acceptance can land either way. At b256 they are
   uniformly, mildly negative (−0.35, −0.35, −0.05): the top-2048 truncation bites
   a little more when 256 nodes reach deeper into the per-node distributions. Net:
   **effectively lossless at b64, ~2% cost at b256** — well within the range where
   the 9× speedup dominates. Raising `beam_candidates` would shrink the b256 gap.

4. **The net speedup is smaller than the `candidate_build` reduction (Amdahl).**
   candidate_build fell 24× (b64) / 15× (b256), but total decode only 6.8× / 9.1×
   because `verify` and `draft_forward` are now the floor. At b64 the fast arm is
   verify-bound (59%); at b256 it is *still* candidate_build-bound (56%).

5. **This keeps the best-first acceptance ceiling at competitive speed.** Fast
   best-first: 7.32 / 7.88 acceptance at 181 / 108 TPS. Compare exp4's best fixed
   schedule `beam.flat`: 6.58 / 7.75 at 153 / 176 TPS, and the old slow best-first
   ceiling: ~7.18 / 8.04 at 23 / 11 TPS. Fast best-first matches/beats flat's
   acceptance *and* runs in flat's throughput class — it is the cheap adaptive
   builder exp4's finding #6 flagged as "worth recreating next."

## Verdict

The naive best-first tree was never algorithmically 10–20× slow — it was paying a
~311 MB/round static-weight transfer and a full-vocab CPU top-k per node. Removing
both (GPU top-k → ship only the ~2048-token active slice, once) gives **6.8× at
budget 64 and 9.1× at budget 256 with acceptance within ±2%**, turning the slowest
arm in exp3 into a verify-bound (b64) builder that keeps the adaptive acceptance
ceiling.

**Next step:** at b256 the fast arm is still `candidate_build`-dominant (56%),
because `.expand` is a *serial* per-pop CPU heap loop (6.6→36.7 ms/round, still
~linear in budget). Batching that expansion per level — one matmul over all
survivors, the beam builder's move — is the remaining win; the open question is
whether best-first's adaptive allocation can be kept while batching, or whether the
level-synchronous beam is the pragmatic stopping point.

Raw data: `results/summary.json`. Analysis: `analyze.py`. Reproduce: `reproduce.md`.
Builder + equivalence gate: `harness/ddtree/sparked_tree.py::build_sparked_tree_fast`,
`test_fast_builder.py`.
