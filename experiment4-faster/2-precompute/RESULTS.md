# Experiment 4-faster / 2 — precomputed transition table (stacked on transfer-less)

> **⚠️ PARTIALLY SUPERSEDED (2026-08-05, harness-6-union).** Finding #4's acceptance
> leak was FIXED: the precompute builder now pools the deduped union (same set as
> `fast`), and the re-run measured exact acceptance parity (8.004 @b64, 8.885 @b256,
> C=512, both arms). The cost moved: union precompute builds an O(L·U²) table, so at
> C=512 it is now *slower* than `fast` (96 vs 143 TPS @b256). The speed findings
> below (per-depth table ≈ free) no longer describe the current builder. The live
> question — fast vs union-precompute vs DDTree at C=128, old-BLOG scale, one GPU —
> is what `modal_benchmark.py` now measures.

Fold the precomputed `[L-1, C, C]` transition table **on top of** the 1-transfer-less
builder, and measure what that one move adds. This is the "together" stack: the
precompute arm keeps every transfer-less fix (GPU top-k, ship one small candidate
slice, no 311 MB static-weight resend) and additionally hoists the per-node
transition matmul out of the heap walk.

Run: H100 + 8 CPU, DSpark-b16 + own markov head, budgets {64, 256},
gsm8k/humaneval/mt-bench × 4 samples × 512 tokens, temp 0, two-pass
(clean TPS / instrumented phases), config fingerprint `7ddaea645ecb`,
CODE_VERSION `harness-5-precompute`. `checks.acceptance_match = true`.

Two arms, same job, same machine, **both at C=512** (so the candidate pool — hence
the tree — is as close to controlled as the two builders allow):
- **`bestfirst.fast`** — 1-transfer-less builder. Its remaining cost is `.expand`:
  a **serial per-pop CPU matmul**, one per popped node, ~linear in budget.
- **`bestfirst.precompute`** — transfer-less **+** the batched table. A node at depth
  *d* always holds an element of the depth *d−1* candidate set, so the whole
  `[L-1, C, C]` transition stack is **one batched GPU matmul before the walk**; the
  heap walk becomes pure CPU array indexing.

## The one move

`build_sparked_tree_precompute` replaces the fast builder's per-pop
`w2_active @ w1_active[prev]` (a serial loop, 48 dependent matmuls/round at b64) with:

1. one `torch.baddbmm` producing every transition the walk could ever ask for,
2. one per-row top-k, one `.cpu()` transfer of the resulting slice,
3. a heap walk that only *reads* that slice — no matmul, no cache miss possible.

Pop order is byte-for-byte the fast builder's, so the **walk is exact** (equivalence
gate `test_precompute_builder.py`: 100% identical trees to the fast builder across
seeds/budgets on synthetic logits, incl. the `markov_head=None` path).

## Results

### Budget 64

| metric | fast | precompute | reduction |
|---|---|---|---|
| **candidate_build** ms/round | 5.33 | 1.79 | **66.4%** |
| — `.expand` (serial per-pop CPU matmul) ms/round | 3.97 | 0.11 | **97.2%** |
| — `.prep` (GPU precompute + one xfer) ms/round | 0.97 | 1.19 | −23% |
| total decode ms/round (all phases) | 29.5 | 25.6 | 13.2% |
| **mean acceptance** (round-wt AGG) | 7.077 | 6.890 | **−2.63%** |
| **clean TPS** (aggregate) | 236.4 | 262.2 | **1.11×** |

Per-dataset TPS: gsm8k 1.11×, humaneval 1.13×, mt-bench 1.11×.

### Budget 256

| metric | fast | precompute | reduction |
|---|---|---|---|
| **candidate_build** ms/round | 20.48 | 4.48 | **78.1%** |
| — `.expand` (serial per-pop CPU matmul) ms/round | 18.27 | 0.43 | **97.7%** |
| — `.prep` (GPU precompute + one xfer) ms/round | 0.95 | 2.14 | −125% |
| total decode ms/round (all phases) | 45.3 | 28.9 | **36.2%** |
| **mean acceptance** (round-wt AGG) | 7.855 | 7.403 | **−5.75%** |
| **clean TPS** (aggregate) | 163.1 | 250.7 | **1.54×** |

Per-dataset TPS: gsm8k 1.38×, humaneval 1.37×, mt-bench **1.74×**.
Dominant phase flips `candidate_build` (45%) → **`verify` (65%)**: the builder is no
longer the floor. At b64 both arms were already verify-bound, which is why the b64
win is smaller — precompute mostly shrinks a phase that was already ~5% of a round.

## Findings

1. **The serial per-pop walk collapses 97%** (b64 3.97→0.11, b256 18.27→0.43 ms/round).
   That was the exact cost 1-transfer-less flagged as its remaining floor, and it is
   the entire point of the precompute. It is *replaced*, not shrunk: `.expand` is now
   a bare CPU read.

2. **The matmul moves into `.prep`, and it is nearly free.** `.prep` grows 0.97→1.19
   (b64) and 0.95→2.14 (b256) — the `[L-1,C,C]` baddbmm plus a top-k slice whose
   transfer scales with budget (k=min(budget,C)). Even at b256 that is 2.1 ms against
   the 18 ms it removed. The C² table is cheap on an H100 at C=512.

3. **Net wall-clock: 1.11× (b64) / 1.54× (b256), acceptance already priced in.** TPS
   is end-to-end, so precompute's lower acceptance (more rounds) is *already* baked
   into these numbers — it is 1.54× faster at b256 **despite** needing ~6% more
   rounds, because each round's build is 16 ms cheaper. The win grows with budget
   because `.expand` was linear in budget and is now flat.

4. **Acceptance is NOT free here: −2.6% (b64) / −5.75% (b256).** This is the honest
   cost, and it is *not* the precompute mechanism (the walk is exact). It is the
   **per-depth candidate restriction the table shape forces.** The fast builder pools
   a deduped **union** of per-depth top-C (~8000 tokens on this model) and lets the
   markov bias promote from that union at *every* depth. The precompute table can
   only afford **per-depth top-C** (512), so a token the bias wants that sits in the
   union but outside *that depth's* top-512 is unreachable. humaneval (structured
   code, strong markov continuations) pays most: −1.26 at b256. On synthetic logits
   with weak bias this gap vanishes (100% identical trees) — it is a real-model,
   strong-bias effect. Matching the union exactly would need a `[L-1, U, U]` table
   (U~8000), i.e. the quadratic-C blow-up the whole approach exists to avoid. So the
   small acceptance cost is intrinsic to keeping the table affordable.

5. **The knob to recover it is C.** Larger C makes per-depth top-C approach the union
   and shrinks the acceptance gap, at quadratic build cost. C=512 sits where the 16 ms
   build cut dwarfs the ~6% acceptance cost; the net is strongly positive at both
   budgets. If the b256 humaneval acceptance loss ever mattered, raising C is the dial.

## Verdict

Stacking the precomputed transition table on the transfer-less builder cuts
`candidate_build` **66% at b64 / 78% at b256** — collapsing the serial per-pop matmul
97% into one batched GPU call — for a **net 1.11× / 1.54× wall-clock speedup**. It
costs **2.6% / 5.75% acceptance**, entirely from the per-depth (vs union) candidate
pool the table requires, recoverable by raising C. At b256 the combined stack is no
longer builder-bound (candidate_build 45%→ verify 65%); it is now a verify-bound
best-first builder that keeps the adaptive acceptance ceiling at H100 throughput.

Raw data: `results/summary.json`. Analysis: `analyze.py`. Charts:
`results/speedup_acceptance.png`, `results/phase_collapse.png`. Reproduce:
`reproduce.md`. Builder + equivalence gate:
`harness/ddtree/sparked_tree.py::build_sparked_tree_precompute`,
`test_precompute_builder.py`.
