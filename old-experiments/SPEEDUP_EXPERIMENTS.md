# Making SparkedTree fast — every attempt, ranked

A consolidated record of the optimization work on the sparked-tree builder.
Sources: [`BLOG.md`](BLOG.md), [`experiments/RESULTS.md`](experiments/RESULTS.md),
[`experiments/FINDINGS.md`](experiments/FINDINGS.md),
[`final_benchmark/REPORT.md`](final_benchmark/REPORT.md),
[`final_benchmark/config.py`](final_benchmark/config.py),
[`concurrency/FINDINGS.md`](concurrency/FINDINGS.md).
Implementation lives in [`ddtree/ddtree_markov.py`](ddtree/ddtree_markov.py).

---

## 0. What we were actually optimizing

A speculative round has four stages: **draft → tree_build → verify → commit**.

Wall-clock = `rounds × round_time`, and the two levers pull against each other:

- `tree_build` is the stage sparked-tree made expensive. It is ~5% of a round.
- **acceptance** decides how many rounds you need. It is the other ~95%.

Almost every mistake in this project came from optimizing the visible 5%.

**Why the tree got expensive.** DDTree assumes depth *d+1* is independent of depth
*d*, so **one top-k table per depth serves every node** — the whole tree is a few
lookups (0.10s). SparkedTree's entire thesis is that this assumption is false:
each node's children get re-scored by the token *that node actually holds*
(`bias = W2 @ W1[prev_token]`). So every node needs its own table. The naive
version cost **2.15s/prompt vs DDTree's 0.10s** — a 21× regression that ate the
whole acceptance win.

Worse, the structure was pathological: best-first can't even *request* node *N*'s
table until node *N−1* is popped. So it wasn't one big GPU call, it was ~48
**dependent** round-trips per round at ~0.26 ms of fixed launch cost each.

---

## 1. The attempts, in order

### ❌ Attempt 1 — Shrink the payload (candidate restriction)

**Idea.** Each round-trip ships a slice of `w2` (the rank-r → vocab matrix). Full
vocab is 78 MB. Restrict to the top-C candidate tokens and it's 17 MB. Memory
traffic is the bottleneck, right?

**Result** (gsm8k, block-16, budget 64, A10G):

| candidates | acceptance | tree_build |
|---|---|---|
| full vocab | 11.562 | 2.73s |
| 4096 | 11.562 | 2.22s |
| 2048 | 11.453 | 2.16s |
| 512 | 11.346 | **2.15s** |

**Verdict: partial, and the diagnosis was wrong.** −21% build time, then it
flatlines — 512 candidates cost the *same* as 4096. Payload was never the issue.
The bottleneck was the **number** of round-trips, not their size. This is the
single most instructive negative in the project: it looked like a memory problem
and was a launch-latency problem.

---

### ❌ Attempt 2 — Wave batching (pop K nodes, expand together)

**Idea.** Keep best-first, but pop *K* nodes at once, batch their table requests
into one call, then push all successors. 7–10× fewer syncs, same tree.

**Result: acceptance collapsed 11.45 → 8.13.**

**Verdict: abandoned.** It is *not* the same tree. Deferring the pushes means a
successor can't outrank later members of its own wave — which is precisely what
best-first ordering *is*. This corrupted the algorithm rather than replacing it.
The lesson that carried forward: **replace the algorithm with a different
principled one, don't approximate it into incoherence.**

---

### ✅ Attempt 3 — Level-synchronous beam (fix the tree shape up front)

**Idea.** Stop letting the tree shape be data-dependent. Decide the width at each
depth *in advance*, then expand a whole level in one batched matmul. Surviving
tokens stay on GPU as a tensor. ~48 syncs/round → **1**.

**Result:** `tree_build` 2.16s → **0.61s (−72%)**.

**Verdict: the big structural win.** This is what made sparked-tree viable at all.
It cost some acceptance (a fixed schedule can't spend budget where the drafter
happens to be uncertain *this* round) but converted an acceptance-only win into
wall-clock parity with DDTree.

---

### ✅✅ Attempt 3a — The FLAT width schedule (the best result in the project)

Fixing the shape up front forces the question: *what shape?* Every intuition says
branch wide near the root, narrow with depth. We instrumented it — logging
`(depth, slot)` for every accepted node across 6 benchmarks — and measurement said
the **exact opposite**:

| depth | accepted node was drafter's top pick | slots for 95% coverage |
|---|---|---|
| 1 | **87%** | **3** |
| 4 | 66% | 13 |
| 8 | 59% | 14 |
| 12 | 48% | 21 |
| 16 | **30%** | **42** |

The drafter is confident near the root and uncertain deep. A geometric decay=0.6
schedule gives `[26,15,9,6,3,2,1,1,1]` — it spends 26 slots where 3 would do and
**zero** slots where 42 are needed.

| schedule (budget 64, gsm8k) | acceptance | speedup | tree_build |
|---|---|---|---|
| geometric decay 0.75 | 9.653 | 6.53x | 0.57s |
| measured `[2,3,7,7,…]` | 10.974 | **7.22x** | 0.65s |
| **flat `[4,4,4,4,…]`** | **11.089** | 7.21x | 0.63s |
| *DDTree reference* | 10.024 | 7.20x | 0.10s |

**Verdict: best single change measured.** +15% acceptance at zero extra cost, and
it's what took the method from 21% *behind* DDTree on wall-clock to parity.

Two corollaries worth keeping:
- **The exact shape matters less than not decaying.** Hand-flattened uniform
  slightly beat the empirically-measured schedule. The failure mode was
  front-loading, not mis-tuning.
- **It unlocked the next optimization for free.** CUDA graph capture requires
  static shapes. Flat is static; geometric and confidence-adaptive are not.

⚠️ Known bug: `flat(16)` at depth 16 gives `[1,1,…,1]` — a chain with zero
branching, which is why budget-16 lost 8.3%. Flat is only valid when
`budget ≥ 2 × depth`; the fix is to truncate depth, not thin the schedule.
(`beam_min_width=2` guards this in the final config.)

---

### ✅✅ Attempt 4 — Precompute the transition table (the elegant one)

**Idea.** The beam level loop was still 16 **dependent** matmuls — level *d* can't
start until level *d−1*'s top-k lands. But note: a parent at depth *d* is *always*
drawn from the depth *d−1* candidate set. The beam has nothing else to choose from.

So the entire transition is a finite **C × C table per depth**, and the whole
`[L−1, C, C]` stack is **one batched matmul with no dependence on the beam at
all**. The normalizing `logsumexp` depends only on `(depth, parent slot)`, so it
folds in too. What remains per level is a gather, an add, and a top-k.

**Result** (H100, budget 64, ms/round):

| builder | ms/round |
|---|---|
| lazy best-first (naive) | 9.55 |
| in-loop matmul, beam | 4.03 |
| **precomputed table, beam** | **1.88** |
| **precomputed table, best-first** | **3.82** |

**Verdict: the win that also rescued best-first.** Best-first's problem was never
arithmetic — it was that each node's table was unknown until the previous pop.
With every table precomputed, the whole heap runs on **CPU against one transfer**.
That gave us back the exact algorithm at 2.5× the beam's cost instead of 5×, and
it is the builder that shipped.

The precompute is **exact** — trees are provably identical to the builder it
replaces.

⚠️ **The knob that bites.** The table is `C × C`, so the candidate pool became a
**quadratic** cost where it used to be linear. Shrinking it is tempting and not
free — measured three independent ways, cutting the pool costs ~3–4% acceptance.
We settled on **C=512**: −4.1% acceptance for a **52% cheaper builder**, which nets
out strongly positive. The penalty for large C is far worse on an A10G than an
H100 — the table trades memory traffic for compute, and only the bigger card has
the headroom.

---

### ⚠️ Attempt 5 — CUDA graph capture

**Idea.** Profiling said the beam level loop is ~100% of `tree_build`:

| component | ms/round | share |
|---|---|---|
| GPU level loop | 3.98 | ~100% |
| transfer / sync | 0.19 | 5% |
| Python tree assembly | 0.06 | **2%** |

So it's bound by *launching* 16 dependent levels, not by data volume or CPU work.
Capture the loop once, replay it.

**Result:**

| budget | eager | graphed | saved | `tree_build` | gap to DDTree |
|---|---|---|---|---|---|
| 64 | 3.594 ms | **1.093 ms** | **70%** | 0.61s → 0.19s | 6.1x → **1.9x** |
| 128 | 3.356 ms | **1.570 ms** | 53% | 0.57s → 0.27s | 6.1x → **2.7x** |

Trees are **bit-identical** to the eager path. Pure speed, zero acceptance cost.

**Verdict: technically excellent, and we didn't ship it as the default.** This is
the counterintuitive headline of the whole optimization effort — see §2.

Two prerequisites: static shapes (came free from the flat schedule), and per-depth
candidates instead of a deduped union (`torch.unique` returns a variable-length
tensor, which breaks capture — and per-depth top-C is slightly tighter anyway).
Capture is cached per `(widths, C, device)`; on failure it logs and falls back to
eager permanently rather than retrying per round.

⚠️ The end-to-end benchmark **could not confirm this**. Per-dataset `tree_build`
moved in the *wrong* direction on half the datasets. Six independently-scheduled
H100 containers carry ~±12% timing variance, which swamps a ~0.4s effect. The
isolated microbenchmark is the trustworthy measurement.

---

### ❌ Attempt 6 — Confidence-head adaptive widths

**Idea.** The flat schedule is the *average* optimal allocation. DSpark's
confidence head predicts P(accept) per position for the *current* round, so widen
per-round instead:

```
reach[d] = prod_{j<d} p_j        # don't widen a depth we'll never reach
value[d] = reach[d] * (1 - p_d)  # don't widen a depth we're already sure about
W_d     ∝ value[d]
```

It produced sensible-looking schedules — easy rounds spread deep
`[2,3,4,6,6,8,…]`, hard rounds hedge early and abandon depth 7+ `[19,20,14,7,3,1]`.

**Result: −2.7% acceptance, −6.6% speed.** Per cell: gsm8k@128 −5.9%/−11.4%,
math500@64 −5.2%/−7.8%, math500@128 −5.0%/−7.6%, alpaca@128 −0.7%/−3.6%. The
single positive cell was alpaca@64 (+3.6% acc / −2.3% spd) — chat, where
round-to-round variance is highest.

**Verdict: failed, for an interesting reason.** The head predicts **chain**
acceptance; a tree needs **coverage**. Those are different objectives — knowing
you'll probably accept position 5 doesn't tell you how many *alternatives* at
position 5 you need to insure against being wrong. It also **broke CUDA graph
capture** (dynamic shapes), so it cost the §5 win too.

---

### ❌ Attempt 7 — Truncate tree depth to 12

**Idea.** Deep nodes rarely get accepted; stop building them.

**Result: no measurable gain.** Depths ≥12 carry **9.7%** of accepted tokens.

**Verdict: dead end.** Consistent with the flat-schedule finding — deep positions
are where the hedging actually pays.

---

### ❌ Attempt 8 — Cap max_fanout at 48

**Idea.** Bound the branching factor to keep tables tighter.

**Result:** traded 0.4% acceptance for build time that **doesn't resolve** in the
harness (per-cell speed noise is ~16%).

**Verdict: not worth it.** Shipped as `max_fanout=0` (= budget, uncapped).

---

### ❌ Attempt 9 — Shrink candidate pool to 256

**Result: −3.6% acceptance for no speed gain.**

**Verdict:** C=512 is the floor. 2048 buys ~4% more acceptance at 4× the build
time — the trade curve is steep on both sides and 512 sits at the knee.

---

## 2. The result that matters most: **the fastest builder is not the best one**

Final benchmark, H100, block 16, 6 datasets × 12 prompts × 512 tokens, batch 1:

| budget 64 | acceptance | speedup | build time |
|---|---|---|---|
| `sparked` (best-first over precomputed table) | **7.832** | **3.21×** | 3.82 ms/round |
| `beam_graphed` (CUDA-graphed beam) | 7.528 | 3.10× | 0.94 ms/round |

The graphed beam is **4× faster at building** and **ends up slower overall**.
0.68s vs 1.70s on tree construction, and it loses.

Because: **builder time is ~5% of a round; acceptance sets how many rounds you
need.** Spending 5% to buy back 95% is the correct trade every time.

This is a trap the project fell into twice — once with the candidate-restriction
misdiagnosis, once by nearly shipping the graphed beam as the default.

Nuance worth recording: at budget **128** the ordering partially flips —
`beam_graphed` posts 3.44× vs `sparked`'s 3.35× despite lower acceptance
(8.095 vs 8.225). At larger budgets the build cost grows fast enough that the
graph starts paying for itself. Both are within the harness's ~5% speed noise, so
we chose on acceptance, which resolves at ~0.5%.

---

## 3. Scoreboard

### By impact on the goal (end-to-end wall clock)

| rank | change | effect | verdict |
|---|---|---|---|
| 🥇 1 | **Flat width schedule** | +15% acceptance, free | Best change in the project. Turned a −21% wall-clock deficit into parity, and unlocked CUDA graphs. |
| 🥈 2 | **Precomputed `[L−1,C,C]` transition table** | 4.03 → 1.88 ms (beam), 9.55 → 3.82 ms (best-first) | Exact, not an approximation. Rescued best-first entirely. Shipped. |
| 🥉 3 | **Level-synchronous beam** | 2.16s → 0.61s (−72%) | The structural fix. Killed 48 dependent syncs/round. |
| 4 | **CUDA graph capture** | −70% builder time, bit-identical trees | Great engineering, kept as an option, not the default. |
| 5 | Candidate restriction to C=512 | −21% build, then flat; −4.1% acc | Half a win — and the diagnosis behind it was wrong. |
| 6 | max_fanout cap | −0.4% acc, unresolvable speed gain | Neutral-to-negative, rejected. |
| 7 | Depth truncation to 12 | nothing | No effect. Depths ≥12 carry 9.7% of accepts. |
| 8 | Candidate pool → 256 | −3.6% acc, no speed gain | Strictly worse. |
| 9 | Confidence-adaptive widths | −2.7% acc, −6.6% spd | Failed *and* broke graph capture. |
| 🔻 10 | **Wave batching** | **11.45 → 8.13 acceptance** | Worst. Corrupted best-first ordering outright. |

### Builder head-to-head (H100, budget 64)

| builder | ms/round | acceptance | shipped? |
|---|---|---|---|
| lazy best-first (naive, per-node round-trips) | 9.55 | 11.56 | ❌ 21× slower than DDTree |
| in-loop matmul, beam | 4.03 | ~11.09 | ❌ superseded |
| **precomputed table, best-first** | **3.82** | **7.83** ᵃ | ✅ **default (`exact-precomputed`)** |
| precomputed table, beam | 1.88 | — | ✅ available |
| + CUDA graph | **0.94** | 7.53 ᵃ | ✅ available (`beam_graphed`) |
| *DDTree reference* | ~0.25 | 7.39 ᵃ | — |

ᵃ 6-dataset H100 mean from the final benchmark; the 11.x figures are gsm8k-only
from the A10G sweep. Not comparable across those two columns — compare within.

---

## 4. Where it landed

```python
SPARKED = dict(tree_mode="exact-precomputed", beam_candidates=512,
               beam_min_width=2, max_fanout=0)
```

Every value decided by a measurement (see [`final_benchmark/config.py`](final_benchmark/config.py)):

| setting | value | decided by |
|---|---|---|
| tree builder | `exact-precomputed` | +6.2% acceptance over flat beam @64 (6/6 datasets), +2.3% @128 |
| candidate pool C | 512 | 256 costs 3.6% acc; 2048 buys 4% at 4× build time |
| min_width | 2 | never emit a `[1]*depth` chain (the budget-16 bug) |
| max_fanout | 0 (= budget) | capping at 48 traded 0.4% acc for unresolvable time |

**Net effect of all this work:** the builder went from a **21× regression vs
DDTree** to a **~1.9–2.7× gap**, and the end-to-end result went from *"beats DDTree
on acceptance, ties on wall-clock (+0.7%)"* to **+17.8% wall-clock, +5.8%
acceptance**.

Stage times show why: our tree costs **more** to build (2.13s vs DDTree's 0.44s)
and that is dwarfed by what higher acceptance saves in verification
(**29.7s vs 39.1s**). Fewer rounds. Verification is where time goes.

---

## 5. The one that got away: it's all batch 1

Every number above is batch 1, and there's a structural problem waiting at
serving concurrency ([`concurrency/FINDINGS.md`](concurrency/FINDINGS.md)):

- Our trees score ~0.06–0.12 accepted tokens per token the target must verify;
  a chain scores ~0.36 — **~5× less compute-efficient per accepted token**. At
  batch 1 the GPU is idle so wide verification is nearly free. At concurrency it
  is not. Measured crossover: **tb64 loses above c≈4, tb128 above c≈2.**
- The builder is **host-resident** — a Python heap walk. SGLang plugins that
  can't run under the overlap scheduler forfeit ~29% throughput on average
  (up to 37% at c=8). An unplanned control confirmed it: a nearly-free
  lookup proposer through a host-side Python loop beat DSpark by 22% at c=1 and
  **lost by 27% at c=32** — a 1.62× relative degradation while doing strictly
  less arithmetic.

So the real remaining optimization isn't shaving more milliseconds off a host-side
builder — it's making the builder **device-resident**. Everything in §1 optimizes
the batch-1 regime; that's a different fix for a different regime.

---

## 6. Lessons that generalize

1. **Measure which axis is the bottleneck before optimizing it.** Candidate
   restriction assumed payload; it was launch count. 512 candidates cost the same
   as 4096 — that one measurement would have saved the whole detour.
2. **Number of dependent GPU round-trips ≫ size of each one.** −21% from shrinking
   calls, −72% from eliminating them.
3. **Don't approximate an algorithm into incoherence — replace it.** Wave batching
   corrupted best-first and lost 29% acceptance. The beam replaced it with a
   different principled algorithm and won.
4. **The fastest component is not the best system.** Builder time is 5% of a round;
   acceptance sets round count. Optimizing the visible 5% at the expense of the
   other 95% is a trap.
5. **Empirical schedules beat intuitive ones, and the intuition was backwards.**
   Everyone's instinct is to branch wide at the root. The drafter is 87% correct
   at depth 1 and 30% at depth 16.
6. **A correctness-preserving optimization is worth more than a faster
   approximation.** The precomputed table is exact and CUDA graphs are
   bit-identical; both survived. Every approximation we tried (wave, small pools,
   depth truncation, adaptive widths) either lost acceptance or did nothing.
7. **Know your noise floor before you claim a win.** Acceptance resolves at ~0.5%
   in this harness; per-cell speed does not resolve below ~16%. That is why the
   CUDA-graph win is a microbenchmark claim and the final builder choice was made
   on acceptance.
