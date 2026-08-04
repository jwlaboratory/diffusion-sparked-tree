# Can we verify less? — offline investigation

Question: the tree verifies 65–129 positions per round to accept ~8 tokens, against
a chain's 17 to accept ~6. That ratio (`RESULTS.md` §9) is what makes trees lose
above c≈4. **Can we spend that width only when it converts?**

No GPU used. Everything here replays accepted-length logs already in the repo
(`experiment1-harness/Results/results_detailed.json`, 6 methods × 3 datasets ×
8 prompts × 512 tokens, budget 64).

Run: `python3 experiment4-speedingup/investigate/<script>.py`

| script | what it does |
|---|---|
| `validate_alignment.py` | **run first** — checks the premise the rest depends on |
| `aligned_headtohead.py` | chain vs tree at shared round boundaries |
| `oracle_selector.py` | prices a perfect per-round chain-or-tree selector |
| `find_signal.py` | is the previous round's length a usable signal? |
| `signal_ceiling.py` | every free history feature + noise tolerance of a real predictor |
| `robustness_and_entropy.py` | divergence robustness + entropy-budget hypothesis |

---

## Method, and its one real weakness

All methods are greedy and near-lossless, so they emit (nearly) the same token
sequence and differ only in how they partition it into rounds. Wherever two
methods share a round boundary they should be in an identical state, making the
next round a controlled head-to-head.

**`validate_alignment.py` shows that premise is only partly true.** Methods
disagree on `n_out` by up to 113 tokens on the same prompt. `BLOG.md` §6 explains
why — the target disagrees with *itself* ~0.5% of the time depending on how many
positions it scores at once, and once one token differs, everything after it
differs. Alignment yield is 16–50% of rounds and front-loaded (median position
0.10–0.43 of the output), exactly the signature of compounding divergence.

So a shared token index late in a generation is **not** a shared context. Every
number below is therefore reported against a **position cutoff sweep**; results
that move with the cutoff are artifacts, results that hold are real. Position 0
(round 1 of each prompt) is the only divergence-free sample by construction.

---

## Finding 1 — RETRACTED: "the tree is not a superset of the chain"

An earlier read of this data found 5.9–9.3% of rounds where the tree accepted
*fewer* tokens than the chain, and attributed it to best-first truncating its own
depth (deep spine nodes carry low path scores and lose to shallow alternatives).

**The robustness sweep kills it.** The negative-gain rate is a pure function of
how much divergence you allow in:

| first X of output | DSpark: tree<chain | DFlash: tree<chain |
|---|---|---|
| 0.05 | **0.0%** | **0.0%** |
| 0.10 | 0.0% | 1.2% |
| 0.25 | 0.6% | 4.4% |
| 0.50 | 1.9% | 4.1% |
| 1.00 | 5.9% | 9.3% |

In the region where the pairing is trustworthy it is **zero**. The tree does
contain the chain's draft, as the theory says it should.

Consequences: the proposed "reserve the greedy spine" fix has no problem left to
solve, and the oracle selector's *acceptance* upside collapses with it (+4.9% →
+0.0–1.2%). The acceptance axis is a dead end. **Only the width axis survives.**

---

## Finding 2 — the width saving is real, ~13–36%

Stable across the cutoff sweep, so not a divergence artifact:

| DFlash b16 vs DDTree, n=432 | acceptance | verify width |
|---|---|---|
| chain | 3.884 | 16 |
| tree | 5.280 | 65 |
| **oracle selector** | 5.28 (no gain) | **56.2** |

A perfect per-round chain-or-tree selector saves **13.5%** of verify width at
zero acceptance loss on block-16, **29.7%** on block-7. Not nothing, but far less
than the 24% the uncorrected read suggested.

The underlying economics, which is the number worth remembering:

> the tree buys **+1.40 tokens for 49 extra scored positions — 35 positions per
> extra accepted token.**

At batch 1 that is fine (a width-65 tree costs 0.87× a width-17 chain — the GPU
is idle either way). At c=32 width 65 costs 2.76×. **This is a concurrency
optimization and worth ~0 at batch 1.**

---

## Finding 3 — history is not the signal

Every feature a decoder already has for free at round start, AUC for "will the
tree gain this round", against a shuffled-label null band of 0.44–0.56:

| feature | DSpark pooled / within-dataset | DFlash pooled / within-dataset |
|---|---|---|
| prev round length | 0.616 / 0.575 | 0.502 / 0.573 |
| mean of last 4 | 0.637 / 0.583 | 0.513 / 0.591 |
| prev saturated? | 0.591 / 0.560 | 0.523 / 0.557 |
| position in output | 0.509 / 0.519 | 0.588 / 0.596 |
| *this round's chain length* | *0.823* | *0.708* |

Pooled numbers that look promising collapse **within dataset** — most of the
apparent signal is just encoding "this is mt-bench". Every threshold policy built
on history loses acceptance faster than it saves width; there is no point on the
frontier that dominates always-tree.

**But the last row shows the information exists** — it is simply about the
*current* round, not the past. Which means it has to come from the drafter.

---

## Finding 4 — the spec for a real predictor is loose

The drafter's `base_logits` are already computed before the tree is built, so a
statistic over them is **free** — no extra forward pass, unlike the confidence
head. What accuracy would it need? Inject gaussian noise into the true chain
length and re-price the policy:

| estimator noise (sd, tokens) | width saved, zero acceptance loss (DSpark) | (DFlash) |
|---|---|---|
| 0.0 (perfect) | −29.7% | −13.5% |
| 1.0 | −27.5% | −13.5% |
| 1.5 | −26.8% | −12.5% |
| 2.0 | −19.9% | −13.5% |
| 3.0 | −8.9% | −9.8% |
| 5.0 | no lossless point | −5.2% |

**Predict chain acceptance to within ±1.5–2 tokens and you capture essentially
the whole benefit.** That is a loose bar — and it is precisely what DSpark's
confidence head was trained to do.

---

## Finding 5 — entropy-based budget shrinking is backwards

The hypothesis: when the drafter is uncertain the target rejects everything
anyway, so shrink the budget and save compute. That predicts tree width should
pay *least* on hard rounds.

Bucketing every aligned round by chain acceptance (what an entropy signal
estimates), DFlash b16:

| chain accepted | share of rounds | mean tree accepted | mean gain | **share of all gain** |
|---|---|---|---|---|
| 1 | 28.9% | 3.12 | **+2.12** | **37.1%** |
| 2 | 19.2% | 3.40 | +1.40 | 16.7% |
| 3 | 15.7% | 4.71 | +1.71 | 17.1% |
| 4–8 | 25.8% | — | +0.4 to +2.3 | 24.5% |
| ≥9 | 10.4% | — | −3.0 to +2.0 | 4.5% |

**Rounds where the chain accepts ≤3 are 64% of rounds and carry 71% of everything
tree width produces.** DSpark b7 agrees: chain ≤3 is 43% of rounds and 73% of the
gain, while saturated rounds (chain=8) are 32% of rounds and contribute **0.0%**.

Shrinking the budget on high-entropy rounds would cut spending exactly where 70%
of the return lives.

The premise fails empirically: high entropy does **not** mean the target rejects
the tree too. At chain=1 the tree still accepts **3.12** tokens — the chain's one
guess was wrong and one of the tree's alternatives was right. Rescuing those
rounds *is the tree's job*.

This is the same shape as the flat-schedule result in `experiments/FINDINGS.md`
§1: intuition says spend where the drafter is confident, measurement says spend
where it is uncertain. Third time this project has hit that wall.

**The correct policy is the mirror image: shrink the budget when the drafter is
CONFIDENT.** Saturated rounds are 32% of DSpark's rounds at mean gain −0.32 —
that is where a chain does the same job for 1/4 the width, and it is the entire
source of Finding 2's saving.

---

## Where that leaves the confidence head

`RESULTS.md` §11 benched it at −2.7% acc / −6.6% spd. The diagnosis there was
that it predicts *chain* acceptance while a tree needs *coverage*. That diagnosis
still holds for the job it was given — **allocating width across depths**.

But the job Finding 2 needs is different: a **scalar gate** on "is this round easy
enough that a chain suffices?" That is a chain-acceptance question, which is the
head's native semantics, used with the **opposite sign** to the original attempt.
Finding 4 says ±2 tokens is good enough. It may need no retraining at all — just
repointing.

Joint training with the tree in the loop would be the stronger version, and is
worth doing if the repointed head lands close but short.

---

## Status

| claim | verdict |
|---|---|
| Tree sometimes accepts less than the chain | ❌ **retracted** — divergence artifact, 0.0% when clean |
| Reserve the greedy spine for +4.9% acceptance | ❌ **withdrawn** — no problem to solve |
| Per-round selector saves verify width | ✅ 13.5% (b16) / 29.7% (b7), zero acceptance loss, oracle |
| Previous-round length predicts it | ❌ AUC ≈ chance within dataset |
| A ±2-token chain-acceptance estimator suffices | ✅ captures the full saving |
| Shrink budget on high-entropy rounds | ❌ **backwards** — those rounds carry 71% of the gain |
| Shrink budget on low-entropy rounds | ✅ this is where the 13–30% actually comes from |

**Caveats.** Block-7 DSpark and DFlash-b16 at budget 64, 8 prompts × 3 datasets —
not the shipped block-16 DSpark at 128. Oracle numbers are upper bounds. n=408/432
aligned rounds, and the trustworthy subset (first 10% of output) is n=99/81.

---

## Built: the instrumentation this analysis was missing

Every number above is an oracle, because the one field needed to turn it into a
policy was never logged. It now is.

**`harness/ddtree/confidence.py`** computes per-round drafter confidence from
`base_draft_logits`, which is already materialized before the tree is built — so
it costs one softmax and **no extra forward pass**, unlike the confidence head.
The headline field is `pred_chain_len`, the drafter's own estimate of greedy-chain
acceptance under exactly the independence assumption DDTree already makes:

```
E[accepted] = sum_{d=1..D} prod_{j<=d} p_j        p_j = top-1 prob at depth d
```

Wired into `sparked_tree_generate` behind `measure_confidence=False`, and placed
**between the `draft` and `tree_build` timers** so it lands in no stage bucket and
cannot move the numbers exp3 exists to measure. Returned as `confidence_by_round`,
paired positionally with `acceptance_lengths`.

That pairing is the real win. This analysis had to align two *different* runs on
token index, which degrades as bf16 divergence accumulates (Finding 1 died of it).
**One run carrying both fields has no alignment problem at all.**

| file | |
|---|---|
| `harness/ddtree/confidence.py` | the statistic (new) |
| `harness/ddtree/sparked_tree.py` | 4-line wiring, off by default |
| `test_confidence.py` | 5 CPU tests — closed form, extremes, monotonicity, degenerate input, and that the call sits outside every stage timer |
| `price_gate.py` | consumes a run and returns go/no-go; `--self-test` verifies it on synthetic data with a planted error of 1.5 tokens |

```bash
python3 experiment4-speedingup/investigate/test_confidence.py   # all passed
python3 experiment4-speedingup/investigate/price_gate.py --self-test
```

### The run to do, and how to read it

Run any tree arm with `measure_confidence=True`, dump `{acc, conf}` per dataset,
then `price_gate.py run.json`. It answers three questions in kill-fastest order:

1. Does `pred_chain_len` track actual acceptance? Spearman < 0.3 → stop.
2. What is its error in tokens? **This is the pass/fail number** — Finding 4 says
   ≤2.0 captures the whole benefit, ≥3.0 buys almost nothing.
3. At each threshold: how many rounds get gated, and were they already at the
   block ceiling? Gating is only safe where they were.

### Then the gate itself

Not yet implemented — it needs a threshold, which needs the run above. One useful
fact for when it is: `build_sparked_tree` already transfers the logits to CPU
(`sparked_tree.py:86`), so reading `pred_chain_len` to steer the budget costs
**no additional sync**.

And it must be built in the direction Finding 5 established — **shrink when the
drafter is confident**, never when it is uncertain.
