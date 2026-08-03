# Sparked Tree — final results

**Qwen3-4B · H100 · block 16 · greedy · 6 datasets × 12 prompts × 512 tokens · batch 1**
Every method verified byte-identical to plain autoregressive decoding.
Full tables in [`REPORT.md`](REPORT.md); configuration and provenance in [`config.py`](config.py).

---

## The three methods

**DSpark** drafts a *chain*. One backbone pass over a masked block produces 16
draft positions, then a small markov head sweeps them serially so draft *k+1* is
conditioned on draft *k*. The target verifies one linear sequence. Cheap, but a
chain has a single point of failure: one wrong token ends the round.

**DDTree** drafts a *tree*. It assumes the distribution at depth *d* is independent
of which token an ancestor holds, so one top-k table per depth serves every node
at that depth. That assumption is what makes it fast — the whole tree is a handful
of table lookups — and it means every sibling offers its children the same
candidates.

**Sparked tree** drafts a tree whose branches are *conditioned on the branch*.
DSpark's markov head says the distribution at depth *d+1* depends on the token
actually drafted at depth *d*, which is precisely what DDTree assumes away. Each
node therefore gets its own children distribution, and the tree is built best-first
over path score, so budget flows to the branches that are actually uncertain.

The cost of that is arithmetic — a per-node bias correction the independence
assumption avoids. The work here was making it cheap: the transitions are
precomputed once per round as a single `[depth, C, C]` table, so the builder does
one batched matmul instead of a chain of dependent ones.

---

## Results

### Acceptance — tokens per round (budget 128)

| dataset | DSpark | DDTree | **sparked** | vs DDTree |
|---|---|---|---|---|
| humaneval | 6.365 | 8.933 | 8.877 | −0.6% |
| mbpp | 6.557 | 8.878 | 8.678 | −2.3% |
| gsm8k | 7.803 | 9.110 | **9.992** | **+9.7%** |
| math500 | 7.833 | 10.410 | 10.171 | −2.3% |
| mt-bench | 3.832 | 4.894 | **5.566** | **+13.7%** |
| alpaca | 4.003 | 4.402 | **6.066** | **+37.8%** |
| **MEAN** | **6.065** | **7.771** | **8.225** | **+5.8%** |

### Speedup vs no drafter

| | DSpark | DDTree | **sparked** |
|---|---|---|---|
| budget 64 | 2.80× | 2.64× | **3.21×** |
| budget 128 | 2.80× | 2.84× | **3.35×** |

| sparked vs | acceptance | speed |
|---|---|---|
| DSpark | **+35.6%** | **+19.6%** (6/6) |
| DDTree | **+5.8%** | **+17.8%** |

---

## What we find

**1. Branch-conditional scoring is the whole effect, and it is large.**
Same drafter, same tree budget, only the scoring differs: independence-scored
gives 3.115 acceptance, branch-conditional gives 7.955 — **+155%**. The same markov
head applied to a *different* backbone makes things worse (−14.4%), so it is not a
general bigram prior; it is a backbone-specific residual, and it only pays when it
guides the model it was trained on.

**2. We now win on speed, which the previous version did not.**
The earlier result beat DDTree on acceptance but tied on wall-clock (+0.7%). Making
the builder ~2× cheaper converts that into **+17.8%**. The mechanism is visible in
the stage times: our tree costs more to build (2.13s vs DDTree's 0.44s) but that is
dwarfed by what higher acceptance saves in verification (**29.7s vs 39.1s**) — fewer
rounds, and verification is where the time actually goes.

**3. The win concentrates in open-ended text, not code.**
Against DDTree we are +37.8% on alpaca and +13.7% on mt-bench, but −0.6% to −2.3%
on humaneval, mbpp and math500. Structured code is predictable enough that a
per-depth table is nearly as good as a per-node one; conversational text is where
the next token genuinely depends on the last one, and that is exactly what
branch-conditional scoring models.

**4. Faster tree construction costs acceptance, and we took the trade knowingly.**
Precomputing the transition table forces a per-depth candidate pool instead of a
pooled union, and a smaller pool costs ~4% acceptance (measured three independent
ways). We give up ~2.6% acceptance against the slow builder and get roughly half
the build time, which nets out strongly positive end-to-end. The precompute itself
is exact — trees are provably identical to the builder it replaces.

**5. A faster builder is not automatically a better one.**
A CUDA-graphed beam builds a tree in 0.68s against best-first's 1.70s, and loses:
7.528 acceptance vs 7.832 at budget 64, ending up slower overall. Builder time is
~5% of a round; acceptance sets how many rounds you need. Optimising the visible
5% at the expense of the other 95% is a trap this project fell into once already.

---

## Limits

**Batch 1 only.** The harness asserts `batch_size == 1`. Acceptance length transfers
to batched serving; wall-clock speedup does not. Our trees score ~0.06 accepted
tokens per token the target must verify, against a chain's ~0.36 — roughly **5×
less compute-efficient per accepted token**. At serving concurrency that ratio is
expected to dominate and invert the result.

**Honest framing:** best-in-class for *latency-bound* speculative decoding — batch 1
to ~8, interactive, local, low-QPS. Unproven and probably unfavourable at
high-throughput serving.

**Measurement.** Two runs of an identical configuration reproduce acceptance to
~0.5% (often exactly — decoding is greedy and seeded) but differ on speed by ~5%
on average and up to 16% on a single dataset. Only 6-dataset means are quoted for
speed; per-dataset speed differences are not claimed.
