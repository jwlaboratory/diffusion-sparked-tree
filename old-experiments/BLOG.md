# SparklingTree: branch-conditional draft trees for block-diffusion speculative decoding

**TL;DR.** DFlash drafts a whole block in one diffusion pass instead of one token at
a time. DDTree turns that block into a *tree* so the target can hedge across several
continuations at once. DSpark adds a tiny markov head that conditions each draft
position on the previous one. SparklingTree combines them: a draft tree whose
branches are scored *conditionally on the branch you are actually on*. On Qwen3-4B
it is **17.8% faster than DDTree** at matched tree budget and **34% faster than
DFlash**, with acceptance gains that grow at temperature 1.0 (**+8.9%, 6/6
datasets**).

---

## 1. Why block diffusion drafters exist

Speculative decoding: a small *draft* model proposes several tokens, and the large
*target* model verifies them in one parallel forward pass. If the target agrees with
the first *k* drafts, you got *k* tokens for the price of one forward pass. Nothing
is approximated — the target's own decoding rule decides what is kept, so the output
is what the target would have produced anyway.
(Background: [speculative decoding from first principles](https://jwlabs.vercel.app/post/speculative-decoding-first-principles).)

The catch is the drafter. Traditionally it is itself autoregressive — MTP, Medusa,
EAGLE — so drafting *k* tokens costs *k* sequential passes. Worse than linear, in
fact, since attention grows with context. The "free" drafter stops being free
exactly when you ask it for the long drafts that make speculation pay.

> **[FIGURE 1]** Draft latency vs draft length: autoregressive drafter climbing
> superlinearly, block-diffusion drafter flat.

**DFlash** replaces the autoregressive drafter with a *block diffusion* model. It
denoises an entire block of 16 positions in one pass, so draft length is nearly free
— generating 16 costs about what generating 1 costs. Per-token draft quality is
lower than an autoregressive drafter's, but you get 16 of them, and in speculative
decoding quantity at fixed latency beats quality.

What DFlash actually produces is a table: one row per block position, one column per
vocabulary entry, holding the **marginal** probability of each token at that
position.

> **[FIGURE 2]** The DFlash output table — block_size rows × vocab columns.

That word *marginal* is the whole story of this post. Row *d* tells you what token
is likely at position *d* **averaged over everything that could happen at the
positions before it**. It does not tell you what is likely at position *d* *given*
that position *d−1* came out a particular way.

DFlash uses the table the obvious way: take the argmax of each row, and you have a
16-token chain to verify.

---

## 2. DDTree: stop betting on one chain

A chain has one point of failure. If draft 3 is wrong, drafts 4–16 are discarded
regardless of how good they were. Acceptance is a prefix, so a single early mistake
costs the whole block.

**DDTree**'s move is to spend the verification budget on a *tree* instead. The target
can verify *B* nodes in one forward pass whether they form a chain or a tree, so you
may as well hedge: put several candidates at position 1, several under each of those
at position 2, and let the target walk whichever path it agrees with.

The elegant part is how DDTree picks the tree. Score a path by the product of its
per-position marginals, and the best *B*-node tree is just the *B* highest-scoring
prefixes — obtainable by a best-first search over the table. One drafter pass, one
tree, no extra model calls.

That relies on an assumption, stated plainly in the paper: the distribution at depth
*d* is treated as **independent** of which token an ancestor holds. It is what makes
DDTree fast — every node at a given depth shares one top-*k* table, so the whole tree
is a handful of lookups.

It is also, strictly, false.

---

## 3. DSpark: the marginals are missing something

**DSpark** attacks the same weakness from the drafter side. It keeps the block
diffusion backbone but adds a **rank-*r* markov head**: a small low-rank correction
that shifts the logits at position *d* based on the token drafted at position *d−1*.

```
corrected_logits[d] = base_logits[d] + W2 @ W1[token[d-1]]
```

Two matrices, a few hundred MB, no attention. DSpark sweeps it serially across the
block — 16 tiny steps, no backbone work — so draft *k+1* is conditioned on draft *k*.
The backbone still runs exactly once.

This recovers some of what the marginals threw away, and it is cheap. But DSpark
applies it to a **chain**, so it still bets everything on one path.

---

## 4. SparklingTree: the two ideas want each other

Here is the observation the whole project rests on.

**DDTree assumes exactly what DSpark's head measures.** DDTree says depth *d+1* is
independent of depth *d*. DSpark's head is a trained estimate of precisely that
dependence. One method's central approximation is the other's central quantity.

So: build DDTree's tree, but score each branch with DSpark's correction applied
*conditionally on that branch*. Every node gets its own children distribution instead
of sharing a per-depth table with its siblings.

> **[FIGURE 3]** Same budget, two trees. DDTree: every sibling offers its children
> the same candidates. SparklingTree: each node's children are re-scored by the token
> that node actually holds.

### Why this is not free

DDTree's independence assumption isn't laziness — it's what makes the tree cheap. One
top-*k* per depth serves every node. Drop it and every node needs its own bias
correction and its own top-*k*.

Our first implementation did exactly that, lazily, one node at a time. It worked and
it was **hopeless**: each materialized node needed a table that could not even be
*requested* until the previous node was popped, so the builder paid ~*B* dependent
GPU round-trips per round. Tree construction went from DDTree's 0.10s to **2.15s**.
The acceptance gain was real and the wall-clock gain was negative.

### Three attempts at making it cheap

**Attempt 1 — batch the walk.** Pop *K* nodes at once, expand them together. Fewer
syncs, same tree… except not the same tree: deferring the pushes means a successor
can't outrank later members of its own wave, which is exactly what best-first *is*.
Acceptance collapsed 11.45 → 8.13. Abandoned.

**Attempt 2 — fix the shape up front (beam).** Decide the width at each depth in
advance, expand a whole level in one batched matmul. One sync per round instead of
*B*. This works, and it forced a useful measurement: we logged where accepted nodes
actually sat in the ranking, and found the drafter is confident near the root (87% of
depth-1 acceptances are its top pick) and uncertain deep (34% at depth 16) — the
opposite of every decaying schedule. A **flat** width schedule beat every geometric
one.

**Attempt 3 — precompute the transitions.** The level loop was still 16 *dependent*
matmuls: level *d* can't start until level *d−1*'s top-k lands. But a parent at depth
*d* is always drawn from the depth *d−1* candidate set — the beam has nothing else to
choose from. So the entire transition is a finite **C × C table per depth**, and the
whole `[L−1, C, C]` stack is one batched matmul with no dependence on the beam at all.
The normalising `logsumexp` depends only on (depth, parent slot), so it folds in too.
What remains per level is a gather, an add and a top-k.

That last idea also rescues best-first. Its problem was never arithmetic — it was that
each node's table was unknown until the previous pop. With every table precomputed,
the whole heap runs on CPU against **one** transfer.

| builder (H100, budget 64) | ms/round |
|---|---|
| lazy best-first | 9.55 |
| in-loop matmul, beam | 4.03 |
| **precomputed table, beam** | **1.88** |
| **+ CUDA graph** | **0.94** |
| **precomputed table, best-first** | **3.82** |

### The knob that bites

The table is **C × C**, so the candidate pool became a *quadratic* cost where it used
to be linear. Shrinking it is tempting and it is not free — measured three independent
ways, cutting the pool costs ~3–4% acceptance. We settled on C=512 and took the trade:
**−4.1% acceptance for a 52% cheaper builder**, which nets out strongly positive
end-to-end. And the penalty for a large C is far worse on an A10G than an H100 — the
table swaps memory traffic for compute, and only the bigger card has the headroom.

One counterintuitive result worth keeping: the **fastest** builder is not the best
one. The CUDA-graphed beam builds a tree in 0.68s against best-first's 1.70s and
*loses* — 7.53 vs 7.83 acceptance, ending up slower overall. Builder time is ~5% of a
round; acceptance decides how many rounds you need.

---

## 5. Results

Qwen3-4B · H100 · block 16 · greedy · 6 datasets × 12 prompts × 512 tokens · batch 1.

| | acceptance | speedup |
|---|---|---|
| DFlash (chain) | 5.570 | 2.50× |
| DSpark (chain) | 6.065 | 2.80× |
| DDTree (tree) | 7.771 | 2.84× |
| **SparklingTree** | **8.225** | **3.35×** |

| SparklingTree vs | speedup |
|---|---|
| DFlash | **+34.2%** |
| DSpark | **+19.6%** |
| DDTree | **+17.8%** |

The advantage holds at every tree budget from 64 to 512, and best-config vs
best-config it is +10.2%. It **grows at temperature 1.0** — +5.8% → **+8.9%**
acceptance, winning **6/6 datasets** — which is the behaviour the theory predicts:
under sampling the next token genuinely depends more on the one just drawn, so
independence costs more.

Where the win comes from is visible in the stage times. Our tree costs *more* to build
(2.13s vs DDTree's 0.44s) and that is dwarfed by what higher acceptance saves in
verification (**29.7s vs 39.1s**). Fewer rounds; verification is where time goes.

### The ablation that matters

Same drafter, same budget, only the scoring rule changes:

| scoring | acceptance |
|---|---|
| independence (DDTree's rule) | 3.115 |
| **branch-conditional (ours)** | **7.955** |

**+155%.** And the same head applied to a *different* backbone makes things worse
(−14.4%) — it is not a general bigram prior, it is a backbone-specific residual.

---

## 6. What this does not show

I spent a full session trying to break this result. Some attacks failed; two landed,
and they bound the claim.

**It is not a portable tree builder.** The scoring needs a drafter trained to leave a
residual for the markov head. Applied to DFlash's drafter it *loses* 14%. This is a
coupled drafter+tree system, not a drop-in improvement to DDTree.

**Our drafter is fine-tuned; DDTree's is not.** We warm-started a block-16 DSpark
drafter on chat data. At matched horizon it is no better than off-the-shelf DFlash-b16
on humaneval, mbpp and math500 — and those are exactly the three datasets where DDTree
*beats* our tree at greedy. Our biggest wins are on chat, which is what we trained on.
The decisive experiment — fine-tune DFlash the same way and re-compare — has not been
run. Until it is, "our tree is better" and "our drafter saw data theirs didn't" both
fit. The strongest evidence for the former is temperature 1.0, where we win the
matched-drafter datasets too.

**Small samples inflated the headline.** Alpaca, which carries our mean, went from
+37.8% to +25.2% when we quadrupled the prompts. The 6-dataset mean drops +5.8% →
+5.0%. The reference papers use 80–164 prompts per dataset; we use 12–48.

**"Byte-identical to autoregressive decoding" is not true — for anyone.** We checked,
and the target model disagrees with *itself* 0.50% of the time depending on how many
positions it processes at once, with no drafter involved. That is bf16 accumulation,
and it means no speculative method in this harness can claim byte-identity. The
acceptance numbers are unaffected — exact-match acceptance compares against the
target's argmax in the pass that verifies — but the wording is wrong wherever it
appears, including in prior work's framing.

**Batch 1 only.** Our trees score ~0.064 accepted tokens per token the target must
verify; a chain scores ~0.36 — roughly **5× less compute-efficient per accepted
token**. At batch 1 the GPU is idle so wide verification is nearly free. At serving
concurrency it is not, and the ranking against *chains* is expected to invert. Against
DDTree it should hold, since at matched budget we verify the same width and accept
more — but that is an argument, not a measurement.

**One target model.** Qwen3-4B only. The drafter is architecture-bound, so 8B/30B
would need new training runs.

Honest framing: **best-in-class for latency-bound speculative decoding** — batch 1 to
~8, interactive, local, low-QPS — on Qwen3-4B, with a drafter you have to train.
Unproven at high-throughput serving, and not a portable replacement for DDTree's tree
builder.

---

## Appendix: things that did not work

| idea | outcome |
|---|---|
| Wave batching (pop K, expand together) | 11.45 → 8.13 acceptance — broke best-first ordering |
| Confidence-head adaptive widths | −2.7% acc, −6.6% speed — head predicts *chain* acceptance, a tree needs *coverage* |
| Cross-model markov head | −14% — backbone-specific residual, not a general prior |
| Truncating tree depth to 12 | no measurable gain; depths ≥12 carry 9.7% of accepted tokens |
| Shrinking candidate pool to 256 | −3.6% acceptance for no speed gain |
| 4× training data | worse on 6/6 — but four variables changed at once, unattributable |
