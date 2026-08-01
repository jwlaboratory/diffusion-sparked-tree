# Diffusion-Sparked Tree — Findings

Markov-guided draft trees for speculative decoding: DDTree's tree verification
driven by DSpark's markov head instead of DDTree's independence assumption.

Target: **Qwen/Qwen3-4B**. Drafters: `z-lab/Qwen3-4B-DFlash-b16`,
`deepseek-ai/dspark_qwen3_4b_block7`, and our block-16 fine-tune.
All methods greedy-lossless (temperature 0, exact-match prefix acceptance),
verified byte-identical to plain autoregressive decoding.

---

## The headline

Validated on 6 datasets (humaneval, mbpp, gsm8k, math500, mt-bench, alpaca),
12 prompts x 512 tokens each, block-16 drafter, flat-schedule beam tree, budget 64.

| comparison | acceptance | wall-clock |
|---|---|---|
| **vs DSpark** (the method we extend) | **+29.4%** | **+19.8%** |
| vs DDTree (prior SOTA here) | +6.4% | **-5.0%** |

**vs DSpark the win is unambiguous and holds on every dataset.** Same drafter, same
verifier, same losslessness guarantee - tree verification with markov-guided branch
scoring instead of chain decoding.

**vs DDTree the result is workload-split, and we lose on average:**

| dataset | acceptance | speedup | verdict |
|---|---|---|---|
| alpaca | +27.3% | +6.7% | **win both** |
| gsm8k | +11.9% | -1.2% | acceptance win, speed tied |
| mt-bench | +7.3% | -4.5% | acceptance win |
| math500 | -0.2% | -7.8% | loss |
| humaneval | -3.5% | -12.1% | loss |
| mbpp | -4.5% | -10.9% | loss |

The split tracks finding 5: **chat favours markov guidance, structured text favours
DFlash's horizon.** Free-form prose has weak positional structure, so a parallel
drafter's independent guesses fall apart (DFlash gets 2.83 acceptance on alpaca)
and intra-block dependency is worth the most. Code and math suit parallel drafting,
and DFlash's block-16 drafter is better trained there than our 600-step fine-tune.

> **Correction.** An earlier read of this comparison ("+10% acceptance, tied speed")
> came from gsm8k alone at 4 prompts - our single best dataset. It did not
> generalise. Single-dataset extrapolation, not the timing noise, was the error.

---

## Unexpected findings

### 1. Tree width should be FLAT, not front-loaded — the drafter is confident early and uncertain deep

Every intuition (and every geometric schedule) says: branch wide near the root
where the tree is small, narrow with depth. **Measurement says the opposite.**

Recording `(depth, slot)` for every accepted node across 6 benchmarks:

| depth | accepted node was drafter's top pick | slots needed for 95% coverage |
|---|---|---|
| 1 | **87%** | **3** |
| 4 | 66% | 13 |
| 8 | 59% | 14 |
| 12 | 48% | 21 |
| 16 | **30%** | **42** |

Near the root the markov-corrected drafter is nearly always right, so extra width
is wasted. Deep positions are where it is genuinely uncertain — and where hedging
pays.

```
geometric decay=0.6 (my guess):  [26, 15, 9, 6, 3, 2, 1, 1, 1]
measured need:                   [2, 3, 7, 7, 5, 6, 6, 4, 4, 3, 3, 3, 2, 3, 4, 2]
```

Effect at equal budget (64 nodes, gsm8k): acceptance **9.653 → 11.089 (+15%)**,
and it is what took the method from 21% *behind* DDTree on wall-clock to parity.

Corollary: **the exact shape matters less than not decaying.** A hand-flattened
uniform schedule slightly beat the measured one (11.089 vs 10.974). The failure
mode was front-loading, not mis-tuning.

### 2. Markov guidance transfers ONLY to the backbone it was trained with

The head is not a general bigram prior — it is a residual correction fitted to one
specific backbone.

| setting | effect on acceptance |
|---|---|
| head guiding **its own** backbone (DSpark) | **+56% to +78%** (6/6 datasets) |
| head bolted onto a **foreign** backbone (DFlash) | **−7% to −17%** (6/6 datasets) |

Adding DSpark's correction to DFlash's already-calibrated logits double-counts the
signal and distorts them. There is no free lunch: you cannot upgrade an existing
drafter by borrowing someone's head. Joint training is load-bearing.

### 3. An independence-assumed tree on the DSpark drafter is WORSE than no tree at all

| method (block-7, 6-dataset mean) | acceptance |
|---|---|
| dspark chain | 5.234 |
| tree, independence-assumed (`nomkv`) | **3.941** ← worse than the chain |
| tree, markov-guided | 6.652 |

DSpark's backbone is trained *expecting* the markov head to correct it, so its raw
logits are a poor tree substrate. Tree verification is not universally beneficial —
it depends on the drafter's scores being calibrated for the way the tree ranks them.

### 4. Trees benefit ~2× more from a longer horizon than chains do

Extending block 7 → 16 (identical prompts, `ddtree`/`dflash` as 0.0% controls):

| dataset | chain gain | tree gain |
|---|---|---|
| gsm8k | +18.5% | **+37.6%** |
| humaneval | +20.0% | **+25.2%** |

A chain just gets a longer thing to break — one wrong token still ends the round.
A tree gets *more places to hedge* at every new depth. Depth and branching compound
rather than add.

### 5. Chat workloads gain nothing from a longer horizon

Same experiment, alpaca: chain **−0.1%**, tree **+5.1%**.

Alpaca acceptance is only ~3.9 of a possible 8 at block-7, so depth 8+ is rarely
reached; adding depth 9–16 provisions capacity that never gets used. **Chat is
limited by per-position quality, not horizon** — an argument for workload-specific
block sizes, and why the block-7 model was already winning on chat.

### 6. The tree-builder bottleneck was the NUMBER of GPU round-trips, not their size

Markov guidance forces per-node candidate tables (a node's children depend on its
own token), which best-first must resolve sequentially: ~48 round-trips per round
at ~0.26 ms fixed cost each.

Two fixes, only one of which worked:

| fix | what it changes | `tree_build` |
|---|---|---|
| candidate restriction | payload per call (78 MB → 17 MB of `w2`) | 2.73s → 2.16s (**−21%**) |
| level-synchronous beam | call *count* (~48 → 1 sync/round) | 2.16s → **0.61s (−72%)** |

Evidence the payload was never the issue: 512 candidates cost the same as 4096
(2.15s vs 2.22s). I initially diagnosed this as memory traffic and was wrong.

The beam builder fixes tree shape up front, so each depth expands in one batched
matmul and surviving tokens stay on the GPU as a tensor — no sync until the tree is
read back once at the end.

### 7. More training compute made the model WORSE

| run | anchors | steps | epochs | train loss | acceptance (gsm8k) |
|---|---|---|---|---|---|
| A10G γ=4 | 32 | 600 | 9 | 0.847 | **10.520** |
| H100 | 128 | 1000 | **31** | **0.522** | 9.926 (−3.5% after control) |

38% lower train loss, ~3.5% worse acceptance. 1000 steps over 2,280 conversations
is 31 epochs — classic overfitting. **Data is the bottleneck, not compute**, which
is a much cheaper constraint to fix than it appeared.

### 8. `loss_decay_gamma` is a null lever

Predicted that `gamma=4` (tuned for block 7) starves depths 8–16 at block 16,
giving them only ~12% of gradient weight. Doubling to `gamma=8` (~27%) changed
acceptance by **−3.2% to +3.0% — noise**. Adam's per-parameter normalization
largely washes out relative loss reweighting; deep positions are capacity/data
limited, not gradient-starved.

### 9. Batching that breaks best-first ordering destroys acceptance

"Wave" mode (pop K, materialize all, batch their tables, then push) cuts syncs
7–10× — and collapses acceptance (11.45 → 8.13). Deferring pushes means a
successor cannot outrank later members of its own wave, which is exactly the
ordering best-first depends on. The beam builder succeeds where wave failed
because it replaces best-first with a *different principled algorithm* rather than
corrupting it.

### 10. The trained markov head is strikingly interpretable

`bias(prev) = W2 @ W1[prev]` is a rank-256 factorization of the full
151936×151936 bigram logit table. Inspecting the real weights:

- `" New"` → York (+16.9), Zealand, Orleans, Jersey
- `"Dis"` → aster, placement, joint — subword completion, the exact failure mode
  parallel drafting has
- `" import"` → Counter, sys, defaultdict, deque, gcd — a competitive-programming
  fingerprint of the training data
- Strongest-bias tokens are brackets/quotes/indentation (near-deterministic
  successors); weakest are rare CJK/emoji, a 100× dynamic range
- 50% of spectral energy in the top **2** singular directions, 90% in 16 of 256

Magnitudes (mean 0.41, p99 3.0) make it a *tiebreaker*, not an override — it flips
positions where the backbone is uncertain and leaves confident ones alone.

---

## Methodology notes that mattered

**Always run an untouched control.** `ddtree_tb64` uses neither the DSpark model
nor the markov head, so it must be constant across arms. It caught three things
that would otherwise have become false claims:

- **Prompt confound.** `benchmark.py` selects prompts via
  `shuffle(seed=0).select(range(max_samples))`, so runs with different sample
  counts use different prompts. The control read 8.481 / 9.014 / 10.024 across
  three runs of the *same method*. Early block-7-vs-block-16 comparisons were
  invalid for this reason; the clean rerun showed controls at exactly 0.0%.
- **~4% wall-clock noise.** The control's *acceptance* is identical run-to-run but
  its *speedup* varies 6.96–7.23x, which is larger than several margins we care
  about. Hence "tied" rather than "beats" against DDTree.
- **GPU numerics drift.** The control moved −2.2% between A10G and H100 (bf16
  kernels differ, shifting the target's logits and thus generated text), so
  cross-GPU model comparisons must subtract it.

**Correctness before speed.** Every builder variant was checked byte-identical to
plain autoregressive greedy decoding before any performance number was taken, and
the batched builder was proven to produce trees identical to the sequential one
across randomized configs.

---

## Open questions

1. **Why does DDTree lose so much more on H100?** Its speedup fell 6.96x → 3.68x
   while ours fell 5.53x → 4.84x, flipping the ordering. Some of this is expected
   (speculative gains shrink on high-bandwidth GPUs because the baseline is less
   memory-bound) but that should hit both methods similarly. Unexplained; the +31%
   H100 win is not yet a claim.
2. **Does the flat schedule hold across all 6 datasets?** Validated on gsm8k only,
   4 samples. Chat may want a different shape given finding 5.
3. **How much does data scale buy?** Finding 7 says data, not compute, is binding.
   10–50× more conversations at ~10 epochs is the untested lever.
4. **Is beam's residual acceptance gap (11.09 vs 11.45 best-first) intrinsic?**
   A fixed schedule cannot spend budget where the drafter happens to be uncertain
   *on this particular round*. An adaptive-but-batched builder might recover it.

---

## Reproduction

```bash
# tree-slot statistics -> width schedule
modal run --detach training/modal_train.py::collect_tree_stats
python experiments/analyze_tree_slots.py stats.json --budget 64 --plot out.html

# schedule comparison
modal run --detach training/modal_train.py::bench_schedule

# clean block-7 vs block-16 (with controls)
modal run --detach training/modal_train.py::bench_horizon

# full 6-dataset sweep, one GPU container per dataset
modal run --detach ddtree/modal_run.py::wide
```

Implementation: [`ddtree/ddtree_markov.py`](../ddtree/ddtree_markov.py)
(`build_markov_tree` best-first, `build_beam_tree` level-synchronous),
[`ddtree/dspark.py`](../ddtree/dspark.py) (DSpark chain decoding),
[`ddtree/model/dspark.py`](../ddtree/model/dspark.py) (drafter, verified
bit-identical to DeepSpec's reference).
