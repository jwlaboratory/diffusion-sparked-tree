# Extending the block size: SparklingTree saturated b=7

Like we showed earlier, DSpark + markov corrector + tree pushes the stock
drafter to the edge of its horizon: per-depth acceptance is still **0.93 at
depth 7** — the last token it's allowed to draft. The drafter isn't running out
of skill, it's running out of runway.

**Figure 0 — the ceiling, directly**
(`results/datascale/report_ceiling_saturation.png`): the distribution of
accepted-tokens-per-round. On gsm8k, **95% of b7 rounds land exactly on the
hard cap of 8** (7 drafted + the verifier's bonus token); on humaneval, 79%.
That is a distribution slammed against a wall, not a drafter at its skill
limit — and the same plot shows the contrast case: on mt-bench only 12% of
rounds touch the cap, so the ceiling was never binding for chat (which
foreshadows exactly where block 16 will and won't pay off). The only way past
the wall is a longer block.

## Retraining DSpark for block 16

`block_size` only controls how many mask tokens are appended per anchor during
training — it changes no parameter shapes. So we warm-started from DeepSpec's
released block-7 checkpoint (`load_state_dict(strict=True)` passes) and
fine-tuned at **block 16**, raising the per-round ceiling from 8 to 17 accepted
tokens. The markov head and confidence head are trained jointly, with the CE
loss computed on markov-corrected logits, so the backbone learns *with* the
correction it will decode with. The fine-tune is almost embarrassingly cheap:
600 steps on 2,400 PerfectBlend conversations, single A10G, a couple of hours.

- Checkpoint: [huggingface.co/shreybirmiwal/Qwen3-4B-DSpark-b16](https://huggingface.co/shreybirmiwal/Qwen3-4B-DSpark-b16)
- Training recipe + Modal pipeline: `experiment2-block16/training/`

One decoding-side change was needed: at a fixed tree budget, a 16-deep tree
spreads the same nodes over more depths (trades width for depth), so we sweep
the budget (64 and 256) to separate "block size helps" from "we under-budgeted
the tree". The bigger budget buys the width back — and the b16 gains grow with it.

## Results (b16 + its own markov head + tree, vs b7 + its head + tree)

**Figure 1 — per-depth acceptance, b7 vs b16** (`report_per_depth_b7_vs_b16.png`):
the extended horizon actually works. Past depth 7, conditional acceptance holds
at **0.85–0.89 through depth 12** and tapers gently to ~0.72–0.81 at depths
14–16. There is no cliff after the old horizon. (Measurement note: compute
per-depth rates from raw acceptance lengths pooled across datasets/budgets —
averaging per-cell rates at deep depths, where only a handful of rounds
survive, manufactures phantom dips.)

**Figure 2 — acceptance change by dataset** (`results/block_size_by_dataset.png`):

| dataset | budget 64 | budget 256 |
|---|---|---|
| gsm8k | **+47%** (7.77 → 11.42) | **+60%** (7.89 → 12.62) |
| humaneval | **+28%** (7.19 → 9.19) | **+36%** (7.54 → 10.28) |
| mt-bench | −0% (4.08 → 4.07) | −3% (4.57 → 4.44) |

Two insights worth calling out:

1. **The markov corrector goes from "nice win" to load-bearing at block 16.**
   With the head off, b16 is *worse* than b7 (−22% to −29% across datasets):
   a 16-token draft gives drafter–target divergence twice as long to compound,
   and the per-parent correction is what keeps deep chains on-manifold. Block
   size and the corrector are not independent upgrades — the longer horizon is
   only usable *because* of the head.
2. **The gains concentrate where continuations are constrained.** Math and code
   are predictable 16 tokens out; open-ended chat (mt-bench) is not — its mean
   acceptance (~4) sits well under the *old* ceiling, so extra runway can't
   help. Block 16 widens the ceiling; it doesn't make chat more predictable.

## Sanity check: the recipe is not data-starved

Since the fine-tune uses only 2,400 conversations, we checked whether the deep
depths were data-limited: three arms with identical config, varying only rows
(2.4k / 10k / 24k, steps epoch-matched at ~5). Pooled mean acceptance: **7.67 /
7.60 / 7.59** — flat at 10× data (`results/datascale/`). Deep-depth acceptance
is bounded by drafter capacity or intrinsic horizon difficulty, not data, and
the cheap 2,400-row recipe is all it takes.
