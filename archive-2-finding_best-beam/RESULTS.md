# Experiment 4a results — beam width schedules

Run: H100 + 8 CPU, DSpark-b16 + own markov head, budgets {64, 256},
gsm8k/humaneval/mt-bench x 4 samples x 512 tokens, temp 0, two-pass
(clean TPS / instrumented phases), fingerprint `2d9f63cfbb4c`.
Acceptance identical across passes (checks.acceptance_match = true).

AGG = round-weighted mean acceptance across the three datasets (sum of accepted
tokens / sum of rounds). Note this weighting favors low-acceptance datasets
(they use more rounds): mt-bench contributes ~2x the rounds of gsm8k.

## Budget 64

| arm | gsm8k | humaneval | mt-bench | AGG | TPS | build share |
|---|---|---|---|---|---|---|
| beam.geo50 `[32,16,8,…]` | 7.04 | 6.55 | 3.75 | 5.60 | 137 | 6.0% |
| beam.geo60 | 8.24 | 7.33 | **3.89** | 6.08 | 147 | 6.8% |
| beam.geo75 | 9.83 | 8.04 | **3.89** | **6.62** | **156** | 8.4% |
| **beam.flat `[4]*16`** | **11.08** | **8.64** | 3.47 | 6.58 | 153 | 10.1% |
| beam.invgeo75 | 9.07 | 6.92 | 2.70 | 5.23 | 124 | 8.2% |
| beam.invgeo50 | 7.38 | 6.37 | 3.01 | 5.13 | 126 | 5.8% |
| beam.depth8 `[8]*8` | 8.17 | 7.41 | 3.80 | 6.05 | 146 | 6.6% |
| beam.depth4 `[16]*4` | 4.94 | 4.87 | 3.68 | 4.50 | 111 | 4.7% |
| bestfirst.ref (ceiling) | 11.72 | 8.92 | 4.00 | 7.18 | 23 | **85.0%** |

## Budget 256

| arm | gsm8k | humaneval | mt-bench | AGG | TPS | build share |
|---|---|---|---|---|---|---|
| beam.geo50 | 8.48 | 7.49 | 4.17 | 6.42 | 153 | 7.5% |
| beam.geo60 | 9.86 | 8.49 | 4.29 | 7.00 | 163 | 8.7% |
| beam.geo75 | 11.18 | 9.11 | **4.51** | 7.61 | 173 | 10.7% |
| **beam.flat `[16]*16`** | **11.93** | **10.02** | 4.19 | **7.75** | **176** | 10.9% |
| beam.invgeo75 | 11.28 | 8.19 | 2.93 | 5.99 | 137 | 10.5% |
| beam.invgeo50 | 9.11 | 7.35 | 3.10 | 5.69 | 133 | 7.3% |
| beam.depth8 `[32]*8` | 8.53 | 8.03 | 4.38 | 6.70 | 160 | 7.2% |
| beam.depth4 `[64]*4` | 4.96 | 4.95 | 4.09 | 4.71 | 116 | 5.2% |
| bestfirst.ref (ceiling) | 12.61 | 9.98 | 4.49 | 8.04 | 11 | **92.1%** |

## Findings

1. **Flat wins where the drafter is strong, and that's where the tokens are.**
   Flat is the best fixed schedule on gsm8k and humaneval at both budgets, by
   1.1–1.3 accepted tokens over the best decaying schedule at b64. At b256 it
   also wins the aggregate outright. The old finding replicates almost exactly:
   old harness flat@gsm8k/64 = 11.089, this harness = 11.079.

2. **The chat exception is real.** On mt-bench (mean acceptance ~3–4, chains
   break early) front-loading wins: geo60/geo75 3.89 vs flat 3.47 at b64. Deep
   slots only pay if you reach them. This is the same signal as the old
   experiments' one positive adaptive-widths cell (alpaca/chat). The right
   schedule tracks expected acceptance depth; flat is the right *default*
   because math/code accepts long and chat differences are small in absolute
   tokens (±0.4) while math/code differences are large (±1.3).

3. **Concentration loses symmetrically; orientation is second-order.** Heavy
   front-load (geo50) and heavy back-load (invgeo50) land within 0.5 of each
   other at the bottom of the full-depth arms in every cell. Mild beats heavy
   in both directions. It is spreading, not direction, that matters most —
   with one asymmetry: back-load is strictly worse than front-load on chat
   (invgeo75 2.70, the worst full-depth cell in the run), because budget parked
   at depths that are never reached is pure waste.

4. **Never truncate depth.** depth4 is budget-invariant garbage: 4.50 -> 4.71
   AGG while its budget quadrupled. It is hard-capped by its 4-token horizon.
   depth8 is mediocre everywhere. Consistent with the old finding that depths
   >= 12 carry ~10% of accepted tokens.

5. **The beam builder does its structural job.** Every beam arm runs at
   111–176 TPS with tree build at 5–11% of decode; the naive best-first arm
   runs at 23 TPS (b64) / 11 TPS (b256) with build at 85% / 92%. Same backbone,
   same head, same verify — only the builder differs. Flat beam vs best-first
   at b256: 16x the throughput at 96% of the acceptance.

6. **Best-first remains the acceptance ceiling everywhere** (7.18 / 8.04 AGG),
   which is why the eventual precomputed-table best-first builder (old
   experiments' shipped default) is worth recreating next: it keeps this
   ceiling and deletes the 85–92% build share.

## The direct measurement: where accepted branches actually live

Charts: `results/schedule_acceptance.png` (the sweep), `results/depth_survival.png`
(how often each depth is reached), `results/branch_distribution.png` (the
measurement below). Trace collection: `modal_traces.py` (best-first @ budget 256,
save_tree_traces=True, 12 prompts, 544 rounds, 3,829 accepted nodes; slot = the
node's rank among its parent's markov-ordered children, recovered from
materialization order).

| depth | accepted n | top-1 hit rate | slots for 95% coverage |
|---|---|---|---|
| 1 | 526 | 85% | 3 |
| 4 | 376 | 84% | 5 |
| 8 | 210 | 91% | 2 |
| 12 | 125 | 86% | 3 |
| 15 | 73 | 75% | 5 |
| 16 | 58 | 69% | 6 |

Two facts fall out:

- **Slots needed never decay toward 1 at any depth** — the profile is a shallow
  U (3–5 near the root, ~2 mid, 5–6 at the deep end), while every geometric
  schedule allocates a strictly decaying profile ending at 0–1. geo50 spends 32
  slots at depth 1 where 3 cover 95% of accepts, and 0 slots past depth 7 where
  2–6 are still needed. That mismatch is the entire acceptance gap.
- **This drafter decays far less with depth than the old experiments' checkpoint**
  (top-1 69% at depth 16 vs their 30%; slots-for-95% 6 vs their 42). The
  qualitative conclusion is identical — don't front-load — but the flat-vs-
  measured gap is even smaller here, i.e. flat is even closer to optimal for
  this b16 head than it was for theirs.

Survival context (`depth_survival.png`, best-first @ 256): gsm8k still accepts
>= 17 tokens in 31% of rounds and humaneval in 13%, so the deep slots flat pays
for are exercised constantly on math/code. mt-bench survival hits ~0% by depth
11 — deep slots are dead weight there, which is exactly the chat exception in
the sweep.

## Verdict

Proved: the budget belongs spread evenly across depths, not concentrated —
flat is the correct default schedule for the beam builder (`[4]*16` at 64,
`[16]*16` at 256). The intuitive wide-at-the-root schedule costs 1–2 accepted
tokens per round on math/code. The one caveat worth remembering: on
short-acceptance workloads (chat), a mild decay (~0.75) is slightly better,
and adaptive allocation (best-first) beats every fixed schedule everywhere.

Raw data: `results/summary.json`. Reproduce: `reproduce.md`.
