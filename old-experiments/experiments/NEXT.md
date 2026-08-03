# Job queue — path to final results

Ordered. Each stage lists its blocker, so nothing starts before its input exists.

---

## Stage 1 — finish the drafter (RUNNING)

`pipeline_h100 --exp-suffix _bigdata` on treehacks/H100:
9,500 conversations (4x prior), ~10 epochs, 1400 steps, 96 anchors, seq 512.

Targets finding 7 (2,280 conversations at 31 epochs overfit) and the one
remaining loss: **humaneval/mbpp**, where DFlash's chain already beats DSpark's,
suggesting a drafter-quality gap rather than a method gap.

**Prediction to check:** gains should concentrate on code/math and barely move
alpaca (where we already win by 20-40% and the limit is per-position quality).

When it lands, one command does publish + full sweep:

```bash
DDTREE_BIG_GPU=H100 modal run --detach training/modal_train.py::finalize \
    --exp-suffix _bigdata
```

---

## Stage 2 — transformers-harness final sweep (blocked on stage 1)

`finalize` runs, on H100, all 6 datasets:

| axis | values |
|---|---|
| methods | baseline (no drafter), dflash, ddtree, dspark, **sparked-tree** |
| tree budget | 32, 64, 128, 256 |
| datasets | humaneval, mbpp, gsm8k, math500, mt-bench, alpaca |

This is the strongest claim available **without** SGLang, and it is single-sequence
only — no concurrency. That limit is the whole reason stage 4 exists.

---

## FINAL BENCHMARK — queued, fires automatically

`queued_final` is running detached on treehacks. It polls the volume for the
bigdata checkpoint, publishes it, then runs the final sweep. No manual step.

| | |
|---|---|
| GPU | **H100** (reference) |
| Drafter | **bigdata block-16** (9,500 conversations, ~10 epochs) |
| Builder | level-synchronous beam, flat width schedule |
| Methods | baseline (no drafter), dflash, ddtree, dspark, **sparked-tree** |
| Datasets | all 6, 12 prompts x 512 tokens |
| Tree budget | 64, 128 |

```bash
DDTREE_BIG_GPU=H100 modal run --detach training/modal_train.py::queued_final
```

Falls back to the newest `step_*` checkpoint if the trainer dies before its own
publish step. Results land in `/vol/results/sweep_*.json` (treehacks volume).

### Reference run (old model)

A matching sweep on the previous best checkpoint (`_best`) is also running. Not
the headline - it exists so "did 4x more data help?" is answerable by diffing two
otherwise identical runs.

---

## QUEUED — confidence-head adaptive tree

`queued_confidence` polls for the same bigdata checkpoint, then runs the ablation.

**Idea.** The flat schedule is the *average* optimal allocation, measured across
many rounds. DSpark's confidence head predicts P(accept) per position for the
*current* round, so width can adapt per-round instead:

```
reach[d] = prod_{j<d} p_j       # do not widen a depth we will never reach
value[d] = reach[d] * (1 - p_d) # do not widen a depth we are already sure about
W_d     ∝ value[d]
```

Measured behaviour of the schedule (budget 64, depth 16):

| round type | widths |
|---|---|
| flat (current) | `[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]` |
| easy (high confidence) | `[2,3,4,6,6,8,6,6,5,5,4,3,2,2,1,1]` — spreads deep |
| hard (low confidence) | `[19,20,14,7,3,1]` — abandons depth 7+, hedges hard early |

Arms are identical apart from width selection: **beam+flat vs confidence-adaptive**,
4 datasets x budgets 64/128. Cost is one extra head forward plus a chain sample
for prev-token context (~0.2-0.45s/run, measured).

Verified lossless on CPU before queueing.

---

## Standing methodology rules

- **H100 is the reference GPU.** Earlier A10G/H100 mixing confounded hardware with
  checkpoint; never compare across GPUs without holding the checkpoint fixed.
- **Always include an untouched control.** `ddtree_tb64` and `dflash` never touch
  the DSpark model. They caught a prompt-selection confound, ~4% wall-clock noise,
  and cross-GPU numerics drift.
- **Losslessness before speed.** Every variant must be byte-identical to plain
  autoregressive greedy before any timing is reported.
- **Never launch `modal run` piped into `head`** — SIGPIPE kills the client before
  it submits the function; the app builds an image and then does nothing.

---

## Deferred (not planned)

- **SGLang integration** — would give real serving numbers at concurrency. Both
  halves exist there (DSpark drafter with markov head; EAGLE tree kernels) but
  nothing joins them; that join is ~3-5 days of work. Dropped for now.
- **Concurrency / batch-size results.** The transformers harness is single-sequence
  (`assert batch_size == 1`), so every number here is batch-1. Acceptance length
  transfers to batched serving; wall-clock speedup does not.
