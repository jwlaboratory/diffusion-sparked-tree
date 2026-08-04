# Archived: gating tree width on a confidence head

**Verdict: does not work. Archived rather than pursued.**

The idea was to cut verification cost by spending tree width only on rounds that
convert — DSpark's confidence head, but tree-aware. It was measured on real
rounds and it fails on two independent axes. This document records why, because
the reason turned out to be more general than the idea.

> **Status: complete.** All 3 datasets, 1082 rounds (gsm8k 232, humaneval 421,
> mt-bench 429). An earlier draft of this document reported the 2-dataset
> partial; mt-bench did not reverse it.

---

## The problem being attacked

The tree verifies 65–129 positions per round to accept ~8 tokens; a chain
verifies 17 to accept ~6. That ~5× compute-efficiency gap
(`old-experiments/RESULTS.md` §9) is what makes trees lose above concurrency ≈4.
If width could be spent only on rounds where it converts, the gap shrinks.

`investigate/FINDINGS.md` established offline that this is not obviously hopeless:
whether width pays *this round* is strongly predictable in principle (AUC
0.71–0.82 from the round's own state), history is useless (AUC ≈ chance), and an
estimator would only need ±1.5–2 tokens of accuracy.

It also established the direction, which is counterintuitive and load-bearing:
**shrink when the drafter is CONFIDENT.** Rounds where the chain accepts ≤3 are
64% of rounds and carry 71% of everything tree width produces, so cutting budget
on uncertain rounds destroys exactly the value it is trying to save.

## What was built

| file | |
|---|---|
| `DDTree/confidence.py` | `round_confidence` (free, one softmax over already-materialized logits) + `TreeAcceptanceHead` (trained, tree-aware) |
| `DDTree/sparked_tree.py` | `measure_confidence` capture + `confidence_gate` |
| `DDTree/run_experiment.py` | `Method.gate` / `Method.head`, strict head loader, per-round capture |
| `modal_benchmark.py` | stage 1, measurement — **ran** |
| `train_head.py` | trains the head and tests whether it earns its place |
| `modal_gated.py` | stage 2, gated vs ungated — **never run**, see below |
| `investigate/` | the offline analysis that motivated all of it, plus 7 CPU tests |

The per-round capture emits `confidence_by_round` paired positionally with
`acceptance_lengths` **from the same run**. That was the methodological point:
the offline analysis had to align two different runs on token index, which
degrades as bf16 divergence accumulates and is what killed its finding 1.

## The result

1082 rounds, block-16 drafter, budget 64. Task: predict this round's **tree**
acceptance from the free confidence features.

| predictor | MAE (tokens) |
|---|---|
| raw `pred_chain_len` | 5.25 |
| **affine-recalibrated free estimator** | **3.18** ← the honest bar |
| trained tree-aware head | 2.99 *(random split — leaks)* |

Leave-one-dataset-out, which is the read that matters:

| held out | n | affine base | head | better by |
|---|---|---|---|---|
| gsm8k | 232 | 4.046 | 3.969 | +0.077 |
| humaneval | 421 | 3.172 | 3.269 | **−0.097** |
| mt-bench | 429 | 3.917 | 2.927 | **+0.990** |
| | | | **mean** | **+0.323** |

The mean looks positive, and it is not. Three checks, and it fails all three:

1. **Nothing meets the spec.** Held-out MAE is 2.93–3.97 tokens against ≤2.0 —
   the range the noise study calls "buys almost nothing". This alone is fatal.
2. **The mean is one cell.** mt-bench carries the entire +0.323; gsm8k is noise
   (+0.077) and humaneval is *negative* (−0.097).
3. **It loses on the leaky split.** On random 80/20, where rounds from the same
   prompt appear in both train and val and the head should look its best, it is
   **worse** than the calibrated baseline (−0.041).

(3) is what explains (2). If the head had learned something real, the leaky split
would show it most clearly. That it does not means the mt-bench cell is the
*affine baseline transferring badly* to a workload unlike the two it was fit on,
which the MLP's nonlinearity partially absorbs. That is not tree-awareness.

> A first version of the verdict logic in `train_head.py` checked only the mean
> gain and a loose MAE floor, and **green-lit this head**. It now requires all
> three checks. Worth recording: an aggregate that looks like a pass can be one
> workload wide.

### A baseline error worth recording

The first version of `train_head.py` compared the head against **raw**
`pred_chain_len`, which is biased by −6 to −7 tokens because it estimates *chain*
acceptance while the label is *tree* acceptance. The head would have shown a
+3.5 token "win" earned entirely by learning that constant offset.

A threshold gate is invariant to affine rescaling, so that win would have been
meaningless. The baseline was changed to a least-squares recalibration fit on
train and scored on val (`fit_affine`), which is the floor the head actually has
to clear. It does not clear it.

### Independent structural confirmation

`price_gate.py` reaches the same conclusion by another route, and this one does
not depend on any model. Gating is only safe on rounds already at the block
ceiling — those needed no width. But the "% gated at ceiling" column never gets
high:

| dataset | estimator sd | best % gated at ceiling | at what saving |
|---|---|---|---|
| gsm8k | 4.01 | 38% (T=5.0) | −10.8% width |
| humaneval | 3.75 | ~20% | −18% width |
| mt-bench | 2.97 | **2.4%** (T=1.0) | −50.6% width |

mt-bench is the clearest failure: at a threshold that would save half the width,
gated rounds were accepting **5.92 of a possible 17** — deep in the region where
the tree is still doing work. At any threshold that saves meaningful width, most
gated rounds were **still using depth** and would lose tokens.

## Why it fails — and this is the transferable part

The diagnosis in `old-experiments/RESULTS.md` §11 was that the confidence head
predicts *chain* acceptance while a tree needs *coverage*. That framing implied a
fix: make the head tree-aware. We did, and it did not help.

The actual reason is one level down. **Tree acceptance is decided by which of 64
branches the target picks.** Drafter confidence is a fact about the *drafter*. The
tree's entire value is hedging against the drafter being wrong — and how well that
hedge lands is a property of the *target*, which is not visible anywhere in the
drafter's logits.

So the mismatch was never chain-vs-tree. It is **drafter-side signal vs
target-side outcome**, and no amount of tree-awareness on the drafter side closes
it. That also explains the original −2.7% / −6.6% failure better than the
chain-vs-tree story did.

This is the third time this project has been wrong about where to spend tree
budget in the same direction (see `old-experiments/SPEEDUP_EXPERIMENTS.md`:
geometric width schedules, confidence-adaptive widths, and now this). The
recurring error is assuming the drafter knows how much help it needs.

## Stage 2 was deliberately not run

`modal_gated.py` is complete and correct, including a `small_always` floor arm —
without it, "gate beats ungated" is indistinguishable from "a smaller tree is just
better here", which have opposite implications.

It was not run. `train_head.py`'s own verdict says the signal is not viable, and
spending H100 time to confirm a negative that two analyses already agree on is not
a good trade. If the gate is ever revisited, the threshold must come from
`price_gate.py` and not from a guess — `GATE_THRESHOLD` is left as `None` and
raises rather than defaulting.

## What is worth keeping

**The instrumentation.** `confidence.py` and the per-round capture cost one
softmax over logits that are already materialized — no extra forward pass — and
they are the field every prior analysis in this repo was missing. Any future
question of the form "what was true about this round?" now has an answer that
does not require aligning two runs.

**The negative itself.** "Predict from the drafter whether the target will need
the hedge" is a natural idea that will occur again. It is now measured.

## What to do instead

The width problem is real; this signal cannot solve it. The alternative that does
not require predicting anything is the **device-resident builder** from
`concurrency/FINDINGS.md` — it attacks the same concurrency ceiling by removing
host-side work rather than by guessing which rounds deserve width.

## Reproduction

```bash
modal run --detach modal_benchmark.py            # stage 1, ~25 min on A100
python3 investigate/price_gate.py results/rounds.json --ceiling 17
python3 train_head.py results/rounds.json --out heads/tree_accept.pt
python3 investigate/test_confidence.py           # 7 CPU tests, no GPU needed
```

Units are checkpointed per `(budget, dataset)` and the volume committed after
each, so an interrupted run resumes. This mattered: the first attempt died when
the local client lost DNS, and 2 of 3 units survived.

Use `--detach`. Without it the app's lifetime is tied to the local client and a
dropped connection kills the container mid-run.
