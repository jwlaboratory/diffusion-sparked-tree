# Diffusion-Sparked Tree — All Results

Markov-guided draft trees for speculative decoding.

**Setup throughout:** Qwen3-4B target · greedy (temperature 0) · exact-match prefix
acceptance · every method verified byte-identical to plain autoregressive decoding.

*All numbers verified against the source JSONs with
[`build_results_table.py`](build_results_table.py); per-run detail in
[`ALL_RESULTS.md`](ALL_RESULTS.md) (472 measurements, 12 runs).*

---

## 1. HEADLINE — final results

Best block-16 drafter · **H100** · beam+flat builder · budget 128 · 6 datasets × 12 prompts × 512 tok

### Acceptance (tokens per round)

| dataset | baseline | DFlash | DSpark | DDTree | **sparked** |
|---|---|---|---|---|---|
| humaneval | 1.000 | 6.452 | 6.465 | 8.765 | **8.899** |
| mbpp | 1.000 | 6.309 | 6.417 | 8.802 | **9.278** |
| gsm8k | 1.000 | 6.646 | 7.901 | 9.154 | **10.630** |
| math500 | 1.000 | 8.110 | 7.604 | 10.418 | **10.590** |
| mt-bench | 1.000 | 3.197 | 3.951 | 5.013 | **5.513** |
| alpaca | 1.000 | 2.901 | 4.110 | 4.300 | **5.709** |
| **MEAN** | 1.000 | 5.602 | 6.075 | 7.742 | **8.437** |

### Speedup vs no drafter

| dataset | DFlash | DSpark | DDTree | **sparked** |
|---|---|---|---|---|
| humaneval | 4.91x | 4.62x | **6.07x** | 5.60x |
| mbpp | 4.62x | 4.38x | **5.43x** | 5.37x |
| gsm8k | 4.86x | 5.45x | 5.48x | **6.01x** |
| math500 | 5.93x | 5.39x | **6.80x** | 6.40x |
| mt-bench | 2.51x | 2.86x | 3.07x | **3.10x** |
| alpaca | 1.98x | 2.26x | 2.44x | **2.75x** |
| **MEAN** | 4.14x | 4.16x | **4.88x** | 4.87x |

**Acceptance: best on 6/6. Speed: best on 3/6.**

| sparked-tree vs | acceptance | speed |
|---|---|---|
| DSpark | **+40.3%** | **+18.2%** |
| DFlash | **+51.2%** | **+17.4%** |
| DDTree | **+9.9%** | +0.7% (tied — see §12) |

---

## 2. TREE BUDGET SWEEP

block-16 · A10G · 6 datasets · flat beam

| budget | DDTree acc | DDTree spd | sparked acc | sparked spd | Δ acc | Δ spd |
|---|---|---|---|---|---|---|
| 16 | 6.747 | 4.85x | 6.190 | 4.13x | −8.3% | −14.8% |
| 32 | 7.209 | 5.21x | 7.398 | 4.89x | +2.6% | −6.1% |
| 64 | 7.833 | 5.54x | 8.066 | 5.20x | +3.0% | −6.1% |
| 128 | 8.065 | 5.10x | 8.662 | 5.05x | +7.4% | −1.0% |
| 256 | 8.364 | 3.67x | **9.083** | **3.75x** | **+8.6%** | **+2.2%** |
| — | DSpark chain | | 6.196 | 4.29x | | |
| — | DFlash chain | | 5.909 | 4.38x | | |

Our margin grows monotonically with budget (−8.3% → +8.6%); we win **both** axes at 256.

⚠️ The budget-16 loss is a **known bug**: `flat(16)` at depth 16 gives
`[1,1,…,1]` — a chain with zero branching. Fix is to truncate depth rather than
thin the schedule. Flat is only valid when `budget ≥ 2 × depth`.

---

## 3. ABLATIONS — what markov guidance is worth

block-7 · A10G · 6-dataset means · budget 64

| drafter | tree? | scoring | acceptance | speedup |
|---|---|---|---|---|
| DFlash | chain | — | 5.457 | 4.04x |
| DFlash | tree | independence = **DDTree** | 7.289 | 5.18x |
| DFlash | tree | + foreign markov head (xmkv) | 6.381 | 3.18x |
| DSpark | chain | markov built-in | 5.234 | 3.81x |
| DSpark | tree | independence (nomkv) | **3.941** | 2.86x |
| DSpark | tree | markov-guided = **ours** | 6.652 | 3.60x |

**Same head, opposite sign:**

| change | | effect |
|---|---|---|
| head guiding **its own** backbone | 3.941 → 6.652 | **+69%** |
| head on a **foreign** backbone | 7.289 → 6.381 | **−12%** |

Per-dataset markov-vs-independence (budget 64): humaneval +56%, mbpp +78%,
gsm8k +67%, math500 +74%, mt-bench +69%, alpaca +70% — **6/6**.

Note the third row: an independence-scored tree on the DSpark drafter (3.941) is
**worse than that drafter's own chain** (5.234), despite verifying 65 tokens
instead of 8. The tree is not a superset of the chain — the chain samples from
markov-corrected logits while the tree ranked raw ones, and DSpark's backbone is
trained to leave a residual the head fills.

⚠️ This table mixes horizons — DSpark block-7 (max 8/round) vs DFlash block-16
(max 16). Not a fair cross-drafter comparison; table 1 is.

---

## 4. TREE-BUILDER OPTIMIZATION

gsm8k · block-16 · budget 64

| builder | acceptance | speedup | tree_build |
|---|---|---|---|
| best-first, full vocab | 11.562 | 5.04x | 2.73s |
| best-first, 4096 candidates | 11.562 | 5.50x | 2.22s |
| best-first, 2048 candidates | 11.453 | 5.60x | 2.16s |
| best-first, 512 candidates | 11.346 | 5.54x | 2.15s |
| beam, geometric decay 0.75 | 9.653 | 6.53x | 0.57s |
| beam, measured `[2,3,7,7…]` | 10.974 | **7.22x** | 0.65s |
| beam, flat `[4,4,5,5…]` | **11.089** | 7.21x | 0.63s |
| *DDTree reference* | 10.024 | 7.20x | 0.10s |
| *DSpark chain* | 8.419 | 5.96x | — |

512 candidates cost the same as 4096 → **the bottleneck was round-trip count, not
payload**. Shrinking each call bought −21%; eliminating calls (level-synchronous
beam, ~48 syncs/round → 1) bought −72%.

---

## 5. TREE-SLOT MEASUREMENT → why flat beats front-loaded

6 datasets · budget 256 · every accepted node recorded as (depth, slot)

| depth | top-1 hit rate | slots for 95% | geometric gave | flat gives |
|---|---|---|---|---|
| 1 | **87%** | 3 | **26** | 4 |
| 2 | 76% | 5 | 15 | 4 |
| 4 | 66% | 13 | 6 | 4 |
| 8 | 59% | 14 | 1 | 4 |
| 12 | 48% | 21 | **0** | 4 |
| 16 | **30%** | 42 | **0** | 4 |

The drafter is confident near the root and uncertain deep — **the opposite of
every decaying schedule**. Corollary: the exact shape matters less than not
decaying (uniform slightly beat the measured schedule, 11.089 vs 10.974).

---

## 6. HARDWARE — A10G vs H100

Identical checkpoints, prompts, and config; **only the GPU differs**.

| dataset | method | A10G | H100 | retained |
|---|---|---|---|---|
| gsm8k | DFlash | 4.90x | 2.85x | 58% |
| | DSpark | 4.75x | 2.98x | 63% |
| | DDTree | 6.17x | 3.16x | **51%** |
| | **sparked** | 4.35x | 3.00x | **69%** |
| humaneval | DDTree | 6.54x | 3.11x | **48%** |
| | **sparked** | 4.40x | 2.85x | **65%** |
| alpaca | DDTree | 2.82x | 1.66x | 59% |
| | **sparked** | 2.92x | 2.79x | **96%** |

Ours vs DSpark by GPU: gsm8k −8.4% → **+0.8%** · humaneval +2.2% → **+18.9%** ·
alpaca +7.2% → **+59.5%**

Higher bandwidth compresses every speedup toward 1x — **we degrade least, DDTree
most**. (Absolute stage-time sums are *not* comparable across GPUs: bf16 numerics
differ, so the two produce different text and token counts. Only per-token `tpot`
is.)

---

## 7. HORIZON — block 7 → block 16

Identical prompts; DDTree/DFlash appear as **0.0% controls**, confirming validity.

| dataset | chain gain | tree gain |
|---|---|---|
| gsm8k | +18.5% | **+37.6%** |
| humaneval | +20.0% | **+25.2%** |
| alpaca | −0.1% | +5.1% |

Trees gain ~2× more from depth than chains — a chain just gets a longer thing to
break, a tree gets more places to hedge at every new depth. Chat gains nothing:
alpaca never reaches depth 8, so depth 9–16 is capacity that is never used.

---

## 8. TRAINING RUNS

| run | data | anchors | steps | epochs | seq | γ | train loss | gsm8k acc |
|---|---|---|---|---|---|---|---|---|
| **A10G γ=4** ← best | 2,280 | 32 | 600 | 9 | 768 | 4 | 0.847 | **10.630** |
| A10G γ=8 | 2,280 | 32 | 600 | 9 | 768 | 8 | 0.904 | ≈same (±3%) |
| H100 | 2,280 | 128 | 1000 | **31** | 1024 | 8 | **0.522** | 10.214 ᶜ |
| bigdata | **9,500** | 96 | 1400 | 12 | 512 | 8 | 0.810 | 10.131 |

**Lower loss ≠ better acceptance.** The H100 run reached the lowest train loss by
memorising — 1000 steps over 2,280 conversations is 31 epochs.

ᶜ Measured at 8 prompts rather than 12, so it shares a configuration with the
others but not a prompt set. Only **A10G γ=4 (10.630) vs bigdata (10.131)** is a
strictly controlled pair.

⚠️ **bigdata was worse on all 6 datasets** (−0.5% to −6.7%) — but it changed four
variables at once (data 2,280→9,500, seq 768→**512**, anchors 32→96, steps
600→1400). Unattributable; the shortened context is the likeliest culprit. Design
error, not a result. **Whether data is the binding constraint remains untested.**

---

## 9. COMPUTE EFFICIENCY — predicts batched serving

| method | acceptance | verify width | accepted/scored | vs chain |
|---|---|---|---|---|
| DFlash chain | 5.602 | 16 | 0.350 | 0.98x |
| DSpark chain | 6.075 | 17 | **0.357** | 1.00x |
| DDTree tb64 | 7.457 | 65 | 0.115 | 0.32x |
| sparked tb64 | 7.842 | 65 | 0.121 | 0.34x |
| DDTree tb128 | 7.742 | 129 | 0.060 | 0.17x |
| sparked tb128 | 8.437 | 129 | 0.065 | 0.18x |

**Chains are ~5× more FLOP-efficient per accepted token.** At batch 1 the GPU is
idle so wide verification is nearly free; at serving concurrency it is not.
Tree-vs-tree we are consistently better (+5.2% @64, +9.0% @128) — our nodes are
less redundant, because DDTree's independence assumption gives every sibling the
*same* children table while branch-conditional scoring gives each its own.

**All results in this document are batch 1.** This ratio is what predicts
inversion at concurrency.

---

## 10. STAGE TIMES — where the work goes

A10G · alpaca · block-7 · seconds summed over prompts

| method | draft | tree_build | verify | commit |
|---|---|---|---|---|
| baseline | — | — | 44.67 | 0.23 |
| DFlash | 3.35 | — | 17.36 | 0.21 |
| DSpark | 2.39 + 0.45 ᵐ | — | 12.45 | 0.13 |
| DDTree tb64 | 2.32 | 0.23 | 12.03 | 0.66 |
| sparked (exact) | 1.77 | **2.97** | **9.15** | 0.50 |
| sparked (wave) | 1.75 | 2.19 | 9.06 | 0.50 |
| nomkv tb64 | 2.91 | 0.26 | 15.02 | 0.83 |
| xmkv tb64 | 2.54 | **7.11** | 12.78 | 0.70 |

ᵐ = serial markov head. We pay **more in tree_build, less in verify** (fewer
rounds). The beam builder cuts build 2.97 → 0.61s, which is what converted an
acceptance-only win into speed parity.

---

## 11. NEGATIVE RESULTS

| experiment | hypothesis | outcome |
|---|---|---|
| Confidence-head adaptive width | per-round widths beat a fixed average | **−2.7% acc, −6.6% spd** — head predicts *chain* acceptance; a tree needs *coverage* |
| Wave batching | fewer syncs, same tree | **11.45 → 8.13 acc** — deferred pushes break best-first ordering |
| `loss_decay_gamma` 4→8 | deep positions gradient-starved | **±3%, noise** — Adam normalises the reweighting away |
| Cross-model markov head | head is a general bigram prior | **−12%, 6/6** — it is a backbone-specific residual |
| 4× training data | data was the binding constraint | **worse 6/6** — but confounded, unattributable |
| Candidate restriction | memory traffic was the bottleneck | **only −21%** — the bottleneck was call count |

Confidence-head per cell: gsm8k@128 −5.9%/−11.4% · math500@64 −5.2%/−7.8% ·
math500@128 −5.0%/−7.6% · alpaca@64 **+3.6%**/−2.3% · alpaca@128 −0.7%/−3.6%.
The single positive cell is chat, where round-to-round variance is highest.

---

## 12. CUDA-GRAPH TREE BUILDER

The beam level loop is ~100% of `tree_build` (H100 profile, budget 64):

| component | ms/round | share |
|---|---|---|
| GPU level loop | 3.98 | ~100% |
| transfer / sync | 0.19 | 5% |
| Python tree assembly | 0.06 | **2%** |

So the builder is bound by *launching* 16 dependent levels, not by data volume or
CPU work. Capturing that loop as a CUDA graph and replaying it:

| budget | eager | graphed | saved | `tree_build` | gap to DDTree |
|---|---|---|---|---|---|
| 64 | 3.594 ms | **1.093 ms** | **70%** | 0.61s → 0.19s | 6.1x → **1.9x** |
| 128 | 3.356 ms | **1.570 ms** | 53% | 0.57s → 0.27s | 6.1x → **2.7x** |

Trees are **bit-identical** to the eager path (tokens, depths, parents, visibility
all equal), so this is pure speed with no effect on acceptance.

Two prerequisites, one of which came free from an earlier finding:

- **Static shapes.** Graph capture requires them. The flat schedule `[4,4,…,4]` is
  static; the geometric and confidence-adaptive schedules vary per round and
  **could not be captured**. The change that improved acceptance (§5) is what
  unlocked the speed fix.
- **Per-depth candidates instead of a deduped union.** `torch.unique` returns a
  variable-length tensor, which breaks capture. A fixed `[depth, C]` table is also
  slightly tighter — each depth gets its own top-C rather than sharing a pool.

Capture happens once per `(widths, C, device)` and is cached; on failure the
builder logs and falls back to eager permanently rather than retrying per round.

⚠️ **The end-to-end benchmark cannot confirm this.** Per-dataset `tree_build`
moved in the *wrong* direction on half the datasets (gsm8k 1.88→2.33s while its
speed improved 11.9%; humaneval 4.10→3.30s while speed dropped 11.9%). Six
independently-scheduled H100 containers carry ~±12% timing variance, which swamps
a ~0.4s effect. The isolated microbenchmark is the trustworthy measurement; the
whole-run speed figure remains **a tie**.

---

## Bottom line

| claim | status |
|---|---|
| Beats **DSpark** (+40.3% acc, +18.2% spd) | ✅ validated, 6/6 datasets |
| Beats **DFlash** (+51.2%, +17.4%) | ✅ validated, 6/6 |
| Beats **DDTree** on acceptance (+9.9%) | ✅ validated, 6/6 |
| Ties DDTree on speed (+0.7%) | ✅ at budget 128; **wins** at 256. Within run-to-run noise — call it a tie |
| Holds at serving concurrency | ❌ **no** — ties at c≈4 and loses above it (budget 64); inverts at c≈2 (budget 128). §9's efficiency math was right; see `concurrency/FINDINGS.md` §5 |

**Honest framing:** best-in-class for *latency-bound* speculative decoding —
but the useful range is now measured, and it is **batch 1 to ~4**, not ~8
(c=4 is a tie at budget 64; budget 128 is already losing by c=2).
Interactive, local, low-QPS. At higher concurrency the chain's 5× compute
efficiency dominates exactly as §9 predicted, and the tree loses outright: 0.52×
at c=32 (budget 64), and that is an upper bound that charges the tree nothing for
building itself.
