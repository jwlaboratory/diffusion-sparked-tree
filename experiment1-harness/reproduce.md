# Experiment 1

## The matrix

Default methods (the 2×2 that carries the claim):

| method | backbone | corrector | verify | role |
|---|---|---|---|---|
| `dflash.tree` | DFlash-b16 | none | tree | foreign OFF (≡ ddtree) |
| `dflash.markov.tree` | DFlash-b16 | DSpark-b7 markov | tree | **foreign ON** |
| `dspark.tree` | DSpark-b7 | none | tree | own OFF |
| `dspark.markov.tree` | DSpark-b7 | DSpark-b7 markov | tree | **own ON** |

The head's effect is the **within-backbone delta**. The claim predicts
`dspark.markov.tree ≫ dspark.tree` and `dflash.markov.tree ≤ dflash.tree`.

### Three outputs prove it

1. **`results[dataset][method].mean_accept`** — acceptance length per method.
   Summarized as `transfer[backbone].accept_pct_change` (own vs foreign %).
2. **`results[...].per_depth_accept`** — conditional accept rate per tree depth,
   reported to `DEPTH_REPORT_LIMIT` (=7). Compares own vs foreign at depths the
   b7 head was actually trained on, so a foreign penalty can't be blamed on
   depth extrapolation.
3. **`corrector_fit[method]`** — the confound-free proof. On each *no-corrector*
   run, at every committed position we compare `argmax(base)` and
   `argmax(base + head.bias(prev))` against the target's real token, plus their
   cross-entropies. Real positions, real previous tokens, no tree/depth
   extrapolation. `delta_hit > 0` / `delta_ce < 0` ⇒ the head fits that backbone.
   Expect it positive on DSpark, ≤0 on DFlash.

---

## Backbones are parameterized (add your own)

Each backbone is `{name, model_id, kind, block_size?}`:

- **`kind`** is intrinsic to the checkpoint: `dflash` (in-place indexing, borrows
  the target's embed/lm_head, drafts `block_size−1`) or `dspark` (next-token
  indexing, owns embed/lm_head, drafts `block_size`, carries a markov head).
- **`block_size`** is a runtime override (default = the checkpoint's config). This
  is why there is **no separate depth cap**: to depth-match a foreign backbone to a
  b7 head, add the same checkpoint at `block_size: 7`, e.g.
  ```python
  {"name": "dflash_b7", "model_id": "z-lab/Qwen3-4B-DFlash-b16", "kind": "dflash", "block_size": 7}
  ```

**Correctors** are auto-derived: every `dspark`-kind backbone with a markov head
exposes `"<name>_markov"`. A corrector is token-only (`bias = W2 @ W1[prev]`), so it
can be spliced onto any backbone — that's what makes `dflash.markov.tree` possible.
The configs confirm both drafters share the same target, `target_layer_ids
[1,9,17,25,33]`, `hidden_size 2560`, `vocab 151936`, `mask_token_id 151669`, so the
splice is well-posed.

---

## Tunables (constants at the top of `modal_benchmark.py`)

| knob | default | notes |
|---|---|---|
| `TARGET` | `Qwen/Qwen3-4B` | verifier |
| `BACKBONES` | dflash_b16, dspark_b7 | each `{name, model_id, kind, block_size?}` |
| `METHODS` | the 2×2 | any `backbone × corrector × verify` cells |
| `CHAIN_ANCHORS` | `False` | also run native `dflash.chain` / `dspark.chain` |
| `PROBE_CORRECTOR` | `dspark_b7_markov` | head used for the fit probe |
| `TASKS` | gsm8k:8, humaneval:8, mt-bench:8 | `(dataset, max_samples)` |
| `TREE_BUDGET` | 64 | |
| `TEMPERATURE` | 0.0 | tree build is greedy top-k regardless of temperature |
| `MAX_NEW_TOKENS` | 512 | paper uses 2048 |
| `SEED` | 0 | |
| `CONFIDENCE_THRESHOLD` | 0.0 | dspark chain only |
| `MEASURE_PER_DEPTH` / `MEASURE_CORRECTOR_FIT` | True | |
| `DEPTH_REPORT_LIMIT` | 7 | per-depth / probe reporting horizon |
| `GPU` / `TIMEOUT_SECONDS` | A100-40GB / 1h | |

---

## How to run

```bash
pip install modal
modal setup                       # one-time auth
cd experiment1-harness
modal run modal_benchmark.py      # add --detach to survive local network drops
```

Output:
- `summary.json` on the `ddtree-results` volume **and** downloaded to
  `experiment1-harness/Results/summary.json`
- transfer + corrector-fit rollups printed to the log

`summary.json` shape:
```json
{
  "config":  { "backbones": {...}, "methods": {...}, "...": "..." },
  "results": { "gsm8k": { "dspark.markov.tree": {"mean_accept": ..., "per_depth_accept": {...}}, "...": "..." } },
  "corrector_fit": { "dflash.tree": {"backbone": "dflash_b16", "overall_depth_le_7": {"delta_hit": ..., "delta_ce": ...}} },
  "transfer": { "dspark_b7": {"accept_pct_change": +NN.N}, "dflash_b16": {"accept_pct_change": -N.N} }
}
```

---

## Determinism & reproducibility notes

- **Temperature 0.** `sample()` (`model/utils.py`) is `argmax` when
  `temperature < 1e-5`; tree building is greedy top-k. Fully deterministic.
- **Seeds.** `run_experiment.run()` seeds `torch` / `torch.cuda` with `SEED`.
  Sample selection is `dataset.shuffle(seed=SEED).select(range(max_samples))`.
- **GPU-independent.** Acceptance length and the corrector-fit probe are pure
  functions of deterministic logits, so an A100-40GB reproduces them exactly.
  Timing/speedup would need matching H100 hardware — not measured here.
- **Metric consistency.** Every method records *new tokens committed per round*
  (chain: `matched+1`; tree: `len(accepted_path)`), so all cells are comparable.
  Absolute acceptance is not comparable across different block sizes — compare the
  **within-backbone %** (`transfer`), which is block-size robust.

### Pinned environment

| component | version / source |
|---|---|
| base image | `nvidia/cuda:12.4.1-devel-ubuntu22.04`, Python 3.11 |
| torch | `2.5.1` (cu124) |
| flash-attn | `2.7.4.post1` prebuilt wheel (cu12 / torch2.5 / cxx11abiFALSE / cp311) |
| transformers | `4.57.1` (≥4.51 for Qwen3, ≥4.56 for the `dtype=` from_pretrained kwarg) |
| datasets | `3.6.0` |

`flash_attn` is mandatory (draft models use `flash_attention_2`). The `huggingface`
Modal secret is attached (models/datasets used are public). HF downloads cache in
the `ddtree-hf-cache` volume.

### Equivalence check (run once, not per experiment)

`dflash.tree` (no corrector) must reduce to the official `ddtree_generate`. Confirm
after touching tree code:

```bash
# on a GPU box / Modal shell, inside DDTree/
python check_ddtree_equiv.py --dataset gsm8k --max-samples 4 --tree-budget 64
```

Exits non-zero if the acceptance streams diverge. This is the only role the official
`benchmark.py` / `ddtree.py` reference path plays now.

---

## Caveats

- **mt-bench** is used single-turn here (first turn only), unlike the official
  multi-turn harness. Fine for acceptance measurement; just not identical prompts.
- **Cross-backbone absolute numbers** differ because block sizes differ (7 vs 16).
  The claim rests on within-backbone deltas and the per-depth / probe analyses,
  all of which are block-size controlled.
