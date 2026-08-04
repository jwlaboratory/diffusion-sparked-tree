# Experiment 2 — block-16 DSpark (exact settings)

> **Experiment 2b (data scaling, 2026-08-04): more training data does NOT raise
> acceptance.** Three arms sharing the `_best` recipe exactly (seq 768, 32
> anchors, γ=4, batch 32, lr 1e-4), varying only PerfectBlend rows with steps
> epoch-matched at ~5: 2,400/360 vs 10,000/1,500 vs 24,000/3,600. Pooled mean
> acceptance: 7.67 / 7.60 / 7.59 (shipped `_best`: 7.62; b7: 6.33) — flat at 10×
> data. The apparent deep-depth "cliff" in per-cell-averaged charts is a
> small-sample artifact: pooled over raw lengths, all b16 arms hold ~0.85-0.89
> through depth 12 and taper mildly to ~0.72-0.80 at 14-16, identically across
> data scales. Deep-depth acceptance is limited by capacity or the horizon
> itself, not data (and not gradient weighting — FINDINGS.md §8). Recipe,
> launcher, and charts: `training/modal_train.py::pipeline_data`,
> `modal_benchmark_datascale.py`, `results/datascale/`.

Does extending the DSpark draft horizon from **block 7 → block 16** raise the
number of tokens the target accepts per verifier call? Two architecturally
identical DSpark drafters (5 draft layers, hidden 2560, `markov_rank` 256, same
`target_layer_ids` / mask token / confidence head) that differ **only** in
`block_size` are compared under identical tree + markov decoding.

- **Verifier / target:** `Qwen/Qwen3-4B`, greedy (temperature 0, deterministic; acceptance length is GPU-independent).
- **Block-16 checkpoint:** [`shreybirmiwal/Qwen3-4B-DSpark-b16`](https://huggingface.co/shreybirmiwal/Qwen3-4B-DSpark-b16), warm-started from `deepseek-ai/dspark_qwen3_4b_block7` (see `training/`).
- Full run/repro instructions, charts, gotchas, and the training recipe: **`reproduce.md`** and **`training/README.md`**.

## Exact method settings (zero ambiguity)

Verified against the code, not assumed (`DDTree/run_experiment.py::build_method_callable`,
`DDTree/sparked_tree.py`, `DDTree/model/dspark.py`). **All four arms are DSpark ×
tree.** There are **no `.chain` arms and no DFlash arms** in this experiment. Both
checkpoints carry a `markov_rank=256` vanilla head, so **both models always own a
markov head** — but a head is only *applied* when the method passes it as the corrector.

| method | backbone (model_id) | kind | eff. `block_size` | markov head | corrector (source) | verify | tree_budget | temperature |
|---|---|---|---|---|---|---|---|---|
| `dspark_b7.tree` | `deepseek-ai/dspark_qwen3_4b_block7` | dspark | **7** | **OFF** (owned but dormant) | `None` | tree | {64, 256} swept | 0.0 |
| `dspark_b7.markov.tree` | `deepseek-ai/dspark_qwen3_4b_block7` | dspark | **7** | **ON** (native — b7's *own* head) | `dspark_b7_markov` | tree | {64, 256} swept | 0.0 |
| `dspark_b16.tree` | `shreybirmiwal/Qwen3-4B-DSpark-b16` | dspark | **16** | **OFF** (owned but dormant) | `None` | tree | {64, 256} swept | 0.0 |
| `dspark_b16.markov.tree` | `shreybirmiwal/Qwen3-4B-DSpark-b16` | dspark | **16** | **ON** (native — b16's *own* head) | `dspark_b16_markov` | tree | {64, 256} swept | 0.0 |

Shared by all four: `draft_mode="dspark"`, target `Qwen/Qwen3-4B` (sdpa, bf16),
`max_new_tokens=512`, `seed=0`, `depth_report_limit=16`. Block size is intrinsic
to each checkpoint (no runtime override; a guard asserts b7=7 / b16=16). For a
dspark tree `depth_limit = block_size`, so **b7 trees reach depth 7, b16 trees
reach depth 16**. The two `*.markov.tree` arms are the headline comparison; the
`*.tree` arms are the markov-off controls that isolate the block-size effect.

## Name ↔ behavior mapping (the known ambiguity, resolved for this repo)

- **`.tree`** (corrector `None`) → markov head **OFF**. `build_sparked_tree`
  branches per-depth-independently. The loaded DSpark model still *owns* a markov
  head; it is simply never passed in, so it never touches the tree. "Markov off"
  means **dormant head, not absent head**.
- **`.markov.tree`** (corrector = the backbone's own head) → markov head **ON**.
  Each node's children are top-k of `base_logits + W2·W1[parent_token]`. The
  corrector is always that **same checkpoint's own** head — never a foreign/spliced one.
- **`.chain`** (none configured here): would call `dspark_generate`, which applies
  the model's markov head **intrinsically** (always on for chain). A hypothetical
  `dspark_b*.chain` is markov-on by construction, with no off switch.
- **DFlash** (none configured here): the code supports `draft_mode="dflash"` and
  splicing a *foreign* markov head onto a DFlash backbone, but experiment 2 never
  does this — both backbones are DSpark and each uses only its own head.
  `dflash.py` / `model/dflash.py` are present only as base classes / timing helpers.
- **Confidence head:** present on both checkpoints but applied by **no** arm here —
  only `dspark_generate` (chain) uses it, and only when `confidence_threshold > 0`
  (default `0.0`).

## The block-16 backbone's own settings

`shreybirmiwal/Qwen3-4B-DSpark-b16` (`training/dspark_block16_qwen3_4b.py`,
`_best` checkpoint): `block_size=16`, `num_draft_layers=5`,
`target_layer_ids=[1,9,17,25,33]`, `mask_token_id=151669`, **`markov_rank=256`,
`markov_head_type='vanilla'`** (jointly trained), confidence head on
(`alpha=1.0`, computed with markov). Warm-started from the block-7 checkpoint,
then fine-tuned (lr 1e-4, `max_train_steps=600`, `num_anchors=32`, bf16). Its
markov head is used **only** in `dspark_b16.markov.tree` (ON) and is **dormant**
in `dspark_b16.tree` (OFF).
