# Experiment 1 — acceptance length: DFlash / DSpark / trees

This experiment uses the official DDTree repo harness (`DDTree/`) to benchmark the
**acceptance-length** difference across drafter × corrector × verify combinations.
"Acceptance length" = mean tokens accepted per verification round (temperature 0,
greedy). No timing — acceptance length is what we compare.

## The six configs

| # | drafter (backbone) | corrector | verify | code path |
|---|--------------------|-----------|--------|-----------|
| 0 | DFlash-b16 | — | chain | `dflash_generate` |
| 2 | DSpark-b7 | DSpark markov (serial) | chain | `dspark_generate` |
| 3 | DFlash-b16 | DSpark markov | tree | `sparked_tree_generate(draft_mode="dflash", markov_head=…)` |
| 4 | DFlash-b16 | — | tree | `sparked_tree_generate(draft_mode="dflash", markov_head=None)` — ≡ DDTree |
| 5 | DSpark-b7 | — | tree | `sparked_tree_generate(draft_mode="dspark", markov_head=None)` |
| 6 | DSpark-b7 | DSpark markov | tree | `sparked_tree_generate(draft_mode="dspark", markov_head=…)` |

- **drafter / backbone** = the small draft transformer that produces per-position
  draft logits (`z-lab/Qwen3-4B-DFlash-b16` or `deepseek-ai/dspark_qwen3_4b_block7`).
- **corrector** = the DSpark markov head, a learned token-only additive logit bias
  (`bias(prev_token)`), applied per step. It is token-only, so it splices onto the
  DFlash backbone with no shape constraint (config 3).
- **target** = `Qwen/Qwen3-4B`, the verifier. Both drafters share this target, its
  `target_layer_ids [1,9,17,25,33]`, and `mask_token_id`, so config 3 is well-posed.

Configs 3–6 share **one** tree code path, so the only variables are drafter and
markov-on/off. The clean contrasts:

- **3 vs 4** — does DSpark's markov head help a DFlash tree (splice)?
- **5 vs 6** — does it help a DSpark tree?
- **2 vs 6** — chain vs tree for the DSpark drafter+markov.

> Comparability caveat: DFlash-b16 drafts 16/round, DSpark-b7 drafts 7 (an 8-wide
> block incl. the anchor), so the per-round acceptance *ceiling* differs (up to 16
> vs up to 8). Compare within a block size (3/4 together, 5/6 together), not 0 vs 2.

## How it runs

`modal_benchmark.py` runs two stages on one GPU, greedy:

- **Stage A** (optional, `RUN_OFFICIAL_REFERENCE`): the unmodified official
  `DDTree/benchmark.py` → `baseline`, `dflash`, `ddtree_tbN`. This validates the
  harness and cross-checks configs 0/4 (config 0 ≡ `dflash`, config 4 ≡
  `ddtree_tb{SPARKED_TREE_BUDGET}`).
- **Stage B**: `DDTree/run_acceptance.py` → the six configs above (loads all three
  models once, loops the datasets).

```bash
pip install modal && modal setup      # one-time
cd experiment1-harness
modal run modal_benchmark.py
```

Run the six configs locally on a GPU box without Modal:

```bash
cd DDTree
python run_acceptance.py --tasks gsm8k:8,humaneval:8,mt-bench:8 \
    --tree-budget 64 --save-json out.json     # add --methods 3,4 for a subset
```

## Output

Written to the `ddtree-results` Modal volume and downloaded to
[`Results/summary.json`](./Results/summary.json):

```json
{
  "config":    { "model": "...", "dflash_draft": "...", "dspark_draft": "...", "...": "..." },
  "results":   { "gsm8k": { "0": 6.85, "2": …, "3": …, "4": …, "5": …, "6": … }, "...": {} },
  "reference": { "gsm8k": { "baseline": 1.0, "dflash": 6.85, "ddtree_tb64": …, "ddtree_tb256": … } }
}
```

`results` holds the six configs (mean acceptance length); `reference` holds the
official numbers for validation. Expected: config 0 ≈ `reference.dflash`, config 4 ≈
`reference.ddtree_tb64`; markov-on ≥ markov-off within a block size if the head helps.

See [`reproduce.md`](./reproduce.md) for the full subset knobs and determinism notes.
The full paper suite (10 datasets, 2 temperatures, larger model pairs) is reached by
expanding the UPPERCASE constants in `modal_benchmark.py`.
