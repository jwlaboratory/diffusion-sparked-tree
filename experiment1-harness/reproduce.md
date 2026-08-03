# Experiment 1

This runs a quick test: 8 prompts per, on multipel speculators, to see  the accepted tokens amount per each


## The methods

All of these run by default (`backbone × corrector × verify`):

| method | backbone | corrector | verify | note |
|---|---|---|---|---|
| `dflash.chain` | DFlash-b16 | none | chain | block drafting, no tree, no markov |
| `dflash.tree` | DFlash-b16 | none | tree | ≡ ddtree |
| `dflash.markov.tree` | DFlash-b16 | DSpark-b7 markov | tree | foreign splice |
| `dspark.chain` | DSpark-b7 | (own markov) | chain | DSpark as intended, native serial |
| `dspark.tree` | DSpark-b7 | none | tree | |
| `dspark.markov.tree` | DSpark-b7 | DSpark-b7 markov | tree | |

The head's effect is the **within-backbone delta** between a backbone's tree runs
with the corrector off vs on: `dspark.markov.tree − dspark.tree` (own) vs
`dflash.markov.tree − dflash.tree` (foreign). The claim predicts own ≫ 0, foreign ≤ 0.
`dspark.chain` / `dflash.chain` are native-drafting references.


## Tunables (constants at the top of `modal_benchmark.py`)

| knob | default | notes |
|---|---|---|
| `TARGET` | `Qwen/Qwen3-4B` | verifier |
| `BACKBONES` | dflash_b16, dspark_b7 | each `{name, model_id, kind, block_size?}` |
| `METHODS` | all six | any `backbone × corrector × verify` cells |
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