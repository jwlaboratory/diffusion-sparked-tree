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

### Exact method settings (verified against the code)

Read the **markov head** column, not the method name. The one subtlety:
`dspark.chain` **uses the DSpark markov head** even though its name has no `.markov`
— the head is DSpark's native drafter, applied *intrinsically* inside
`dspark_generate` (`model.sample_draft_tokens → self.markov_head`), not via the
`corrector` slot. So `corrector=None` on `dspark.chain` does **not** mean "no markov".

Backbones (block_size = checkpoint config, no runtime override):

- **DFlash-b16** = `z-lab/Qwen3-4B-DFlash-b16`, kind `dflash`, effective `block_size=16`.
  In-place indexing (hidden `i` → token `i`); borrows the **target's** `embed_tokens`/`lm_head`;
  drafts `block_size-1 = 15` tokens; has **no** markov head.
- **DSpark-b7** = `deepseek-ai/dspark_qwen3_4b_block7`, kind `dspark`, effective `block_size=7`.
  Next-token indexing (hidden `i` → token `i+1`); **owns** `embed_tokens`/`lm_head`;
  drafts `block_size = 7` tokens; carries a `VanillaMarkovHead`.
- Corrector `dspark_b7_markov` = the DSpark-b7 head, token-only (`bias = W2 @ W1[prev]`),
  so it can be spliced onto **either** backbone.

| method | backbone (id / kind / eff block_size) | drafter indexing | markov head (ON/OFF + source) | corrector slot | verify | tree_budget | temp |
|---|---|---|---|---|---|---|---|
| `dflash.chain` | `z-lab/Qwen3-4B-DFlash-b16` / dflash / 16 | in-place, 15 drafts | **OFF** (none) | none (disallowed) | chain | n/a | 0.0 |
| `dflash.tree` | `z-lab/Qwen3-4B-DFlash-b16` / dflash / 16 | in-place, depth ≤ 15 | **OFF** (none) | None | tree | 64 | 0.0 |
| `dflash.markov.tree` | `z-lab/Qwen3-4B-DFlash-b16` / dflash / 16 | in-place, depth ≤ 15 | **ON — FOREIGN** (dspark_b7 head on DFlash) | `dspark_b7_markov` | tree | 64 | 0.0 |
| `dspark.chain` | `deepseek-ai/dspark_qwen3_4b_block7` / dspark / 7 | next-token, 7 drafts | **ON — NATIVE, intrinsic** (own head in `dspark_generate`) | none (ignored) | chain | n/a | 0.0 |
| `dspark.tree` | `deepseek-ai/dspark_qwen3_4b_block7` / dspark / 7 | next-token, depth ≤ 7 | **OFF** (none) | None | tree | 64 | 0.0 |
| `dspark.markov.tree` | `deepseek-ai/dspark_qwen3_4b_block7` / dspark / 7 | next-token, depth ≤ 7 | **ON — NATIVE** (own head conditions each tree node) | `dspark_b7_markov` | tree | 64 | 0.0 |

Notes:
- **markov OFF** = raw backbone logits: a per-depth-independent tree (`markov_head=None`
  in `sparked_tree_generate` skips the `bias(prev)` term), or a parallel-argmax chain.
- **corrector slot** is the `corrector` field in the method spec. `dflash.chain` forbids it
  (`dflash_generate` has no corrector); `dspark.chain` ignores it (head is intrinsic).
- **temperature 0.0** everywhere (greedy). `confidence_threshold=0.0`, so `dspark.chain`'s
  confidence-truncation head is inactive and never shortens the 7-token proposal.
- The corrector-fit **probe** (`PROBE_CORRECTOR = dspark_b7_markov`) runs measurement-only
  on the two no-corrector tree methods (`dflash.tree`, `dspark.tree`) and never alters the
  tree that is actually built.


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