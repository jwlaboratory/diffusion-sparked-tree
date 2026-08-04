# Experiment 1

This runs a quick test: 8 prompts per, on multipel speculators, to see  the accepted tokens amount per each


## The methods

All of these run by default (`backbone × corrector × verify`):

| method | backbone | corrector | verify | note |
|---|---|---|---|---|
| `dflash.chain` | DFlash-b16 | none | chain | block drafting, no tree, no markov |
| `dflash.markov.chain` | DFlash-b16 | DSpark-b7 markov | chain | foreign head swept over the chain block |
| `dflash.tree` | DFlash-b16 | none | tree | ≡ ddtree |
| `dflash.markov.tree` | DFlash-b16 | DSpark-b7 markov | tree | foreign splice |
| `dspark.nomarkov.chain` | DSpark-b7 | none (`markov="off"`) | chain | intrinsic head ablated: parallel argmax |
| `dspark.chain` | DSpark-b7 | (own markov) | chain | DSpark as intended, native serial |
| `dspark.tree` | DSpark-b7 | none | tree | |
| `dspark.markov.tree` | DSpark-b7 | DSpark-b7 markov | tree | |
| `dspark_b16.nomarkov.chain` | DSpark-b16 | none (`markov="off"`) | chain | intrinsic head ablated: parallel argmax |
| `dspark_b16.chain` | DSpark-b16 | (own markov) | chain | our exp2 fine-tune, native serial |
| `dspark_b16.tree` | DSpark-b16 | none | tree | |
| `dspark_b16.markov.tree` | DSpark-b16 | DSpark-b16 markov | tree | own head, 16-deep tree |

The three `*.markov.chain` / `*.nomarkov.chain` rows complete the
`chain/tree × markov off/on` grid per backbone: `dflash.markov.chain` sweeps the
foreign b7 head serially over DFlash's chain block (the chain analogue of
`dflash.markov.tree`), and the `nomarkov.chain` rows disable the head DSpark
normally applies intrinsically, giving the true no-markov chain baselines.

The head's effect is the **within-backbone delta** between a backbone's tree runs
with the corrector off vs on: `dspark.markov.tree − dspark.tree` /
`dspark_b16.markov.tree − dspark_b16.tree` (own) vs
`dflash.markov.tree − dflash.tree` (foreign). The claim predicts own ≫ 0, foreign ≤ 0.
`dspark.chain` / `dspark_b16.chain` / `dflash.chain` are native-drafting references.

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
- **DSpark-b16** = `shreybirmiwal/Qwen3-4B-DSpark-b16`, kind `dspark`, effective `block_size=16`.
  Same architecture as DSpark-b7 (warm-started from it in experiment 2, see
  `../experiment2-block16/training/`); drafts `block_size = 16` tokens; carries its
  **own** jointly-trained `VanillaMarkovHead` (`dspark_b16_markov`).
- Corrector `dspark_b7_markov` = the DSpark-b7 head, token-only (`bias = W2 @ W1[prev]`),
  so it can be spliced onto **any** backbone. `dspark_b16_markov` is the analogous
  head from the b16 checkpoint (used only on its own backbone here).

| method | backbone (id / kind / eff block_size) | drafter indexing | markov head (ON/OFF + source) | corrector slot | verify | tree_budget | temp |
|---|---|---|---|---|---|---|---|
| `dflash.chain` | `z-lab/Qwen3-4B-DFlash-b16` / dflash / 16 | in-place, 15 drafts | **OFF** (none) | None | chain | n/a | 0.0 |
| `dflash.markov.chain` | `z-lab/Qwen3-4B-DFlash-b16` / dflash / 16 | in-place, 15 drafts | **ON — FOREIGN** (dspark_b7 head swept serially over the chain block) | `dspark_b7_markov` | chain | n/a | 0.0 |
| `dflash.tree` | `z-lab/Qwen3-4B-DFlash-b16` / dflash / 16 | in-place, depth ≤ 15 | **OFF** (none) | None | tree | 64 | 0.0 |
| `dflash.markov.tree` | `z-lab/Qwen3-4B-DFlash-b16` / dflash / 16 | in-place, depth ≤ 15 | **ON — FOREIGN** (dspark_b7 head on DFlash) | `dspark_b7_markov` | tree | 64 | 0.0 |
| `dspark.nomarkov.chain` | `deepseek-ai/dspark_qwen3_4b_block7` / dspark / 7 | next-token, 7 drafts | **OFF** (intrinsic head ablated via `markov="off"`; parallel argmax) | none (disallowed) | chain | n/a | 0.0 |
| `dspark.chain` | `deepseek-ai/dspark_qwen3_4b_block7` / dspark / 7 | next-token, 7 drafts | **ON — NATIVE, intrinsic** (own head in `dspark_generate`) | none (disallowed) | chain | n/a | 0.0 |
| `dspark.tree` | `deepseek-ai/dspark_qwen3_4b_block7` / dspark / 7 | next-token, depth ≤ 7 | **OFF** (none) | None | tree | 64 | 0.0 |
| `dspark.markov.tree` | `deepseek-ai/dspark_qwen3_4b_block7` / dspark / 7 | next-token, depth ≤ 7 | **ON — NATIVE** (own head conditions each tree node) | `dspark_b7_markov` | tree | 64 | 0.0 |
| `dspark_b16.nomarkov.chain` | `shreybirmiwal/Qwen3-4B-DSpark-b16` / dspark / 16 | next-token, 16 drafts | **OFF** (intrinsic head ablated via `markov="off"`; parallel argmax) | none (disallowed) | chain | n/a | 0.0 |
| `dspark_b16.chain` | `shreybirmiwal/Qwen3-4B-DSpark-b16` / dspark / 16 | next-token, 16 drafts | **ON — NATIVE, intrinsic** (own head in `dspark_generate`) | none (disallowed) | chain | n/a | 0.0 |
| `dspark_b16.tree` | `shreybirmiwal/Qwen3-4B-DSpark-b16` / dspark / 16 | next-token, depth ≤ 16 | **OFF** (none) | None | tree | 64 | 0.0 |
| `dspark_b16.markov.tree` | `shreybirmiwal/Qwen3-4B-DSpark-b16` / dspark / 16 | next-token, depth ≤ 16 | **ON — NATIVE** (own head conditions each tree node) | `dspark_b16_markov` | tree | 64 | 0.0 |

Notes:
- **markov OFF** = raw backbone logits: a per-depth-independent tree (`markov_head=None`
  in `sparked_tree_generate` skips the `bias(prev)` term), or a parallel-argmax chain.
- **corrector slot** is the `corrector` field in the method spec. On a dflash chain it is
  swept serially over the block (`dflash.markov.chain`); dspark chains forbid it (their
  head is intrinsic — set `markov: "off"` in the method spec to ablate it instead).
- **temperature 0.0** everywhere (greedy). `confidence_threshold=0.0`, so `dspark.chain`'s
  confidence-truncation head is inactive and never shortens the 7-token proposal.
- The corrector-fit **probe** (`PROBE_CORRECTOR = dspark_b7_markov`) runs measurement-only
  on the no-corrector tree methods (`dflash.tree`, `dspark.tree`, `dspark_b16.tree`) and
  never alters the tree that is actually built. Note the probe head is always the **b7**
  head, so the `dspark_b16` row measures the b7 head's fit on the b16 backbone.


## Tunables (constants at the top of `modal_benchmark.py`)

| knob | default | notes |
|---|---|---|
| `TARGET` | `Qwen/Qwen3-4B` | verifier |
| `BACKBONES` | dflash_b16, dspark_b7, dspark_b16 | each `{name, model_id, kind, block_size?}` |
| `METHODS` | all twelve | any `backbone × corrector × verify` cells (+ chain `markov: "off"`) |
| `PROBE_CORRECTOR` | `dspark_b7_markov` | head used for the fit probe |
| `TASKS` | gsm8k:8, humaneval:8, mt-bench:8 | `(dataset, max_samples)` |
| `TREE_BUDGET` | 64 | |
| `TEMPERATURE` | 0.0 | tree build is greedy top-k regardless of temperature |
| `MAX_NEW_TOKENS` | 512 | paper uses 2048 |
| `SEED` | 0 | |
| `CONFIDENCE_THRESHOLD` | 0.0 | dspark chain only |
| `MEASURE_PER_DEPTH` / `MEASURE_CORRECTOR_FIT` | True | |
| `DEPTH_REPORT_LIMIT` | 16 | per-depth / probe reporting horizon (reaches the b16 tree depth) |
| `GPU` / `TIMEOUT_SECONDS` | A100-40GB / 3h | |

---

## How to run

```bash
pip install modal
modal setup                       # one-time auth
cd experiment1-harness
modal run modal_benchmark.py      # add --detach to survive local network drops
```

To add/re-run a subset without recomputing the rest, pass `--methods` — the fresh
raws are merged into the existing local `Results/results_detailed.json` and the
summary is rebuilt over the union (this is how the three `*markov*.chain` cells
were added on top of the original nine-method run):

```bash
modal run --detach modal_benchmark.py \
  --methods "dflash.markov.chain,dspark.nomarkov.chain,dspark_b16.nomarkov.chain"
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