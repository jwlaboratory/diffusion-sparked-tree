# Reproducing DFlash / DSpark / tree acceptance lengths on Modal (Experiment 1)

This measures **acceptance length** across six drafter × corrector × verify
configs (see [`README.md`](./README.md) for the table), using the official DDTree
harness (`DDTree/`) plus our `dspark.py` / `sparked_tree.py`, on
[Modal](https://modal.com), over a **small subset** of the paper's benchmark
suite, at **temperature 0** (greedy, deterministic).

Launcher: [`modal_benchmark.py`](./modal_benchmark.py).

---

## What it runs

Two stages on one GPU (see the module docstring in `modal_benchmark.py`):

**Stage A — official reference** (unmodified `DDTree/benchmark.py`, single sdpa
pass per dataset), optional via `RUN_OFFICIAL_REFERENCE`:

| method key    | what it is                                           |
|---------------|------------------------------------------------------|
| `baseline`    | `block_size=1` reference (autoregressive-style draft) |
| `dflash`      | **DFlash** block-diffusion draft                     |
| `ddtree_tbN`  | **DFlash + DDTree** draft tree, one per tree budget N |

**Stage B — the six configs** (`DDTree/run_acceptance.py`), which drive DSpark and
the naive sparked tree the official harness cannot produce (configs 0/2/3/4/5/6 in
the README). It loads the target + both drafters once and loops the datasets.

Stage A cross-validates Stage B: config `0` ≡ `dflash`, and config `4` ≡
`ddtree_tb{SPARKED_TREE_BUDGET}` (both use the DFlash backbone with no corrector).
We deliberately skip the extra `--flash-attn` pass from the paper's
`run_benchmark.sh`: it only changes which run wins the timing "best-of" pick and
**does not affect acceptance length**, the quantity we reproduce.

### The subset (vs. the full paper)

| dimension      | full paper (`run_benchmark.sh`)            | this reproduction        |
|----------------|--------------------------------------------|--------------------------|
| datasets       | 10 (gsm8k, math500, aime24/25, humaneval, mbpp, livecodebench, swe-bench, mt-bench, alpaca) | **3**: gsm8k, humaneval, mt-bench |
| samples/dataset| 30–164                                     | **8**                    |
| model/draft    | Qwen3-4B, Qwen3-8B, Qwen3-Coder-30B pairs  | **Qwen3-4B only** (DFlash-b16 + DSpark-b7 drafters) |
| tree budgets   | 16,32,64,128,256,512,1024                  | **64,256**               |
| temperature    | 0.0 and 1.0                                | **0.0 only**             |
| max new tokens | 2048                                       | **512**                  |
| GPUs           | 8× (torchrun)                              | **1** (single process)   |

All subset knobs live as UPPERCASE constants at the top of `modal_benchmark.py`.
To reproduce the *full* paper, expand `TASKS`, `MODEL_NAME`/`DRAFT_NAME`,
`TREE_BUDGET`, and set `MAX_NEW_TOKENS = 2048`.

---

## How to run

```bash
pip install modal
modal setup                       # one-time auth
cd experiment1-harness
modal run modal_benchmark.py
```

Output:
- Stage A raw run files: Modal volume `ddtree-results` at `/results/runs/*.pt`
- Stage B six-config file: `/results/runs/sparked__*.json`
- Aggregated `summary.json` (`results` = six configs, `reference` = official
  numbers): written to the volume **and** downloaded locally to
  `experiment1-harness/Results/summary.json`
- A six-config mean-acceptance-length table printed to the container log

Both stages are **resumable**: existing `runs/*.pt` and `sparked__*.json` are
skipped, so a re-run only fills gaps.

---

## Determinism & reproducibility notes

- **Temperature 0.** `benchmark.py` defaults to `--temperature 0.0`; `sample()`
  (`model/utils.py`) uses `argmax` when `temperature < 1e-5`, so decoding is
  greedy and deterministic.
- **Seeds.** `benchmark.py` sets `random`, `numpy`, `torch`, and
  `torch.cuda` seeds to 0, and enables `cudnn.deterministic = True`,
  `cudnn.benchmark = False`.
- **Fixed sample selection.** When a dataset is larger than `max_samples`, the
  harness does `dataset.shuffle(seed=0).select(range(max_samples))` — the same 8
  items every run.
- **Acceptance length is GPU-independent.** At temperature 0, acceptance length
  is a pure function of the (deterministic) model logits, so it reproduces
  exactly on any GPU. An **A100-40GB** (the default `GPU`) therefore reproduces
  the paper's acceptance numbers.
- **Timing/speedup is NOT reproduced here.** Wall-clock speedup depends on
  hardware; the paper used H100. If you want comparable timing, set
  `GPU = "H100"` in `modal_benchmark.py` and add the `--flash-attn` pass. Note
  the smaller `MAX_NEW_TOKENS = 512` also affects timing, not the per-round
  acceptance mean.

### Pinned environment

| component     | version / source                                             |
|---------------|-------------------------------------------------------------|
| base image    | `nvidia/cuda:12.4.1-devel-ubuntu22.04`, Python 3.11         |
| torch         | `2.5.1` (cu124 wheel index)                                  |
| flash-attn    | `2.7.4.post1` prebuilt wheel (cu12 / torch2.5 / cxx11abiFALSE / cp311) |
| transformers  | `4.57.1` (needs ≥4.51 for Qwen3, ≥4.56 for the `dtype=` from_pretrained kwarg the harness uses) |
| datasets      | `3.6.0`                                                      |

`flash_attn` is **mandatory**: the DFlash draft model always uses
`flash_attention_2`, and `benchmark.py` raises if it is not importable. The
pinned prebuilt wheel avoids a ~30-minute from-source build. If you bump
`TORCH_VERSION`, update `FLASH_ATTN_WHEEL` to the matching wheel from the
[flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases).

### Models & data

- Target `Qwen/Qwen3-4B` and drafts `z-lab/Qwen3-4B-DFlash-b16` (DFlash, block 16)
  and `deepseek-ai/dspark_qwen3_4b_block7` (DSpark + markov + confidence, block 7)
  are public HF models; datasets (`openai/gsm8k`, `openai/openai_humaneval`,
  `HuggingFaceH4/mt_bench_prompts`) are public. No token is normally required. The
  DSpark checkpoint loads into `DSparkDraftModel` with **zero missing/unexpected
  keys** (verified against its `model.safetensors` header).
- The launcher attaches this workspace's existing `huggingface` Modal secret
  (belt-and-suspenders for any gated asset). If you run in a different
  workspace, create a `huggingface` secret containing `HF_TOKEN` or edit the
  `secrets` list in `modal_benchmark.py`.
- HF downloads are cached in the `ddtree-hf-cache` Modal volume, so re-runs skip
  re-downloading.

### Other notes

- **Single process.** We run `benchmark.py` directly (no `torchrun`). Its
  distributed layer no-ops when `RANK` is unset (it prints a warning — expected).
- **Inline C++ KV-cache compaction.** DDTree tries to JIT-build a small C++
  extension for cache compaction; the `build-essential` + CUDA-devel image
  supports it, and it **falls back to a pure-Python path** if the build fails.
  Either way, acceptance length is identical (it's a perf optimization). Pass
  `--disable-cpp-compact-cache` to force the Python path.
- **Resumable.** Existing `/results/runs/*.pt` files are skipped, so a re-run
  only fills in missing datasets.

---

## Expected shape of results

`summary.json` maps each dataset to mean acceptance length (tokens accepted per
verification round). `results` holds the six configs (keyed by id `0/2/3/4/5/6`);
`reference` holds the official numbers for validation:

```json
{
  "config": { "...": "..." },
  "results": {
    "gsm8k": { "0": >1, "2": >1, "3": …, "4": …, "5": …, "6": … },
    "humaneval": { "...": "..." },
    "mt-bench":  { "...": "..." }
  },
  "reference": {
    "gsm8k": { "baseline": ~1.0, "dflash": >1, "ddtree_tb64": ≥dflash, "ddtree_tb256": ≥ddtree_tb64 }
  }
}
```

Expected patterns:

- **Validation:** config `0` ≈ `reference.dflash`; config `4` ≈ `reference.ddtree_tb64`
  (same DFlash backbone, no corrector — a built-in cross-check of `sparked_tree`).
- **Trees ≥ chains:** `4 ≥ 0` and (`5`,`6`) ≥ `2`-ceiling within their block size —
  a draft tree accepts at least as many tokens/round as the flat chain.
- **Markov contribution** (the thing this experiment isolates): compare markov-on vs
  markov-off *within a block size* — config `3` vs `4` (DFlash tree) and `6` vs `5`
  (DSpark tree). A positive gap means the corrector helps that backbone.

> Do not compare across block sizes (e.g. `0`/`3`/`4` at block 16 vs `2`/`5`/`6` at
> block 7): the acceptance ceiling differs (up to 16 vs 8), so raw cross-block
> numbers are not apples-to-apples.
