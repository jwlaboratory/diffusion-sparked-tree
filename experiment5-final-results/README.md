# Experiment 5 — Final Results

Head-to-head on **one machine, one job** (deliberately unsharded — TPS ratios are
only trustworthy within a single GPU; the sharded launcher was deleted after
cross-shard TPS was measured to swing ~20% at identical work): **AR · DFlash ·
DSpark · DDTree · SparklingTree (ours)**, reporting **mean acceptance** and **net
wall-clock speedup vs AR**.

SparklingTree = DSpark-b16 + its markov head + the best-first **union** tree
builder (post `harness-6-union` fix; per-depth-precompute results predating the
fix are archived in `../_archive_old_results_pre_union/` and are NOT citable).
DDTree is the official reference (DFlash backbone, corrector-free).

## TODO before the citable run (details in run_final.py docstring)

1. **Builder** — `best-first-fast` vs `best-first-precompute`: same tree post-fix,
   different build cost. Decided by the one-GPU repro in
   `../experiment4-faster/2-precompute/` (in flight).
2. **Temperature** — temp 0 is standard; the branch-conditional edge is largest at
   temp 1.0 (old BLOG: +8.9% acceptance, 6/6 datasets). If temp-0 aggregate does
   not clear DDTree, lead with temp 1.0 and report temp 0 alongside. No dataset
   cherry-picking.

The paper's conclusion cites `results/summary.json` produced by this script only.

## Benchmarking practices

We follow **DDTree benchmarking practices** (the official repo, pinned at our
embedded clone commit `f6d9bb8`):

1. **C++ KV compaction ON** — upstream's default (`maybe_enable_cpp_compact(True)`,
   enforced in `harness/runner/driver.py`). The old final benchmark disabled it
   (`--disable-cpp-compact-cache`); that deviation distorted its wall-clock numbers
   and is documented as an erratum in `old-experiments/`.
2. **Sync-on timing always recorded** — the `instrumented` pass reproduces the
   DDTree repo's methodology exactly (per-stage `torch.cuda.synchronize()` barriers,
   upstream `dflash.py:157`), so those numbers are directly comparable to the
   DDTree paper's. The `clean` pass (same run, barriers off) is reported alongside
   as the unbiased headline TPS. Every citable run executes BOTH passes.

## How to run

1. **Set the tunables** at the top of `run_final.py` — hardware (`GPU`), our builder
   config (`C`, `K`, `BUDGETS`), which methods to include, the speedup baseline, and
   the datasets. `C` / `K` / `B` come from the exp4 csweep winner.
2. Smoke test (minutes): `modal run run_final.py --smoke`
3. Full run (detached, resumable): `modal run --detach run_final.py --spawn`
4. Fetch + analyze:
   ```bash
   modal volume get ddtree-results final/summary.json results/summary.json
   python analyze_final.py results/summary.json --baseline DFlash --ours SparklingTree
   ```

Per-unit checkpointed and fingerprinted over config + CODE_VERSION, so a re-run
resumes and changing any tunable (C/K/B/datasets) starts a fresh cache namespace.

## Tunables cheat-sheet

| knob | meaning | where |
|---|---|---|
| `C` | candidate pool per depth (`beam_candidates`) | SparklingTree only |
| `K` | max fanout per node (`max_fanout`, 0 = budget) | SparklingTree only |
| `BUDGETS` | tree node budget(s) B | DDTree + SparklingTree |
| `SPEEDUP_BASELINE` | which method's TPS is the 1× denominator | analysis |
| `DATASETS` | `[[name, n_samples], ...]` | all |

Speedup is **N× vs autoregressive** by default (`SPEEDUP_BASELINE="Autoregressive"`)
— the same denominator the DDTree/DFlash papers use (their 8.2×). The `Autoregressive`
arm is plain target greedy decode (one forward per token, acceptance ≡ 1.0). Set
`SPEEDUP_BASELINE` to `"DFlash"`/`"DDTree"` for a speculator-relative view instead.

## Alignment with the DDTree paper (arXiv 2604.12989, Table 2)

The default tunables reproduce the paper's protocol: **10 benchmarks** (math500,
gsm8k, aime24, aime25, humaneval, mbpp, livecodebench, swe-bench, mt-bench, alpaca)
at their full test-set sizes (30–164 samples), **max_new_tokens=2048**, **budget
sweep {16,32,64,128,256,512,1024}**, block size 16, temp 0. This is a large
multi-hour H100 run — trim `DATASETS` / `BUDGETS` for quick iteration. DDTree's own
builder is a best-first heap tree, so SparklingTree (same algorithm + the DSpark
markov corrector) is a direct, fair comparison.

## Output

`analyze_final.py` prints, per budget: each method's acceptance, clean TPS, speedup
vs baseline, and dominant phase — then SparklingTree's acceptance/wall-clock
advantage over every other method.

Smoke result (n=1, 64 tokens — noisy, structure only): SparklingTree led on both
acceptance and TPS; real numbers come from the full run once C/K are fixed.
