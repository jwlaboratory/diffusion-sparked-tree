# Concurrency benchmark

Resolves the one unproven row in `experiments/RESULTS.md`: *"holds at serving
concurrency — untested, all results batch 1."*

## Design

Only two of the four methods exist in SGLang. The builtin `SpeculativeAlgorithm`
enum is `DFLASH / DSPARK / EAGLE / EAGLE3 / FROZEN_KV_MTP / STANDALONE / NGRAM`.
DDTree and sparked-tree are not there.

They can arrive as plugins through `SpeculativeAlgorithm.register(...)`, but a
plugin registered `supports_overlap=False` only runs under
`--disable-overlap-schedule` — while the builtins default to the V2 overlap
scheduler, worth **>33% on a single B200**. Compare naively and the chains get a
third of a speedup that has nothing to do with chains-vs-trees.

So: **every arm at overlap off** (the fair comparison), **plus the chains at
overlap on** (the price the trees pay).

| arm | algorithm | overlap | status |
|---|---|---|---|
| `baseline_ov` / `baseline_noov` | none | on / off | runnable |
| `dflash_ov` / `dflash_noov` | `DFLASH` | on / off | runnable |
| `dspark_ov` / `dspark_noov` | `DSPARK` | on / off | checkpoint open, see below |
| `ddtree_noov` | plugin | off only | plugin runs; needs the DDTree builder wired |
| `sparked_noov` | plugin | off only | plugin runs; needs the DSpark drafter wired |

The baseline on/off pair is a control: it separates *overlap helps decoding* from
*overlap helps speculative decoding*, which are not the same number.

## Run

```bash
modal run --detach concurrency/run_concurrency.py::sweep
```

One GPU container per arm; the ladder runs against one warm server. To run a
subset:

```bash
modal run --detach concurrency/run_concurrency.py::sweep --arms dflash_ov,dflash_noov
```

Then:

```bash
python3 concurrency/report.py /vol/results/CONCURRENCY_H100_<ts>.json --out concurrency/REPORT.md
```

## What this run does and does not establish

**Does:** the chain baseline at concurrency, and the overlap delta per method.
That delta is the decision input — if overlap is worth 30% to DSpark and
sparked-tree beats DSpark by 12% with overlap off, sparked-tree loses in
production and the device-resident builder port is mandatory. If it is worth 5%,
the port is optional.

**Does not:** compare trees against chains. Both tree arms still lack a real
drafter, so the report prints them as explicit blocked rows rather than omitting
them. The plugin itself is written and validated — see below.

## Open questions

1. **DSpark checkpoint format.** Our block-16 drafter is warm-started from
   `deepseek-ai/dspark_qwen3_4b_block7` and fine-tuned through DeepSpec
   (`training/modal_train.py`); it lives on the volume, not the Hub. Whether
   SGLang's `DSPARK` worker loads it is unverified. `config.py` currently points
   at the published block-7 base, which runs but compares block-7 against our
   block-16 batch-1 numbers. Resolve before quoting DSpark cross-harness.
   The DFlash arm has no such problem — `z-lab/Qwen3-4B-DFlash-b16` is the same
   checkpoint the batch-1 harness uses.

2. **Pin the image.** `run_concurrency.py` pins `SGLANG_TAG` deliberately —
   `supports_overlap=False` is already deprecated upstream and the V1 worker path
   is gone, so the plugin contract can move between releases. Override with
   `SGLANG_TAG=...` if you bump it.

## The plugin

`sparked_plugin/` registers `SPARKED` and **runs**: server boots, TARGET_VERIFY
CUDA-graph capture works, and output is byte-identical to greedy no-speculation
decoding (`test_plugin_e2e.py`, H100). Acceptance 1.375 with a stand-in proposer.

```bash
modal run concurrency/test_plugin_e2e.py::run
```

Two things to know before extending it:

- **`CustomSpecAlgo` is missing three methods the scheduler calls
  unconditionally** — `create_future_map`, `need_topk`,
  `carries_draft_hidden_states`. The conformance guard only checks `is_*` /
  `supports_*` names, and `init_overlap` runs regardless of the overlap flag, so
  every plugin dies at startup without them. `algo.py` copies the enum versions.
- **The launcher needs a `__main__` guard.** sglang spawns its scheduler with
  multiprocessing spawn, which re-imports the launcher in the child — which is
  also how the plugin gets registered in the scheduler process.

What remains is the drafter: `LookupTreeSource` is a prompt-lookup fixture that
exists to make acceptance non-zero so the machinery can be tested.
`DSparkTreeSource` is the real one and is not implemented — the right shape is to
subclass `DSparkWorkerV2`, keep its drafter and KV injection, and replace only
its chain verify.

## Why the verify path was reusable

**The template is NGRAM, not EAGLE.**

`NgramVerifyInput` is a shipping non-EAGLE tree with our exact shape profile —
node-budgeted with no depth cap (*"spec_steps is meaningless for this tree"*) and
`tree_topk = -1`, the documented sentinel for *"irregular tree, no fixed
per-level branching"*. `EagleVerifyInput.max_tree_depth` carries the matching
invitation: *"Algorithms with other tree shapes override this."* The verify
kernel is generic over shape: `verify_tree_greedy_func` walks the tree through
`retrieve_index` / `retrieve_next_token` / `retrieve_next_sibling`.

Our builder already emits every field:

| SGLang | ours |
|---|---|
| `draft_token` | `node_token_ids`, root prepended |
| `custom_mask` | `visibility`, reshaped to their flat layout |
| `positions` | `node_depths + seq_len` |
| `retrieve_index` / `_next_token` / `_next_sibling` | `parents` + `child_maps`, re-encoded first-child / next-sibling |
| `draft_token_num` | `1 + tree_budget` |
| `max_tree_depth` / `tree_topk` | `draft_token_num` / `-1` |

Three things make this smaller than it looks. The DSpark drafter half already
exists upstream in `dspark_components/`. Temperature 0.0 — which
`final_benchmark/config.py` locks — hits the greedy verify path, skipping the
rejection-sampling branch and its `draft_probs` entirely. And DDTree vs
sparked-tree is the *same plugin* with `build_ddtree_tree` swapped for the markov
builder, so one piece of work lands both arms.

And the `child_maps` → first-child/next-sibling conversion — flagged early as the
main correctness risk — turned out not to be ours at all. `ngram_worker.py`
builds *only* the mask and calls `reconstruct_indices_from_tree_mask`, which
derives `retrieve_index` / `retrieve_next_token` / `retrieve_next_sibling` from
it. Our `visibility` already matches that mask convention
(`visibility[i, :i] = visibility[parent[i], :i]`), so the bridge is a reshape,
and `NgramVerifyInput` is reused verbatim rather than subclassed — its defaults
(`max_tree_depth == draft_token_num`, `tree_topk == -1`) are already right for an
irregular budgeted tree.

The one conversion that remained — our dense `[N, N]` mask → their flat layouts —
is covered by `test_sparked_bridge.py` (432 trees, CPU) and
`test_kernel_agreement.py` (494 trees against SGLang's own kernel, GPU), with
`test_negative_control.py` proving those tests can fail.

## Why overlap stays off even so

`supports_overlap=True` asserts no host sync in the hot path. Upstream holds that
line hard: `dspark_worker_v2.py` is 767 lines with zero `.cpu()` / `.item()` /
`.tolist()`, and `dflash_worker_v2.py` maintains a device-side
`_compute_compact_draft_seq_lens` alongside its `_host` variant rather than
accept one.

Our builders all sync:

- `exact-precomputed` (the arm `final_benchmark/config.py` locks) — numpy
  `heapq` walk, host-side by construction.
- graphed beam — the level loop is on device, but `int(root_token_id)` goes in
  at `ddtree_markov.py:851` and `.cpu().numpy()` comes out at
  `ddtree_markov.py:855`, then `_tree_structure_from_levels` assembles
  `list[int]` / `list[dict]` on the host.

At batch 1 that sync is 0.19 ms, ~5% of build, and worth paying — §12 measures
the graphed builder end-to-end at 1.093 ms against 3.594 ms eager. Under the
overlap scheduler it stops being a cost and becomes a barrier.
