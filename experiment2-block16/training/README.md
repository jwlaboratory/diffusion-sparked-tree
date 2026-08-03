# Fine-tuning the block-16 DSpark drafter

The recipe that produced [`shreybirmiwal/Qwen3-4B-DSpark-b16`](https://huggingface.co/shreybirmiwal/Qwen3-4B-DSpark-b16):
a DSpark drafter for Qwen3-4B whose draft horizon is extended from block 7 to
block 16, warm-started from DeepSpec's released block-7 checkpoint.

Benchmarking is a separate harness (`../modal_benchmark.py`). Nothing in this
directory scores the model.

## What runs

`modal_train.py` is a four-stage Modal pipeline; every artifact persists on the
`ddtree-train` volume so stages are skippable on re-run:

1. **data** — `download_and_split.py` pulls a 2400-row PerfectBlend subset of chat
   conversations and holds out a 5% eval split.
2. **target cache** — `prepare_target_cache.py` precomputes the target model's
   hidden states at the layers the drafter reads (`target_layer_ids`). This is a
   small, run-specific cache, not DeepSpec's full ~38TB official one.
3. **warm-start train** — `train_warmstart.py` runs DeepSpec's trainer with the
   block-16 config, loading the block-7 weights first.
4. **publish** — the newest checkpoint is copied to
   `/vol/ckpt_final/dspark_block16<suffix>`, ready for `from_pretrained`.

Files:

| file | role |
|---|---|
| `modal_train.py` | the Modal pipeline (`pipeline`, `pipeline_h100`, `publish`) |
| `dspark_block16_qwen3_4b.py` | DeepSpec config: block 7→16, fine-tune schedule |
| `train_warmstart.py` | DeepSpec `train.py` + a warm-start monkeypatch |

## Why warm-starting from block 7 is valid

`block_size` only controls how many mask tokens are appended per anchor during
training; it does not change any parameter shape. So the block-7 drafter's weights
load into the block-16 model with `load_state_dict(strict=True)` — a strict load
that would fail on any architectural mismatch. We start from a trained drafter and
teach it the longer horizon, rather than paying for a from-scratch run.

## The config, honestly

The A10G run trades anchors for fitting in 24GB. The DSpark loss materializes
float32 `[num_anchors, block_size, vocab]` probability tensors (~311MB each at 32
anchors × block 16), so `num_anchors` is cut from DeepSpec's 512 (8×H100) to 32.
Other block-16 settings: `lr 1e-4`, `max_train_steps 600`, `markov_rank 256`,
confidence head on, `loss_decay_gamma 4.0`. See `dspark_block16_qwen3_4b.py` for
the full set and the reasoning in its docstring.

## `_best` vs `_bigdata`: why the smaller run ships

The shipped model is `_best`: the A10G run above, 2,400 downloaded rows / 2,280
after the 5% eval split. A larger arm (`_bigdata`, via `pipeline_h100`) scored
**worse on all six benchmark datasets** (−0.5% to −6.7%), which is why `_best` is
what ships.

That is a selection criterion, not a finding. `_bigdata` moved five variables at
once:

| | `_best` | `_bigdata` |
|---|---|---|
| conversations | 2,280 | 9,500 |
| sequence length | 768 | **512** |
| anchors | 32 | 96 |
| steps | 600 | 1400 |
| loss decay γ | 4 | 8 |

So the regression is **unattributable** — the shortened context is at least as
likely a culprit as the extra data. The prior phase logged this as a design error
rather than a result, and **whether data is the binding constraint remains
untested**. Do not cite this as evidence that more data hurts. Source:
`old-experiments/experiments/RESULTS.md` §8, and
`old-experiments/final_benchmark/config.py` for the checkpoint choice.

A related caution from the same table: the H100 arm reached the *lowest* training
loss (0.522) and still lost on acceptance — 1000 steps over 2,280 conversations is
31 epochs of memorisation. Train loss is not the metric here; acceptance is.

## How to run

```bash
pip install modal
modal setup                                   # one-time auth

# the shipped model: A10G, 2400 rows -> _best
modal run --detach modal_train.py::pipeline --exp-suffix _best

# the larger arm -> _bigdata (scored worse; see the caveat above).
# 9,500 is the conversation count recorded for that run in RESULTS.md §8; the
# exact original invocation was not committed, so treat this line as a
# reconstruction rather than a verbatim replay.
modal run --detach modal_train.py::pipeline_h100 \
    --exp-suffix _bigdata --num-samples 10000 --cache-tag _bigdata
```

`--detach` lets the run survive a dropped local connection. `publish` can be
called on its own to re-publish the latest checkpoint for a given `--exp-suffix`.

HF auth comes from the `huggingface` Modal secret / `HF_TOKEN` env var. Never
hardcode a token.

## Cost

The A10G `pipeline` run (`_best`) is a single-GPU job with a 12h timeout ceiling;
in practice the 600-step fine-tune plus the one-time data/cache build finish well
inside that. The `pipeline_h100` arm is more expensive per hour (H100, 80GB) and
processes 4× the data. Exact wall-clock and dollar cost depend on your Modal GPU
pricing and are not pinned here.
