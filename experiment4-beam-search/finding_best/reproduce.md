# Experiment 4a — finding the best beam width schedule

## Question

The level-synchronous beam builder (`harness/ddtree/sparked_tree.py::build_beam_tree`)
fixes the tree shape up front: `widths[d]` nodes at depth `d`, one batched matmul per
level, one GPU sync per round. That removes the ~budget dependent round-trips that made
the naive best-first tree ~20x more expensive than DDTree's — but it forces a choice the
naive builder never had to make: **how should the node budget be distributed across
depths?**

Intuition says front-load (branch wide at the root, deep nodes are rarely reached).
The old experiments (old-experiments/SPEEDUP_EXPERIMENTS.md §3a) measured the opposite:
the drafter is 87% top-1-correct at depth 1 and only 30% at depth 16, so flat beat every
decaying schedule. This experiment re-proves that on the current harness.

## Arms

All arms: DSpark-b16 backbone + its own markov head, tree verify. Only the schedule
differs (beam arms) or the builder (reference arm).

| arm | schedule @ budget 64 | tests |
|---|---|---|
| beam.geo50 | [32,16,8,4,2,1,1] | heavy front-load |
| beam.geo60 | [26,15,9,6,3,2,1,1,1] | the old default decay |
| beam.geo75 | [16,12,9,7,5,4,3,2,2,1,1,1,1] | mild front-load |
| beam.flat | [4]*16 | uniform (hypothesized winner) |
| beam.invgeo75 | reversed geo75 | mild back-load — mirror control, same width multiset |
| beam.invgeo50 | reversed geo50 | heavy back-load |
| beam.depth8 | [8]*8 | uniform, 2x wide / half deep |
| beam.depth4 | [16]*4 | uniform, 4x wide / quarter deep |
| bestfirst.ref | adaptive (naive builder) | acceptance ceiling any fixed schedule chases |

Two orthogonal axes: the geo/flat/invgeo family varies *orientation* (where the width
goes) at full depth; the depth8/depth4 arms vary *depth-vs-width* at uniform shape.

Budgets 64 and 256 (matching exp3), gsm8k/humaneval/mt-bench x 4 samples, 512 tokens,
temp 0, H100 + 8 CPU. Note: geometric schedules can under-spend by a node or two to
rounding (geo75@256 spends 254/256); flat/flat_depth always spend exactly the budget.

## Metrics & how to read them

- **mean acceptance length** — the headline. Resolves at ~0.5% in this harness. The
  schedule question is an acceptance question: builder time is ~5% of a round.
- clean-pass TPS — reported, but per-cell speed noise is ~16%; don't rank on it.
- instrumented phase shares — confirms the beam killed candidate_build dominance
  (naive arm was 88–95% candidate_build in exp3).

## Run

```bash
cd experiment4-beam-search/finding_best
modal run modal_benchmark.py --smoke     # minutes; validates end-to-end
modal run modal_benchmark.py --spawn     # full run, fire-and-forget, checkpointed
# progress: modal volume ls ddtree-results beamsched/cache/<fingerprint>
# fetch:    modal volume get ddtree-results beamsched/summary.json results/summary.json
```

Per-(budget, dataset) units are checkpointed to the `ddtree-results` volume; an
interruption costs at most one unit. Resume by re-running the same command.

## Implementation notes

- Beam builder + schedules added to the shared harness (`sparked_tree.py`): schedule
  specs are resolved once per generate call via `resolve_width_schedule`; the builder
  keeps survivors on GPU between levels and does ONE `.cpu()` transfer at the end.
- Wired through `Method.tree_kwargs` -> `sparked_tree_generate(tree_mode="beam",
  beam_schedule=..., beam_candidates=2048)`. The best-first path is untouched
  (defaults preserve exp3 behavior); driver CODE_VERSION bumped to `harness-3-beam`
  so no stale exp3 cache unit can be resumed against the new code.
- CPU sanity tests for the builder (structure, visibility, schedules, mirror
  property, guards) ran green locally before launch.
