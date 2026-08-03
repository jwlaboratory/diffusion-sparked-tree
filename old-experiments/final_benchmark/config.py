"""The single locked configuration for the final benchmark.

Every value here was chosen by a measurement, not a default. The provenance
column is the experiment that decided it; anything without one is inherited from
the published setup and was not re-litigated.

    setting            value              decided by
    -----------------  -----------------  ------------------------------------
    checkpoint         _best              reproduces published §1 to +0.1%;
                                          _bigdata is worse on all 6 datasets
    block size         16                 §7: trees gain ~2x what chains gain
                                          from the longer horizon
    tree builder       exact-precomputed  +6.2% acceptance over flat beam at
                                          budget 64 (6/6 datasets), +2.3% @128
    candidate pool C   512                256 costs 3.6% acceptance; 2048 buys
                                          4% more at 4x the build time
    min_width          2                  never emit a [1]*depth chain; binds
                                          only below budget 32
    max_fanout         0 (= budget)       capping at 48 traded 0.4% acceptance
                                          for build time that does not resolve
    temperature        0.0                greedy, exact-match acceptance
    GPU                H100               standing methodology rule: H100 is the
                                          reference GPU

Why best-first rather than the CUDA-graphed beam: the graphed beam builds a tree
in 0.94 ms against best-first's 3.82 ms (H100, budget 64), but builder time is
~5% of the round while acceptance sets the number of rounds. Acceptance also
resolves at ~0.5% in this harness whereas per-cell speed does not resolve below
~16%, so the acceptance win is the measurable one. The graphed beam stays
available for latency-critical use and is benchmarked alongside as `beam_graphed`.
"""

TARGET = "Qwen/Qwen3-4B"
DFLASH_DRAFT = "z-lab/Qwen3-4B-DFlash-b16"
CHECKPOINT = "/vol/ckpt_final/dspark_block16_best"

# The released DSpark drafter. Needed because benchmark.py derives BOTH the
# `dspark` chain row and the `dsparktree_*` rows from one checkpoint: pointing it
# at our fine-tune makes the "DSpark" row our own drafter with the tree removed,
# which is the right ablation but is NOT the published DSpark baseline. Running it
# as a separate arm gives the real one. DFlash and DDTree never touch this model,
# so they are identical across arms and act as the bridge that makes the two
# runs comparable.
DSPARK_RELEASED = "deepseek-ai/dspark_qwen3_4b_block7"

DATASETS = ["humaneval", "mbpp", "gsm8k", "math500", "mt-bench", "alpaca"]
TREE_BUDGETS = "64,128"
MAX_SAMPLES = 12
MAX_NEW_TOKENS = 512

# the arm under test, and the one it is derived from
SPARKED = dict(tree_mode="exact-precomputed", beam_candidates=512, beam_min_width=2, max_fanout=0)
BEAM_GRAPHED = dict(tree_mode="beam", beam_candidates=512, beam_min_width=2, cuda_graph=True)
# baseline arm: released DSpark checkpoint, so the `dspark` row is the real method
DSPARK_PAPER = dict(tree_mode="beam", beam_candidates=512, beam_min_width=2,
                    checkpoint=DSPARK_RELEASED)

# Methods reported. `baseline` is plain autoregressive decoding and exists only to
# normalise speedups; dflash is free (the harness runs it regardless).
REPORTED = ["baseline", "dflash", "dspark", "ddtree", "sparked"]
