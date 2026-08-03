"""Locked configuration for the concurrency benchmark.

RESULTS.md closes with one unproven row: "holds at serving concurrency —
untested, all results batch 1". This harness is the experiment that resolves it.

The design problem is that only two of the four methods exist in SGLang. The
builtin `SpeculativeAlgorithm` enum is DFLASH / DSPARK / EAGLE / EAGLE3 /
FROZEN_KV_MTP / STANDALONE / NGRAM — DDTree and sparked-tree are not there and
must arrive as plugins via `SpeculativeAlgorithm.register(...)`.

That creates a fairness trap. A plugin registered with `supports_overlap=False`
only runs under `--disable-overlap-schedule`, while the two builtins run on the
V2 overlap scheduler by default. Overlap is worth >33% on a single B200 (LMSYS,
"next-generation speculative decoding"), so the naive comparison would hand the
chains a third of a speedup that has nothing to do with chains-vs-trees — and
since §9 already predicts trees lose at concurrency, we would read a scheduler
artifact as confirmation of our own hypothesis.

So every arm is measured with overlap OFF, and the two chains are *additionally*
measured with overlap ON:

    overlap off    the fair 4-way comparison. Everyone equally handicapped.
    overlap on     chains only. The delta against their own off-run prices what
                   the trees are forfeiting by not being device-resident.

The second half is the actionable one. If overlap is worth 30% to DSpark and
sparked-tree beats DSpark by 12% with overlap off, sparked-tree loses in
production and the builder port is mandatory, not optional. If overlap is worth
5%, the port is a nice-to-have. That number is the input to the decision.

    setting        value            why
    -------------  ---------------  ---------------------------------------
    target         Qwen/Qwen3-4B    same as the batch-1 benchmark
    dataset        sharegpt         standard serving mix; the batch-1 harness
                                    used task datasets, which do not stress
                                    the scheduler
    concurrency    1..32            LMSYS used 1-32 for DFlash, so our numbers
                                    are methodology-comparable to theirs
    GPU            H100             standing methodology rule
    overlap        both             see above

Nothing here is swept. This run exists to produce one comparison.
"""

TARGET = "Qwen/Qwen3-4B"
DFLASH_DRAFT = "z-lab/Qwen3-4B-DFlash-b16"

# Unresolved: our block-16 DSpark drafter is warm-started from
# deepseek-ai/dspark_qwen3_4b_block7 and fine-tuned through DeepSpec
# (training/modal_train.py), and lives on the volume rather than on the Hub.
# Whether SGLang's DSPARK worker loads that checkpoint format is the one open
# question in this harness — see README. Point at the published block-7 base to
# get a running DSPARK arm, at the cost of comparing block-7 against our
# block-16 batch-1 numbers.
DSPARK_DRAFT = "deepseek-ai/dspark_qwen3_4b_block7"

CONCURRENCY = [1, 2, 4, 8, 16, 32]

# Enough completed requests at every concurrency to reach steady state. At c=1
# the floor dominates; at c=32 the per-slot term does.
MIN_PROMPTS = 64
PROMPTS_PER_SLOT = 10
WARMUP_REQUESTS = 8

DATASET = "sharegpt"
SEED = 0


def num_prompts(concurrency: int) -> int:
    return max(MIN_PROMPTS, PROMPTS_PER_SLOT * concurrency)


# `overlap=None` means the arm cannot express the setting at all, which is a
# result in itself and is rendered as such rather than dropped from the table.
ARMS = {
    # control: plain autoregressive decoding. Normalises speedups, and its own
    # on/off delta separates "overlap helps decoding" from "overlap helps
    # speculative decoding" — they are not the same number.
    "baseline_ov":   dict(algo=None,     draft=None,         overlap=True),
    "baseline_noov": dict(algo=None,     draft=None,         overlap=False),

    "dflash_ov":     dict(algo="DFLASH", draft=DFLASH_DRAFT, overlap=True),
    "dflash_noov":   dict(algo="DFLASH", draft=DFLASH_DRAFT, overlap=False),

    "dspark_ov":     dict(algo="DSPARK", draft=DSPARK_DRAFT, overlap=True),
    "dspark_noov":   dict(algo="DSPARK", draft=DSPARK_DRAFT, overlap=False),

    # --- the width-cost sweep. Every arm here is `capped` so the widths are
    # compared under one memory regime; dspark_capped is the chain reference,
    # re-run under the same caps rather than reused from the uncapped sweep.
    "dspark_capped":  dict(algo="DSPARK", draft=DSPARK_DRAFT, overlap=False,
                           capped=True),
    "tree_w17_noov":  dict(algo="SPARKED", draft=None, overlap=False, plugin=True,
                           num_draft_tokens=17, capped=True),
    "tree_w65_noov":  dict(algo="SPARKED", draft=None, overlap=False, plugin=True,
                           num_draft_tokens=65, capped=True),
    "tree_w129_noov": dict(algo="SPARKED", draft=None, overlap=False, plugin=True,
                           num_draft_tokens=129, capped=True),
}

# The sweep never exceeds c=32, so capping here costs no measured concurrency.
CAP_MAX_RUNNING = 32
CAP_MEM_FRACTION = 0.75

# Every rung of the width sweep sees the SAME prompt set, so a cross-rung trend
# is not confounded by workload. (The main sweep scales prompts with
# concurrency, which is right for throughput but wrong for comparing rungs.)
FIXED_PROMPTS = 96
REPEATS = 3

# Why the tree arms are measurable before the drafter is wired.
#
#     throughput  =  acceptance / round_time
#
# Section 3 of FINDINGS measured acceptance flat to within 3% across a 32x batch
# range: it is a property of the drafter and the tree, not of the scheduler. And
# round_time is a property of the VERIFY WIDTH and the batch -- the target model
# does not know or care which proposer produced the tokens it is verifying.
#
# So the two terms separate. Acceptance for the real sparked tree is already
# measured at batch 1 (RESULTS.md section 1: 7.842 at budget 64 vs 6.075 for the
# DSpark chain). What was never measured is how round_time scales with width as
# concurrency rises -- exactly the term section 9's compute-efficiency argument
# is about. The plugin supplies real trees at real widths, so it measures that
# term correctly no matter how good its proposals are.
#
# What this does NOT give: an end-to-end sparked-tree throughput number. The
# prediction below is a model, and it inherits every assumption in it.
ACCEPTANCE_BATCH1 = {      # RESULTS.md section 1 / section 9, H100, batch 1
    "dflash_chain": 5.602,
    "dspark_chain": 6.075,
    "ddtree_tb64": 7.457,
    "sparked_tb64": 7.842,
    "ddtree_tb128": 7.742,
    "sparked_tb128": 8.437,
}
WIDTH_OF_ARM = {"dspark_capped": 17, "tree_w17_noov": 17,
                "tree_w65_noov": 65, "tree_w129_noov": 129}

# Acceptance re-measured in the SAME setting the cost sweep ran in: the published
# block-7 drafter (not our block-16 checkpoint), on the two chat-shaped datasets
# this repo supports. `validate_acceptance.py`, via ddtree/benchmark.py unchanged.
#
# Absolute values sit well below RESULTS.md (5.4 vs 7.8) exactly as expected for
# block-7 on chat rather than block-16 on task data. The prediction consumes only
# the RATIO, and the ratio came out HIGHER than the batch-1 splice assumed:
#
#     tb64   1.386 (alpaca) / 1.334 (mt-bench) -> 1.360   vs 1.291 assumed  (+5.4%)
#     tb128  1.407 (alpaca) / 1.396 (mt-bench) -> 1.402   vs 1.389 assumed  (+0.9%)
#
# So the first crossover was CONSERVATIVE toward the chain, not toward our own
# hypothesis. Using the measured ratios moves tb64's c=4 point from 0.94 to 0.99
# -- still below 1.0, so the crossover rung does not move, but it is now marginal
# rather than clear, and that should be said out loud.
MEASURED_ACCEPTANCE = {          # H100, block-7 drafter, mean over the 2 datasets
    "sparked_tb64": 1.360,       # ratio vs the DSpark chain, not an absolute
    "sparked_tb128": 1.402,
}
MEASURED_ACCEPTANCE_RAW = {
    "alpaca":   {"dspark": 3.931, "sparked_tb64": 5.449, "sparked_tb128": 5.531},
    "mt-bench": {"dspark": 3.799, "sparked_tb64": 5.069, "sparked_tb128": 5.305},
}

# Declared, not yet runnable. Kept in the config so the report prints an
# explicit blocked row: a table that silently omits two of four arms reads as
# though the comparison was made.
#
# Both are unblocked by ONE plugin. The template is NGRAM, not EAGLE:
# NgramVerifyInput is a shipping non-EAGLE tree whose shape profile is ours —
# node-budgeted with no depth cap ("spec_steps is meaningless for this tree")
# and tree_topk = -1, the documented sentinel for "irregular tree, no fixed
# per-level branching". EagleVerifyInput.max_tree_depth carries the matching
# invitation: "Algorithms with other tree shapes override this."
#
# The verify kernel is generic over tree shape — verify_tree_greedy_func walks
# retrieve_index / retrieve_next_token / retrieve_next_sibling, a first-child /
# next-sibling encoding our parents + child_maps convert to mechanically. At
# temperature 0.0 (which config locks) we hit that greedy path and skip the
# rejection-sampling branch and its draft_probs entirely.
BLOCKED_ARMS = {
    "ddtree_noov": (
        "no DDTREE builtin; needs a plugin via SpeculativeAlgorithm.register(). "
        "Same plugin as sparked_noov with build_ddtree_tree swapped in for the "
        "markov builder — one plugin unblocks both arms."
    ),
    "sparked_noov": (
        "plugin not yet written. Verify path is reusable (see above); the work "
        "is a V2 worker plus a SpecInput subclass. Overlap stays off regardless: "
        "the builder syncs at int(root_token_id) in and .cpu().numpy() out "
        "(ddtree_markov.py:851,855), plus the numpy heapq walk in the shipped "
        "exact-precomputed arm."
    ),
}

# The comparisons the report is built to make. Ratio of output tok/s, per
# concurrency level.
DELTAS = [
    ("overlap value, baseline", "baseline_ov", "baseline_noov"),
    ("overlap value, DFlash",   "dflash_ov",   "dflash_noov"),
    ("overlap value, DSpark",   "dspark_ov",   "dspark_noov"),
    ("DSpark vs DFlash (off)",  "dspark_noov", "dflash_noov"),
]
