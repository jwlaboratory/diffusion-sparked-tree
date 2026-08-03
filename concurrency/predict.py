"""Predict sparked-tree throughput at concurrency from measured width cost.

    python3 concurrency/predict.py concurrency/results_width.json

The identity this rests on:

    throughput  =  acceptance / round_time

Both terms are measurable separately, and neither needs the other:

  * **acceptance** is a property of the drafter and the tree. FINDINGS section 3
    measured it flat to within 3% across a 32x batch range, so a batch-1 number
    carries over to serving unchanged. The ratios used here were re-measured on
    the SAME block-7 drafter and chat workload as the cost sweep
    (`validate_acceptance.py`), so the two halves no longer come from different
    checkpoints.
  * **round_time** is a property of the verify width and the batch. The target
    model verifies `width` tokens per request per round; it neither knows nor
    cares which proposer produced them. So a plugin with mediocre proposals
    measures the cost of a width-65 tree exactly as well as a good one would.

That separation is what makes this run meaningful before the DSpark drafter is
wired into the plugin. What it produces is a *prediction*, not a measurement of
sparked-tree throughput, and it inherits two assumptions worth stating plainly:

  1. round_time depends on width but not on tree SHAPE. Two width-65 trees with
     different branching verify the same number of tokens through the same
     attention mask; only the mask bits differ. Believed safe, not verified.
  2. Our builder's own cost is not in here. It is host-resident (~3.8 ms/round
     best-first, RESULTS.md section 12) and does NOT amortise across a batch, so
     the real arm is SLOWER than this predicts -- increasingly so with
     concurrency. Every number below is therefore an upper bound.

The obvious objection, checked rather than waved away: the stand-in proposer
fills only ~24 of 129 slots at the widest setting, so most verify rows are
padding. That does not deflate the cost. `worker.py` builds the FULL_MASK prefix
as `torch.ones((N, seq_len))` over **all** N rows -- padded rows are isolated
only within the tree block, and still attend the entire committed prefix. The
target therefore runs a genuine width-N forward with full prefix attention on
every row, which is where essentially all the cost is; the tree block is N x N
against a prefix of hundreds to thousands. Sparse trees and dense trees of the
same width cost the same.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402


def round_time_ms(row):
    """ms per decode round = (ms per output token) x (tokens per round)."""
    tpot, accept = row.get("mean_tpot_ms"), row.get("accept_length")
    if not tpot or not accept:
        return None
    return tpot * accept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--out", default="concurrency/PREDICTION.md")
    args = parser.parse_args()

    payload = json.load(open(args.json_path))
    results, concurrency = payload["results"], payload["concurrency"]

    def get(arm, c, key="output_throughput"):
        rows = results.get(arm) or {}
        return (rows.get(str(c)) or rows.get(c) or {}).get(key)

    def rt(arm, c):
        rows = results.get(arm) or {}
        return round_time_ms(rows.get(str(c)) or rows.get(c) or {})

    out = [
        "# Predicted sparked-tree throughput at concurrency",
        "",
        f"`{payload['gpu']}` · `{payload['sglang_image']}` · `{payload['target']}` "
        f"· `{payload['dataset']}`",
        "",
        "Derived, not measured end to end. See the module docstring of",
        "`predict.py` for the identity and its two assumptions. Every number here",
        "is an **upper bound**: the host-resident builder's own per-round cost is",
        "not included, and it does not amortise across a batch.",
        "",
        "## Measured: round time by verify width (ms/round)",
        "",
        "| arm | width | " + " | ".join(f"c={c}" for c in concurrency) + " |",
        "|---|---|" + "---|" * len(concurrency),
    ]
    for arm, width in cfg.WIDTH_OF_ARM.items():
        cells = [f"{rt(arm, c):.2f}" if rt(arm, c) else "—" for c in concurrency]
        out.append(f"| {arm} | {width} | " + " | ".join(cells) + " |")

    out += ["", "## Measured: cost of width, relative to the width-17 chain", "",
            "| width | " + " | ".join(f"c={c}" for c in concurrency) + " |",
            "|---|" + "---|" * len(concurrency)]
    ref = "dspark_capped"
    for arm, width in cfg.WIDTH_OF_ARM.items():
        if arm == ref:
            continue
        cells = []
        for c in concurrency:
            a, b = rt(arm, c), rt(ref, c)
            cells.append(f"{a / b:.2f}x" if a and b else "—")
        out.append(f"| {width} | " + " | ".join(cells) + " |")

    out += ["", "## Predicted: sparked-tree vs the DSpark chain", "",
            "`(acceptance ratio) / (round-time ratio)`, using acceptance ratios",
            f"**re-measured on the same block-7 drafter and chat workload as the",
            f"cost sweep**: {cfg.MEASURED_ACCEPTANCE['sparked_tb64']}x at budget 64, "
            f"{cfg.MEASURED_ACCEPTANCE['sparked_tb128']}x at 128. (The batch-1 splice "
            f"assumed 1.291 and 1.389 — it understated the tree.)",
            "",
            "| arm | " + " | ".join(f"c={c}" for c in concurrency) + " |",
            "|---|" + "---|" * len(concurrency)]

    # The direct ratio rt(tree)/rt(chain) is WRONG, and the first sweep made it
    # obvious: tree_w17 came out FASTER than the width-17 DSpark chain (7.37 vs
    # 9.88 ms/round at c=1). Same verify width, so that gap is not width -- it is
    # the DSpark drafter forward, which our stand-in proposer does not pay.
    #
    # So isolate the width increment using two arms that differ ONLY in width and
    # share the same (near-free) proposer, then add it to the real chain:
    #
    #     round_time_sparked(c) ~= rt(dspark_chain, c) + [rt(tree_wN, c) - rt(tree_w17, c)]
    #
    # This keeps the drafter cost exactly once and attributes the rest to width.
    predictions = {}
    for arm, acc_key, width in (("tree_w65_noov", "sparked_tb64", 65),
                                ("tree_w129_noov", "sparked_tb128", 129)):
        # Prefer the ratio measured on the same checkpoint and workload as the
        # cost sweep; fall back to the batch-1 splice only if it is missing.
        acc_ratio = cfg.MEASURED_ACCEPTANCE.get(
            acc_key,
            cfg.ACCEPTANCE_BATCH1[acc_key] / cfg.ACCEPTANCE_BATCH1["dspark_chain"])
        cells, series = [], {}
        for c in concurrency:
            wide, narrow, chain = rt(arm, c), rt("tree_w17_noov", c), rt(ref, c)
            if not wide or not narrow or not chain:
                cells.append("—")
                continue
            width_delta = wide - narrow
            pred_round_time = chain + width_delta
            if pred_round_time <= 0:
                cells.append("—")
                continue
            pred = acc_ratio / (pred_round_time / chain)
            series[c] = pred
            cells.append(f"**{pred:.2f}x**" if pred >= 1.0 else f"{pred:.2f}x")
        predictions[acc_key] = series
        out.append(f"| {acc_key} (w={width}) | " + " | ".join(cells) + " |")

    out += ["", "Width increment isolated from the two arms that differ only in "
            "width (`tree_w65`/`tree_w129` minus `tree_w17`), then added to the "
            "measured DSpark chain round time — so the drafter forward is counted "
            "once, not zero times or twice. See the code comment for why the "
            "direct ratio would be wrong."]

    # Does round_time actually grow linearly in width? The additive model above
    # assumes the increment 17->65 and 17->129 come from the same per-token cost.
    # Three widths is enough to check: fit through (17, 65) and see what it
    # predicts at 129. A large miss means the cost is superlinear in width and
    # the tb128 prediction is optimistic.
    out += ["", "## Model check: is round time linear in width?", "",
            "The additive model assumes a constant per-verify-token cost. Fit the",
            "slope on widths 17->65, extrapolate to 129, compare to measured.", "",
            "| c | slope (ms/token) | predicted w=129 | measured w=129 | error |",
            "|---|---|---|---|---|"]
    linearity = []
    for c in concurrency:
        a17, a65, a129 = (rt("tree_w17_noov", c), rt("tree_w65_noov", c),
                          rt("tree_w129_noov", c))
        if not (a17 and a65 and a129):
            out.append(f"| {c} | — | — | — | — |")
            continue
        slope = (a65 - a17) / (65 - 17)
        pred129 = a17 + slope * (129 - 17)
        err = (pred129 - a129) / a129
        linearity.append(abs(err))
        out.append(f"| {c} | {slope:.3f} | {pred129:.1f} | {a129:.1f} | "
                   f"{err * 100:+.0f}% |")
    if linearity:
        worst = max(linearity) * 100
        verdict = ("linear enough to trust the additive model"
                   if worst < 20 else
                   "NOT linear -- the additive model is only a rough guide, and "
                   "the wider arm's prediction is the less reliable one")
        out.append("")
        out.append(f"Worst extrapolation error **{worst:.0f}%** — {verdict}.")

    out += ["", "## Crossover", ""]
    for key, series in predictions.items():
        below = [c for c, v in series.items() if v < 1.0]
        if not series:
            out.append(f"- **{key}**: no data.")
        elif not below:
            out.append(f"- **{key}**: stays above 1.0x through c="
                       f"{max(series)} — no crossover in the measured range.")
        else:
            first = min(below)
            margin = ("a tie rather than a loss" if series[first] >= 0.97
                      else "a clear loss")
            out.append(f"- **{key}**: falls below 1.0x at **c={first}** "
                       f"({series[first]:.2f}x — {margin} at that rung). Above it "
                       f"the DSpark chain wins on throughput, even granting the "
                       f"tree its full measured acceptance advantage and charging "
                       f"it nothing for building the tree.")
    out += ["",
            "Both lines are upper bounds. Adding the builder's ~3.8 ms/round of "
            "host-serial work — which does not amortise — moves every crossover "
            "left, and moves it further left the larger the batch.",
            ""]

    Path(args.out).write_text("\n".join(out))
    print("\n".join(out))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
