"""Turn an instrumented run into a go/no-go on the confidence gate.

Consumes `confidence_by_round` (added by harness/ddtree/confidence.py) paired
positionally with `acceptance_lengths` from the SAME run. That pairing is the
whole point: the offline work in FINDINGS.md had to align two different runs on
token index, which broke down as bf16 divergence accumulated. One run carrying
both fields has no alignment problem at all.

Three questions, in the order that can kill the idea fastest:

  1. Does pred_chain_len track what actually happened? If not, stop.
  2. What is its error in tokens? FINDINGS.md section 4 says +-1.5-2 captures the
     whole benefit and +-3 mostly kills it. This is the pass/fail number.
  3. At each gate threshold, how many rounds get shrunk and what were they
     accepting? Rounds already saturating the block are free to shrink; rounds
     mid-range are not.

Input JSON: {"<dataset>": {"acc": [...], "conf": [{"pred_chain_len": ..}, ..]}}
with acc[i] and conf[i] describing the same round.

    python3 price_gate.py run.json
    python3 price_gate.py --self-test      # synthetic, verifies this script
"""
import argparse
import json
import math
import random
import sys


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    return pearson(rank(xs), rank(ys))


def report(name, acc, conf, ceiling, w_tree, w_small):
    pred = [c["pred_chain_len"] for c in conf]
    n = len(acc)
    print("=" * 74)
    print(f"{name}   n={n} rounds   block ceiling {ceiling}")
    print("=" * 74)

    # ---- 1. does the free signal track reality? ----
    r = pearson(pred, acc)
    rho = spearman(pred, acc)
    print(f"\n  1. pred_chain_len vs actual accepted: pearson {r:+.3f}  spearman {rho:+.3f}")
    if rho < 0.3:
        print("     VERDICT: signal too weak, gate is dead. Stop here.")
    elif rho < 0.5:
        print("     VERDICT: marginal.")
    else:
        print("     VERDICT: usable.")

    # ---- 2. error in tokens, against the +-1.5-2 spec ----
    err = [p - a for p, a in zip(pred, acc)]
    bias = sum(err) / n
    sd = math.sqrt(sum((e - bias) ** 2 for e in err) / max(n - 1, 1))
    print(f"\n  2. estimator error: bias {bias:+.2f} tokens, sd {sd:.2f} tokens")
    verdict = ("captures the full benefit" if sd <= 2.0 else
               "degraded but usable" if sd <= 3.0 else
               "too noisy -- FINDINGS section 4 says this buys almost nothing")
    print(f"     vs the +-1.5-2 token spec: {verdict}")

    # ---- 3. threshold sweep ----
    print(f"\n  3. gate: when pred_chain_len >= T, build {w_small}-wide instead of {w_tree}-wide")
    print(f"     {'T':>5s} {'%gated':>7s} {'mean acc on gated':>18s} "
          f"{'%gated at ceiling':>18s} {'mean width':>11s} {'saving':>8s}")
    for T in [t / 2 for t in range(2, 2 * ceiling + 1)]:
        gated = [a for a, p in zip(acc, pred) if p >= T]
        if not gated:
            continue
        frac = len(gated) / n
        at_ceiling = sum(1 for a in gated if a >= ceiling) / len(gated)
        width = frac * w_small + (1 - frac) * w_tree
        print(f"     {T:5.1f} {frac:7.1%} {sum(gated)/len(gated):18.2f} "
              f"{at_ceiling:18.1%} {width:11.1f} {width/w_tree - 1:+8.1%}")
    print("\n     Read the 'mean acc on gated' column: gating is safe only where it")
    print("     is at or near the ceiling, because those rounds needed no width.")
    print()


def self_test():
    """Synthetic run with a KNOWN structure, to verify this script reports it."""
    rng = random.Random(0)
    ceiling, n = 16, 900
    acc, conf = [], []
    for _ in range(n):
        # latent difficulty -> true accepted length; estimator sees it with sd~1.5
        true = min(ceiling, max(1, int(rng.gauss(7, 4))))
        acc.append(true)
        conf.append({"pred_chain_len": max(0.0, true + rng.gauss(0, 1.5))})
    print("SELF-TEST: synthetic data, estimator sd planted at 1.5 tokens,")
    print("strong rank correlation by construction. The report below should")
    print("recover sd ~= 1.5 and call the signal usable.\n")
    report("synthetic", acc, conf, ceiling, 65, 17)

    pred = [c["pred_chain_len"] for c in conf]
    assert spearman(pred, acc) > 0.7, "self-test: correlation not recovered"
    err = [p - a for p, a in zip(pred, acc)]
    sd = math.sqrt(sum((e - sum(err) / n) ** 2 for e in err) / (n - 1))
    assert 1.2 < sd < 1.8, f"self-test: sd not recovered ({sd})"
    print("self-test passed: planted sd 1.5 recovered as "
          f"{sd:.2f}, correlation recovered.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ceiling", type=int, default=16)
    ap.add_argument("--tree-width", type=int, default=65)
    ap.add_argument("--small-width", type=int, default=17)
    args = ap.parse_args()

    if args.self_test or not args.run_json:
        self_test()
        if not args.run_json:
            return
    data = json.load(open(args.run_json))
    for name, payload in data.items():
        acc, conf = payload["acc"], payload["conf"]
        if len(acc) != len(conf):
            print(f"{name}: SKIP -- acc has {len(acc)} rounds, conf has {len(conf)}; "
                  "these must be the same run", file=sys.stderr)
            continue
        report(name, acc, conf, args.ceiling, args.tree_width, args.small_width)


if __name__ == "__main__":
    main()
