"""Does tree width actually convert? Replay accepted-length logs.

Every method in this harness is greedy and byte-identical to plain autoregressive
decoding, so all methods emit the SAME token sequence and merely partition it into
rounds differently. That gives a token-index alignment axis: wherever two methods
share a round boundary they are in an identical state (same context, same
position), so their NEXT round is a controlled head-to-head.

Three questions:
  A. How much of the tree budget sits at depths the round never reaches?
  B. How often does a round saturate the block (ceiling = block_size + 1)?
  C. At shared boundaries, how often does the tree accept exactly what the
     chain would have -- i.e. the width bought nothing?
"""
from collections import Counter

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP1 = ROOT / "experiment1-harness/Results/results_detailed.json"

# (chain method, tree method, depth_limit, ceiling, label)
PAIRS = [
    ("dspark.chain", "dspark.markov.tree", 7, 8, "DSpark b7: chain vs markov tree"),
    ("dflash.chain", "dflash.tree", 15, 16, "DFlash b16: chain vs DDTree"),
]


def cuts(lengths):
    """Round-boundary token positions -> length of the round starting there."""
    out, pos = {}, 0
    for n in lengths:
        out[pos] = n
        pos += n
    return out, pos


def main():
    d = json.loads(EXP1.read_text())
    per_dataset = d["per_dataset"]
    budget = d["cfg"]["tree_budget"]
    datasets = list(per_dataset)

    # ---------- A + B: distributions, no alignment needed ----------
    print("=" * 78)
    print("A/B. ROUND-LENGTH DISTRIBUTION, SATURATION, AND WASTED DEPTH")
    print("=" * 78)
    print(f"tree_budget={budget}, flat schedule assumed (equal width at every depth)\n")
    print(f"{'method':<22s} {'dataset':<10s} {'rounds':>7s} {'mean':>6s} "
          f"{'satur':>7s} {'len<=2':>7s} {'wasted':>7s}")
    print("-" * 78)
    for name, depth_limit, ceiling in [
        ("dspark.chain", 7, 8), ("dspark.markov.tree", 7, 8),
        ("dflash.chain", 15, 16), ("dflash.tree", 15, 16),
        ("dflash.markov.tree", 15, 16),
    ]:
        for ds in datasets:
            m = per_dataset[ds]["methods"].get(name)
            if not m:
                continue
            L = m["lengths"]
            n = len(L)
            mean = sum(L) / n
            sat = sum(1 for x in L if x >= ceiling) / n
            short = sum(1 for x in L if x <= 2) / n
            # depths beyond the accepted length were provisioned but never reached
            wasted = sum(max(0, ceiling - x) for x in L) / (n * ceiling)
            print(f"{name:<22s} {ds:<10s} {n:7d} {mean:6.2f} {sat:6.1%} "
                  f"{short:6.1%} {wasted:6.1%}")
        print()

    # ---------- C: aligned head-to-head ----------
    print("=" * 78)
    print("C. CHAIN vs TREE AT SHARED ROUND BOUNDARIES")
    print("=" * 78)
    print("At a shared boundary both methods have identical context, so the next")
    print("round is a controlled comparison. 'gain 0' = the tree accepted exactly")
    print("what the chain did => all tree width was wasted that round.\n")

    for chain_name, tree_name, depth_limit, ceiling, label in PAIRS:
        print(f"--- {label} ---")
        tot = Counter()
        gains = []
        first_round = []          # position 0: zero selection bias
        for ds in datasets:
            meths = per_dataset[ds]["methods"]
            if chain_name not in meths or tree_name not in meths:
                continue
            ds_gains = []
            for cs, ts in zip(meths[chain_name]["per_sample"],
                              meths[tree_name]["per_sample"]):
                cmap, cend = cuts(cs["acc"])
                tmap, tend = cuts(ts["acc"])
                # drop the final round of each: max_new_tokens truncates it
                horizon = min(cend, tend) - max(ceiling, 16)
                shared = [p for p in cmap if p in tmap and p < horizon]
                for p in shared:
                    g = tmap[p] - cmap[p]
                    ds_gains.append(g)
                    tot[g] += 1
                    if p == 0:
                        first_round.append(g)
            if ds_gains:
                zero = sum(1 for g in ds_gains if g <= 0) / len(ds_gains)
                print(f"  {ds:<10s} n={len(ds_gains):4d}  "
                      f"gain 0: {zero:5.1%}   mean gain: "
                      f"{sum(ds_gains)/len(ds_gains):+.2f} tokens")
            gains += ds_gains

        if gains:
            zero = sum(1 for g in gains if g <= 0) / len(gains)
            print(f"  {'ALL':<10s} n={len(gains):4d}  gain 0: {zero:5.1%}   "
                  f"mean gain: {sum(gains)/len(gains):+.2f} tokens")
            print(f"    gain distribution: "
                  f"{ {k: tot[k] for k in sorted(tot)} }")
        if first_round:
            z = sum(1 for g in first_round if g <= 0) / len(first_round)
            print(f"    [unbiased subset] round 1 of every prompt: n={len(first_round)}, "
                  f"gain 0: {z:.1%}, mean gain {sum(first_round)/len(first_round):+.2f}")
        print()


if __name__ == "__main__":
    main()
