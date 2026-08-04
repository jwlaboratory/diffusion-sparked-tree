"""Two things: repair the alignment caveat, then test entropy-based budget cuts.

PART 1 -- validate_alignment.py showed methods DISAGREE on n_out (spread up to
113 tokens). So the "all methods emit the same text" premise is only true until
the first bf16 divergence; after that a shared token index is NOT a shared
context and the pairing is broken. Alignment yield is front-loaded (median
position 0.10-0.43 of the output), exactly as compounding divergence predicts.

We cannot detect divergence directly -- only round lengths were logged, not
token ids. So instead: sweep a cutoff on position-in-output and check whether
the estimate is stable. Position 0 (round 1 of each prompt) is the only sample
that is divergence-free by construction.

PART 2 -- the proposal: when the drafter is uncertain, the target will reject
everything anyway, so shrink the budget and save compute. That predicts gain
from tree width should be LOW on hard rounds. We have no logged entropy, but
chain acceptance is the quantity entropy would be predicting, so bucketing by
it tests the same hypothesis.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP1 = ROOT / "experiment1-harness/Results/results_detailed.json"

PAIRS = [
    ("dspark.chain", "dspark.markov.tree", 8, 17, 65, "DSpark b7 (block 7)"),
    ("dflash.chain", "dflash.tree", 16, 16, 65, "DFlash b16 vs DDTree (block 16)"),
]


def cuts(lengths):
    out, pos = {}, 0
    for n in lengths:
        out[pos] = n
        pos += n
    return out, pos


def collect(per_dataset, chain_name, tree_name, ceiling):
    rows = []
    for ds, payload in per_dataset.items():
        meths = payload["methods"]
        if chain_name not in meths or tree_name not in meths:
            continue
        for cs, ts in zip(meths[chain_name]["per_sample"],
                          meths[tree_name]["per_sample"]):
            cmap, cend = cuts(cs["acc"])
            tmap, tend = cuts(ts["acc"])
            horizon = min(cend, tend) - max(ceiling, 16)
            for p in tmap:
                if p in cmap and p < horizon:
                    rows.append({
                        "ds": ds,
                        "frac": p / max(tend, 1),
                        "round_idx": None,
                        "chain": cmap[p],
                        "tree": tmap[p],
                        "gain": tmap[p] - cmap[p],
                    })
            # tag round index within the prompt
            order = sorted(r for r in tmap)
            idx = {p: i for i, p in enumerate(order)}
            for r in rows[-len([p for p in tmap if p in cmap and p < horizon]):]:
                pass
    return rows


def main():
    d = json.loads(EXP1.read_text())
    per_dataset = d["per_dataset"]

    for chain_name, tree_name, ceiling, w_chain, w_tree, label in PAIRS:
        rows = collect(per_dataset, chain_name, tree_name, ceiling)
        print("=" * 76)
        print(f"{label}")
        print("=" * 76)

        print("\n  PART 1. IS THE ESTIMATE STABLE AS DIVERGENCE ACCUMULATES?")
        print("  Restricting to boundaries in the first X of the output. Early")
        print("  boundaries are the ones least likely to sit after a divergence.\n")
        print(f"    {'first X of output':>18s} {'n':>5s} {'gain=0':>8s} {'tree<chain':>11s} "
              f"{'mean gain':>10s} {'oracle acc':>11s} {'oracle wid':>11s}")
        for cut in [0.05, 0.10, 0.25, 0.50, 1.00]:
            sub = [r for r in rows if r["frac"] <= cut]
            if len(sub) < 15:
                continue
            n = len(sub)
            z = sum(1 for r in sub if r["gain"] == 0) / n
            neg = sum(1 for r in sub if r["gain"] < 0) / n
            mg = sum(r["gain"] for r in sub) / n
            t_acc = sum(r["tree"] for r in sub) / n
            o_acc = sum(max(r["chain"], r["tree"]) for r in sub) / n
            fc = sum(1 for r in sub if r["tree"] <= r["chain"]) / n
            o_w = fc * w_chain + (1 - fc) * w_tree
            print(f"    {cut:18.2f} {n:5d} {z:8.1%} {neg:11.1%} {mg:+10.2f} "
                  f"{o_acc/t_acc - 1:+10.1%} {o_w/w_tree - 1:+10.1%}")

        print("\n  PART 2. DOES TREE WIDTH PAY OFF *LESS* ON HARD ROUNDS?")
        print("  The proposal says yes: high entropy => rejected anyway => shrink budget.")
        print("  Bucketing by chain acceptance, which is what an entropy signal estimates.\n")
        print(f"    {'chain accepted':>14s} {'n':>5s} {'share':>7s} {'mean tree':>10s} "
              f"{'mean gain':>10s} {'share of all gain':>18s}")
        total_gain = sum(max(r["gain"], 0) for r in rows)
        buckets = {}
        for r in rows:
            buckets.setdefault(min(r["chain"], ceiling), []).append(r)
        for k in sorted(buckets):
            grp = buckets[k]
            g = sum(max(x["gain"], 0) for x in grp)
            print(f"    {k:14d} {len(grp):5d} {len(grp)/len(rows):7.1%} "
                  f"{sum(x['tree'] for x in grp)/len(grp):10.2f} "
                  f"{sum(x['gain'] for x in grp)/len(grp):+10.2f} "
                  f"{g/total_gain:18.1%}")
        print()


if __name__ == "__main__":
    main()
