"""Is there a zero-cost signal for "will tree width pay this round?"

The oracle selector (oracle_selector.py) is worth -24% verify width at +4.9%
acceptance, but it needs perfect foresight. This asks whether a signal you
already have for free -- the PREVIOUS round's accepted length -- predicts it.

Free matters. The confidence head costs an extra forward pass and was already
measured at -2.7% acc / -6.6% spd when pointed at width allocation. History
costs nothing: you just observed it.

Controls:
  * within-dataset AUC, so the signal is not merely encoding "this is mt-bench"
  * a shuffled-label null, so AUC is compared against chance on this sample size
"""
import json
import random
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
    """Aligned rounds with a causal feature: the tree's own previous round length.

    Only history is used, so this is what a deployed policy could actually see.
    """
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
            tlens = ts["acc"]
            pos = 0
            for j, n in enumerate(tlens):
                if j > 0 and pos in cmap and pos < horizon:
                    rows.append({
                        "ds": ds,
                        "prev": tlens[j - 1],
                        "prev2": tlens[j - 2] if j > 1 else tlens[j - 1],
                        "chain": cmap[pos],
                        "tree": n,
                        "gain": n - cmap[pos],
                    })
                pos += n
    return rows


def auc(scores, labels):
    """P(score of a positive > score of a negative), ties counted as half."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return wins / (len(pos) * len(neg))


def null_auc(scores, labels, trials=400, seed=0):
    """Distribution of AUC under label shuffling -> what chance looks like here."""
    rng = random.Random(seed)
    out = []
    lab = list(labels)
    for _ in range(trials):
        rng.shuffle(lab)
        out.append(auc(scores, lab))
    out.sort()
    return out[int(0.025 * trials)], out[int(0.975 * trials)]


def main():
    d = json.loads(EXP1.read_text())
    per_dataset = d["per_dataset"]

    for chain_name, tree_name, ceiling, w_chain, w_tree, label in PAIRS:
        rows = collect(per_dataset, chain_name, tree_name, ceiling)
        n = len(rows)
        print("=" * 76)
        print(f"{label}   n={n} aligned rounds with history")
        print("=" * 76)

        # ---- does the feature separate the classes at all? ----
        print("\n  mean previous-round length, split by what happened next:")
        for name, sel in [("tree gained (>0)", lambda r: r["gain"] > 0),
                          ("tree tied (=0)", lambda r: r["gain"] == 0),
                          ("tree lost (<0)", lambda r: r["gain"] < 0)]:
            grp = [r for r in rows if sel(r)]
            if grp:
                print(f"    {name:<18s} n={len(grp):4d}  prev={sum(r['prev'] for r in grp)/len(grp):5.2f}"
                      f"   this-round chain={sum(r['chain'] for r in grp)/len(grp):5.2f}")

        # ---- discrimination, pooled and within-dataset ----
        scores = [-r["prev"] for r in rows]          # low prev => expect tree to help
        labels = [r["gain"] > 0 for r in rows]
        a = auc(scores, labels)
        lo, hi = null_auc(scores, labels)
        print(f"\n  AUC(prev-round length -> tree gains)  pooled : {a:.3f}"
              f"   [null 95% band {lo:.3f}-{hi:.3f}]")

        for ds in sorted({r['ds'] for r in rows}):
            sub = [r for r in rows if r["ds"] == ds]
            s = [-r["prev"] for r in sub]
            y = [r["gain"] > 0 for r in sub]
            a_ds = auc(s, y)
            print(f"    within {ds:<10s} n={len(sub):4d}  AUC {a_ds:.3f}"
                  f"   (base rate {sum(y)/len(y):.1%})")

        # ---- what a threshold policy actually buys ----
        print(f"\n  Policy: if previous round accepted >= T, run the CHAIN this round.")
        print(f"  {'T':>3s} {'%chain':>7s} {'accept':>7s} {'width':>7s} "
              f"{'vs always-tree':>16s}")
        base_acc = sum(r["tree"] for r in rows) / n
        for T in range(1, ceiling + 2):
            use_chain = [r["prev"] >= T for r in rows]
            acc = sum(r["chain"] if u else r["tree"] for r, u in zip(rows, use_chain)) / n
            frac = sum(use_chain) / n
            width = frac * w_chain + (1 - frac) * w_tree
            print(f"  {T:3d} {frac:7.1%} {acc:7.3f} {width:7.1f} "
                  f"{acc/base_acc - 1:+7.1%} acc {width/w_tree - 1:+7.1%} wid")

        orc = sum(max(r["chain"], r["tree"]) for r in rows) / n
        fc = sum(1 for r in rows if r["tree"] <= r["chain"]) / n
        print(f"  {'ORC':>3s} {fc:7.1%} {orc:7.3f} "
              f"{fc*w_chain+(1-fc)*w_tree:7.1f} "
              f"{orc/base_acc - 1:+7.1%} acc "
              f"{(fc*w_chain+(1-fc)*w_tree)/w_tree - 1:+7.1%} wid   <- perfect foresight")
        print()


if __name__ == "__main__":
    main()
