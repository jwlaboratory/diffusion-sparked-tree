"""Two questions the threshold sweep left open.

1. Is ANY history feature predictive, or is the whole idea dead? Tests every
   free feature derivable from what a decoder already knows at round start.

2. If history fails, the signal has to come from the drafter. The drafter's
   base_logits are already computed before the tree is built, so a statistic
   over them is free -- no extra forward pass, unlike the confidence head.
   That statistic would be an estimate of THIS round's chain acceptance, which
   separates the classes cleanly (2.96 vs 6.72 tokens).

   So: how accurate would such an estimator have to be? Inject gaussian noise
   into the true chain length and re-price the policy. That converts "try a
   confidence signal" into a spec: predict chain acceptance to within +-N tokens
   or do not bother.
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
            tl = ts["acc"]
            pos = 0
            for j, n in enumerate(tl):
                if j >= 2 and pos in cmap and pos < horizon:
                    hist = tl[max(0, j - 4):j]
                    rows.append({
                        "ds": ds,
                        "prev": tl[j - 1],
                        "prev2": tl[j - 2],
                        "mean4": sum(hist) / len(hist),
                        "max4": max(hist),
                        "min4": min(hist),
                        "sat_prev": 1.0 if tl[j - 1] >= ceiling else 0.0,
                        "tokpos": pos,
                        "chain": cmap[pos],
                        "tree": n,
                        "gain": n - cmap[pos],
                    })
                pos += n
    return rows


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return wins / (len(pos) * len(neg))


def null_band(scores, labels, trials=400, seed=0):
    rng = random.Random(seed)
    lab, out = list(labels), []
    for _ in range(trials):
        rng.shuffle(lab)
        out.append(auc(scores, lab))
    out.sort()
    return out[int(0.025 * trials)], out[int(0.975 * trials)]


FEATURES = ["prev", "prev2", "mean4", "max4", "min4", "sat_prev", "tokpos"]


def main():
    d = json.loads(EXP1.read_text())
    per_dataset = d["per_dataset"]
    rng = random.Random(0)

    for chain_name, tree_name, ceiling, w_chain, w_tree, label in PAIRS:
        rows = collect(per_dataset, chain_name, tree_name, ceiling)
        n = len(rows)
        labels = [r["gain"] > 0 for r in rows]
        tree_acc = sum(r["tree"] for r in rows) / n

        print("=" * 74)
        print(f"{label}   n={n}")
        print("=" * 74)
        print("\n  1. EVERY FREE HISTORY FEATURE (AUC for 'tree will gain')")
        lo, hi = null_band([-r["prev"] for r in rows], labels)
        print(f"     chance band on this sample: {lo:.3f} - {hi:.3f}\n")
        for f in FEATURES:
            a = auc([-r[f] for r in rows], labels)
            a = max(a, 1 - a)          # allow either sign; best case for the feature
            within = []
            for ds in sorted({r["ds"] for r in rows}):
                sub = [r for r in rows if r["ds"] == ds]
                if len(sub) < 20:
                    continue
                x = auc([-r[f] for r in sub], [r["gain"] > 0 for r in sub])
                within.append(max(x, 1 - x))
            wa = sum(within) / len(within) if within else float("nan")
            flag = "" if a < hi else "  <- above chance"
            print(f"     {f:<9s} pooled {a:.3f}   within-dataset mean {wa:.3f}{flag}")

        # contemporaneous chain length: not observable, but bounds any estimator
        a_c = auc([-r["chain"] for r in rows], labels)
        print(f"\n     [not observable] this-round chain length: AUC {a_c:.3f}")
        print("     -> the information exists, it is just not in the history.")

        # ---- 2. how good must a chain-acceptance estimator be? ----
        print("\n  2. NOISE TOLERANCE OF A DRAFTER-SIDE CHAIN-ACCEPTANCE ESTIMATOR")
        print("     policy: run the chain when predicted chain acceptance >= T")
        print("     reported: best width saving with NO acceptance loss vs always-tree\n")
        print(f"     {'noise sd':>9s} {'best T':>7s} {'accept':>8s} {'width':>7s} "
              f"{'width saved':>12s}")
        for sd in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            best = None
            for T in [t / 2 for t in range(2, 2 * ceiling + 4)]:
                accs, widths = [], []
                reps = 1 if sd == 0 else 40
                for _ in range(reps):
                    tot_a = tot_w = 0.0
                    for r in rows:
                        hat = r["chain"] + (rng.gauss(0, sd) if sd else 0.0)
                        if hat >= T:
                            tot_a += r["chain"]
                            tot_w += w_chain
                        else:
                            tot_a += r["tree"]
                            tot_w += w_tree
                    accs.append(tot_a / n)
                    widths.append(tot_w / n)
                a = sum(accs) / len(accs)
                w = sum(widths) / len(widths)
                if a >= tree_acc - 1e-9 and (best is None or w < best[2]):
                    best = (T, a, w)
            if best:
                T, a, w = best
                print(f"     {sd:9.1f} {T:7.1f} {a:8.3f} {w:7.1f} "
                      f"{w/w_tree - 1:+11.1%}")
            else:
                print(f"     {sd:9.1f} {'--':>7s} {'--':>8s} {'--':>7s} "
                      f"{'no lossless point':>12s}")
        print()


if __name__ == "__main__":
    main()
