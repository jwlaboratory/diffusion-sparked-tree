"""What would a perfect per-round chain-or-tree selector buy?

Builds on cascade_analysis.py: at every shared round boundary we know both what
the chain accepted and what the tree accepted from an identical context. So we
can price an oracle that picks the better one per round.

Also diagnoses the negative-gain rounds (tree accepts FEWER tokens than the
chain), which should not happen if the tree contained the chain's draft.
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


def main():
    d = json.loads(EXP1.read_text())
    per_dataset = d["per_dataset"]
    datasets = list(per_dataset)

    for chain_name, tree_name, ceiling, w_chain, w_tree, label in PAIRS:
        rows = []           # (chain_len, tree_len)
        for ds in datasets:
            meths = per_dataset[ds]["methods"]
            if chain_name not in meths or tree_name not in meths:
                continue
            for cs, ts in zip(meths[chain_name]["per_sample"],
                              meths[tree_name]["per_sample"]):
                cmap, cend = cuts(cs["acc"])
                tmap, tend = cuts(ts["acc"])
                horizon = min(cend, tend) - max(ceiling, 16)
                for p in cmap:
                    if p in tmap and p < horizon:
                        rows.append((cmap[p], tmap[p]))

        n = len(rows)
        chain_acc = sum(c for c, _ in rows) / n
        tree_acc = sum(t for _, t in rows) / n
        oracle_acc = sum(max(c, t) for c, t in rows) / n
        # oracle picks the chain whenever the tree would not beat it
        pick_chain = [(c, t) for c, t in rows if t <= c]
        frac_chain = len(pick_chain) / n
        oracle_width = frac_chain * w_chain + (1 - frac_chain) * w_tree

        print("=" * 74)
        print(f"{label}   n={n} aligned rounds")
        print("=" * 74)
        print(f"  chain            acceptance {chain_acc:6.3f}   verify width {w_chain:5d}")
        print(f"  tree             acceptance {tree_acc:6.3f}   verify width {w_tree:5d}")
        print(f"  ORACLE selector  acceptance {oracle_acc:6.3f}   verify width "
              f"{oracle_width:5.1f}   (picks chain {frac_chain:.1%} of rounds)")
        print()
        print(f"  vs tree:  acceptance {oracle_acc/tree_acc - 1:+.1%},  "
              f"verify width {oracle_width/w_tree - 1:+.1%}")
        print(f"  tree buys {tree_acc - chain_acc:+.2f} tokens for "
              f"{w_tree - w_chain} extra scored positions "
              f"= {(w_tree - w_chain)/(tree_acc - chain_acc):.0f} positions per extra token")
        print()

        # --- diagnose the rounds where the tree LOST ---
        neg = [(c, t) for c, t in rows if t < c]
        zero = [(c, t) for c, t in rows if t == c]
        pos = [(c, t) for c, t in rows if t > c]
        print(f"  tree WORSE : {len(neg)/n:5.1%}  mean chain len on these rounds "
              f"{sum(c for c,_ in neg)/max(len(neg),1):5.2f}  "
              f"(tree got {sum(t for _,t in neg)/max(len(neg),1):5.2f})")
        print(f"  tree EQUAL : {len(zero)/n:5.1%}  mean chain len "
              f"{sum(c for c,_ in zero)/max(len(zero),1):5.2f}")
        print(f"  tree BETTER: {len(pos)/n:5.1%}  mean chain len "
              f"{sum(c for c,_ in pos)/max(len(pos),1):5.2f}  "
              f"(tree got {sum(t for _,t in pos)/max(len(pos),1):5.2f})")
        print(f"  overall mean chain len {chain_acc:.2f}")
        print()
        print("  If the tree contained the chain's draft, 'tree WORSE' would be 0%.")
        print("  A high chain length on those rounds => best-first spent budget on")
        print("  width and truncated its own depth below where the chain reached.")
        print()


if __name__ == "__main__":
    main()
