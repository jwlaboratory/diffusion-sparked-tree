"""Sanity checks on the token-index alignment that all of these results rest on.

The alignment assumes every method emits the SAME token sequence (all are greedy
and lossless), so a shared round boundary means a shared context. BLOG.md section 6
reports that byte-identity is ~99.5%, not 100% -- bf16 accumulation makes the
target disagree with itself depending on how many positions it scores at once.

If that broke the alignment badly, the head-to-head numbers would be junk. Three
checks:
  1. do methods agree on total generated length per prompt?
  2. does a method compared against ITSELF yield exactly zero gain? (control)
  3. how many aligned boundaries do we actually get, and where?
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP1 = ROOT / "experiment1-harness/Results/results_detailed.json"


def cuts(lengths):
    out, pos = {}, 0
    for n in lengths:
        out[pos] = n
        pos += n
    return out, pos


def main():
    d = json.loads(EXP1.read_text())
    per_dataset = d["per_dataset"]

    print("=" * 72)
    print("1. DO METHODS AGREE ON OUTPUT LENGTH PER PROMPT?")
    print("=" * 72)
    print("   (all methods lossless => same text => same n_out)\n")
    for ds, payload in per_dataset.items():
        meths = payload["methods"]
        names = list(meths)
        n_prompts = len(meths[names[0]]["per_sample"])
        mismatch = 0
        maxdiff = 0
        for i in range(n_prompts):
            outs = [meths[m]["per_sample"][i]["n_out"] for m in names]
            if len(set(outs)) > 1:
                mismatch += 1
                maxdiff = max(maxdiff, max(outs) - min(outs))
        print(f"   {ds:<10s} {n_prompts} prompts, "
              f"{mismatch} with differing n_out across the 6 methods, "
              f"max spread {maxdiff} tokens")

    print("\n" + "=" * 72)
    print("2. SELF-CONTROL: a method aligned against itself")
    print("=" * 72)
    print("   must produce gain=0 at 100% of boundaries, n = all rounds\n")
    for ds, payload in per_dataset.items():
        m = payload["methods"]["dflash.tree"]
        tot = zero = 0
        for s in m["per_sample"]:
            cmap, _ = cuts(s["acc"])
            for p, v in cmap.items():
                tot += 1
                zero += (v - cmap[p] == 0)
        print(f"   {ds:<10s} {zero}/{tot} boundaries with gain 0 "
              f"({'PASS' if zero == tot else 'FAIL'})")

    print("\n" + "=" * 72)
    print("3. ALIGNMENT YIELD")
    print("=" * 72)
    print("   how much of each run is usable, and is it front-loaded?\n")
    for chain_name, tree_name, ceiling in [
        ("dspark.chain", "dspark.markov.tree", 8),
        ("dflash.chain", "dflash.tree", 16),
    ]:
        print(f"   {chain_name} vs {tree_name}")
        for ds, payload in per_dataset.items():
            meths = payload["methods"]
            shared = 0
            tree_rounds = 0
            frac_pos = []
            for cs, ts in zip(meths[chain_name]["per_sample"],
                              meths[tree_name]["per_sample"]):
                cmap, cend = cuts(cs["acc"])
                tmap, tend = cuts(ts["acc"])
                horizon = min(cend, tend) - max(ceiling, 16)
                tree_rounds += len(tmap)
                for p in tmap:
                    if p in cmap and p < horizon:
                        shared += 1
                        frac_pos.append(p / max(tend, 1))
            mid = sorted(frac_pos)[len(frac_pos) // 2] if frac_pos else float("nan")
            print(f"     {ds:<10s} {shared:4d} aligned / {tree_rounds:4d} tree rounds "
                  f"= {shared/tree_rounds:5.1%}   median position in output: {mid:.2f}")
        print()


if __name__ == "__main__":
    main()
