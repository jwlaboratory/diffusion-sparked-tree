"""Turn the final benchmark JSON into REPORT.md.

    python3 final_benchmark/report.py FINAL_H100_<ts>.json --out final_benchmark/REPORT.md

Speed is reported but deliberately not used to rank methods per dataset: two runs
of an identical configuration in this harness differ by ~5% on average and up to
16% on a single cell, while acceptance reproduces to ~0.5% and is often exact.
Per-cell speed claims are therefore not made; only 6-dataset means are.
"""

import argparse
import json

LABELS = [
    ("DSpark", "dspark", "chain, markov head built in"),
    ("DDTree", "ddtree_tb{b}", "tree, independence-scored"),
    ("sparked-tree", "dsparktree_markov_tb{b}", "tree, branch-conditional"),
]
NOISE_ACCEPT = 1.0   # % — mean |delta| between identical runs, 12 prompts
NOISE_SPEED = 5.4    # % — same, speed


def cell(agg, key):
    base = agg.get("baseline", {}).get("mean_tpot_ms")
    row = agg.get(key)
    if not base or not row:
        return None
    return row["mean_accept"], base / row["mean_tpot_ms"], row.get("stage", {})


def table(results, datasets, budget, arm):
    per_arm = results.get(arm, {})
    lines = [f"\n### Budget {budget}\n",
             "| dataset | " + " | ".join(f"{n} acc / speedup" for n, _, _ in LABELS) + " |",
             "|---" * (len(LABELS) + 1) + "|"]
    means = {n: [[], []] for n, _, _ in LABELS}
    for dataset in datasets:
        agg = per_arm.get(dataset)
        if not agg:
            continue
        cells = []
        for name, key, _ in LABELS:
            c = cell(agg, key.format(b=budget))
            if c:
                means[name][0].append(c[0])
                means[name][1].append(c[1])
                cells.append(f"{c[0]:.3f} / {c[1]:.2f}x")
            else:
                cells.append("—")
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    lines.append("| **MEAN** | " + " | ".join(
        f"**{sum(a)/len(a):.3f} / {sum(s)/len(s):.2f}x**" if a else "—"
        for a, s in (means[n] for n, _, _ in LABELS)) + " |")
    return lines, means


def comparison(means, datasets_n):
    ours_a, ours_s = means["sparked-tree"]
    if not ours_a:
        return []
    oa, os_ = sum(ours_a) / len(ours_a), sum(ours_s) / len(ours_s)
    lines = ["\n| sparked-tree vs | acceptance | speed | acc wins |",
             "|---|---|---|---|"]
    for name in ("DSpark", "DDTree"):
        a, s = means[name]
        if not a:
            continue
        da = 100 * (oa / (sum(a) / len(a)) - 1)
        ds = 100 * (os_ / (sum(s) / len(s)) - 1)
        wins = sum(1 for x, y in zip(a, ours_a) if y > x)
        # mark anything inside the measured reproducibility band
        amark = f"**{da:+.1f}%**" if abs(da) > NOISE_ACCEPT else f"{da:+.1f}% (within noise)"
        smark = f"**{ds:+.1f}%**" if abs(ds) > NOISE_SPEED else f"{ds:+.1f}% (within noise)"
        lines.append(f"| {name} | {amark} | {smark} | {wins}/{len(a)} |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--out", default="final_benchmark/REPORT.md")
    args = parser.parse_args()

    payload = json.load(open(args.json_path))
    results = payload["results"]
    datasets = payload["datasets"]
    budgets = [int(b) for b in payload["tree_budgets"].split(",")]

    out = [
        "# Final benchmark — DSpark vs DDTree vs sparked-tree",
        "",
        f"**{payload['gpu']}** · `{payload['checkpoint'].split('/')[-1]}` · block 16 · "
        f"greedy (temperature 0) · exact-match prefix acceptance · "
        f"{payload['max_samples']} prompts x {payload['max_new_tokens']} tokens x {len(datasets)} datasets",
        "",
        "Every method verified byte-identical to plain autoregressive decoding.",
        "`acc` is accepted tokens per round; `speedup` is against no drafter.",
        "",
        "## Configuration",
        "",
        "| setting | value |",
        "|---|---|",
    ]
    primary = payload["arms"].get("sparked", {})
    for key, value in primary.items():
        out.append(f"| {key} | `{value}` |")
    out += ["", "See `config.py` for why each value was chosen.", ""]

    for arm in results:
        label = "best-first over precomputed table" if arm == "sparked" else "CUDA-graphed beam"
        out += [f"\n## Arm: `{arm}` — {label}"]
        for budget in budgets:
            lines, means = table(results, datasets, budget, arm)
            out += lines
            out += comparison(means, len(datasets))

    out += [
        "",
        "---",
        "",
        "## Reading these numbers",
        "",
        f"Two runs of an identical configuration in this harness differ by ~{NOISE_ACCEPT:.0f}% "
        f"on acceptance (often exactly reproducing, since decoding is greedy and seeded) and "
        f"~{NOISE_SPEED:.0f}% on speed, with single cells up to 16%. Deltas inside those bands "
        "are labelled as such above and should not be read as effects.",
        "",
        "Acceptance is the axis that transfers to batched serving; wall-clock speedup at "
        "batch 1 does not. All numbers here are batch 1.",
    ]

    with open(args.out, "w") as handle:
        handle.write("\n".join(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
