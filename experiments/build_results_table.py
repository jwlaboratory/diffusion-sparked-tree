"""Assemble every benchmark run from the session into one master table.

    python experiments/build_results_table.py <json_dir> [--md out.md] [--html out.html]

Each run is tagged with its full configuration (GPU, drafter checkpoint, tree
builder, budget, prompts) because those varied across the session and several
early comparisons were confounded by exactly that.
"""

import argparse
import json
import os

# Every run, with the configuration that produced it.
# file -> (run id, GPU, drafter, tree builder, prompts x tokens, notes)
RUNS = [
    ("wide_block7.json", dict(
        run="wide-block7", gpu="A10G", drafter="block-7 (released)", builder="best-first (exact)",
        prompts="16 x 512", note="first broad sweep; markov-vs-independence ablation")),
    ("block16_result.json", dict(
        run="block16-g4", gpu="A10G", drafter="block-16 gamma=4", builder="best-first (exact)",
        prompts="6 x 384", note="first block-16 model; best acceptance of the session")),
    ("block16_g8.json", dict(
        run="block16-g8", gpu="A10G", drafter="block-16 gamma=8", builder="best-first (exact)",
        prompts="6 x 384", note="loss_decay_gamma ablation - null result")),
    ("h100_bench.json", dict(
        run="block16-h100-overfit", gpu="H100", drafter="block-16 H100 (31 epochs)", builder="best-first (exact)",
        prompts="6 x 384", note="lower train loss (0.522) but WORSE acceptance - overfit")),
    ("beam.json", dict(
        run="beam-sweep", gpu="A10G", drafter="block-16 gamma=4", builder="exact vs beam (decay sweep)",
        prompts="4 x 384", note="gsm8k only; first beam builder measurements")),
    ("final.json", dict(
        run="final-a10g", gpu="A10G", drafter="block-16 gamma=4", builder="beam + flat",
        prompts="12 x 512", note="6-dataset validation; corrected the earlier gsm8k-only claim")),
    ("a10g_ctrl.json", dict(
        run="gpu-ctrl-A10G", gpu="A10G", drafter="block-7 (released)", builder="best-first (exact)",
        prompts="8 x 384", note="GPU control arm - identical config to gpu-ctrl-H100")),
    ("h100_ctrl.json", dict(
        run="gpu-ctrl-H100", gpu="H100", drafter="block-7 (released)", builder="best-first (exact)",
        prompts="8 x 384", note="GPU control arm - isolates hardware from checkpoint")),
    ("h100_best.json", dict(
        run="h100-bestcfg", gpu="H100", drafter="block-16 H100 (overfit)", builder="beam + flat",
        prompts="8 x 384", note="3 datasets; first H100 run with the good builder")),
    ("final_best.json", dict(
        run="FINAL", gpu="H100", drafter="block-16 gamma=4 (best)", builder="beam + flat",
        prompts="12 x 512", note="**headline result**")),
    ("final_bigdata.json", dict(
        run="final-bigdata", gpu="H100", drafter="block-16 bigdata (9.5k convs)", builder="beam + flat",
        prompts="12 x 512", note="4x data BUT also seq 768->512, anchors 32->96 - confounded, worse")),
    ("confidence.json", dict(
        run="confidence", gpu="A10G", drafter="block-16 bigdata", builder="beam flat vs confidence-adaptive",
        prompts="10 x 512", note="confidence-head widths: -2.7% accept, -6.6% speed")),
]

PRETTY = {
    "baseline": "no drafter", "dflash": "DFlash chain", "dspark": "DSpark chain",
}
# verify width per round (tokens the target must score), for compute efficiency
WIDTH = {"dflash": 16, "dspark": 17, "baseline": 1}


def label(method):
    if method in PRETTY:
        return PRETTY[method]
    for prefix, name in (
        ("ddtree_xmkv_tb", "DDTree+foreign head tb"), ("ddtree_tb", "DDTree tb"),
        ("dsparktree_nomkv_tb", "tree, independence tb"), ("dsparktree_markov_tb", "sparked-tree tb"),
        ("dsparktree_wave_tb", "sparked-tree wave tb"),
    ):
        if method.startswith(prefix):
            return name + method[len(prefix):]
    return method


def width(method):
    if method in WIDTH:
        return WIDTH[method]
    for prefix in ("ddtree_tb", "dsparktree_nomkv_tb", "dsparktree_markov_tb",
                   "dsparktree_wave_tb", "ddtree_xmkv_tb"):
        if method.startswith(prefix):
            return int(method[len(prefix):]) + 1
    return None


def unwrap(payload):
    """Different runners nested their results differently."""
    for key in ("per_dataset", "results"):
        if isinstance(payload, dict) and key in payload:
            return payload[key]
    return payload


def rows_for(path, meta):
    with open(path) as handle:
        data = unwrap(json.load(handle))
    out = []
    for dataset, agg in data.items():
        if not isinstance(agg, dict):
            continue
        # confidence run nests one level deeper: dataset -> mode -> methods
        modes = agg if any(k in ("beam", "confidence") for k in agg) else {"": agg}
        for mode, methods in modes.items():
            if not isinstance(methods, dict):
                continue
            base = methods.get("baseline", {}).get("mean_tpot_ms")
            for method, r in methods.items():
                if not isinstance(r, dict) or "mean_accept" not in r:
                    continue
                w = width(method)
                out.append(dict(
                    **meta, dataset=dataset, mode=mode, method=method, label=label(method),
                    accept=r["mean_accept"], tpot=r["mean_tpot_ms"],
                    speedup=(base / r["mean_tpot_ms"]) if base else None,
                    width=w, efficiency=(r["mean_accept"] / w) if w else None,
                    tree_build=r.get("stage", {}).get("tree_build") or r.get("stage_times", {}).get("tree_build"),
                ))
    return out


def to_markdown(rows):
    lines = ["# Every benchmark run\n",
             "All methods greedy-lossless (temperature 0), verified byte-identical to plain",
             "autoregressive decoding. `efficiency` = accepted tokens per token the target must",
             "score — the metric that predicts behaviour at serving concurrency.\n"]
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    for run, rs in by_run.items():
        m = rs[0]
        lines += [f"\n## {run}\n",
                  f"**GPU** {m['gpu']} · **drafter** {m['drafter']} · **builder** {m['builder']} · "
                  f"**prompts** {m['prompts']}  \n{m['note']}\n",
                  "| dataset | mode | method | accept | tpot ms | speedup | verify width | eff | tree_build s |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda x: (x["dataset"], x["mode"], -x["accept"])):
            sp = f"{r['speedup']:.2f}x" if r["speedup"] else "-"
            ef = f"{r['efficiency']:.3f}" if r["efficiency"] else "-"
            tb = f"{r['tree_build']:.2f}" if r["tree_build"] else "-"
            w = r["width"] or "-"
            lines.append(f"| {r['dataset']} | {r['mode'] or '-'} | {r['label']} | {r['accept']:.3f} | "
                         f"{r['tpot']:.2f} | {sp} | {w} | {ef} | {tb} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_dir")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    rows, missing = [], []
    for fname, meta in RUNS:
        path = os.path.join(args.json_dir, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        try:
            rows.extend(rows_for(path, meta))
        except Exception as exc:
            missing.append(f"{fname} ({exc})")

    print(f"{len(rows)} measurements from {len({r['run'] for r in rows})} runs")
    if missing:
        print("missing/failed:", missing)

    if args.md:
        with open(args.md, "w") as handle:
            handle.write(to_markdown(rows))
        print(f"wrote {args.md}")

    # console summary: the headline run only
    final = [r for r in rows if r["run"] == "FINAL"]
    if final:
        print(f"\n=== FINAL ===\n{'dataset':11s} {'method':22s} {'accept':>7s} {'speedup':>8s} {'eff':>6s}")
        for r in sorted(final, key=lambda x: (x["dataset"], -x["accept"])):
            if r["method"] == "baseline" or (r["width"] or 0) > 200:
                continue
            print(f"{r['dataset']:11s} {r['label']:22s} {r['accept']:7.3f} "
                  f"{r['speedup']:7.2f}x {r['efficiency']:6.3f}")


if __name__ == "__main__":
    main()
