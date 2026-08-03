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

    # --- precomputed-transition builder (this session). Every run below uses the
    # block-16 bigdata drafter and the C x C precomputed table; arms differ only in
    # the knob named in `note`.
    ("ablations_block16.json", dict(
        run="ablations-block16", gpu="A10G", drafter="block-16 bigdata", builder="best-first (lazy, C=2048)",
        prompts="6 x 384", note="markov-vs-independence at ONE horizon (fixes the block-7/16 mix)")),
    ("builder_candidates.json", dict(
        run="builder-candidates", gpu="A10G", drafter="block-16 bigdata", builder="best-first, candidate sweep",
        prompts="4 x 384", note="gsm8k only; DDTree control identical (10.024) across all four arms")),
    ("beam_candidates.json", dict(
        run="beam-candidates", gpu="A10G", drafter="block-16 bigdata", builder="beam + flat, C sweep",
        prompts="10 x 512", note="C=256 costs 3.6% accept vs 512; C=2048 gains 4% at 4x build")),
    ("depth_sweep.json", dict(
        run="depth-sweep", gpu="A10G", drafter="block-16 bigdata", builder="beam + flat, min_width sweep",
        prompts="10 x 512", note="depth truncation: no measurable gain; budget-16 chain fix is large")),
    ("shape_fanout_full.json", dict(
        run="shape-fanout-full", gpu="A10G", drafter="block-16 bigdata", builder="beam vs best-first-precomputed",
        prompts="10 x 512", note="best-first shape +4.3% accept at fanout=budget")),
    ("shape_fanout48.json", dict(
        run="shape-fanout-48", gpu="A10G", drafter="block-16 bigdata", builder="beam vs best-first-precomputed",
        prompts="10 x 512", note="fanout capped at 48; +3.9% accept, build 1.44s -> 1.02s")),
    ("sweep_h100_bigdata.json", dict(
        run="sweep-h100-bigdata", gpu="H100", drafter="block-16 bigdata", builder="beam + flat, precomputed C=512",
        prompts="12 x 512", note="budget sweep; WRONG checkpoint for headline comparison (see _best runs)")),
    ("sweep_a10g_bigdata.json", dict(
        run="sweep-a10g-bigdata", gpu="A10G", drafter="block-16 bigdata", builder="beam + flat, precomputed C=512",
        prompts="12 x 512", note="budget sweep + the A10G half of the hardware comparison")),
    ("sweep_a10g_replicate.json", dict(
        run="sweep-a10g-replicate", gpu="A10G", drafter="block-16 bigdata", builder="beam + flat, precomputed C=512",
        prompts="12 x 512", note="accidental exact replicate of the row above - the noise-floor measurement")),
    ("sweep_bestfirst_bigdata.json", dict(
        run="sweep-bestfirst-bigdata", gpu="A10G", drafter="block-16 bigdata", builder="best-first-precomputed C=512",
        prompts="12 x 512", note="tree shape: +6.2% acceptance over beam @64 (6/6), +2.3% @128")),

    # --- the _best checkpoint control triple. Same GPU, checkpoint and prompts;
    # only the tree builder differs, so the deltas are attributable to it alone.
    ("best_ctrl_incumbent.json", dict(
        run="best-CONTROL-incumbent", gpu="H100", drafter="block-16 best", builder="in-loop matmul, union C=2048",
        prompts="12 x 512", note="**reproduces published headline to +0.1% - validates the whole chain**")),
    ("best_beam_precomputed.json", dict(
        run="best-beam-precomputed", gpu="H100", drafter="block-16 best", builder="beam + flat, precomputed C=512",
        prompts="12 x 512", note="cost of the candidate-scheme change alone: -4.1% accept, -52% build")),
    ("best_bestfirst_a10g.json", dict(
        run="best-bestfirst-a10g", gpu="A10G", drafter="block-16 best", builder="best-first-precomputed C=512",
        prompts="12 x 512", note="cross-GPU vs the control - indicative only, superseded by final_benchmark")),
    ("depth_minwidth1.json", dict(
        run="depth-minwidth1", gpu="A10G", drafter="block-16 bigdata", builder="beam + flat, min_width 1 vs 2",
        prompts="12 x 512", note="the budget-16 chain fix measured against the ACTUAL old behaviour: +1.5%")),
    ("horizon.json", dict(
        run="horizon-block7-16", gpu="A10G", drafter="block-7 released vs block-16 bigdata", builder="best-first (lazy)",
        prompts="6 x 384", note="identical prompts; ddtree_tb64 is the 0.0% control")),
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
        # Several runs nest one level deeper: dataset -> arm -> methods, where the
        # arm key is the swept value (mode name, candidate count, min_width...).
        # Detect structurally rather than by name: an arm is a dict of methods, so
        # it is the level that owns "baseline".
        nested = all(isinstance(v, dict) and "baseline" in v for v in agg.values()) if agg else False
        modes = agg if nested else {"": agg}
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
