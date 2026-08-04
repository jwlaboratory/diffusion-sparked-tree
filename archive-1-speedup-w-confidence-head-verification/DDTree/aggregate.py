"""Turn raw per-round acceptance lengths into the summary the charts read.

Pure Python -- no torch, no GPU. That is the point: the benchmark checkpoints raw
acceptance lengths per (budget, dataset) unit, and everything downstream (means,
per-depth rates, the block-size rollup) is a deterministic function of those
lists. So a summary can be rebuilt at any time, from a partial run, on a laptop,
without paying for the GPU again.

Both paths import this module -- the live run in run_experiment.py and the offline
rebuild in results/build_summary.py -- so there is exactly one implementation of
the rollup and the two cannot drift apart.

A unit's raw payload is {method_name: [accepted_tokens_per_round, ...]}.
"""

from statistics import mean


def per_depth_accept(acceptance_lengths: list[int], depth_limit: int) -> dict[int, float]:
    """Conditional accept rate at depth d = P(reach d+1 | reach d).

    acceptance_lengths store L = new tokens per round; a round reaches tree depth d
    when L >= d. So rate(d) = #(L >= d+1) / #(L >= d)."""
    rates = {}
    for d in range(1, depth_limit + 1):
        reached = sum(1 for a in acceptance_lengths if a >= d)
        deeper = sum(1 for a in acceptance_lengths if a >= d + 1)
        rates[d] = (deeper / reached) if reached else None
    return rates


def split_method(name: str) -> tuple[str, str]:
    """'dspark_b16.markov.tree' -> ('dspark_b16', 'markov.tree').

    The remainder after the backbone IS the arm name, which is what the block-size
    rollup groups on: two backbones sharing an arm differ only in block size.
    """
    backbone, _, arm = name.partition(".")
    return backbone, arm


def block_size_of(backbone: str) -> int:
    """'dspark_b16' -> 16. The block size is carried in the backbone's name."""
    tail = backbone.rsplit("_b", 1)[-1]
    return int(tail)


def build_summary(
    units: dict[tuple[int, str], dict[str, list[int]]],
    depth_limit: int = 16,
    config_extra: dict | None = None,
) -> dict:
    """units[(budget, dataset)] -> {method: [lengths]}  ==>  summary dict.

    Tolerates a partial matrix: only budgets/datasets actually present are
    reported, and the block-size rollup averages over whatever datasets exist for
    that budget. Missing cells are omitted rather than imputed.
    """
    budgets = sorted({b for b, _ in units})
    datasets_by_budget = {b: [d for bb, d in units if bb == b] for b in budgets}
    methods = sorted({m for u in units.values() for m in u}, key=list(
        {m: None for u in units.values() for m in u}).index)

    summary = {
        "config": {
            "depth_report_limit": depth_limit,
            "tree_budgets": budgets,
            "tasks": sorted({d for _, d in units}),
            "backbones": {
                bb: {"block_size": block_size_of(bb)}
                for bb in sorted({split_method(m)[0] for m in methods})
            },
            "methods": {
                m: {"backbone": split_method(m)[0],
                    "corrector": f"{split_method(m)[0]}_markov" if "markov" in m else None,
                    "verify": "tree"}
                for m in methods
            },
            "metric": "mean_acceptance_length",
            **(config_extra or {}),
        },
        "results": {},
        "block_size": {},
    }

    for b in budgets:
        bkey = str(b)
        summary["results"][bkey] = {}
        for d in datasets_by_budget[b]:
            raw = units[(b, d)]
            summary["results"][bkey][d] = {
                m: {
                    "mean_accept": (mean(v) if v else 0.0),
                    "rounds": len(v),
                    "per_depth_accept": per_depth_accept(v, depth_limit),
                }
                for m, v in raw.items()
            }

    # Block-size rollup per budget: group methods by arm, compare smallest vs
    # largest block size within that arm.
    arms: dict[str, dict[int, str]] = {}
    for m in methods:
        bb, arm = split_method(m)
        arms.setdefault(arm, {})[block_size_of(bb)] = m

    for b in budgets:
        bkey = str(b)
        res = summary["results"][bkey]
        datasets = datasets_by_budget[b]
        summary["block_size"][bkey] = {}
        for arm, by_block in arms.items():
            if len(by_block) < 2:
                continue
            small_bs, large_bs = min(by_block), max(by_block)
            small, large = by_block[small_bs], by_block[large_bs]
            usable = [d for d in datasets if small in res[d] and large in res[d]]
            if not usable:
                continue

            per_dataset = {}
            for d in usable:
                s = res[d][small]["mean_accept"]
                l = res[d][large]["mean_accept"]
                per_dataset[d] = {
                    "small": s, "large": l, "delta": l - s,
                    "pct_change": ((l - s) / s * 100.0) if s else None,
                }
            s_all = mean([res[d][small]["mean_accept"] for d in usable])
            l_all = mean([res[d][large]["mean_accept"] for d in usable])
            summary["block_size"][bkey][arm] = {
                "small_block": small_bs, "large_block": large_bs,
                "small_method": small, "large_method": large,
                "per_dataset": per_dataset,
                "overall": {
                    "small": s_all, "large": l_all, "delta": l_all - s_all,
                    "pct_change": ((l_all - s_all) / s_all * 100.0) if s_all else None,
                },
            }

    return summary


def print_rollups(summary: dict) -> None:
    if not summary.get("block_size"):
        return
    print("\n" + "=" * 64)
    print("Block size: acceptance-length change from a longer draft horizon")
    print("=" * 64)
    for bkey in sorted(summary["block_size"], key=int):
        print(f"  tree_budget {bkey}:")
        for arm, r in summary["block_size"][bkey].items():
            o = r["overall"]
            pct = o["pct_change"]
            tail = "n/a" if pct is None else f"({pct:+.1f}%)"
            print(f"    {arm:<16} b{r['small_block']}={o['small']:.3f} -> "
                  f"b{r['large_block']}={o['large']:.3f}  delta={o['delta']:+.3f}  {tail}")
            # Per-dataset, because the effect is task-dependent: a longer horizon
            # only pays where completions reach deep tree positions, so an average
            # over a mixed task set can hide a sign flip underneath it.
            for d, pd in r["per_dataset"].items():
                dpct = pd["pct_change"]
                dtail = "n/a" if dpct is None else f"({dpct:+.1f}%)"
                print(f"      {d:<14} {pd['small']:.3f} -> {pd['large']:.3f}  "
                      f"delta={pd['delta']:+.3f}  {dtail}")
