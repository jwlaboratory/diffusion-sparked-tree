"""Derive an optimal beam width schedule from measured tree hits.

Input: the JSON produced by `modal run training/modal_train.py::collect_tree_stats`,
which records (depth, slot) for every accepted node across real workloads.

The question a width schedule answers is: how many candidate slots does depth d
need before the token that actually gets accepted is inside the tree? If depth-1
acceptances are nearly always slot 0, spending 26 of a 64-node budget there is
waste that belongs deeper.

Outputs a coverage table, a proposed schedule under a node budget, and (with
--plot) a standalone HTML chart.

    python experiments/analyze_tree_slots.py stats.json --budget 64 --plot out.html
"""

import argparse
import json
from collections import Counter, defaultdict


def load(path):
    with open(path) as handle:
        data = json.load(handle)
    per_dataset = {}
    for dataset, payload in data.items():
        slots = [tuple(x) for x in payload["accepted_slots"]]
        per_dataset[dataset] = {
            "slots": slots,
            "depth_reached": payload.get("depth_reached", []),
            "mean_accept": payload.get("mean_accept"),
            "tree_budget": payload.get("tree_budget"),
        }
    return per_dataset


def coverage_table(slots):
    """P(slot < k | node at this depth was accepted), per depth."""
    by_depth = defaultdict(Counter)
    for depth, slot in slots:
        by_depth[depth][slot] += 1
    table = {}
    for depth in sorted(by_depth):
        counter = by_depth[depth]
        total = sum(counter.values())
        cum, curve = 0, []
        for slot in range(max(counter) + 1):
            cum += counter.get(slot, 0)
            curve.append(cum / total)
        table[depth] = {"total": total, "curve": curve, "max_slot": max(counter)}
    return table


def depth_mass_table(slots, depth_limit=16):
    """Share of all accepted nodes living at each depth, and the tail beyond it.

    Distinct from the coverage table above, which answers "given that a node at
    depth d was accepted, how far down the ranking was it?". That is a conditional
    and says nothing about how often depth d is reached at all - so a depth can
    show a low hit rate while still carrying real acceptance mass. Truncating the
    tree trades away exactly this mass, so it is the number that decides whether
    truncation is free.
    """
    by_depth = Counter(depth for depth, _ in slots)
    total = sum(by_depth.values())
    rows, tail = [], total
    for depth in range(1, depth_limit + 1):
        count = by_depth.get(depth, 0)
        rows.append({
            "depth": depth,
            "count": count,
            "share": count / total if total else 0.0,
            "share_at_or_beyond": tail / total if total else 0.0,
        })
        tail -= count
    return rows


def slots_for_coverage(curve, target):
    for index, value in enumerate(curve):
        if value >= target:
            return index + 1
    return len(curve)


def propose_schedule(table, budget, target=0.95):
    """Give each depth the slots it needs for `target` coverage, weighted by how
    often that depth is reached at all, then scale to the node budget."""
    need = {d: slots_for_coverage(v["curve"], target) for d, v in table.items()}
    reach = {d: v["total"] for d, v in table.items()}
    if not need:
        return []
    total_reach = max(reach.values())
    raw = {d: need[d] * (reach[d] / total_reach) for d in need}
    scale = budget / max(sum(raw.values()), 1e-9)
    schedule, remaining = [], budget
    for depth in sorted(raw):
        if remaining <= 0:
            break
        width = max(1, min(remaining, round(raw[depth] * scale)))
        schedule.append(width)
        remaining -= width
    if remaining > 0 and schedule:
        schedule[0] += remaining
    return schedule


def geometric_schedule(budget, depth_limit, decay=0.6):
    widths, remaining, width = [], budget, max(1.0, budget * (1.0 - decay))
    for _ in range(depth_limit):
        if remaining <= 0:
            break
        w = max(1, min(remaining, int(round(width))))
        widths.append(w)
        remaining -= w
        width *= decay
    return widths


def render_html(per_dataset, tables, proposals, budget, path):
    datasets = sorted(tables)
    max_depth = max((max(t) for t in tables.values() if t), default=1)
    palette = ["#4C8DFF", "#FF8A4C", "#39B87A", "#C86DD7", "#E0524A", "#26A5A5"]

    def bars(depth_counts, colour):
        peak = max(depth_counts.values()) if depth_counts else 1
        out = []
        for depth in range(1, max_depth + 1):
            value = depth_counts.get(depth, 0)
            height = 100 * value / peak if peak else 0
            out.append(
                f'<div class="bar-wrap"><div class="bar" style="height:{height:.1f}%;background:{colour}" '
                f'title="depth {depth}: {value} accepted"></div><span>{depth}</span></div>'
            )
        return "".join(out)

    sections = []
    for i, dataset in enumerate(datasets):
        table = tables[dataset]
        colour = palette[i % len(palette)]
        depth_counts = {d: v["total"] for d, v in table.items()}
        rows = []
        for depth in sorted(table):
            curve = table[depth]["curve"]
            cells = "".join(
                f"<td>{curve[k]:.0%}</td>" if k < len(curve) else "<td>-</td>" for k in range(8)
            )
            rows.append(
                f"<tr><td class='d'>{depth}</td><td>{table[depth]['total']}</td>{cells}"
                f"<td class='need'>{slots_for_coverage(curve, 0.95)}</td></tr>"
            )
        sections.append(f"""
        <section>
          <h2><span class="dot" style="background:{colour}"></span>{dataset}</h2>
          <div class="chart">{bars(depth_counts, colour)}</div>
          <p class="cap">accepted nodes per depth</p>
          <table>
            <thead><tr><th>depth</th><th>n</th>
              {''.join(f'<th>&le;{k}</th>' for k in range(8))}<th>need@95%</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
          <p class="cap">cumulative P(accepted node was in the first k slots)</p>
        </section>""")

    prop_rows = "".join(
        f"<tr><td>{name}</td><td class='sched'>{sched}</td><td>{sum(sched)}</td></tr>"
        for name, sched in proposals.items()
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Tree slot hits</title><style>
 body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:32px;
   background:#0f1115;color:#e6e8eb}}
 h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:15px;margin:28px 0 10px;font-weight:600}}
 .sub{{color:#9aa3ad;margin:0 0 24px}}
 .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}}
 .chart{{display:flex;align-items:flex-end;gap:3px;height:130px;padding:8px;background:#161a21;
   border-radius:8px}}
 .bar-wrap{{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;height:100%}}
 .bar{{width:100%;border-radius:3px 3px 0 0;min-height:2px}}
 .bar-wrap span{{font-size:10px;color:#6d7681;margin-top:4px}}
 .cap{{color:#6d7681;font-size:12px;margin:6px 0 0}}
 table{{border-collapse:collapse;margin-top:14px;font-size:12.5px;width:100%}}
 th,td{{padding:5px 9px;text-align:right;border-bottom:1px solid #222831}}
 th{{color:#9aa3ad;font-weight:600}} td.d{{text-align:left;color:#9aa3ad}}
 td.need{{color:#FFC44C;font-weight:600}} td.sched{{text-align:left;font-family:ui-monospace,monospace}}
 section{{max-width:900px}}
 .box{{background:#161a21;border-radius:8px;padding:16px 18px;margin-top:26px;max-width:900px}}
</style></head><body>
<h1>Which tree slots actually get accepted?</h1>
<p class="sub">Budget {budget} nodes &middot; measured on real workloads. "need@95%" = slots required
at that depth for the accepted token to be inside the tree 95% of the time.</p>
{''.join(sections)}
<div class="box"><h2 style="margin-top:0">Proposed width schedules</h2>
<table><thead><tr><th style="text-align:left">source</th>
  <th style="text-align:left">widths by depth</th><th>total</th></tr></thead>
<tbody>{prop_rows}</tbody></table></div>
</body></html>"""
    with open(path, "w") as handle:
        handle.write(html)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_json")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    per_dataset = load(args.stats_json)
    tables = {ds: coverage_table(v["slots"]) for ds, v in per_dataset.items()}

    all_slots = [s for v in per_dataset.values() for s in v["slots"]]
    combined = coverage_table(all_slots)

    print(f"{'depth':>5s} {'accepted':>9s} {'<=0':>6s} {'<=1':>6s} {'<=2':>6s} {'<=3':>6s} "
          f"{'<=7':>6s} {'need@' + str(int(args.target * 100)) + '%':>9s}")
    for depth in sorted(combined):
        curve = combined[depth]["curve"]
        get = lambda k: f"{curve[k]:.0%}" if k < len(curve) else "-"
        print(f"{depth:5d} {combined[depth]['total']:9d} {get(0):>6s} {get(1):>6s} {get(2):>6s} "
              f"{get(3):>6s} {get(7):>6s} {slots_for_coverage(curve, args.target):9d}")

    depth_limit = max(combined) if combined else 16

    print(f"\n{'depth':>5s} {'accepted':>9s} {'share':>7s} {'>= this depth':>14s}")
    for row in depth_mass_table(all_slots, depth_limit):
        print(f"{row['depth']:5d} {row['count']:9d} {row['share']:6.1%} {row['share_at_or_beyond']:13.1%}")
    print("\n'>= this depth' is what a truncation at that depth gives up.")

    proposals = {
        "measured (this data)": propose_schedule(combined, args.budget, args.target),
        "geometric decay=0.6 (current)": geometric_schedule(args.budget, depth_limit, 0.6),
        "geometric decay=0.5": geometric_schedule(args.budget, depth_limit, 0.5),
    }
    for name, sched in proposals.items():
        print(f"\n{name}: {sched}  (sum={sum(sched)})")

    if args.plot:
        print(f"\nwrote {render_html(per_dataset, tables, proposals, args.budget, args.plot)}")


if __name__ == "__main__":
    main()
