"""Render the findings dashboard (standalone HTML, no dependencies).

    python experiments/make_charts.py out.html

Data is inlined from the measured runs so the chart is reproducible without the
Modal volume; see FINDINGS.md for provenance of each number.
"""

import sys

# ---- measured data -----------------------------------------------------------

# Tree slot hits: 6 datasets, budget 256, block-16 drafter.
SLOT_TOP1 = [87, 76, 69, 66, 64, 63, 61, 59, 59, 54, 50, 48, 49, 39, 33, 30]
SLOT_NEED95 = [3, 5, 11, 13, 11, 15, 16, 14, 14, 13, 15, 21, 18, 28, 43, 42]
SLOT_COUNT = [1420, 1344, 1220, 1082, 952, 829, 733, 635, 542, 464, 400, 334, 261, 211, 168, 124]
GEOMETRIC = [26, 15, 9, 6, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
MEASURED = [2, 3, 7, 7, 5, 6, 6, 4, 4, 3, 3, 3, 2, 3, 4, 2]

# Schedule sweep: gsm8k, budget 64, block-16 A10G drafter. (accept, speedup, build_s)
SCHEDULES = [
    ("best-first (exact)", 11.453, 5.51, 2.15),
    ("geometric decay 0.75", 9.653, 6.53, 0.57),
    ("measured [2,3,7,7..]", 10.974, 7.22, 0.65),
    ("flat [4,4,5,5..]", 11.089, 7.21, 0.63),
]
SCHEDULE_REF = [("ddtree_tb64", 10.024, 7.20), ("dspark chain", 8.419, 5.96)]

# Markov ablation, tb64, block-7, 6 datasets: (dataset, nomkv, markov)
ABLATION = [
    ("humaneval", 4.477, 6.990), ("mbpp", 4.028, 7.184), ("gsm8k", 4.585, 7.647),
    ("math500", 4.316, 7.495), ("mt-bench", 3.071, 5.204), ("alpaca", 3.170, 5.393),
]
# Cross-model transfer, tb64: (dataset, ddtree, ddtree+foreign markov head)
CROSS = [
    ("humaneval", 8.426, 7.667), ("mbpp", 8.403, 7.110), ("gsm8k", 8.481, 7.577),
    ("math500", 9.670, 8.060), ("mt-bench", 4.622, 4.025), ("alpaca", 4.132, 3.848),
]
# Horizon block7 -> block16, identical prompts: (dataset, chain7, chain16, tree7, tree16)
HORIZON = [
    ("gsm8k", 6.663, 7.893, 7.647, 10.520),
    ("humaneval", 5.707, 6.850, 7.145, 8.942),
    ("alpaca", 3.873, 3.870, 5.263, 5.529),
]
# Builder optimization path: (label, build_s, speedup)
BUILDER = [
    ("full vocab, best-first", 2.73, 5.04),
    ("+ candidate restriction", 2.16, 5.60),
    ("+ beam (geometric)", 0.57, 6.53),
    ("+ beam (flat schedule)", 0.63, 7.21),
]

C = {"blue": "#4C8DFF", "orange": "#FF8A4C", "green": "#39B87A", "red": "#E0524A",
     "purple": "#C86DD7", "gold": "#FFC44C", "grey": "#4a5260"}


def bars(items, colour_fn, label_fn, height=150, value_fn=None):
    peak = max(v for _, v in items) or 1
    out = []
    for name, value in items:
        h = 100 * value / peak
        out.append(
            f'<div class="bw"><div class="bv">{value_fn(value) if value_fn else value}</div>'
            f'<div class="b" style="height:{h:.1f}%;background:{colour_fn(name, value)}"></div>'
            f'<span>{label_fn(name)}</span></div>'
        )
    return f'<div class="chart" style="height:{height}px">{"".join(out)}</div>'


def grouped(items, series, colours, height=160, fmt="{:.2f}"):
    """items: list of (label, v1, v2, ...) — one cluster per label."""
    peak = max(max(row[1:]) for row in items) or 1
    clusters = []
    for row in items:
        label, values = row[0], row[1:]
        segs = "".join(
            f'<div class="b sub" style="height:{100 * v / peak:.1f}%;background:{c}" '
            f'title="{s}: {fmt.format(v)}"><em>{fmt.format(v)}</em></div>'
            for v, c, s in zip(values, colours, series)
        )
        clusters.append(f'<div class="cluster"><div class="cbars">{segs}</div><span>{label}</span></div>')
    legend = "".join(
        f'<span class="lg"><i style="background:{c}"></i>{s}</span>' for s, c in zip(series, colours)
    )
    return f'<div class="legend">{legend}</div><div class="chart grouped" style="height:{height}px">{"".join(clusters)}</div>'


def build(path):
    depths = list(range(1, 17))

    # 1. confidence vs depth
    conf = bars(
        [(str(d), SLOT_TOP1[d - 1]) for d in depths],
        lambda n, v: C["green"] if v >= 60 else (C["gold"] if v >= 45 else C["red"]),
        lambda n: n, value_fn=lambda v: f"{v}%",
    )
    need = bars(
        [(str(d), SLOT_NEED95[d - 1]) for d in depths],
        lambda n, v: C["blue"], lambda n: n,
    )

    # 2. schedule shape: what I guessed vs what the data says
    sched_rows = [(str(d), GEOMETRIC[d - 1], MEASURED[d - 1]) for d in depths]
    sched = grouped(sched_rows, ["geometric decay 0.6 (guess)", "measured need"],
                    [C["red"], C["green"]], height=150, fmt="{:.0f}")

    # 3. schedule outcomes
    sched_acc = grouped([(n, a) for n, a, _, _ in SCHEDULES], ["acceptance"], [C["blue"]], 150)
    sched_spd = grouped([(n, s) for n, _, s, _ in SCHEDULES], ["speedup x"], [C["orange"]], 150)
    sched_bld = grouped([(n, b) for n, _, _, b in SCHEDULES], ["tree_build s"], [C["purple"]], 130)

    # 4. ablation + cross transfer
    abl = grouped(ABLATION, ["independence tree", "markov-guided tree"], [C["grey"], C["green"]], 165)
    cross = grouped(CROSS, ["DDTree (own scores)", "+ foreign markov head"], [C["blue"], C["red"]], 165)

    # 5. horizon
    hz = grouped([(d, c7, c16, t7, t16) for d, c7, c16, t7, t16 in HORIZON],
                 ["chain block-7", "chain block-16", "tree block-7", "tree block-16"],
                 [C["grey"], C["blue"], "#7a8494", C["green"]], 175)

    # 6. builder path
    bld = grouped([(n, b, s) for n, b, s in BUILDER], ["tree_build s", "speedup x"],
                  [C["purple"], C["orange"]], 160)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Diffusion-Sparked Tree — findings</title><style>
:root{{color-scheme:dark light}}
body{{font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:32px 28px 60px;
 background:#0e1015;color:#e8eaee;max-width:1080px}}
h1{{font-size:24px;margin:0 0 6px;letter-spacing:-.3px}}
h2{{font-size:16px;margin:38px 0 4px;letter-spacing:-.2px}}
.sub{{color:#98a1ad;margin:0 0 6px;font-size:13px}}
.key{{background:#161a22;border-left:3px solid {C['gold']};padding:11px 15px;margin:10px 0 16px;
 border-radius:0 7px 7px 0;font-size:13.5px;color:#d6dbe2}}
.key b{{color:{C['gold']}}}
.chart{{display:flex;align-items:flex-end;gap:4px;background:#141821;border-radius:9px;padding:12px 10px 6px;margin-top:10px}}
.bw{{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;height:100%}}
.b{{width:100%;border-radius:3px 3px 0 0;min-height:2px}}
.bv{{font-size:9.5px;color:#8891a0;margin-bottom:3px}}
.bw span{{font-size:10px;color:#6f7887;margin-top:5px}}
.chart.grouped{{gap:10px}}
.cluster{{flex:1;display:flex;flex-direction:column;justify-content:flex-end;height:100%}}
.cbars{{display:flex;align-items:flex-end;gap:2px;height:100%}}
.b.sub{{flex:1;position:relative}}
.b.sub em{{position:absolute;top:-14px;left:50%;transform:translateX(-50%);font-style:normal;
 font-size:9px;color:#8891a0;white-space:nowrap}}
.cluster>span{{text-align:center;font-size:10.5px;color:#6f7887;margin-top:6px}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;font-size:11.5px;color:#98a1ad}}
.lg i{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:middle}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
@media(max-width:760px){{.row{{grid-template-columns:1fr}}}}
.foot{{color:#6f7887;font-size:12px;margin-top:44px;border-top:1px solid #222833;padding-top:14px}}
</style></head><body>

<h1>Diffusion-Sparked Tree</h1>
<p class="sub">Markov-guided draft trees for speculative decoding &middot; Qwen3-4B &middot;
all methods greedy-lossless</p>

<h2>1. The drafter is confident near the root, uncertain deep</h2>
<p class="sub">Every accepted node across 6 benchmarks, recorded as (depth, slot).</p>
<div class="key">At depth 1 the accepted token is the drafter's <b>top pick 87%</b> of the time —
3 candidate slots cover 95%. By depth 16 that falls to <b>30%</b>, needing <b>42</b> slots.</div>
<div class="row">
 <div><p class="sub">top-1 hit rate by depth</p>{conf}</div>
 <div><p class="sub">slots needed for 95% coverage</p>{need}</div>
</div>

<h2>2. So tree width should be flat — not front-loaded</h2>
<div class="key">The geometric schedule I guessed spends <b>26 of 64 nodes at depth 1</b>, which
needs only 3, and gives depths 10–16 <b>nothing</b> — exactly where hedging pays.</div>
{sched}

<h2>3. Fixing the schedule closed the entire wall-clock gap</h2>
<p class="sub">gsm8k, budget 64. Reference: ddtree_tb64 = 10.02 accept / 7.20x &middot; dspark chain = 8.42 / 5.96x</p>
<div class="key">Flat scheduling recovers <b>96% of best-first's acceptance at 30% of its build cost</b>,
taking the method from <b>21% behind</b> DDTree on speed to <b>parity</b>, while staying ~10% ahead on acceptance.</div>
<div class="row">
 <div><p class="sub">acceptance</p>{sched_acc}</div>
 <div><p class="sub">speedup vs plain autoregressive</p>{sched_spd}</div>
</div>
<p class="sub">tree_build cost (seconds)</p>{sched_bld}

<h2>4. Markov guidance is the whole effect — and it does not transfer</h2>
<div class="key">Same drafter, same budget, only tree construction differs: <b>+56% to +78%</b> on 6/6
datasets. But bolt the same head onto a <b>different</b> backbone and it <b>hurts</b> on 6/6.
Joint training is load-bearing, not incidental.</div>
<div class="row">
 <div><p class="sub">markov vs independence (same DSpark drafter)</p>{abl}</div>
 <div><p class="sub">foreign head applied to DFlash/DDTree</p>{cross}</div>
</div>

<h2>5. Trees gain ~2× more from a longer horizon than chains do</h2>
<p class="sub">block 7 → 16, identical prompts, ddtree/dflash controls at exactly 0.0%</p>
<div class="key">gsm8k: chain <b>+18.5%</b>, tree <b>+37.6%</b>. A chain just gets a longer thing to
break; a tree gets more places to hedge at every new depth. Chat (alpaca) gains
<b>nothing</b> — it never reaches depth 8, so horizon is not its bottleneck.</div>
{hz}

<h2>6. The builder bottleneck was round-trip COUNT, not payload</h2>
<div class="key">Shrinking each call's payload (candidate restriction) bought <b>−21%</b>.
Eliminating calls (level-synchronous beam: ~48 syncs/round → <b>1</b>) bought <b>−72%</b>.
512 candidates cost the same as 4096 — the data was never the problem.</div>
{bld}

<p class="foot">Full write-up in <code>experiments/FINDINGS.md</code>.
Controls (<code>ddtree_tb64</code>, <code>dflash</code>) are untouched by the DSpark model and
appear in every comparison; they caught a prompt-selection confound, ~4% wall-clock
noise, and cross-GPU numerics drift.</p>
</body></html>"""

    with open(path, "w") as handle:
        handle.write(html)
    return path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "findings.html"
    print(f"wrote {build(out)}")
