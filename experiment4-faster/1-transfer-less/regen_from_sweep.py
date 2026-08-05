"""Regenerate the 1-transfer-less budget-64 charts at the swept best-K (=256),
sourcing from the C-sweep run so the `fast` bar is identical to the one that will
appear in the 2-precompute charts (same machine, same run, n=8).

Maps sweep arms  bestfirst.ref -> bestfirst.ref,  fast.c<K> -> bestfirst.fast."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent / "2-precompute" / "csweep" / "results" / "summary.json"
K = 256
FAST_ARM = f"fast.c{K}"

sys.path.insert(0, str(HERE))
import make_charts as mc

s = json.loads(SWEEP.read_text())
b = "64"

def remap(section):
    out = {}
    for ds, arms in s["results"][section][b].items():
        out[ds] = {"bestfirst.ref": arms["bestfirst.ref"], "bestfirst.fast": arms[FAST_ARM]}
    return {b: out}

mini = {
    "config": s["config"],
    "results": {"clean": remap("clean"), "instrumented": remap("instrumented")},
    "timing": {b: {"bestfirst.ref": s["timing"][b]["bestfirst.ref"],
                   "bestfirst.fast": s["timing"][b][FAST_ARM]}},
    "checks": s.get("checks", {}),
}

outdir = HERE / "results"; outdir.mkdir(exist_ok=True)
mc.fig_speedup_acceptance(mini, [64], outdir / "speedup_acceptance_b64.png",
    suptitle="Transfer only top-K",
    subtext=f"SparklingTree_b16, h100, bench x8, top-K={K}")
mc.fig_phase_collapse(mini, [64], outdir / "phase_collapse_b64.png")
print(f"regenerated 1-transfer-less b64 charts at K={K} from sweep")
