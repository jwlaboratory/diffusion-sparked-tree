"""Experiment 5 — FINAL RESULTS.

Head-to-head: DFlash vs DSpark vs DDTree vs SparklingTree (ours), one job, one
machine, reporting mean acceptance and net wall-clock speedup. Set the tunables
below (C / K / B come from the exp4 csweep winner) and run.

    modal run run_final.py --smoke          # tiny end-to-end validation, minutes
    modal run --detach run_final.py --spawn # full run, detached (survives CLI)

Then:  python analyze_final.py results/summary.json
"""

import json
from pathlib import Path

import modal

# ═══════════════════════════════════════════════════════════════════════════ #
#  TUNABLES  —  edit these                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

# --- hardware ---
GPU = "H100"
CPU = 8

# --- SparklingTree (ours): fill C / K / B from the exp4 csweep winner ---------
OUR_TREE_MODE = "best-first-precompute"   # the builder that shipped from exp4/2
C             = 512        # candidate pool per depth        (beam_candidates)  <-- set from csweep
K             = 0          # max fanout per node, 0 = budget  (max_fanout)       <-- set from csweep
# B: tree node budget(s). Paper sweeps {16,32,64,128,256,512,1024}, block size 16.
BUDGETS       = [16, 32, 64, 128, 256, 512, 1024]   # applies to DDTree + SparklingTree

# --- which methods to include in the comparison ------------------------------
INCLUDE_AUTOREGRESSIVE = True   # plain target decode -- the 1× speedup baseline
INCLUDE_DFLASH         = True   # DFlash chain drafter (DFlash-b16)
INCLUDE_DSPARK         = True   # DSpark chain drafter (DSpark-b16, intrinsic markov)
INCLUDE_DDTREE         = True   # official DDTree reference (DFlash-b16, corrector-free)
INCLUDE_SPARKLINGTREE  = True   # ours: DSpark-b16 + markov head + precompute tree

# --- speedup denominator: every method's TPS is divided by this method's TPS --
#     "Autoregressive" gives the canonical N×-vs-AR number the DDTree/DFlash
#     papers report (their 8.2× is exactly this). Set to "DFlash"/"DDTree" for a
#     speculator-relative view instead.
SPEEDUP_BASELINE = "Autoregressive"

# --- evaluation : full DDTree-paper protocol (arXiv 2604.12989, Table 2) ------
#     10 benchmarks, full test sets, 2048 new tokens. Trim for quick runs (or use
#     `modal run run_final.py --smoke`). This is a large multi-hour H100 sweep.
DATASETS = [
    ["math500", 128], ["gsm8k", 128], ["aime24", 30], ["aime25", 30],
    ["humaneval", 164], ["mbpp", 128], ["livecodebench", 128], ["swe-bench", 128],
    ["mt-bench", 80], ["alpaca", 128],
]
MAX_NEW_TOKENS = 2048
TEMPERATURE    = 0.0
SEED           = 0
WARMUP_TOKENS  = 256

# --- models (usually leave as-is) --------------------------------------------
TARGET            = "Qwen/Qwen3-4B"
DFLASH_MODEL_ID   = "z-lab/Qwen3-4B-DFlash-b16"
DSPARK_MODEL_ID   = "shreybirmiwal/Qwen3-4B-DSpark-b16"

# ═══════════════════════════════════════════════════════════════════════════ #
#  Assembly (derived from the tunables above)                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

_BACKBONE_DEFS = {
    "dflash_b16": {"name": "dflash_b16", "model_id": DFLASH_MODEL_ID, "kind": "dflash"},
    "dspark_b16": {"name": "dspark_b16", "model_id": DSPARK_MODEL_ID, "kind": "dspark"},
}

# AR needs no backbone of its own; pin it to one we already load for another arm
# (build_backbones folds it into the needed set, so AR-only still loads it).
_AR_BACKBONE = "dspark_b16" if (INCLUDE_DSPARK or INCLUDE_SPARKLINGTREE) else "dflash_b16"


def build_methods() -> list[dict]:
    m = []
    if INCLUDE_AUTOREGRESSIVE:
        # Plain target decode -- no drafter. It attaches to an already-loaded
        # backbone only to satisfy the schema (the callable ignores `model` and
        # drives `target` directly). Backbone is chosen in build_backbones so it is
        # always one we actually load.
        m.append({"name": "Autoregressive", "backbone": _AR_BACKBONE, "corrector": None,
                  "verify": "autoregressive"})
    if INCLUDE_DFLASH:
        m.append({"name": "DFlash", "backbone": "dflash_b16", "corrector": None, "verify": "chain"})
    if INCLUDE_DSPARK:
        # DSpark's markov head is INTRINSIC: dspark_generate drafts via
        # model.sample_draft_tokens (a serial markov sweep), so the head is ALWAYS
        # applied on a dspark chain. corrector=None is correct here -- the corrector
        # slot only routes a *pluggable* head into verify="tree"; for a chain it is
        # ignored (setting it would be a silent no-op, not an enable). DSpark = the
        # markov head along a chain.
        m.append({"name": "DSpark", "backbone": "dspark_b16", "corrector": None, "verify": "chain"})
    if INCLUDE_DDTREE:
        m.append({"name": "DDTree", "backbone": "dflash_b16", "corrector": None, "verify": "ddtree"})
    if INCLUDE_SPARKLINGTREE:
        # SparklingTree = the SAME dspark markov head, but applied across a TREE.
        # A tree builder needs the head passed EXPLICITLY (verify="tree" has no
        # intrinsic head), so corrector MUST be set -- corrector=None here would
        # silently build a head-less, per-depth-independent tree. This is the only
        # difference from the DSpark arm above: same backbone, same head, tree not chain.
        m.append({"name": "SparklingTree", "backbone": "dspark_b16",
                  "corrector": "dspark_b16_markov", "verify": "tree",
                  "tree_kwargs": {"tree_mode": OUR_TREE_MODE, "beam_candidates": C, "max_fanout": K}})
    if not m:
        raise ValueError("no methods selected -- enable at least one INCLUDE_* flag")
    return m


def build_backbones(methods: list[dict]) -> list[dict]:
    needed = {m["backbone"] for m in methods}
    return [_BACKBONE_DEFS[n] for n in sorted(needed)]


CACHE_DIR = "/results/final/cache"
TIMEOUT_SECONDS = 8 * 60 * 60


def build_run_config() -> dict:
    methods = build_methods()
    return {
        "target": TARGET,
        "backbones": build_backbones(methods),
        "methods": methods,
        "tasks": DATASETS,
        "tree_budgets": BUDGETS,
        "passes": ["clean", "instrumented"],
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "confidence_threshold": 0.0,
        "measure_corrector_fit": False,
        "warmup_tokens": WARMUP_TOKENS,
        "discard_first_sample": True,
        "cache_dir": CACHE_DIR,
        "force": False,
        "gpu": GPU,
        "cpu": CPU,
    }


# ═══════════════════════════════════════════════════════════════════════════ #
#  Image / app                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

TORCH_VERSION = "2.5.1"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-"
    "cp311-cp311-linux_x86_64.whl"
)

HERE = Path(__file__).parent
HARNESS_DIR = HERE.parent / "harness"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(f"torch=={TORCH_VERSION}", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install("transformers==4.57.1", "datasets==3.6.0",
                 "numpy", "loguru", "tqdm", "ninja", "typing_extensions", "hf_transfer")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(HARNESS_DIR.as_posix(), remote_path="/root/harness")
)

app = modal.App("ddtree-exp5-final")
hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


@app.function(image=image, gpu=GPU, cpu=CPU, timeout=TIMEOUT_SECONDS,
              volumes={"/cache": hf_cache, "/results": results_vol}, secrets=secrets)
def run_experiment(cfg: dict) -> dict:
    import sys
    sys.path.insert(0, "/root/harness/ddtree")
    sys.path.insert(0, "/root/harness/runner")
    import driver
    summary = driver.run(cfg, on_checkpoint=results_vol.commit)
    out = Path("/results/final/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


# ═══════════════════════════════════════════════════════════════════════════ #
#  Final results table                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

def print_final_table(summary: dict, baseline: str = SPEEDUP_BASELINE) -> None:
    """acceptance + net wall-clock speedup for every method, per budget."""
    timing = summary.get("timing", {})
    if not timing:
        print("(no timing in summary)")
        return
    for bkey in sorted(timing, key=int):
        arms = timing[bkey]
        base_tps = arms.get(baseline, {}).get("tps_clean")
        print("\n" + "=" * 78)
        print(f"FINAL RESULTS — tree budget {bkey}   (speedup baseline: {baseline})")
        print("=" * 78)
        print(f"  {'method':<16}{'acceptance':>12}{'TPS (clean)':>14}{'speedup':>11}{'dominant':>16}")
        # order: baseline first, then by TPS desc
        names = sorted(arms, key=lambda n: (-(arms[n]['tps_clean'] or 0)))
        for name in names:
            r = arms[name]
            acc = _accept(summary, bkey, name)
            tps = r["tps_clean"]
            spd = (tps / base_tps) if base_tps else float("nan")
            print(f"  {name:<16}{acc:>12.3f}{tps:>14.2f}{spd:>10.2f}×"
                  f"{(r.get('dominant_phase') or '-'):>16}")


def _accept(summary: dict, bkey: str, name: str) -> float:
    by_ds = summary["results"]["clean"][bkey]
    a, r = 0.0, 0
    for arms in by_ds.values():
        e = arms.get(name)
        if e:
            a += e["mean_accept"] * e["rounds"]; r += e["rounds"]
    return a / r if r else float("nan")


@app.local_entrypoint()
def main(smoke: bool = False, spawn: bool = False):
    cfg = build_run_config()
    if smoke:
        cfg["tasks"] = [["gsm8k", 1]]
        cfg["max_new_tokens"] = 64
        cfg["tree_budgets"] = [64]
        cfg["warmup_tokens"] = 32

    if spawn:
        call = run_experiment.spawn(cfg)
        print(f"spawned: {call.object_id}")
        print("progress:  modal volume ls ddtree-results final/cache")
        print("fetch:     modal volume get ddtree-results final/summary.json")
        return

    summary = run_experiment.remote(cfg)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    name = "summary_smoke.json" if smoke else "summary.json"
    (out_dir / name).write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {out_dir / name}")

    if not summary.get("checks", {}).get("acceptance_match", True):
        print("WARNING: acceptance mismatch:", summary["checks"]["mismatched_units"])
    print_final_table(summary)
