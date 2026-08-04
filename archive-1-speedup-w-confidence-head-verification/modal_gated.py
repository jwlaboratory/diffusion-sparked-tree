"""Experiment 4, stage 2: does the gate actually pay?

Blocked on stage 1 -- GATE_THRESHOLD below must come from `price_gate.py`, not
from a guess. Set it, then:

    modal run experiment4-speedup-verification/modal_gated.py

Four arms, all the same drafter and corrector, differing only in how the tree is
sized:

    ungated              full budget every round               <- control
    gated_free           shrink when free pred_chain_len >= T
    gated_head           shrink when the trained head says so
    small_always         the shrunk budget every round         <- floor

`small_always` is the arm that makes the result interpretable. If the gate beats
ungated it could be because gating is smart, or simply because a smaller tree is
better on this workload -- and those have opposite implications. Without the floor
arm the two are indistinguishable, which is the mistake
`old-experiments/experiments/FINDINGS.md` flags in its methodology notes: always
run the control that isolates the mechanism.

H100, not A100. Stage 1 claimed only acceptance and confidence, which are
deterministic at temperature 0 and therefore GPU-independent. This stage claims
WALL-CLOCK, and the standing methodology rule is that H100 is the reference GPU
and cross-GPU timing comparisons are invalid.

Read the result on two axes:
  * acceptance -- resolves to ~0.5% here, so a real effect is visible
  * speed -- does NOT resolve below ~5% per cell (16% worst case), so only the
    mean across datasets is quotable, and small deltas are ties
"""

import json
from pathlib import Path

import modal

TARGET = "Qwen/Qwen3-4B"
BLOCK16_MODEL_ID = "shreybirmiwal/Qwen3-4B-DSpark-b16"

# ---- FILL FROM STAGE 1 ------------------------------------------------------ #
# price_gate.py prints a threshold sweep; pick the T whose gated rounds were
# already at/near the block ceiling, since those are the rounds that needed no
# width. Leaving this None is a hard error rather than a silent default: a wrong
# threshold produces a plausible-looking number that means nothing.
GATE_THRESHOLD = None
FULL_BUDGET = 64
SMALL_BUDGET = 16
# Path on the results volume, written by train_head.py then uploaded.
HEAD_PATH = "/results/exp4/heads/tree_accept.pt"
USE_TRAINED_HEAD = False   # flip once train_head.py says the head beats the baseline
# ----------------------------------------------------------------------------- #

BACKBONES = [
    {"name": "dspark_b16", "model_id": BLOCK16_MODEL_ID, "kind": "dspark"},
]

TASKS = [["gsm8k", 8], ["humaneval", 8], ["mt-bench", 8]]
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
SEED = 0
CONFIDENCE_THRESHOLD = 0.0
MEASURE_PER_DEPTH = True
DEPTH_REPORT_LIMIT = 16
CACHE_DIR = "/results/exp4/gated_cache"

GPU = "H100"
TIMEOUT_SECONDS = 6 * 60 * 60

TORCH_VERSION = "2.5.1"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-"
    "cp311-cp311-linux_x86_64.whl"
)

HERE = Path(__file__).parent
DDTREE_DIR = HERE / "DDTree"


def build_methods() -> list[dict]:
    if GATE_THRESHOLD is None:
        raise SystemExit(
            "GATE_THRESHOLD is unset. Run stage 1 (modal_benchmark.py), then\n"
            "  python3 investigate/price_gate.py results/rounds.json --ceiling 16\n"
            "and set the threshold from its sweep. Do not guess it.")

    base = dict(backbone="dspark_b16", corrector="dspark_b16_markov", verify="tree")
    methods = [
        {**base, "name": "ungated", "measure_confidence": True},
        {**base, "name": "gated_free", "measure_confidence": True,
         "gate": [GATE_THRESHOLD, SMALL_BUDGET]},
        {**base, "name": "small_always", "measure_confidence": True,
         # threshold below any achievable prediction => shrinks every round
         "gate": [-1e9, SMALL_BUDGET]},
    ]
    if USE_TRAINED_HEAD:
        methods.insert(3, {**base, "name": "gated_head", "measure_confidence": True,
                           "gate": [GATE_THRESHOLD, SMALL_BUDGET],
                           "head": "tree_accept"})
    return methods


def build_run_config() -> dict:
    cfg = {
        "target": TARGET,
        "backbones": BACKBONES,
        "methods": build_methods(),
        "tasks": TASKS,
        "tree_budgets": [FULL_BUDGET],
        "cache_dir": CACHE_DIR,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "measure_per_depth": MEASURE_PER_DEPTH,
        "depth_report_limit": DEPTH_REPORT_LIMIT,
    }
    if USE_TRAINED_HEAD:
        cfg["acceptance_heads"] = {"tree_accept": HEAD_PATH}
    return cfg


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .pip_install(f"torch=={TORCH_VERSION}", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install(
        "transformers==4.57.1",
        "datasets==3.6.0",
        "numpy", "loguru", "tqdm", "ninja", "typing_extensions", "hf_transfer",
    )
    .env({
        "HF_HOME": "/cache/hf",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(DDTREE_DIR.as_posix(), remote_path="/root/DDTree")
)

app = modal.App("exp4-gated")

hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


@app.function(
    image=image, gpu=GPU, timeout=TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache, "/results": results_vol}, secrets=secrets,
)
def run_gated(cfg: dict) -> dict:
    import sys

    sys.path.insert(0, "/root/DDTree")
    import run_experiment as exp

    summary = exp.run(cfg, on_checkpoint=results_vol.commit)
    out = Path("/results/exp4/gated_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main():
    summary = run_gated.remote(build_run_config())
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "gated_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nSaved {out_dir / 'gated_summary.json'}\n")
    print(f"{'arm':<16}{'accept':>9}{'tpot ms':>10}{'gated %':>9}{'mean budget':>13}")
    print("-" * 57)
    for bkey, per_ds in summary.get("results", {}).items():
        agg = {}
        for entries in per_ds.values():
            for name, e in entries.items():
                a = agg.setdefault(name, {"acc": [], "tpot": [], "gf": [], "mb": []})
                a["acc"].append(e.get("mean_accept", 0.0))
                if "tpot_ms" in e:
                    a["tpot"].append(e["tpot_ms"])
                if "gated_frac" in e:
                    a["gf"].append(e["gated_frac"])
                    a["mb"].append(e["mean_budget"])
        for name, a in agg.items():
            m = lambda k: (sum(a[k]) / len(a[k])) if a[k] else float("nan")
            print(f"{name:<16}{m('acc'):9.3f}{m('tpot'):10.2f}"
                  f"{m('gf') * 100 if a['gf'] else 0:9.1f}{m('mb') if a['mb'] else FULL_BUDGET:13.1f}")
    print("\nSpeed does not resolve below ~5% per cell -- read the mean, not cells.")
    print("Compare gated vs BOTH ungated and small_always: beating ungated while")
    print("also beating small_always is the only result that shows the gate is")
    print("selecting well rather than a smaller tree simply being better here.")
