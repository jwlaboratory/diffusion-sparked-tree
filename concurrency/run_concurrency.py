"""Concurrency benchmark: DFlash and DSpark chains, overlap scheduler on and off.

    modal run --detach concurrency/run_concurrency.py::sweep

One GPU container per arm. The server launch dominates the cost, so an arm pays
it once and runs the whole concurrency ladder against the same warm process --
the inverse of final_benchmark/run_final.py, where the dataset was the expensive
axis and the arms shared a container.

Every setting comes from config.py, which records why each was chosen.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

for _candidate in (Path(__file__).resolve().parent, Path("/root/concurrency")):
    if (_candidate / "config.py").exists():
        sys.path.insert(0, str(_candidate))
        break
import config as cfg  # noqa: E402

app = modal.App("sparked-tree-concurrency")

# Pinned, not :latest. `supports_overlap=False` is already deprecated upstream
# and the V1 worker path has been removed, so the plugin contract this harness
# is built around can move between releases. Bump deliberately.
#
# v0.5.16 verified by scratch probe: sglang 0.5.16, register_algorithm present
# with the (name, *, supports_overlap, validate_server_args, spec_class)
# signature, SpeculativeAlgorithm.register classmethod present, NgramVerifyInput
# carrying tree_topk / max_tree_depth, verify_tree_greedy_func exported, and
# both bench_serving paths importable. Builtin enum is
# DFLASH/DSPARK/EAGLE/EAGLE3/FROZEN_KV_MTP/STANDALONE/NGRAM/NONE — no tree algo,
# as expected.
SGLANG_TAG = os.environ.get("SGLANG_TAG", "lmsysorg/sglang:v0.5.16-cu129")

image = (
    modal.Image.from_registry(SGLANG_TAG)
    .pip_install("requests")
    .add_local_dir(Path(__file__).parent, remote_path="/root/concurrency")
)

vol = modal.Volume.from_name("ddtree-train", create_if_missing=True)
hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)

GPU = os.environ.get("CONCURRENCY_GPU", "H100")
PORT = 30000


# Plugin arms cannot go through `python -m sglang.launch_server`: registration
# must precede prepare_server_args, and the scheduler is spawned (not forked), so
# the child re-imports this file -- which is how SPARKED gets registered in the
# scheduler process too. Hence the module-scope import plus the __main__ guard.
PLUGIN_LAUNCHER = '''
import os, sys
sys.path.insert(0, "/root/concurrency")
import sparked_plugin
from sglang.launch_server import run_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
'''
PLUGIN_LAUNCHER_PATH = "/root/launch_with_plugin.py"


def _launch_server(arm_cfg: dict) -> subprocess.Popen:
    """Start sglang and block until it answers /health."""
    import requests

    if arm_cfg.get("plugin"):
        Path(PLUGIN_LAUNCHER_PATH).write_text(PLUGIN_LAUNCHER)
        cmd = [sys.executable, PLUGIN_LAUNCHER_PATH]
    else:
        cmd = [sys.executable, "-m", "sglang.launch_server"]
    cmd += [
        "--model-path", cfg.TARGET,
        "--port", str(PORT),
        "--host", "127.0.0.1",
    ]
    if arm_cfg["algo"] is not None:
        cmd += ["--speculative-algorithm", arm_cfg["algo"]]
        if arm_cfg.get("draft"):
            cmd += ["--speculative-draft-model-path", arm_cfg["draft"]]
        if arm_cfg.get("num_draft_tokens"):
            cmd += ["--speculative-num-draft-tokens",
                    str(arm_cfg["num_draft_tokens"])]
    if not arm_cfg["overlap"]:
        cmd.append("--disable-overlap-schedule")

    # Tree verify at width 129 reserves 2*129 KV tokens per request per decode and
    # captures verify graphs at bs*width tokens; at the default max_bs=256 that
    # OOMs an 80GB H100 before the model is even warm. Capping to the sweep's own
    # ceiling costs nothing (we never exceed c=32) -- but it must be applied to
    # EVERY arm in a cost comparison, not just the ones that would crash, or the
    # widths are measured under different memory regimes.
    if arm_cfg.get("capped"):
        cmd += ["--max-running-requests", str(cfg.CAP_MAX_RUNNING),
                "--cuda-graph-max-bs-decode", str(cfg.CAP_MAX_RUNNING),
                "--mem-fraction-static", str(cfg.CAP_MEM_FRACTION),
                # The first sweep showed 3x swings on IDENTICAL prompts
                # (tree_w17: 7.37 -> 8.41 -> 22.65 ms/round at c=1,2,4, which
                # share the same 64 prompts). Prefix reuse against a warm server
                # varies by rung and is the prime suspect; turning the radix
                # cache off costs absolute speed uniformly and buys
                # rung-to-rung comparability, which is the only thing this
                # sweep is for.
                "--disable-radix-cache"]

    print("$ " + " ".join(cmd), flush=True)
    env = dict(os.environ, HF_HOME="/hfcache", TOKENIZERS_PARALLELISM="false")
    proc = subprocess.Popen(cmd, env=env)

    deadline = time.time() + 1800
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before becoming healthy")
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                print("server healthy", flush=True)
                return proc
        except Exception:
            pass
        time.sleep(5)

    proc.kill()
    raise TimeoutError("server did not become healthy within 30 min")


def _bench(concurrency: int, rep: int = 0, num_prompts: int = None) -> dict:
    """One rung of the ladder. Returns bench_serving's own metrics dict."""
    out_file = f"/tmp/bench_c{concurrency}_r{rep}.jsonl"
    if os.path.exists(out_file):
        os.remove(out_file)

    cmd = [
        sys.executable, "-m", "sglang.benchmark.serving",
        "--backend", "sglang",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--dataset-name", cfg.DATASET,
        "--num-prompts", str(num_prompts or cfg.num_prompts(concurrency)),
        "--max-concurrency", str(concurrency),
        "--warmup-requests", str(cfg.WARMUP_REQUESTS),
        "--seed", str(cfg.SEED),
        "--disable-tqdm",
        "--output-file", out_file,
    ]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, env=dict(os.environ, HF_HOME="/hfcache"), check=True)

    # bench_serving appends one JSON object per run; ours is the last line.
    with open(out_file) as handle:
        return json.loads(handle.read().strip().splitlines()[-1])


@app.function(image=image, gpu=GPU, timeout=14400, volumes={"/vol": vol, "/hfcache": hf_cache})
def one_arm(arm: str) -> dict:
    """Full concurrency ladder for one arm, against one warm server."""
    arm_cfg = cfg.ARMS[arm]
    print(f"=== {arm}: algo={arm_cfg['algo']} overlap={arm_cfg['overlap']} ===", flush=True)

    proc = _launch_server(arm_cfg)
    rows = {}
    try:
        repeats = cfg.REPEATS if arm_cfg.get("capped") else 1
        fixed = cfg.FIXED_PROMPTS if arm_cfg.get("capped") else None
        for concurrency in cfg.CONCURRENCY:
            observations = []
            for rep in range(repeats):
                metrics = _bench(concurrency, rep=rep, num_prompts=fixed)
                observations.append({
                    "output_throughput": metrics.get("output_throughput"),
                    "mean_tpot_ms": metrics.get("mean_tpot_ms"),
                    "p99_tpot_ms": metrics.get("p99_tpot_ms"),
                    "mean_ttft_ms": metrics.get("mean_ttft_ms"),
                    "mean_itl_ms": metrics.get("mean_itl_ms"),
                    "accept_length": metrics.get("accept_length"),
                    "completed": metrics.get("completed"),
                })
                print(f"[{arm}] c={concurrency} rep{rep}: "
                      f"{observations[-1]['output_throughput']:.1f} tok/s, "
                      f"tpot {observations[-1]['mean_tpot_ms']:.2f} ms", flush=True)
            # Median, not mean -- and the repeats showed exactly why.
            #
            # rep0 is systematically the outlier at every rung where variance
            # appears, and rep1/rep2 then agree to within 1-3%:
            #     dspark  c=16  [14.13, 4.03, 4.16]
            #     tree_w17 c=16 [41.21,  9.38,  9.33]
            #     tree_w65 c=4  [ 8.41,  7.11,  7.12]
            # Never rep1 or rep2. That is a per-batch-size warmup cost -- CUDA
            # graph capture and allocation for a batch shape the server has not
            # served yet. --warmup-requests warms the server, not each rung.
            #
            # This is also the whole explanation for the first sweep: it ran ONE
            # repeat per rung, so every number it produced was a cold rep0. The
            # "3x variance on identical prompts" was not noise, it was warmup.
            #
            # Median of three discards a single outlier by construction, so it
            # returns a warm value without needing to hard-code "drop rep0".
            def med(key):
                vals = sorted(o[key] for o in observations if o.get(key) is not None)
                return vals[len(vals) // 2] if vals else None
            rows[concurrency] = {k: med(k) for k in observations[0]}
            rows[concurrency]["observations"] = observations
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()

    return rows


@app.function(image=image, gpu=GPU, timeout=5400, volumes={"/vol": vol, "/hfcache": hf_cache})
def validate(arm: str = "dflash_noov") -> dict:
    """One arm, one rung, few prompts. Catches config errors for the price of a
    short container instead of a full ladder, and warms the HF cache volume so
    the real sweep does not pay the model download six times over."""
    arm_cfg = cfg.ARMS[arm]
    proc = _launch_server(arm_cfg)
    try:
        out_file = "/tmp/validate.jsonl"
        subprocess.run([
            sys.executable, "-m", "sglang.benchmark.serving",
            "--backend", "sglang", "--host", "127.0.0.1", "--port", str(PORT),
            "--dataset-name", cfg.DATASET, "--num-prompts", "8",
            "--max-concurrency", "2", "--seed", str(cfg.SEED),
            "--disable-tqdm", "--output-file", out_file,
        ], env=dict(os.environ, HF_HOME="/hfcache"), check=True)
        with open(out_file) as handle:
            metrics = json.loads(handle.read().strip().splitlines()[-1])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"VALIDATE OK: {metrics.get('output_throughput')} tok/s, "
          f"accept_length={metrics.get('accept_length')}, "
          f"completed={metrics.get('completed')}", flush=True)
    return metrics


@app.function(image=image, timeout=28800, volumes={"/vol": vol, "/hfcache": hf_cache})
def sweep(arms: str = ",".join(cfg.ARMS)) -> dict:
    """Fan out one container per arm; gather into a single report JSON."""
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    unknown = [a for a in arm_list if a not in cfg.ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known: {sorted(cfg.ARMS)}")

    calls = {arm: one_arm.spawn(arm=arm) for arm in arm_list}
    print(f"spawned {len(calls)} containers", flush=True)

    results = {}
    for arm, call in calls.items():
        try:
            results[arm] = call.get()
            print(f"[done] {arm}", flush=True)
        except Exception as exc:
            print(f"!! {arm} FAILED: {exc}", flush=True)

    payload = {
        "gpu": GPU,
        "sglang_image": SGLANG_TAG,
        "target": cfg.TARGET,
        "dataset": cfg.DATASET,
        "concurrency": cfg.CONCURRENCY,
        "arms": {a: cfg.ARMS[a] for a in arm_list},
        "blocked_arms": cfg.BLOCKED_ARMS,
        "results": results,
    }
    os.makedirs("/vol/results", exist_ok=True)
    out = f"/vol/results/CONCURRENCY_{GPU}_{int(time.time())}.json"
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2)
    vol.commit()
    print(f"\nsaved {out}\nCONCURRENCY BENCHMARK COMPLETE", flush=True)
    return payload
