"""Does the SPARKED plugin actually run, and is it lossless?

    modal run concurrency/test_plugin_e2e.py::run

Three questions, in order of how much they'd cost to get wrong:

  1. Does the server boot with a plugin-registered algorithm? This is where the
     missing `create_future_map` / `need_topk` / `carries_draft_hidden_states`
     would bite -- `init_overlap` runs unconditionally, so the gap is fatal even
     with --disable-overlap-schedule.
  2. Does it generate at all -- i.e. is the verify/accept/commit wiring right?
  3. **Is the output identical to no-speculation greedy decoding?** Speculative
     decoding is lossless by construction, so any divergence means the tree, the
     mask, or the accept path is wrong. This is the check that matters: a subtly
     wrong mask degrades quality without crashing.

The tree source here is LookupTreeSource, not the DSpark drafter -- so this
validates the plugin machinery and the bridge, not the sparked tree itself.
Acceptance above 1.0 proves the tree path is live rather than silently
degenerating to single-token decoding.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App("sparked-plugin-e2e")
TAG = "lmsysorg/sglang:v0.5.16-cu129"
image = (
    modal.Image.from_registry(TAG)
    .pip_install("requests", "loguru")
    # The DSpark-backed arms import our real builders out of ddtree/.
    .add_local_dir(str(Path(__file__).resolve().parent.parent / "ddtree"),
                   remote_path="/root/ddtree")
    .add_local_dir(Path(__file__).parent, remote_path="/root/concurrency")
)
hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)

TARGET = "Qwen/Qwen3-4B"
PORT = 31000

PROMPTS = [
    "Write a Python function that reverses a linked list. Then explain it.",
    "List the first 10 prime numbers, then list them again in reverse order.",
    "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):",
]

# sglang.launch_server has no main(); its CLI body lives under __main__. Mirror
# it here, with the plugin imported first -- prepare_server_args resolves the
# algorithm name, so registration has to precede it.
# The __main__ guard is load-bearing: sglang spawns its scheduler with
# multiprocessing spawn, which re-imports this file in the child. Without the
# guard the child re-runs launch and dies in _check_not_importing_main. The
# plugin import stays at module scope precisely so the child gets it too --
# that is how SPARKED is registered in the scheduler process.
LAUNCHER = '''
import os, sys
sys.path.insert(0, "/root/concurrency")
sys.path.insert(0, "/root/ddtree")     # build_markov_tree_precomputed lives here
import sparked_plugin          # registers SPARKED (in parent AND spawned child)
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


def _wait_healthy(proc, port, timeout=1800):
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early, code {proc.returncode}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(5)
    proc.kill()
    raise TimeoutError("server never became healthy")


def _generate(port, prompts):
    """Greedy, fixed length, so runs are directly comparable."""
    import requests
    outputs = []
    for prompt in prompts:
        r = requests.post(
            f"http://127.0.0.1:{port}/generate",
            json={"text": prompt,
                  "sampling_params": {"temperature": 0.0, "max_new_tokens": 96}},
            timeout=600)
        r.raise_for_status()
        outputs.append(r.json()["text"])
    return outputs


def _server_info(port):
    import requests
    try:
        return requests.get(f"http://127.0.0.1:{port}/get_server_info", timeout=10).json()
    except Exception as exc:
        return {"error": str(exc)}


def _run_server(extra_args, port, launcher_file, extra_env=None):
    env = dict(os.environ, HF_HOME="/hfcache", TOKENIZERS_PARALLELISM="false")
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, launcher_file, "--model-path", TARGET,
           "--port", str(port), "--host", "127.0.0.1"] + extra_args
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env)
    _wait_healthy(proc, port)
    return proc


@app.function(image=image, gpu="H100", timeout=7200, volumes={"/hfcache": hf_cache})
def run(algo: str = "SPARKED", draft: str = None, width: int = 17) -> dict:
    launcher_file = "/root/launch_with_plugin.py"
    Path(launcher_file).write_text(LAUNCHER)
    result = {}

    # --- 1. baseline: no speculation, greedy. The reference output. ---
    proc = _run_server([], PORT, launcher_file)
    try:
        baseline = _generate(PORT, PROMPTS)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
    result["baseline_lens"] = [len(t) for t in baseline]
    print("baseline done", flush=True)

    # --- 2. SPARKED plugin ---
    try:
        extra = ["--speculative-algorithm", algo,
                 "--disable-overlap-schedule",
                 "--max-running-requests", "8",
                 "--mem-fraction-static", "0.75"]
        env_extra = None
        if draft:
            # DSpark-backed arms: the drafter keeps its own gamma (derived from
            # the checkpoint's block size); tree width travels out of band.
            extra += ["--speculative-draft-model-path", draft]
            env_extra = {"SPARKED_TREE_WIDTH": str(width)}
            # Staged: the target verify graph cannot build a dummy tree
            # SpecInput for a DSpark-flagged algorithm (KeyError 'cache_seqlens')
            # -- that dummy comes from the is_ngram() branch, which we gave up to
            # keep the DSpark loader paths. Graphs affect SPEED, not output, so
            # validate losslessness first and treat capture as separate work.
            extra += ["--disable-cuda-graph"]
        else:
            extra += ["--speculative-num-draft-tokens", str(width)]
        proc = _run_server(extra, PORT + 1, launcher_file, extra_env=env_extra)
    except Exception as exc:
        result["boot"] = f"FAILED: {type(exc).__name__}: {exc}"
        print(f"PLUGIN BOOT FAILED: {exc}", flush=True)
        return result

    result["boot"] = "ok"
    try:
        sparked = _generate(PORT + 1, PROMPTS)
        info = _server_info(PORT + 1)
        result["accept_length"] = (info.get("internal_states", [{}])[0]
                                   .get("avg_spec_accept_length"))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()

    matches = [a == b for a, b in zip(baseline, sparked)]
    result["lossless"] = all(matches)
    result["per_prompt_match"] = matches
    if not all(matches):
        for i, ok in enumerate(matches):
            if not ok:
                result[f"diff_{i}"] = {"baseline": baseline[i][:400],
                                       "sparked": sparked[i][:400]}
    print(f"lossless={result['lossless']} accept_length={result.get('accept_length')}",
          flush=True)
    return result


@app.local_entrypoint()
def main(algo: str = "SPARKED", draft: str = "", width: int = 17):
    out = run.remote(algo=algo, draft=(draft or None), width=width)
    print("\n=== plugin e2e ===")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("diff_")}, indent=2))
    for k, v in out.items():
        if k.startswith("diff_"):
            print(f"\n--- {k} ---\nbaseline: {v['baseline'][:300]}\nsparked : {v['sparked'][:300]}")
    if out.get("boot") != "ok":
        raise SystemExit("plugin did not boot")
    if not out.get("lossless"):
        raise SystemExit("plugin output diverged from greedy baseline")
    print("\nPLUGIN E2E PASSED: boots, generates, lossless vs greedy baseline")
