"""Capture a streaming demo trace: Autoregressive vs SparklingTree, one gsm8k
prompt, one H100. Writes per-round (elapsed_seconds, decoded_text_so_far)
checkpoints for each method to the results volume; make_gif.py renders the
side-by-side streaming GIF from them (DDTree-paper style).

    modal run modal_demo.py
    modal volume get ddtree-results demo/demo.json results/demo.json
"""

import json
from pathlib import Path

import modal

TARGET = "Qwen/Qwen3-4B"
DSPARK_MODEL_ID = "shreybirmiwal/Qwen3-4B-DSpark-b16"
DFLASH_MODEL_ID = "z-lab/Qwen3-4B-DFlash-b16"

# Chat (alpaca-style) -- SparklingTree's best domain vs the field. Long-form
# instruction so the four methods spread out visibly.
PROMPT = ("Write a detailed one-week healthy living plan: for each day give a "
          "meal idea and one exercise, with a short explanation of why it helps.")

MAX_NEW_TOKENS = 450
BUDGET = 64          # final config
C = 128
K = 64

GPU = "H100"
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
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(HARNESS_DIR.as_posix(), remote_path="/root/harness")
)

app = modal.App("ddtree-demo-gif")
hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


def _checkpoints(result, tokenizer, num_input: int) -> list:
    """Per-round (elapsed_s, text_so_far) from the harness result contract.

    Token i's timestamp: round_timestamps are cumulative from decode start;
    round r commits acceptance_lengths[r] tokens. The prefill token rides at t=0.
    """
    gen = result.output_ids[0].tolist()[num_input:]
    ckpts = [(0.0, "")]
    done = 1                                     # prefill/anchor token
    for accepted, t in zip(result.acceptance_lengths, result.round_timestamps):
        done = min(done + accepted, len(gen))
        ckpts.append((float(t), tokenizer.decode(gen[:done], skip_special_tokens=True)))
    return ckpts


@app.function(image=image, gpu=GPU, cpu=8, timeout=1800,
              volumes={"/cache": hf_cache, "/results": results_vol}, secrets=secrets)
def run_demo() -> dict:
    import sys
    sys.path.insert(0, "/root/harness/ddtree")
    sys.path.insert(0, "/root/harness/runner")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ddtree import maybe_enable_cpp_compact, ddtree_generate
    from timing import set_timing
    from autoregressive import autoregressive_generate
    from dspark import dspark_generate
    from sparked_tree import sparked_tree_generate
    from backbones import load_backbones, build_correctors

    set_timing(False)                    # clean timing: no per-stage barriers
    maybe_enable_cpp_compact(True)
    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, attn_implementation="sdpa", dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET)

    cfg = {"backbones": [
        {"name": "dspark_b16", "model_id": DSPARK_MODEL_ID, "kind": "dspark"},
        {"name": "dflash_b16", "model_id": DFLASH_MODEL_ID, "kind": "dflash"},
    ], "target": TARGET}
    backbones = load_backbones(cfg, target, device)
    correctors = build_correctors(backbones)
    sp = backbones["dspark_b16"]
    fl = backbones["dflash_b16"]
    head = correctors["dspark_b16_markov"]

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    eos = tokenizer.eos_token_id

    runners = {
        "ar": lambda inp, n: autoregressive_generate(
            target=target, input_ids=inp, max_new_tokens=n,
            stop_token_ids=[eos], temperature=0.0),
        "dspark": lambda inp, n: dspark_generate(
            model=sp.model, target=target, input_ids=inp, block_size=sp.eff_block_size,
            confidence_threshold=0.0, max_new_tokens=n,
            stop_token_ids=[eos], temperature=0.0),
        "ddtree": lambda inp, n: ddtree_generate(
            model=fl.model, target=target, input_ids=inp, mask_token_id=fl.model.mask_token_id,
            block_size=fl.eff_block_size, tree_budget=BUDGET, max_new_tokens=n,
            stop_token_ids=[eos], temperature=0.0),
        "st": lambda inp, n: sparked_tree_generate(
            model=sp.model, target=target, input_ids=inp, mask_token_id=sp.model.mask_token_id,
            block_size=sp.eff_block_size, tree_budget=BUDGET, markov_head=head,
            draft_mode="dspark", tree_mode="best-first-precompute",
            beam_candidates=C, max_fanout=K, max_new_tokens=n,
            stop_token_ids=[eos], temperature=0.0),
    }

    # Warm every path (JIT, caches) before the recorded runs.
    warm = tokenizer.encode("Warmup", return_tensors="pt").to(device)
    for fn in runners.values():
        fn(warm, 96)
    torch.cuda.synchronize()

    def stats(r):
        toks = int(r.num_output_tokens) if hasattr(r, "num_output_tokens") else len(r.acceptance_lengths)
        t = float(r.total_decode_time)
        return {"tokens": toks, "seconds": t, "tps": toks / t}

    n_in = int(ids.shape[1])
    out = {
        "prompt": PROMPT,
        "config": {"budget": BUDGET, "C": C, "K": K, "max_new_tokens": MAX_NEW_TOKENS,
                   "target": TARGET, "gpu": GPU},
    }
    for key, fn in runners.items():
        r = fn(ids, MAX_NEW_TOKENS)
        out[key] = {"checkpoints": _checkpoints(r, tokenizer, n_in), **stats(r)}
        print(f"{key}: {out[key]['tokens']} tok in {out[key]['seconds']:.2f}s = {out[key]['tps']:.1f} tok/s")
    p = Path("/results/demo/demo.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    results_vol.commit()
    print(f"AR: {out['ar']['tokens']} tok in {out['ar']['seconds']:.2f}s = {out['ar']['tps']:.1f} tok/s")
    print(f"ST: {out['st']['tokens']} tok in {out['st']['seconds']:.2f}s = {out['st']['tps']:.1f} tok/s")
    return out


@app.local_entrypoint()
def main():
    out = run_demo.remote()
    local = HERE / "results"
    local.mkdir(exist_ok=True)
    (local / "demo.json").write_text(json.dumps(out))
    print("saved demo/results/demo.json")
