"""Trace collection for the branch-distribution measurement.

Runs the best-first sparked tree (budget 256 -- the widest explorer we have) with
save_tree_traces=True and dumps every round's tree + accepted path. The analysis
then recovers, for every ACCEPTED node, its (depth, slot) -- slot being the rank
of the node among its parent's markov-ranked children -- which is the direct
measurement of where tree budget actually pays: the old experiments'
analyze_tree_slots methodology (drafter 87% top-1 at depth 1, 30% at depth 16).

Best-first materializes a parent's children strictly in rank order (the sibling
chain pushes rank r+1 only after rank r pops), so a node's slot is simply the
count of earlier-materialized siblings -- recoverable from the parents array
alone, no extra bookkeeping in the decode loop.

Per-prompt checkpointing to the results volume; rerun to resume.

Usage:
    modal run --detach modal_traces.py --spawn
"""

import json
from pathlib import Path

import modal

TARGET = "Qwen/Qwen3-4B"
BACKBONE = {"name": "dspark_b16", "model_id": "shreybirmiwal/Qwen3-4B-DSpark-b16", "kind": "dspark"}
TASKS = [["gsm8k", 4], ["humaneval", 4], ["mt-bench", 4]]
TREE_BUDGET = 256
MAX_NEW_TOKENS = 512
SEED = 0
OUT_DIR = "/results/beamsched/traces"

GPU = "H100"
CPU = 8
TORCH_VERSION = "2.5.1"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-"
    "cp311-cp311-linux_x86_64.whl"
)

HERE = Path(__file__).parent
HARNESS_DIR = HERE.parent.parent / "harness"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(f"torch=={TORCH_VERSION}", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install(
        "transformers==4.57.1", "datasets==3.6.0",
        "numpy", "loguru", "tqdm", "ninja", "typing_extensions", "hf_transfer",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(HARNESS_DIR.as_posix(), remote_path="/root/harness")
)

app = modal.App("ddtree-exp4-traces")
hf_cache = modal.Volume.from_name("ddtree-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]


@app.function(
    image=image, gpu=GPU, cpu=CPU, timeout=2 * 60 * 60,
    volumes={"/cache": hf_cache, "/results": results_vol}, secrets=secrets,
)
def collect_traces() -> list[str]:
    import os
    import sys

    sys.path.insert(0, "/root/harness/ddtree")
    sys.path.insert(0, "/root/harness/runner")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from model import load_and_process_dataset
    from sparked_tree import sparked_tree_generate
    from backbones import load_backbones

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")

    target = AutoModelForCausalLM.from_pretrained(
        TARGET, attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(TARGET)

    backbones = load_backbones({"backbones": [BACKBONE]}, target, device)
    bb = backbones["dspark_b16"]

    def encode(prompt: str):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return tokenizer.encode(text, return_tensors="pt").to(device)

    def generate(ids, max_new, traces):
        return sparked_tree_generate(
            model=bb.model, target=target, input_ids=ids,
            mask_token_id=bb.model.mask_token_id, max_new_tokens=max_new,
            block_size=bb.eff_block_size, stop_token_ids=[tokenizer.eos_token_id],
            temperature=0.0, tree_budget=TREE_BUDGET,
            markov_head=bb.markov_head, draft_mode="dspark",
            save_tree_traces=traces,
        )

    _ = generate(encode("Warmup"), 256, False)  # kernels, allocator, draft KV
    print("[warmup] done", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for dataset_name, max_samples in TASKS:
        dataset = load_and_process_dataset(dataset_name)
        if len(dataset) > max_samples:
            dataset = dataset.shuffle(seed=SEED).select(range(max_samples))
        for i, row in enumerate(dataset):
            path = f"{OUT_DIR}/{dataset_name}__{i}.json"
            if os.path.exists(path):
                print(f"[resume] {path} exists", flush=True)
                written.append(path)
                continue
            result = generate(encode(row["turns"][0]), MAX_NEW_TOKENS, True)
            json.dump(
                {
                    "dataset": dataset_name, "sample": i,
                    "tree_budget": TREE_BUDGET,
                    "acceptance_lengths": [int(a) for a in result.acceptance_lengths],
                    "round_trees": result.round_trees,
                },
                open(path, "w"),
            )
            results_vol.commit()
            written.append(path)
            print(f"[checkpoint] {path} ({len(result.acceptance_lengths)} rounds)", flush=True)
    return written


@app.local_entrypoint()
def main(spawn: bool = False):
    if spawn:
        call = collect_traces.spawn()
        print(f"spawned: {call.object_id}")
        print("progress: modal volume ls ddtree-results beamsched/traces")
        return
    written = collect_traces.remote()
    print(f"wrote {len(written)} trace files")
