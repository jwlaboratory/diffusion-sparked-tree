"""DeepSpec train.py with one change: warm-start the drafter from the released
block-7 checkpoint (deepseek-ai/dspark_qwen3_4b_block7).

Valid because the drafter's parameter shapes are block-size-agnostic: block_size
only controls how many mask tokens are appended per anchor during training.
load_state_dict(strict=True) enforces exact architectural match.

Run exactly like train.py:  python train_warmstart.py --config <cfg> [--opts ...]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deepspec.trainer.dspark_trainer as dspark_trainer_module

_original_build = dspark_trainer_module.Qwen3DSparkTrainer._build_draft_model

WARMSTART_REPO = os.environ.get("WARMSTART_REPO", "deepseek-ai/dspark_qwen3_4b_block7")


def _build_draft_model_warmstart(self, *, target_config, model_args):
    model = _original_build(self, target_config=target_config, model_args=model_args)
    if WARMSTART_REPO and WARMSTART_REPO.lower() != "none":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        weights_path = hf_hub_download(WARMSTART_REPO, "model.safetensors")
        state_dict = load_file(weights_path)
        result = model.load_state_dict(state_dict, strict=True)
        print(f"[warmstart] loaded {WARMSTART_REPO}: {result}", flush=True)
    return model


dspark_trainer_module.Qwen3DSparkTrainer._build_draft_model = _build_draft_model_warmstart

# ---- verbatim tail of DeepSpec train.py ----
from train import main  # noqa: E402  (DeepSpec's train.py, resolved via sys.path)

if __name__ == "__main__":
    torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())
