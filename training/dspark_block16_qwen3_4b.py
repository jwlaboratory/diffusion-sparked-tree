"""DSpark block-16 fine-tune config for Qwen3-4B (warm-started from block-7).

Derived from DeepSpec's config/dspark/dspark_qwen3_4b.py with:
  - block_size 7 -> 16 (the whole point: extend the draft horizon)
  - fine-tune schedule instead of from-scratch (lower lr, fixed max steps)
  - single-GPU-friendly sizes (A10G): seq 1024, num_anchors 256, batch accum 32
  - torch_compile off for robustness on a cursory run

The markov head and confidence head stay enabled and jointly trained - the CE
loss is computed on markov-corrected logits (teacher-forced), so the backbone
learns WITH the correction it will have at inference time.
"""
import os

from deepspec.trainer import Qwen3DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, QWEN_3_4B

project_name = "deepspec"
exp_name = "dspark_block16_qwen3_4b_warmstart"
seed = 42

model = dict(
    target_model_name_or_path=QWEN_3_4B,
    block_size=16,
    num_draft_layers=5,
    target_layer_ids=[1, 9, 17, 25, 33],
    mask_token_id=151669,
    # loss materializes float32 probs of [num_anchors, block_size, vocab]:
    # 32*16*151936*4B = 311MB/tensor. 256 anchors OOMs a 24GB A10G (DeepSpec
    # trains block-7 at 512 anchors on 8xH100).
    num_anchors=32,

    ## markov head (jointly trained, same as released checkpoint)
    markov_rank=256,
    markov_head_type='vanilla',

    ## confidence head
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    ## loss
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=Qwen3DSparkTrainer,
    lr=1.0e-4,
    warmup_ratio=0.03,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=32,
    num_train_epochs=100,      # bounded by max_train_steps
    max_train_steps=600,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=200,
)

data = dict(
    target_cache_path="/vol/target_cache_block16",
    chat_template="qwen",
    max_length=768,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    return cfg
