"""Backbone loading for the shared harness: DFlash and DSpark, with the guards.

Merged from the two experiment drivers: exp1 contributed the dflash branch, exp2
contributed the RoPE normalization + inv_freq assertion and the block-size guard.
Exp2's guard was keyed by backbone name and therefore silently skipped any backbone
it did not know; the table here covers all three exp3 backbones.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoConfig

from model import DFlashDraftModel, DSparkDraftModel


@dataclass
class Backbone:
    name: str
    model_id: str
    kind: str                         # "dflash" | "dspark"
    block_size: Optional[int] = None  # runtime override; None -> checkpoint config
    # filled in by load_backbones():
    model: object = None
    eff_block_size: int = 0
    markov_head: object = None


EXPECTED_ROPE_THETA = 1000000.0
EXPECTED_BLOCK_SIZE = {"dflash_b16": 16, "dspark_b7": 7, "dspark_b16": 16}


def load_config(model_id: str):
    """Load a checkpoint config, normalizing RoPE across transformers majors.

    The DSpark checkpoints were serialized by transformers v5, which moved RoPE
    settings into a nested `rope_parameters` dict and stopped emitting a top-level
    `rope_theta`. Pinned transformers 4.57.1 knows nothing about that field and
    silently falls back to rope_theta=10000.0 while these drafters were trained at
    1000000.0 -- nothing raises, the drafts just get quietly worse. DFlash ships a
    top-level rope_theta and passes through here unchanged, which is exactly why a
    mixed dflash+dspark run NEEDS this: without it the dflash arms would run
    correctly and the dspark arms degraded, and the resulting timing/acceptance
    comparison would look entirely plausible.

    Do NOT "clean this up" while the image pins transformers < 5.
    """
    config = AutoConfig.from_pretrained(model_id)
    params = getattr(config, "rope_parameters", None)
    if params:
        rope_type = params.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(
                f"{model_id}: rope_type={rope_type!r} needs an explicit scaling port; "
                "only 'default' is handled here."
            )
        config.rope_theta = float(params["rope_theta"])
    return config


def load_backbones(cfg: dict, target, device) -> dict[str, Backbone]:
    backbones: dict[str, Backbone] = {}
    for spec in cfg["backbones"]:
        bb = Backbone(**spec)
        config = load_config(bb.model_id)
        if bb.kind == "dflash":
            # Borrows the target's embed_tokens/lm_head; carries no markov head.
            bb.model = DFlashDraftModel.from_pretrained(
                bb.model_id, config=config,
                attn_implementation="flash_attention_2", dtype=torch.bfloat16,
            ).to(device).eval()
            bb.markov_head = None
        elif bb.kind == "dspark":
            bb.model = DSparkDraftModel.from_pretrained(
                bb.model_id, config=config,
                attn_implementation="flash_attention_2", dtype=torch.bfloat16,
            ).to(device).eval()
            bb.markov_head = bb.model.markov_head
        else:
            raise ValueError(f"backbone {bb.name!r}: unknown kind {bb.kind!r}")
        bb.eff_block_size = int(bb.block_size) if bb.block_size else int(bb.model.block_size)

        # Silent-corruption guards, not sanity theater. Recover the rotary base the
        # model will ACTUALLY use from the inv_freq buffer (base = inv_freq[1] **
        # -len(inv_freq)), independent of whatever load_config() wrote -- it fires
        # even if the normalization above is deleted.
        inv_freq = bb.model.rotary_emb.inv_freq.detach().double()
        base = float(inv_freq[1] ** (-inv_freq.numel()))
        if abs(base - EXPECTED_ROPE_THETA) / EXPECTED_ROPE_THETA > 1e-3:
            raise ValueError(
                f"backbone {bb.name!r} ({bb.model_id}): effective rotary base is {base:g}, "
                f"expected {EXPECTED_ROPE_THETA:g}. See load_config()."
            )
        expected_bs = EXPECTED_BLOCK_SIZE.get(bb.name)
        if expected_bs is not None and bb.eff_block_size != expected_bs:
            raise ValueError(
                f"backbone {bb.name!r} ({bb.model_id}): block_size is "
                f"{bb.eff_block_size}, expected {expected_bs}."
            )
        print(f"loaded {bb.name}: kind={bb.kind} block_size={bb.eff_block_size} "
              f"rope_theta={base:g} markov_head={'yes' if bb.markov_head is not None else 'no'}",
              flush=True)

        backbones[bb.name] = bb
    return backbones


def build_correctors(backbones: dict[str, Backbone]) -> dict[str, object]:
    """Every dspark backbone with a markov head exposes '<name>_markov'."""
    correctors: dict[str, object] = {}
    for bb in backbones.values():
        if bb.markov_head is not None:
            correctors[f"{bb.name}_markov"] = bb.markov_head
    return correctors
