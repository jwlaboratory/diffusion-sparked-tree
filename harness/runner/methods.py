"""Method dispatch: {backbone, corrector, verify} -> a generate callable.

verify values:
  "chain"  -- dflash_generate (dflash) / dspark_generate (dspark). On a DSpark
              backbone the markov head is applied INTRINSICALLY (always on); the
              corrector field is ignored. On DFlash there is no corrector slot at
              all, so setting one raises.
  "tree"   -- sparked_tree_generate, markov_head = the named corrector (or None
              for a per-depth-independent tree).
  "ddtree" -- ddtree_generate, the official DDTree reference implementation.
              DFlash-only, corrector-free. This is deliberately NOT
              sparked_tree(dflash, markov=None): the sparked builder pays a
              per-node full-vocab CPU top-k that build_ddtree_tree does not, and
              routing DDTree through it would charge the baseline ~10x tree_build
              overhead it does not have.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from dflash import dflash_generate
from dspark import dspark_generate
from ddtree import ddtree_generate
from sparked_tree import sparked_tree_generate

from backbones import Backbone


@dataclass
class Method:
    name: str
    backbone: str                     # key into backbones
    corrector: Optional[str] = None   # corrector name, or None
    verify: str = "tree"              # "chain" | "tree" | "ddtree"
    tree_kwargs: Optional[dict] = None  # extra kwargs for sparked_tree_generate
                                        # (e.g. tree_mode/beam_schedule); verify="tree" only


def needs_budget(method: Method) -> bool:
    """Chain arms have no tree_budget knob; tree-shaped arms do."""
    return method.verify in ("tree", "ddtree")


def build_method_callable(
    method: Method,
    backbones: dict[str, Backbone],
    correctors: dict[str, object],
    target,
    eos_id: int,
    cfg: dict,
    tree_budget: Optional[int],
) -> Callable:
    bb = backbones[method.backbone]
    model = bb.model
    bs = bb.eff_block_size
    common = dict(
        max_new_tokens=cfg["max_new_tokens"],
        stop_token_ids=[eos_id],
        temperature=cfg["temperature"],
    )

    if method.tree_kwargs and method.verify != "tree":
        raise ValueError(f"{method.name}: tree_kwargs only applies to verify='tree'")

    corrector_head = None
    if method.corrector is not None:
        if method.corrector not in correctors:
            raise ValueError(f"method {method.name!r}: unknown corrector {method.corrector!r}")
        corrector_head = correctors[method.corrector]

    if method.verify == "chain":
        if bb.kind == "dflash":
            if method.corrector is not None:
                raise ValueError(f"{method.name}: dflash chain has no corrector slot")
            return lambda ids: dflash_generate(
                model=model, target=target, input_ids=ids,
                mask_token_id=model.mask_token_id, block_size=bs, **common,
            )
        # dspark chain applies its own markov head intrinsically; corrector ignored.
        return lambda ids: dspark_generate(
            model=model, target=target, input_ids=ids, block_size=bs,
            confidence_threshold=cfg["confidence_threshold"], **common,
        )

    if method.verify == "ddtree":
        if bb.kind != "dflash":
            raise ValueError(f"{method.name}: verify='ddtree' expects a dflash backbone")
        if method.corrector is not None:
            raise ValueError(f"{method.name}: ddtree has no corrector slot")
        return lambda ids: ddtree_generate(
            model=model, target=target, input_ids=ids, mask_token_id=model.mask_token_id,
            block_size=bs, tree_budget=tree_budget, **common,
        )

    if method.verify == "tree":
        # probe_markov_head is intentionally never passed: the corrector-fit probe
        # runs a per-token full-vocab loop inside the decode window and would make
        # exactly the cheap arms look slow in a timing experiment.
        tree_kwargs = dict(method.tree_kwargs or {})
        return lambda ids: sparked_tree_generate(
            model=model, target=target, input_ids=ids, mask_token_id=model.mask_token_id,
            block_size=bs, tree_budget=tree_budget,
            markov_head=corrector_head, draft_mode=bb.kind,
            **tree_kwargs, **common,
        )

    raise ValueError(f"method {method.name!r}: unknown verify {method.verify!r}")
