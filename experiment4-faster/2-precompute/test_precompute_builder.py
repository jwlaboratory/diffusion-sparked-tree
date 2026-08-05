"""Correctness gate for build_sparked_tree_precompute (CPU, no GPU needed).

The precompute builder must build the SAME best-first tree as build_sparked_tree_fast
at the same candidate set -- it only changes where the transition arithmetic runs
(one batched matmul before the walk vs a serial per-pop matmul). It is NOT bitwise
identical: baddbmm reduces in a different order than the per-node w2 @ w1[prev], so
paths that score within float32 epsilon can diverge. We therefore assert a HIGH tree
agreement rate rather than bit-identity, and also cover the markov_head=None path.

NB: this gates the WALK (pop order), not real-model acceptance parity. Both builders
use C=512, but the fast builder pools a deduped UNION of per-depth top-C while this
builder uses PER-DEPTH top-C (the table needs a fixed [L,C] shape). On synthetic
logits the markov bias is too weak to promote a token across that gap, so trees are
100% identical here; on the real model the stronger bias makes the union slightly
richer, which is the ~2-6% acceptance delta measured in RESULTS.md (finding #4).

Run: cd harness/ddtree && python .../2-precompute/test_precompute_builder.py
(or from repo root with harness/ddtree on PYTHONPATH).
"""

import os
import sys
import types

import torch

# Stub the heavy deps sparked_tree/ddtree import at module load -- the builders under
# test need only torch/numpy/heapq and the light timing helpers.
def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m

_stub("transformers", AutoModelForCausalLM=object, DynamicCache=object)
_stub("model", sample=None, extract_context_feature=None, DFlashDraftModel=object)
_stub("dflash", dflash_generate=None)
_stub("loguru", logger=types.SimpleNamespace(info=lambda *a, **k: None,
                                             warning=lambda *a, **k: None,
                                             debug=lambda *a, **k: None))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "harness", "ddtree"))

import timing
timing.set_timing(False)  # CPU-only test: no cuda.synchronize barriers

from sparked_tree import build_sparked_tree, build_sparked_tree_fast, build_sparked_tree_precompute


class _MarkovHead:
    """Minimal stand-in exposing .markov_w1.weight / .markov_w2.weight."""

    def __init__(self, w1, w2):
        self.markov_w1 = type("W", (), {"weight": w1})()
        self.markov_w2 = type("W", (), {"weight": w2})()


def _tree_key(node_token_ids, node_depths, parents):
    """Order-independent structural signature: {(parent_token_path... )}.

    We key each node by (depth, token, parent_index) which fully determines the tree
    up to node numbering (best-first emits in score order, identical across builders
    when scores agree)."""
    toks = node_token_ids.tolist()
    deps = node_depths.tolist()
    return tuple(sorted((int(d), int(t), int(p)) for d, t, p in zip(deps, toks, parents[1:])))


def _make_inputs(seed, depth=16, vocab=2000, rank=32):
    g = torch.Generator().manual_seed(seed)
    base_logits = torch.randn(depth, vocab, generator=g)
    w1 = torch.randn(vocab, rank, generator=g) * 0.1
    w2 = torch.randn(vocab, rank, generator=g) * 0.1
    head = _MarkovHead(w1, w2)
    root = int(torch.randint(0, vocab, (1,), generator=g).item())
    return base_logits, head, root


def test_agreement():
    print(f"{'seed':>4} {'budget':>6} {'C':>5} {'agree%':>7} {'nodes(fast/pre)':>16}")
    total_agree, total_nodes = 0, 0
    for seed in range(6):
        base_logits, head, root = _make_inputs(seed)
        for budget in (64, 256):
            for C in (512,):
                f_tok, f_dep, f_par, *_ = build_sparked_tree_fast(base_logits, head, root, budget, C)
                p_tok, p_dep, p_par, *_ = build_sparked_tree_precompute(base_logits, head, root, budget, C)
                fk, pk = _tree_key(f_tok, f_dep, f_par), _tree_key(p_tok, p_dep, p_par)
                inter = len(set(fk) & set(pk))
                denom = max(len(fk), len(pk), 1)
                agree = 100.0 * inter / denom
                total_agree += inter
                total_nodes += denom
                print(f"{seed:>4} {budget:>6} {C:>5} {agree:>6.1f}% {len(f_tok):>7}/{len(p_tok):<8}")
    overall = 100.0 * total_agree / total_nodes
    print(f"\nOverall node agreement (fast vs precompute, C=512): {overall:.2f}%")
    assert overall > 95.0, f"tree agreement too low: {overall:.2f}%"


def test_reference_equivalence():
    """THE correctness gate: precompute at C >= vocab must reproduce the exact
    build_sparked_tree tree (ground truth), up to float32 reduction-order epsilon.

    At C >= vocab the per-depth top-C candidate set IS the whole vocab, so precompute
    and build_sparked_tree normalise / top-k over the same support and must agree on
    every popped node. They are not bit-identical: precompute's batched baddbmm reduces
    the [C,R]@[R,C] transition in a different order than build_sparked_tree's per-node
    w2 @ w1[prev], so paths within float32 epsilon can flip -- hence >99% (not 100%),
    with any residual traced to near-tie scores. This ties the precompute WALK back to
    the reference builder, not merely to the fast builder."""
    print("\nreference equivalence (precompute C>=vocab vs build_sparked_tree):")
    total_agree, total_nodes = 0, 0
    for seed in range(6):
        base_logits, head, root = _make_inputs(seed, depth=6, vocab=256, rank=16)
        for budget in (32, 64):
            C = 256  # == vocab -> per-depth top-C covers the entire vocabulary
            r_tok, r_dep, r_par, *_ = build_sparked_tree(base_logits, head, root, budget)
            p_tok, p_dep, p_par, *_ = build_sparked_tree_precompute(base_logits, head, root, budget, C)
            rk, pk = _tree_key(r_tok, r_dep, r_par), _tree_key(p_tok, p_dep, p_par)
            inter = len(set(rk) & set(pk))
            denom = max(len(rk), len(pk), 1)
            total_agree += inter
            total_nodes += denom
            print(f"  seed={seed} budget={budget} agree={100.0 * inter / denom:>5.1f}%  "
                  f"nodes ref/pre={len(r_tok)}/{len(p_tok)}")
    overall = 100.0 * total_agree / total_nodes
    print(f"Overall precompute-vs-reference agreement: {overall:.2f}%")
    assert overall > 99.0, f"precompute diverges from build_sparked_tree: {overall:.2f}%"


def test_reference_equivalence_no_corrector():
    """Same ground-truth gate on the markov_head=None path: at C >= vocab the
    per-depth-independent precompute tree must match build_sparked_tree exactly (no
    matmul at all here, so this one is expected to be essentially 100%)."""
    total_agree, total_nodes = 0, 0
    for seed in range(4):
        base_logits, _, root = _make_inputs(seed, depth=6, vocab=256, rank=16)
        for budget in (32, 64):
            r_tok, r_dep, r_par, *_ = build_sparked_tree(base_logits, None, root, budget)
            p_tok, p_dep, p_par, *_ = build_sparked_tree_precompute(base_logits, None, root, budget, 256)
            rk, pk = _tree_key(r_tok, r_dep, r_par), _tree_key(p_tok, p_dep, p_par)
            inter = len(set(rk) & set(pk))
            total_agree += inter
            total_nodes += max(len(rk), len(pk), 1)
    overall = 100.0 * total_agree / total_nodes
    print(f"no-corrector precompute-vs-reference agreement: {overall:.2f}%")
    assert overall > 99.0, f"no-corrector precompute diverges: {overall:.2f}%"


def test_no_corrector_path():
    """markov_head=None must produce a valid tree (per-depth-independent)."""
    base_logits, _, root = _make_inputs(0)
    tok, dep, par, cmaps, vis, sub = build_sparked_tree_precompute(base_logits, None, root, 64, 512)
    assert len(tok) == 64, len(tok)
    assert vis.shape == (65, 65), vis.shape
    assert par[0] == -1
    print(f"no-corrector path OK: {len(tok)} nodes, visibility {tuple(vis.shape)}")


def test_empty_budget():
    base_logits, head, root = _make_inputs(0)
    tok, dep, par, cmaps, vis, sub = build_sparked_tree_precompute(base_logits, head, root, 0, 512)
    assert len(tok) == 0 and vis.shape == (1, 1)
    print("empty-budget path OK")


if __name__ == "__main__":
    test_empty_budget()
    test_no_corrector_path()
    test_agreement()
    test_reference_equivalence()
    test_reference_equivalence_no_corrector()
    print("\nAll precompute builder checks passed.")
