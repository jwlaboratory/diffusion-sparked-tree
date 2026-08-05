"""Local (CPU, no GPU) correctness gate for build_sparked_tree_fast.

The load-bearing test: with the candidate restriction DISABLED (candidates=0),
build_sparked_tree_fast must return trees byte-for-byte identical to
build_sparked_tree. We also check that a vocab-covering candidate budget is still
identical, that a moderate budget yields a subset/close match, and that the
markov_head=None path works.

Run:  <repo>/... testvenv/bin/python test_fast_builder.py
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent.parent / "harness"
sys.path.insert(0, (HARNESS / "ddtree").as_posix())

from sparked_tree import build_sparked_tree, build_sparked_tree_fast  # noqa: E402
from model.dspark import VanillaMarkovHead  # noqa: E402
from timing import set_timing  # noqa: E402

# CPU-only test: cuda_time() must not try to sync a non-existent device.
set_timing(False)


def make_head(vocab, rank, seed):
    torch.manual_seed(seed)
    head = VanillaMarkovHead(vocab_size=vocab, markov_rank=rank)
    # Randomize the (default-initialized) weights so the bias is non-trivial.
    with torch.no_grad():
        head.markov_w1.weight.normal_(0.0, 0.5)
        head.markov_w2.weight.normal_(0.0, 0.5)
    return head


def trees_equal(a, b):
    (ta, da, pa, ca, va, _) = a
    (tb, db, pb, cb, vb, _) = b
    return (
        torch.equal(ta, tb)
        and torch.equal(da, db)
        and pa == pb
        and ca == cb
        and torch.equal(va, vb)
    )


def accepted_node_set(res):
    """The (depth, token_id) multiset of nodes -- used for subset comparison."""
    toks, depths = res[0].tolist(), res[1].tolist()
    return set(zip(depths, toks))


def main():
    vocab, depth_limit, rank = 200, 6, 8
    n_fail = 0

    # ---- Gate 1: candidates=0 EXACTLY equals build_sparked_tree ----------------
    print("Gate 1: candidates=0 exact equivalence (with markov head)")
    for seed in range(6):
        for budget in (4, 16, 32, 64):
            torch.manual_seed(1000 + seed)
            base = torch.randn(depth_limit, vocab)
            head = make_head(vocab, rank, seed)
            root = int(torch.randint(0, vocab, (1,)).item())
            ref = build_sparked_tree(base, head, root, budget)
            fast = build_sparked_tree_fast(base, head, root, budget, candidates=0)
            ok = trees_equal(ref, fast)
            if not ok:
                n_fail += 1
                print(f"  FAIL seed={seed} budget={budget}")
    print(f"  {'ok' if n_fail == 0 else 'FAILURES'}")

    # ---- Gate 2: candidates >= vocab still exact -------------------------------
    print("Gate 2: candidates>=vocab exact equivalence")
    g2_fail = 0
    for seed in range(4):
        for budget in (16, 64):
            torch.manual_seed(2000 + seed)
            base = torch.randn(depth_limit, vocab)
            head = make_head(vocab, rank, seed)
            root = int(torch.randint(0, vocab, (1,)).item())
            ref = build_sparked_tree(base, head, root, budget)
            fast = build_sparked_tree_fast(base, head, root, budget, candidates=vocab)
            if not trees_equal(ref, fast):
                g2_fail += 1
                print(f"  FAIL seed={seed} budget={budget}")
    n_fail += g2_fail
    print(f"  {'ok' if g2_fail == 0 else 'FAILURES'}")

    # ---- Gate 3: markov_head=None path, candidates=0 exact ---------------------
    print("Gate 3: markov_head=None exact equivalence (candidates=0)")
    g3_fail = 0
    for seed in range(4):
        for budget in (16, 64):
            torch.manual_seed(3000 + seed)
            base = torch.randn(depth_limit, vocab)
            root = int(torch.randint(0, vocab, (1,)).item())
            ref = build_sparked_tree(base, None, root, budget)
            fast = build_sparked_tree_fast(base, None, root, budget, candidates=0)
            if not trees_equal(ref, fast):
                g3_fail += 1
                print(f"  FAIL seed={seed} budget={budget}")
            # Also with a restriction, no-head path must still be a subset of ref.
            fast_r = build_sparked_tree_fast(base, None, root, budget, candidates=64)
            if not accepted_node_set(fast_r).issubset(accepted_node_set(ref)):
                g3_fail += 1
                print(f"  FAIL(subset) seed={seed} budget={budget}")
    n_fail += g3_fail
    print(f"  {'ok' if g3_fail == 0 else 'FAILURES'}")

    # ---- Gate 4: moderate candidates -> subset / close match -------------------
    print("Gate 4: moderate candidates -> near-equivalence (documented diffs)")
    total_ref, total_shared = 0, 0
    for seed in range(6):
        for budget in (16, 64):
            torch.manual_seed(4000 + seed)
            base = torch.randn(depth_limit, vocab)
            head = make_head(vocab, rank, seed)
            root = int(torch.randint(0, vocab, (1,)).item())
            ref = build_sparked_tree(base, head, root, budget)
            fast = build_sparked_tree_fast(base, head, root, budget, candidates=64)
            rset, fset = accepted_node_set(ref), accepted_node_set(fast)
            total_ref += len(rset)
            total_shared += len(rset & fset)
    frac = total_shared / max(total_ref, 1)
    print(f"  shared (depth,token) nodes vs ref: {total_shared}/{total_ref} = {frac:.3f}")
    print("  (restriction is lossy only when a bias-promoted token falls outside")
    print("   the top-C base candidates; with C=64 over a 200-vocab it is small.)")

    print()
    if n_fail == 0:
        print("ALL EXACT-EQUIVALENCE GATES PASSED")
        return 0
    print(f"{n_fail} FAILURES")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
