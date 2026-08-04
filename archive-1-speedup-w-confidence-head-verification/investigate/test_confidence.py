"""CPU tests for harness/ddtree/confidence.py.

No GPU and no model: round_confidence() is a pure function of a logits tensor, so
it is fully testable here. What must hold before this is worth a GPU hour:

  1. pred_chain_len matches the closed form computed independently in float64
  2. it behaves correctly at both extremes (near-deterministic vs uniform)
  3. it is monotone in drafter confidence -- the property the policy relies on
  4. no NaN/Inf on degenerate input, and the transfer helper preserves order
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "DDTree"))
from confidence import FIELDS, round_confidence, stack_rounds  # noqa: E402

VOCAB = 512
DEPTH = 16


def reference_pred_len(logits: torch.Tensor) -> float:
    """Closed form in float64, written from the docstring rather than the code."""
    lp = logits.double().log_softmax(dim=-1)
    running, total = 0.0, 0.0
    for d in range(lp.shape[0]):
        running += float(lp[d].max())
        total += math.exp(running)
    return total


def peaked_logits(depth, vocab, sharpness, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(depth, vocab, generator=g)
    x[:, 0] += sharpness          # make token 0 the winner by a tunable margin
    return x


def test_matches_closed_form():
    for sharp in [0.0, 2.0, 6.0, 12.0]:
        x = peaked_logits(DEPTH, VOCAB, sharp)
        got = float(round_confidence(x)[FIELDS.index("pred_chain_len")])
        want = reference_pred_len(x)
        assert abs(got - want) < 1e-3, f"sharp={sharp}: {got} vs {want}"
    print("  ok  pred_chain_len matches an independent float64 closed form")


def test_extremes():
    # near-deterministic: every depth is certain -> expect ~= depth_limit
    x = torch.full((DEPTH, VOCAB), -30.0)
    x[:, 0] = 30.0
    v = round_confidence(x)
    pred = float(v[FIELDS.index("pred_chain_len")])
    assert pred > DEPTH - 0.01, pred
    assert float(v[FIELDS.index("root_p1")]) > 0.999
    assert float(v[FIELDS.index("root_entropy")]) < 1e-3

    # uniform: nothing is known -> chain dies immediately, entropy = log(vocab)
    x = torch.zeros(DEPTH, VOCAB)
    v = round_confidence(x)
    pred = float(v[FIELDS.index("pred_chain_len")])
    assert pred < 0.01, pred
    assert abs(float(v[FIELDS.index("root_entropy")]) - math.log(VOCAB)) < 1e-4
    print(f"  ok  extremes: deterministic -> {DEPTH}.0, uniform -> ~0, "
          f"entropy -> log(V)={math.log(VOCAB):.3f}")


def controlled_p1(depth, vocab, p1):
    """Logits whose top-1 probability is exactly p1 at every depth.

    Deliberately not `peaked_logits`: with random logits over a 512 vocab the max
    of the noise is ~3.5, so a planted peak below that never actually wins and
    confidence does not vary at all -- which is what made the first version of
    this test fail against correct code.
    """
    rest = math.log((1.0 - p1) / (vocab - 1))
    x = torch.full((depth, vocab), rest)
    x[:, 0] = math.log(p1)
    return x


def test_monotone_in_confidence():
    """The policy is a threshold on this value, so ordering is the load-bearing bit."""
    levels = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    preds = [float(round_confidence(controlled_p1(DEPTH, VOCAB, p))[0]) for p in levels]
    assert preds == sorted(preds), preds
    assert preds[0] < 1.0 < preds[-1]

    # closed form for a constant-p1 block: sum_{d=1..D} p^d
    for p, got in zip(levels, preds):
        want = sum(p ** d for d in range(1, DEPTH + 1))
        assert abs(got - want) < 1e-3, (p, got, want)
    print("  ok  monotone in drafter confidence, and matches sum_d p^d: "
          + " -> ".join(f"{p:.2f}" for p in preds))


def test_degenerate_and_transfer():
    assert round_confidence(torch.zeros(0, VOCAB)).shape == (len(FIELDS),)
    v = round_confidence(torch.full((4, VOCAB), -1e4))
    assert torch.isfinite(v).all(), v

    rows = [round_confidence(peaked_logits(DEPTH, VOCAB, s, seed=s)) for s in range(5)]
    out = stack_rounds(rows)
    assert len(out) == 5 and set(out[0]) == set(FIELDS)
    for i, row in enumerate(rows):
        assert abs(out[i]["pred_chain_len"] - float(row[0])) < 1e-5
    assert stack_rounds([]) == []
    print("  ok  degenerate input finite; stack_rounds preserves order and fields")


def test_flag_is_inert():
    """measure_confidence=False must not even construct the stats."""
    text = (Path(__file__).resolve().parents[1]
            / "DDTree/sparked_tree.py").read_text()
    assert "if measure_confidence:" in text, "flag not gated"
    call = text.index("confidence_rounds.append")
    draft_close = text.index('stage_times["draft"] += draft_stage_elapsed')
    build_open = text.index("tree_build_start = cuda_time()")
    assert draft_close < call < build_open, (
        "measurement must sit between the draft and tree_build timers so it "
        "lands in no stage bucket")
    # The gate DOES belong in tree_build: sizing the tree is part of building it.
    assert build_open < text.index("if gate_active:"), "gate must be inside tree_build"
    print("  ok  measurement outside every stage timer; gate charged to tree_build")


def test_gate_direction():
    """Finding 5: shrink on CONFIDENT rounds. A sign flip here silently cuts budget
    on the rounds carrying 71% of what tree width produces, and no other test in
    this stack would catch it."""
    text = (Path(__file__).resolve().parents[1]
            / "DDTree/sparked_tree.py").read_text()
    # Slice from the gate forward. `node_token_ids, node_depths` also appears back
    # in build_sparked_tree's return, so the end marker must be searched AFTER the
    # gate start or the slice comes out empty and the assertions vacuously pass.
    gate_at = text.index("if gate_active:")
    body = text[gate_at:text.index("node_token_ids, node_depths", gate_at)]
    assert body.strip(), "gate body slice is empty -- the markers moved"
    assert "predicted >= threshold" in body, (
        "gate must shrink when predicted acceptance is HIGH; 'predicted <= "
        "threshold' would cut budget exactly where the return lives")
    assert "round_budget = small_budget" in body
    print("  ok  gate shrinks on high predicted acceptance (correct direction)")


def test_tree_acceptance_head():
    from confidence import TreeAcceptanceHead
    torch.manual_seed(0)
    head = TreeAcceptanceHead()

    feats = torch.randn(64, len(FIELDS))
    head.set_normalization(feats.mean(0), feats.std(0))
    out = head(feats)
    assert out.shape == (64,), out.shape
    assert (out >= 1.0).all(), "acceptance is >= 1; the softplus floor must hold"

    # single-round path: exactly how the decode loop calls it
    one = head(torch.randn(len(FIELDS)))
    assert one.shape == () and float(one) >= 1.0

    # a degenerate (zero) std must not produce NaN
    head.set_normalization(torch.zeros(len(FIELDS)), torch.zeros(len(FIELDS)))
    assert torch.isfinite(head(feats)).all()
    print("  ok  TreeAcceptanceHead: shapes, >=1 floor, scalar path, zero-std safe")


if __name__ == "__main__":
    print("confidence.py")
    test_matches_closed_form()
    test_extremes()
    test_monotone_in_confidence()
    test_degenerate_and_transfer()
    test_flag_is_inert()
    test_gate_direction()
    test_tree_acceptance_head()
    print("\nall passed")
