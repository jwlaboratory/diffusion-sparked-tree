"""Train the tree-aware acceptance head, and check whether it earns its place.

The framing: DSpark already ships a confidence head. It predicts CHAIN acceptance,
and `old-experiments/RESULTS.md` section 11 pointed it at per-depth width
allocation -- a coverage question it cannot answer -- where it measured -2.7%
acceptance / -6.6% speed. This head answers a question it can: how many tokens
will THIS ROUND'S TREE accept? That is the quantity the gate needs, and unlike the
original it has the tree in the loop.

The bar it has to clear is not zero. `confidence.round_confidence` already gives
`pred_chain_len` for free, so the head is only worth its complexity if it beats
that baseline on held-out data. This script reports both and says so plainly.

Two splits, because they answer different questions:
  * random      -- optimistic. Rounds within a prompt are correlated, so this
                   leaks and should be read as an upper bound.
  * leave-one-dataset-out -- the honest one. FINDINGS.md finding 5 shows the
                   gate's value is strongly workload-dependent, so a head that
                   only works on the workload it saw is not usable.

    python3 train_head.py results/rounds.json --out heads/tree_accept.pt
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "DDTree"))
from confidence import FIELDS, TreeAcceptanceHead  # noqa: E402


def load_rounds(path):
    """rounds.json -> {dataset: (features [N,F], labels [N])}."""
    raw = json.loads(Path(path).read_text())
    by_ds = {}
    for key, rec in raw.items():
        # key looks like "b64__gsm8k__n8::dspark_b16.markov.tree"
        unit = key.split("::")[0]
        ds = unit.split("__")[1] if "__" in unit else unit
        acc, conf = rec["acc"], rec["conf"]
        if len(acc) != len(conf):
            raise ValueError(
                f"{key}: {len(acc)} accepted lengths vs {len(conf)} confidence rows. "
                "These must be the same rounds; a mismatch means the capture is "
                "misaligned and every number downstream would be wrong.")
        x = torch.tensor([[c[f] for f in FIELDS] for c in conf], dtype=torch.float32)
        y = torch.tensor(acc, dtype=torch.float32)
        if ds in by_ds:
            x = torch.cat([by_ds[ds][0], x])
            y = torch.cat([by_ds[ds][1], y])
        by_ds[ds] = (x, y)
    return by_ds


def mae(pred, y):
    return float((pred - y).abs().mean())


def baseline_mae(x, y):
    """The free estimator used directly as a tree-acceptance predictor."""
    return mae(x[:, FIELDS.index("pred_chain_len")], y)


def fit_affine(xtr, ytr):
    """Least-squares a*pred_chain_len + b, fit on train only.

    This is the baseline that matters. Raw pred_chain_len is enormously biased
    (~-6 to -7 tokens) because it estimates CHAIN acceptance while the label is
    TREE acceptance -- two different quantities. A head compared against the raw
    estimator would "win" simply by learning that constant offset, which is not
    tree-awareness and would not justify its existence.

    A threshold gate is also invariant to an affine rescaling, so this is the
    honest floor: whatever the free signal can do once it is allowed to be
    calibrated. The head has to beat THIS.
    """
    p = xtr[:, FIELDS.index("pred_chain_len")]
    n = len(p)
    mp, my = p.mean(), ytr.mean()
    var = ((p - mp) ** 2).sum()
    a = (((p - mp) * (ytr - my)).sum() / var) if float(var) > 1e-9 else torch.tensor(0.0)
    b = my - a * mp
    return float(a), float(b)


def affine_mae(a, b, x, y):
    return mae(a * x[:, FIELDS.index("pred_chain_len")] + b, y)


def fit(xtr, ytr, xva, yva, epochs=300, hidden=32, seed=0, verbose=False):
    torch.manual_seed(seed)
    head = TreeAcceptanceHead(hidden=hidden)
    head.set_normalization(xtr.mean(0), xtr.std(0))
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    # L1, not L2: the gate thresholds the prediction, so typical-case error in
    # tokens is what matters and squared error would let rare hard rounds dominate.
    lossf = torch.nn.L1Loss()

    best, best_state = float("inf"), None
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        loss = lossf(head(xtr), ytr)
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            head.eval()
            with torch.no_grad():
                v = mae(head(xva), yva)
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in head.state_dict().items()}
            if verbose and ep % 50 == 0:
                print(f"      ep {ep:4d}  train {float(loss):.3f}  val {v:.3f}")
    head.load_state_dict(best_state)
    return head, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds_json")
    ap.add_argument("--out", default="heads/tree_accept.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=32)
    args = ap.parse_args()

    by_ds = load_rounds(args.rounds_json)
    X = torch.cat([x for x, _ in by_ds.values()])
    Y = torch.cat([y for _, y in by_ds.values()])
    print(f"loaded {len(Y)} rounds from {len(by_ds)} datasets: "
          + ", ".join(f"{d}({len(by_ds[d][1])})" for d in sorted(by_ds)))
    print(f"tree acceptance: mean {float(Y.mean()):.2f}, "
          f"min {float(Y.min()):.0f}, max {float(Y.max()):.0f}")

    a_all, b_all = fit_affine(X, Y)
    print("\nBASELINES -- the free pred_chain_len as a TREE-acceptance predictor")
    print(f"  raw            MAE {baseline_mae(X, Y):6.3f} tokens  "
          "(predicts chain acceptance, so hugely biased)")
    print(f"  affine-recal   MAE {affine_mae(a_all, b_all, X, Y):6.3f} tokens  "
          f"(y ~ {a_all:.2f}*pred + {b_all:.2f})  <- the bar the head must clear")
    print("  A threshold gate is invariant to affine rescaling, so the recalibrated")
    print("  number is the honest floor; beating only the raw one proves nothing.")
    print("  (FINDINGS.md section 4: <=2.0 captures the full gate benefit, "
          ">=3.0 buys almost nothing)")

    # ---- honest split: hold out a whole dataset ----
    print("\nLEAVE-ONE-DATASET-OUT (the honest read -- does it transfer?)")
    print(f"  {'held out':<12s} {'n':>6s} {'affine base':>12s} {'head':>8s} {'better by':>11s}")
    transfer = []
    for held in sorted(by_ds):
        xva, yva = by_ds[held]
        xtr = torch.cat([x for d, (x, _) in by_ds.items() if d != held])
        ytr = torch.cat([y for d, (_, y) in by_ds.items() if d != held])
        _, v = fit(xtr, ytr, xva, yva, epochs=args.epochs, hidden=args.hidden)
        aa, bb = fit_affine(xtr, ytr)          # calibrate on TRAIN, score on VAL
        b = affine_mae(aa, bb, xva, yva)
        transfer.append((b, v))
        print(f"  {held:<12s} {len(yva):6d} {b:10.3f} {v:8.3f} {b - v:+11.3f}")
    mean_gain = sum(b - v for b, v in transfer) / len(transfer)
    print(f"  mean improvement over the free estimator: {mean_gain:+.3f} tokens")

    # ---- optimistic split, and the checkpoint we actually ship ----
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(Y), generator=g)
    cut = int(0.8 * len(Y))
    tr, va = perm[:cut], perm[cut:]
    head, v = fit(X[tr], Y[tr], X[va], Y[va],
                  epochs=args.epochs, hidden=args.hidden, verbose=True)
    aa, bb = fit_affine(X[tr], Y[tr])
    b = affine_mae(aa, bb, X[va], Y[va])
    print(f"\nRANDOM 80/20 (optimistic -- rounds within a prompt are correlated)")
    print(f"  baseline {b:.3f}   head {v:.3f}   better by {b - v:+.3f} tokens")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "hidden": args.hidden,
                "val_mae": v, "baseline_mae": b, "fields": list(FIELDS)}, out)
    print(f"\nsaved {out}")

    print("\nVERDICT")
    if mean_gain <= 0.05:
        print("  The head does NOT beat the free estimator on held-out workloads.")
        print("  Ship the free pred_chain_len gate; this head is not worth its")
        print("  complexity or its per-round forward.")
    elif min(v for _, v in transfer) > 3.0:
        print("  Both estimators miss the +-2 token spec on held-out workloads.")
        print("  The gate is not viable on this signal -- do not run stage 2.")
    else:
        print(f"  The head beats the free estimator by {mean_gain:.2f} tokens on")
        print("  held-out workloads. Use it for the stage-2 gate.")


if __name__ == "__main__":
    main()
