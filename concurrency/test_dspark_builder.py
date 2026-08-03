"""Can our tree builder drive SGLang's DSpark markov head?

    modal run concurrency/test_dspark_builder.py::run

This is the last unknown between the working plugin and a real sparked-tree arm.
The plugin currently runs on a stand-in proposer; the real one has to take
DSpark's block logits and our markov head and produce a tree. Two questions:

  1. Does `build_markov_tree_precomputed` accept SGLang's markov head unmodified?
     Ours (ddtree/model/dspark.py:68) and theirs (srt/models/dspark.py:77)
     both declare markov_w1 = nn.Embedding(vocab, rank) and
     markov_w2 = nn.Linear(rank, vocab, bias=False) -- the builder reads
     `.markov_w1.weight` / `.markov_w2.weight`, so it should. Should is not is.
  2. Do the trees it emits survive the bridge and SGLang's own kernel?

Question 2 is the one worth the GPU: the trees here come from the real
best-first builder over real logits, not from the synthetic topologies
test_kernel_agreement.py generates. If the shapes our builder actually produces
round-trip, the remaining work is worker wiring, not algorithm risk.
"""

import sys
from pathlib import Path

import modal

app = modal.App("sparked-dspark-builder")
TAG = "lmsysorg/sglang:v0.5.16-cu129"
REPO = Path(__file__).resolve().parent.parent
image = (
    modal.Image.from_registry(TAG)
    .pip_install("loguru")
    .add_local_dir(str(REPO / "ddtree"), remote_path="/root/ddtree")
    .add_local_dir(str(REPO / "concurrency"), remote_path="/root/concurrency")
)


@app.function(image=image, gpu="L4", timeout=3600)
def run() -> dict:
    import torch

    sys.path.insert(0, "/root/ddtree")
    sys.path.insert(0, "/root/concurrency")

    from sglang.srt.models.dspark import VanillaMarkov
    from sgl_kernel.speculative import reconstruct_indices_from_tree_mask
    from sparked_bridge import (
        batch_qlen_mask, parents_from_retrieve, tree_to_draft_tokens,
        tree_to_qlen_mask,
    )

    result = {}
    device = torch.device("cuda")
    torch.manual_seed(0)

    # SGLang's head, not ours. Random weights: we are testing the interface and
    # the tree shapes, not acceptance quality.
    VOCAB, RANK, BLOCK = 4096, 128, 16
    head = VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK).to(device).eval()
    result["head_class"] = type(head).__name__
    result["has_w1_w2"] = hasattr(head, "markov_w1") and hasattr(head, "markov_w2")
    result["w1_shape"] = list(head.markov_w1.weight.shape)
    result["w2_shape"] = list(head.markov_w2.weight.shape)

    try:
        from ddtree_markov import build_markov_tree_precomputed
        from ddtree import build_ddtree_tree
    except Exception as exc:
        result["import"] = f"!! {type(exc).__name__}: {exc}"
        return result
    result["import"] = "ok"

    # Both arms share this plumbing; only the builder differs. Testing both here
    # is what makes "one plugin, two arms" a checked claim rather than a plan.
    BUILDERS = {
        "sparked": lambda logits, budget, root: build_markov_tree_precomputed(
            logits, budget, head, root, candidates=512),
        "ddtree": lambda logits, budget, root: build_ddtree_tree(logits, budget)[:5],
    }

    failures, checked, shapes = [], 0, []
    for arm, build in BUILDERS.items():
      for budget in (16, 32, 64):
        N = budget + 1
        for seed in range(6):
            torch.manual_seed(seed)
            # Peaked, not uniform: a flat distribution makes every builder emit
            # the same degenerate fan and would hide shape bugs.
            base_logits = (torch.randn(BLOCK, VOCAB, device=device) * 3.0)
            root = int(torch.randint(0, VOCAB, (1,)).item())

            try:
                node_tokens, node_depths, parents, _child_maps, visibility = build(
                    base_logits, budget, root)
            except Exception as exc:
                failures.append(f"{arm}/budget={budget}/seed={seed}: builder raised "
                                f"{type(exc).__name__}: {exc}")
                continue

            n = len(parents)
            fanouts = {}
            for p in parents[1:]:
                fanouts[p] = fanouts.get(p, 0) + 1
            shapes.append({"arm": arm, "budget": budget, "nodes": n,
                           "max_depth": int(node_depths.max()) if node_depths.numel() else 0,
                           "max_fanout": max(fanouts.values()) if fanouts else 0})

            # bridge -> SGLang kernel -> is it still our tree?
            mask = tree_to_qlen_mask(visibility, N)
            tokens = tree_to_draft_tokens(root, node_tokens, N)
            tree_mask = batch_qlen_mask([mask]).to(device)
            seq_lens = torch.tensor([128], dtype=torch.int64, device=device)
            positions = torch.empty((N,), dtype=torch.int64, device=device)
            ri = torch.empty((1, N), dtype=torch.int64, device=device)
            nt = torch.empty((1, N), dtype=torch.int64, device=device)
            ns = torch.empty((1, N), dtype=torch.int64, device=device)
            reconstruct_indices_from_tree_mask(
                tree_mask, seq_lens, positions, ri, nt, ns, 1, N)
            torch.cuda.synchronize()

            checked += 1
            got = parents_from_retrieve(nt[0].cpu().numpy(), ns[0].cpu().numpy())
            if got[:n] != list(parents):
                failures.append(
                    f"{arm}/budget={budget}/seed={seed}: kernel tree != builder tree\n"
                    f"  kernel={got[:n]}\n  builder={list(parents)}")
                continue
            pos = positions.cpu().numpy()
            want = [128] + [128 + int(d) for d in node_depths.tolist()]
            if pos[:n].tolist() != want[:n]:
                failures.append(f"{arm}/budget={budget}/seed={seed}: positions differ")
            if int(tokens[0]) != root:
                failures.append(f"{arm}/budget={budget}/seed={seed}: root token lost")

    result.update(checked=checked, num_failures=len(failures),
                  failures=failures[:10], shapes=shapes[:36])
    print(f"checked {checked} real builder trees, {len(failures)} failures")
    return result


@app.local_entrypoint()
def main():
    out = run.remote()
    print("\n=== dspark builder x sglang markov head ===")
    for k in ("head_class", "has_w1_w2", "w1_shape", "w2_shape", "import",
              "checked", "num_failures"):
        print(f"  {k}: {out.get(k)}")
    print("\n  tree shapes (arm / budget / nodes / max_depth / max_fanout):")
    seen = set()
    for s in out.get("shapes", []):
        key = (s["arm"], s["budget"])
        if key in seen:
            continue
        seen.add(key)
        print(f"    {s['arm']:>8} / {s['budget']:>4} / {s['nodes']:>4} / "
              f"{s['max_depth']:>3} / {s['max_fanout']:>3}")
    if out.get("import") != "ok":
        raise SystemExit(f"builder import failed: {out.get('import')}")
    if out.get("num_failures"):
        for f in out["failures"]:
            print("FAIL " + f)
        raise SystemExit(1)
    print("\nDSPARK BUILDER OK: our builder drives SGLang's head; trees round-trip")
