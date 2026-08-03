"""Negative control for test_kernel_agreement: can that test actually fail?

    modal run concurrency/test_negative_control.py::run

A green agreement run means nothing unless a wrong mask turns it red. Three
deliberate corruptions, each a convention mistake we could plausibly have made.

RESULT (H100/L4, sglang 0.5.16), and the reason this file is worth keeping:

    transposed            25/25 detected   <- caught
    diagonal_only         25/25 detected   <- caught
    exclusive_ancestors    0/25 detected   <- NOT caught
    correct                0/25 flagged    <- no false positives

`exclusive_ancestors` clears the diagonal, and `reconstruct_indices_from_tree_mask`
does not read it: the parent/child structure lives entirely in the off-diagonal
ancestor bits. So the agreement test covers tree STRUCTURE and nothing else.

That is a limitation of the test, not a defect in the bridge -- our diagonal is
correct by construction (`visibility[i, i] = True`, ddtree_markov.py:637) and is
asserted directly by test_sparked_bridge's "node visible to itself" check. But it
matters, because the same mask is reused as the attention mask on the FULL_MASK
path, where a cleared diagonal WOULD be wrong: a draft token that cannot attend
itself gets wrong logits. Nothing downstream of the kernel would flag it.

So the two tests are not redundant and neither subsumes the other:
    CPU test  -> the diagonal is set, and the mask encodes our tree
    GPU test  -> SGLang reconstructs that same tree from our mask
"""

import sys
from pathlib import Path

import modal

app = modal.App("sparked-negative-control")
TAG = "lmsysorg/sglang:v0.5.16-cu129"
image = modal.Image.from_registry(TAG).add_local_dir(
    str(Path(__file__).resolve().parent), remote_path="/root/concurrency"
)


@app.function(image=image, gpu="L4", timeout=1800)
def run() -> dict:
    import random
    import torch

    sys.path.insert(0, "/root/concurrency")
    from sparked_bridge import batch_qlen_mask, parents_from_retrieve, tree_to_qlen_mask
    from test_sparked_bridge import best_first_like, visibility_from_parents
    from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

    device = torch.device("cuda")
    N = 65

    def kernel_parents(mask_2d, n):
        tree_mask = batch_qlen_mask([mask_2d]).to(device)
        seq_lens = torch.tensor([64], dtype=torch.int64, device=device)
        positions = torch.empty((N,), dtype=torch.int64, device=device)
        ri = torch.empty((1, N), dtype=torch.int64, device=device)
        nt = torch.empty((1, N), dtype=torch.int64, device=device)
        ns = torch.empty((1, N), dtype=torch.int64, device=device)
        reconstruct_indices_from_tree_mask(
            tree_mask, seq_lens, positions, ri, nt, ns, 1, N)
        torch.cuda.synchronize()
        try:
            return parents_from_retrieve(nt[0].cpu().numpy(), ns[0].cpu().numpy())[:n]
        except Exception as exc:
            return f"walk-error: {exc}"

    results = {}
    for name in ("transposed", "exclusive_ancestors", "diagonal_only", "correct"):
        mismatches = 0
        for seed in range(25):
            parents = best_first_like(N, random.Random(seed))
            vis = visibility_from_parents(parents)
            mask = tree_to_qlen_mask(vis, N)
            if name == "transposed":
                mask = mask.T.contiguous()
            elif name == "exclusive_ancestors":       # drop the self bit
                mask = mask.clone()
                mask.fill_diagonal_(False)
            elif name == "diagonal_only":             # forget ancestors entirely
                mask = torch.eye(N, dtype=torch.bool)
            got = kernel_parents(mask, len(parents))
            if got != parents:
                mismatches += 1
        results[name] = f"{mismatches}/25 detected as wrong"
        print(f"{name}: {results[name]}")
    return results


@app.local_entrypoint()
def main():
    out = run.remote()
    print("\n=== negative control ===")
    for k, v in out.items():
        print(f"{k}: {v}")
    # exclusive_ancestors is expected NOT to be caught -- the kernel ignores the
    # diagonal. Asserting the known result keeps this honest: if a future sglang
    # starts reading the diagonal, this flips and we find out.
    expected_caught = {"transposed", "diagonal_only"}
    expected_missed = {"exclusive_ancestors"}

    problems = []
    for name in expected_caught:
        if out[name].startswith("0/"):
            problems.append(f"{name} went undetected -- agreement test lost its teeth")
    for name in expected_missed:
        if not out[name].startswith("0/"):
            problems.append(f"{name} is now detected -- kernel began reading the "
                            "diagonal; re-read the mask contract")
    if not out["correct"].startswith("0/"):
        problems.append("correct masks reported wrong -- the test itself is broken")

    if problems:
        for line in problems:
            print("PROBLEM: " + line)
        raise SystemExit(1)
    print("CONTROL PASSED (structure corruptions caught; diagonal known-uncovered)")
