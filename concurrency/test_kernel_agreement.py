"""Does SGLang's own kernel recover our tree from our mask?

    modal run concurrency/test_kernel_agreement.py::run

The CPU tests prove our mask encodes our tree. They cannot prove SGLang reads it
the same way -- a transposed convention, an exclusive-vs-inclusive ancestor path,
or a different root rule would pass every CPU test and produce silently wrong
acceptance in the server. This runs the real
`sgl_kernel.speculative.reconstruct_indices_from_tree_mask` over our real masks
and checks the tree it reconstructs is the tree we built.

Failure here means the plugin would verify against the wrong tree, which shows up
as degraded acceptance rather than a crash -- the expensive kind of bug.
"""

import sys
from pathlib import Path

import modal

app = modal.App("sparked-kernel-agreement")
TAG = "lmsysorg/sglang:v0.5.16-cu129"
image = (
    modal.Image.from_registry(TAG)
    .add_local_dir(Path(__file__).parent, remote_path="/root/concurrency")
)


@app.function(image=image, gpu="L4", timeout=3600)
def run() -> dict:
    import random

    import numpy as np
    import torch

    sys.path.insert(0, "/root/concurrency")
    from sparked_bridge import (
        batch_qlen_mask,
        parents_from_retrieve,
        tree_to_qlen_mask,
    )
    from test_sparked_bridge import (
        balanced,
        best_first_like,
        chain,
        deep_narrow,
        depths_from_parents,
        star,
        visibility_from_parents,
    )
    from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

    device = torch.device("cuda")
    failures, checked = [], 0

    def check_batch(parent_sets, N, label):
        """One kernel call over a batch of trees; compare each against its source."""
        nonlocal checked
        bs = len(parent_sets)
        masks, seq_lens = [], []
        for i, parents in enumerate(parent_sets):
            vis = visibility_from_parents(parents)
            masks.append(tree_to_qlen_mask(vis, N))
            seq_lens.append(64 + 7 * i)

        tree_mask = batch_qlen_mask(masks).to(device)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        positions = torch.empty((bs * N,), dtype=torch.int64, device=device)
        retrieve_index = torch.empty((bs, N), dtype=torch.int64, device=device)
        retrieve_next_token = torch.empty((bs, N), dtype=torch.int64, device=device)
        retrieve_next_sibling = torch.empty((bs, N), dtype=torch.int64, device=device)

        reconstruct_indices_from_tree_mask(
            tree_mask, seq_lens_t, positions, retrieve_index,
            retrieve_next_token, retrieve_next_sibling, bs, N,
        )
        torch.cuda.synchronize()

        nt = retrieve_next_token.cpu().numpy()
        ns = retrieve_next_sibling.cpu().numpy()
        pos = positions.cpu().numpy().reshape(bs, N)

        for i, parents in enumerate(parent_sets):
            n = len(parents)
            checked += 1
            try:
                got = parents_from_retrieve(nt[i], ns[i])
            except Exception as exc:
                failures.append(f"{label}[{i}]: walk failed: {exc}")
                continue
            # Compare only the real nodes; padded rows are isolated by design.
            if got[:n] != parents:
                failures.append(
                    f"{label}[{i}] n={n}: parents differ\n  kernel={got[:n]}\n  ours  ={parents}")
                continue
            depths = [0] + depths_from_parents(parents).tolist()
            want = [seq_lens[i] + d for d in depths]
            if pos[i][:n].tolist() != want:
                failures.append(
                    f"{label}[{i}] n={n}: positions differ\n  kernel={pos[i][:n].tolist()}\n  ours  ={want}")

    # Regular shapes, batched, exactly at budget.
    for N in (17, 33, 65, 129):
        for name, gen in (("chain", chain), ("star", star),
                          ("balanced", balanced), ("deep_narrow", deep_narrow)):
            check_batch([gen(N)], N, f"{name}/N={N}")

    # Irregular best-first shapes -- the case tree_topk=-1 exists for.
    for N in (17, 65, 129):
        for seed in range(60):
            r = random.Random(seed)
            check_batch([best_first_like(N, r)], N, f"best_first/N={N}/seed={seed}")

    # Real batching: several different trees in one kernel call.
    for N in (33, 65):
        for seed in range(30):
            r = random.Random(1000 + seed)
            bs = r.randrange(2, 9)
            check_batch([best_first_like(N, random.Random(seed * 100 + k))
                         for k in range(bs)], N, f"batch{bs}/N={N}/seed={seed}")

    # Under-budget trees, exercising the padding path.
    for N in (65, 129):
        for n in (1, 2, 5, 33, N - 1):
            check_batch([best_first_like(n, random.Random(n))], N, f"pad/n={n}/N={N}")

    result = {"checked": checked, "failures": failures[:25],
              "num_failures": len(failures)}
    print(f"checked {checked} trees, {len(failures)} failures")
    for line in failures[:25]:
        print("  FAIL " + line)
    return result


@app.local_entrypoint()
def main():
    out = run.remote()
    print(f"\n=== {out['checked']} trees checked, {out['num_failures']} failures ===")
    if out["num_failures"]:
        for line in out["failures"]:
            print("FAIL " + line)
        raise SystemExit(1)
    print("KERNEL AGREEMENT: our mask reconstructs to our tree")
