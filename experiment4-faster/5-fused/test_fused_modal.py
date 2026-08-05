"""Modal GPU validation for the fused transition-table CUDA kernel.

Proves harness/ddtree/fused_table.py::fused_bias_lse_topk is a drop-in for the
torch add+logsumexp+topk slab in sparked_tree.py::_union_transition_topk, then times
the two head-to-head. Three checks on one H100:

  PARITY (op-level): random base [L,U] / bias [U,U], torch path vs fused. The add is a
    single fp32 op in both and the kernel's radix-sort tie rule matches torch.topk, so
    the selected columns (slots) must be EXACTLY equal; vals differ only by lse
    reduction-order epsilon (allclose atol 1e-5).
  PARITY (tree-level): build_sparked_tree_precompute with use_fused True vs False on
    synthetic strong-bias logits -- the whole builder must build the same best-first
    tree (expect ~100% node agreement).
  MICROBENCH: torch slab vs fused op at L=16, U=2048, each k, 500 iters, CUDA events.

The nvcc compile (~2-5 min) is the long pole. Run SYNCHRONOUSLY:
    modal run test_fused_modal.py
The report is also written to the ddtree-results volume (fused/report.json), so if the
CLI session drops mid-run just re-run -- the volume JSON persists.
"""

import json
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Image / app (mirrors experiment4-faster/2-precompute/modal_benchmark.py)     #
# --------------------------------------------------------------------------- #

GPU = "H100"
TORCH_VERSION = "2.5.1"

HERE = Path(__file__).parent
HARNESS_DIR = HERE.parent.parent / "harness"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .pip_install(f"torch=={TORCH_VERSION}", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy", "loguru", "ninja", "typing_extensions")
    .add_local_dir(HARNESS_DIR.as_posix(), remote_path="/root/harness")
)

app = modal.App("ddtree-exp4-fused")
results_vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)


# --------------------------------------------------------------------------- #
# Remote validation                                                           #
# --------------------------------------------------------------------------- #

@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    timeout=60 * 60,
    volumes={"/results": results_vol},
)
def validate() -> dict:
    import sys
    import types

    sys.path.insert(0, "/root/harness/ddtree")

    import torch

    from fused_table import load_fused_module, fused_supported

    # --- helpers copied from 2-precompute/test_precompute_builder.py -----------
    # (copied, not imported, to avoid that module's import-time dependency stubs.)
    class _MarkovHead:
        def __init__(self, w1, w2):
            self.markov_w1 = type("W", (), {"weight": w1})()
            self.markov_w2 = type("W", (), {"weight": w2})()

    def _tree_key(node_token_ids, node_depths, parents):
        toks = node_token_ids.tolist()
        deps = node_depths.tolist()
        return tuple(sorted((int(d), int(t), int(p)) for d, t, p in zip(deps, toks, parents[1:])))

    def _make_inputs(seed, depth=16, vocab=2000, rank=32, bias_scale=0.1, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        base_logits = torch.randn(depth, vocab, generator=g)
        w1 = torch.randn(vocab, rank, generator=g) * bias_scale
        w2 = torch.randn(vocab, rank, generator=g) * bias_scale
        head = _MarkovHead(w1.to(device), w2.to(device))
        root = int(torch.randint(0, vocab, (1,), generator=g).item())
        return base_logits.to(device), head, root

    def torch_slab_topk(base, bias, k):
        """Exact torch path of _union_transition_topk (all depths in one batch)."""
        rows = base.unsqueeze(1)                     # [L, 1, U]
        slab = rows + bias.unsqueeze(0)              # [L, U, U]
        slab = slab - torch.logsumexp(slab, dim=-1, keepdim=True)
        return torch.topk(slab, k=k, dim=-1)         # (vals, slots)

    report: dict = {"build": {}, "parity_op": [], "parity_tree": [], "microbench": []}

    # --- build the extension (long pole) --------------------------------------
    module = load_fused_module()
    if module is None:
        report["build"]["ok"] = False
        report["build"]["error"] = "load_fused_module() returned None (build failed / no CUDA)"
        return report
    report["build"]["ok"] = True

    dev = "cuda"
    L = 16

    # --- PARITY (op-level) -----------------------------------------------------
    op_all_slots_exact = True
    op_max_vals_diff = 0.0
    for seed in range(6):
        g = torch.Generator(device=dev).manual_seed(seed)
        for U in (1500, 2048, 4096):
            base = torch.randn(L, U, generator=g, device=dev, dtype=torch.float32)
            bias = torch.randn(U, U, generator=g, device=dev, dtype=torch.float32) * 0.1
            for k in (64, 128, 256):
                if not fused_supported(U, k):
                    continue
                t_vals, t_slots = torch_slab_topk(base, bias, k)
                f_vals, f_slots = module.fused_bias_lse_topk(base.contiguous(), bias.contiguous(), k)
                torch.cuda.synchronize()
                t_slots_i = t_slots.to(torch.int32)
                f_slots_i = f_slots.to(torch.int32)
                slots_exact = bool(torch.equal(t_slots_i, f_slots_i))
                n_diff = int((t_slots_i != f_slots_i).sum().item())
                frac_diff = n_diff / t_slots_i.numel()
                vals_diff = float((t_vals - f_vals).abs().max().item())
                vals_close = bool(torch.allclose(t_vals, f_vals, atol=1e-5))
                op_all_slots_exact = op_all_slots_exact and slots_exact
                op_max_vals_diff = max(op_max_vals_diff, vals_diff)
                report["parity_op"].append({
                    "seed": seed, "U": U, "k": k,
                    "slots_exact": slots_exact,
                    "slots_frac_diff": frac_diff, "slots_n_diff": n_diff,
                    "vals_max_abs_diff": vals_diff, "vals_allclose_1e-5": vals_close,
                })
    op_max_frac_diff = max((x["slots_frac_diff"] for x in report["parity_op"]), default=0.0)
    report["parity_op_summary"] = {
        "all_slots_exact": op_all_slots_exact,
        "max_slots_frac_diff": op_max_frac_diff,
        "max_vals_abs_diff": op_max_vals_diff,
    }

    # --- PARITY (tree-level): whole builder, use_fused True vs False -----------
    # sparked_tree (and ddtree) import transformers/model/dflash at module load; this
    # image is deliberately lean (no transformers). Stub those the same way
    # 2-precompute/test_precompute_builder.py does -- the builder only needs
    # torch/numpy/heapq + the light timing helpers.
    def _stub(name, **attrs):
        m = types.ModuleType(name)
        for kk, vv in attrs.items():
            setattr(m, kk, vv)
        sys.modules[name] = m

    _stub("transformers", AutoModelForCausalLM=object, DynamicCache=object)
    _stub("model", sample=None, extract_context_feature=None, DFlashDraftModel=object)
    _stub("dflash", dflash_generate=None)

    from sparked_tree import build_sparked_tree_precompute

    tree_total_agree, tree_total_nodes = 0, 0
    for seed in range(6):
        base_logits, head, root = _make_inputs(
            seed, depth=16, vocab=2000, rank=32, bias_scale=3.0, device=dev)
        for budget in (64, 256):
            C = 128
            n_tok, n_dep, n_par, *_ = build_sparked_tree_precompute(
                base_logits, head, root, budget, C, use_fused=False)
            f_tok, f_dep, f_par, *_ = build_sparked_tree_precompute(
                base_logits, head, root, budget, C, use_fused=True)
            nk, fk = _tree_key(n_tok, n_dep, n_par), _tree_key(f_tok, f_dep, f_par)
            inter = len(set(nk) & set(fk))
            denom = max(len(nk), len(fk), 1)
            tree_total_agree += inter
            tree_total_nodes += denom
            report["parity_tree"].append({
                "seed": seed, "budget": budget, "C": C,
                "agree_pct": 100.0 * inter / denom,
                "nodes_torch": len(n_tok), "nodes_fused": len(f_tok),
            })
    report["parity_tree_summary"] = {
        "overall_agree_pct": 100.0 * tree_total_agree / max(tree_total_nodes, 1),
    }

    # --- MICROBENCH: torch slab vs fused, L=16 U=2048 --------------------------
    U = 2048
    g = torch.Generator(device=dev).manual_seed(123)
    base = torch.randn(L, U, generator=g, device=dev, dtype=torch.float32)
    bias = torch.randn(U, U, generator=g, device=dev, dtype=torch.float32) * 0.1
    ITERS = 500

    def time_call(fn):
        for _ in range(10):  # warmup
            fn()
        torch.cuda.synchronize()
        start_ev, end_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(ITERS):
            fn()
        end_ev.record()
        torch.cuda.synchronize()
        return start_ev.elapsed_time(end_ev) * 1000.0 / ITERS  # microseconds/call

    for k in (64, 128, 256):
        torch_us = time_call(lambda: torch_slab_topk(base, bias, k))
        fused_us = time_call(lambda: module.fused_bias_lse_topk(base.contiguous(), bias.contiguous(), k))
        report["microbench"].append({
            "L": L, "U": U, "k": k,
            "torch_us_per_call": torch_us,
            "fused_us_per_call": fused_us,
            "speedup": torch_us / fused_us if fused_us else None,
        })

    # --- persist + print -------------------------------------------------------
    out = Path("/results/fused/report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    results_vol.commit()

    print("\n" + "=" * 68)
    print("FUSED TABLE KERNEL VALIDATION")
    print("=" * 68)
    print(f"build ok: {report['build']['ok']}")
    print(f"\nPARITY (op): all slots bit-exact = {op_all_slots_exact}   "
          f"max slot-entry mismatch frac = {op_max_frac_diff:.2e}   "
          f"max |vals| diff = {op_max_vals_diff:.2e}")
    print(f"PARITY (tree): overall node agreement = "
          f"{report['parity_tree_summary']['overall_agree_pct']:.2f}%")
    print("\nMICROBENCH (L=16, U=2048, 500 iters, us/call):")
    print(f"  {'k':>5} {'torch':>12} {'fused':>12} {'speedup':>9}")
    for row in report["microbench"]:
        print(f"  {row['k']:>5} {row['torch_us_per_call']:>10.2f}us "
              f"{row['fused_us_per_call']:>10.2f}us {row['speedup']:>8.2f}x")
    print("=" * 68)

    return report


@app.local_entrypoint()
def main():
    report = validate.remote()
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {out_dir / 'report.json'}")
