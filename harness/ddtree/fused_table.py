"""Opt-in fused CUDA kernel for the sparked-tree transition-table build.

sparked_tree.py::_union_transition_topk resolves, per (depth d, parent-slot p), the
row  x[c] = base_active[d, c] + bias[p, c]  over the union of U candidate columns,
log-softmax-normalises it, and keeps the top-k. The torch path does this as three
global-memory passes over a materialised [n, U, U] slab (add, logsumexp, topk) --
~256 MB at U~2000 -- which is most of the precompute builder's `.prep` cost.

This module fuses those three passes into ONE kernel with no [n, U, U]
materialisation: one CUDA block per (d, p) row loads base[d, :] and bias[p, :],
computes x in registers, block-reduces the logsumexp, then does an EXACT top-k via a
CUB block radix sort whose composite key reproduces torch.topk's tie rule (descending
value, lowest column index wins on ties). base+bias is a single fp32 add in both
paths, so the selected column set is bit-identical to the torch path; only the lse
reduction order differs (epsilon in `vals`).

Follows the inline-compiled extension pattern of ddtree.py::load_cpp_compact_module:
load_inline with graceful fallback (returns None) on any build failure, so the caller
transparently keeps the torch path. nvcc lives on the Modal CUDA-devel image; local
Macs have no CUDA, so load_fused_module() simply returns None there.
"""

from functools import lru_cache

from loguru import logger


# 256 threads x 16 items/thread = 4096 columns max. k <= 256.
_THREADS = 256
_ITEMS_PER_THREAD = 16
_MAX_UNION = _THREADS * _ITEMS_PER_THREAD  # 4096
_MAX_K = 256


_CUDA_SOURCE = r"""
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

// 256 threads; items/thread is templated so a row of U columns sorts over
// ceil(U/256) rounded up to {4,8,12,16} keys instead of always 4096 (no padding
// waste at small U).
#define FT_THREADS 256

// monotonic_flip: map float bits to a uint32 that sorts in ascending float order.
// (sign flip for positives, full flip for negatives -- the standard radix trick.)
__device__ __forceinline__ unsigned int monotonic_flip(float v) {
    unsigned int b = __float_as_uint(v);
    return b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

// One block per (depth, parent-slot) row. Grid = L * U.
template <int FT_ITEMS>
__global__ void fused_bias_lse_topk_kernel(
        const float* __restrict__ base,   // [L, U]
        const float* __restrict__ bias,   // [U, U]
        float* __restrict__ vals,         // [L, U, k]
        int*   __restrict__ slots,        // [L, U, k]
        int L, int U, int k) {
    const long row = blockIdx.x;          // 0 .. L*U-1
    const int d = (int)(row / U);
    const int p = (int)(row % U);
    const float* base_row = base + (long)d * U;
    const float* bias_row = bias + (long)p * U;

    typedef cub::BlockRadixSort<unsigned long long, FT_THREADS, FT_ITEMS> BlockRadixSortT;
    typedef cub::BlockReduce<float, FT_THREADS> BlockReduceT;
    __shared__ union {
        typename BlockRadixSortT::TempStorage sort;
        typename BlockReduceT::TempStorage reduce;
    } temp;
    __shared__ float s_max;
    __shared__ float s_lse;

    float xvals[FT_ITEMS];
    unsigned long long keys[FT_ITEMS];

    const float NEG_INF = -__int_as_float(0x7f800000);  // -inf

    // Blocked arrangement: thread t owns columns [t*ITEMS, t*ITEMS+ITEMS).
    #pragma unroll
    for (int i = 0; i < FT_ITEMS; ++i) {
        int c = threadIdx.x * FT_ITEMS + i;
        xvals[i] = (c < U) ? (base_row[c] + bias_row[c]) : NEG_INF;
    }

    // --- logsumexp over the row (max then sum-exp) ---
    float local_max = NEG_INF;
    #pragma unroll
    for (int i = 0; i < FT_ITEMS; ++i) local_max = fmaxf(local_max, xvals[i]);
    float block_max = BlockReduceT(temp.reduce).Reduce(local_max, cub::Max());
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();
    float row_max = s_max;

    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < FT_ITEMS; ++i) {
        int c = threadIdx.x * FT_ITEMS + i;
        if (c < U) local_sum += __expf(xvals[i] - row_max);
    }
    __syncthreads();  // reuse temp.reduce
    float block_sum = BlockReduceT(temp.reduce).Sum(local_sum);
    if (threadIdx.x == 0) s_lse = row_max + logf(block_sum);
    __syncthreads();
    float row_lse = s_lse;

    // --- exact top-k via block radix sort ---
    // key = (~flip(x) << 32) | column: ascending key == descending value, and on
    // equal values the ascending column index wins -- torch.topk's tie rule.
    #pragma unroll
    for (int i = 0; i < FT_ITEMS; ++i) {
        int c = threadIdx.x * FT_ITEMS + i;
        if (c < U) {
            unsigned int hi = ~monotonic_flip(xvals[i]);
            keys[i] = ((unsigned long long)hi << 32) | (unsigned int)c;
        } else {
            keys[i] = 0xFFFFFFFFFFFFFFFFULL;  // pad sorts to the very end
        }
    }
    __syncthreads();  // reuse temp for the sort
    BlockRadixSortT(temp.sort).Sort(keys);  // ascending, blocked output arrangement

    // Blocked output: thread t holds global ranks [t*ITEMS, t*ITEMS+ITEMS).
    float* vals_row = vals + row * k;
    int* slots_row = slots + row * k;
    #pragma unroll
    for (int i = 0; i < FT_ITEMS; ++i) {
        int g = threadIdx.x * FT_ITEMS + i;
        if (g < k) {
            int col = (int)(keys[i] & 0xFFFFFFFFULL);
            float x = base_row[col] + bias_row[col];  // exact, matches selection
            vals_row[g] = x - row_lse;
            slots_row[g] = col;
        }
    }
}

std::vector<torch::Tensor> fused_bias_lse_topk(torch::Tensor base, torch::Tensor bias, int64_t k) {
    TORCH_CHECK(base.is_cuda() && bias.is_cuda(), "base and bias must be CUDA tensors");
    TORCH_CHECK(base.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32,
                "base and bias must be float32");
    TORCH_CHECK(base.dim() == 2 && bias.dim() == 2, "base must be [L,U], bias [U,U]");
    base = base.contiguous();
    bias = bias.contiguous();
    const int L = (int)base.size(0);
    const int U = (int)base.size(1);
    TORCH_CHECK(bias.size(0) == U && bias.size(1) == U, "bias must be [U,U] matching base");
    TORCH_CHECK(U <= FT_THREADS * 16, "U exceeds fused-kernel capacity (4096)");
    TORCH_CHECK(k >= 1 && k <= 256 && k <= U, "k must be in [1, min(256, U)]");

    auto vals = torch::empty({L, U, (long)k}, base.options());
    auto slots = torch::empty({L, U, (long)k}, base.options().dtype(torch::kInt32));

    dim3 grid((unsigned)((long)L * U));
    dim3 block(FT_THREADS);
    auto stream = at::cuda::getCurrentCUDAStream();
    // Smallest items/thread whose block capacity (256 * items) covers U -> the sort
    // touches no more padding than one extra tile's worth.
    float* bp = base.data_ptr<float>();
    float* xp = bias.data_ptr<float>();
    float* vp = vals.data_ptr<float>();
    int* sp = slots.data_ptr<int>();
    auto s = stream.stream();
    if (U <= FT_THREADS * 4)
        fused_bias_lse_topk_kernel<4><<<grid, block, 0, s>>>(bp, xp, vp, sp, L, U, (int)k);
    else if (U <= FT_THREADS * 8)
        fused_bias_lse_topk_kernel<8><<<grid, block, 0, s>>>(bp, xp, vp, sp, L, U, (int)k);
    else if (U <= FT_THREADS * 12)
        fused_bias_lse_topk_kernel<12><<<grid, block, 0, s>>>(bp, xp, vp, sp, L, U, (int)k);
    else
        fused_bias_lse_topk_kernel<16><<<grid, block, 0, s>>>(bp, xp, vp, sp, L, U, (int)k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {vals, slots};
}
"""

_CPP_DECL = "std::vector<torch::Tensor> fused_bias_lse_topk(torch::Tensor base, torch::Tensor bias, int64_t k);"


@lru_cache(maxsize=1)
def load_fused_module():
    """Build (once) and return the inline CUDA extension, or None on any failure.

    None is the graceful-fallback signal: no CUDA toolchain (local Mac), build error,
    or missing CUB -- the caller keeps the torch path. Cached so the ~2-5 min nvcc
    compile happens at most once per process."""
    try:
        import torch
        from torch.utils.cpp_extension import load_inline
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"torch.utils.cpp_extension unavailable; fused table disabled. {exc}")
        return None

    if not torch.cuda.is_available():
        logger.warning("CUDA unavailable; fused table disabled (torch path only).")
        return None

    try:
        module = load_inline(
            name="ddtree_fused_table_ext_v2",
            cpp_sources=[_CPP_DECL],
            cuda_sources=[_CUDA_SOURCE],
            functions=["fused_bias_lse_topk"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr"],
            verbose=False,
        )
        logger.info("Loaded inline CUDA fused transition-table extension.")
        return module
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to build fused CUDA table extension; using torch path. {exc}")
        return None


def fused_supported(union_size: int, topk: int) -> bool:
    """Shape guard: the kernel supports U <= 4096, 1 <= k <= min(256, U)."""
    return 1 <= topk <= min(_MAX_K, union_size) and union_size <= _MAX_UNION
