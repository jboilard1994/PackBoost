#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

static constexpr int LANES = 32;

// Shared-memory index: layout [nodes, 2, LANES]
static __device__ __forceinline__ int SH_idx(int node, int ch, int lane) {
    return (node * 2 + ch) * LANES + lane;
}

// H layout: [nf, nodes_tot, 2, LANES]
static __device__ __forceinline__ int64_t* Hptr(
    int64_t* H, int nodes_tot, int f, int node, int chan, int lane)
{
    return H + (((size_t)f * nodes_tot + node) * 2 + chan) * LANES + lane;
}

static __device__ __forceinline__ void atomicAdd_i64(int64_t* addr, int64_t val) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(val));
}

// one k-step, kept warp-uniform
static __device__ __forceinline__ void step_k(
    int k,
    unsigned fmask,
    int32_t y_val,
    uint32_t lf_word,
    uint32_t xs_word,
    int d,
    const int* __restrict__ depth_offset,
    int d_thresh,
    int nodes_tot,
    int f,
    int lane,
    int* __restrict__ s_hist,
    int64_t* __restrict__ H)
{
    const int32_t yk = __shfl_sync(fmask, y_val, k);
    const int32_t lk = (d == 0)
        ? 0
        : (int32_t)__shfl_sync(fmask, (int32_t)lf_word, k);

    const int     node = lk + depth_offset[d];
    const int32_t v    = (int32_t)((xs_word >> k) & 1u);

    if (d < d_thresh) {
        s_hist[SH_idx(node, 0, lane)] += v * yk;
        s_hist[SH_idx(node, 1, lane)] += v;
    } else if (v) {
        atomicAdd_i64(Hptr(H, nodes_tot, f, node, 0, lane), (int64_t)yk);
        atomicAdd_i64(Hptr(H, nodes_tot, f, node, 1, lane), (int64_t)1);
    }
}

// full-width fast path: k = 0..31 always
static __device__ __forceinline__ void process_full_tile(
    unsigned fmask,
    int32_t y_val,
    uint32_t lf_word,
    uint32_t xs_word,
    int d,
    const int* __restrict__ depth_offset,
    int d_thresh,
    int nodes_tot,
    int f,
    int lane,
    int* __restrict__ s_hist,
    int64_t* __restrict__ H)
{
    #pragma unroll
    for (int k = 0; k < LANES; ++k) {
        step_k(k, fmask, y_val, lf_word, xs_word, d,
               depth_offset, d_thresh, nodes_tot, f, lane, s_hist, H);
    }
}

// tail path: k = 0..n_valid-1
static __device__ __forceinline__ void process_tail_tile(
    int n_valid,
    unsigned fmask,
    int32_t y_val,
    uint32_t lf_word,
    uint32_t xs_word,
    int d,
    const int* __restrict__ depth_offset,
    int d_thresh,
    int nodes_tot,
    int f,
    int lane,
    int* __restrict__ s_hist,
    int64_t* __restrict__ H)
{
    for (int k = 0; k < n_valid; ++k) {
        step_k(k, fmask, y_val, lf_word, xs_word, d,
               depth_offset, d_thresh, nodes_tot, f, lane, s_hist, H);
    }
}

__global__ void _h_sm(
    const uint32_t* __restrict__ XS,
    const int16_t*  __restrict__ Y,
    const uint16_t* __restrict__ LF,
    int64_t*        __restrict__ H,
    int nf, int cols_32M, int N, int Dm,
    int max_depth, int nodes_tot, int d_thresh,
    int stride_per_block)
{
    const int f    = blockIdx.x;
    const int bi   = blockIdx.y;
    const int lane = threadIdx.x;

    if (f >= nf || lane >= LANES) return;

    const int smem_nodes = (1 << d_thresh) - 1;
    extern __shared__ int s_hist[];

    for (int n = 0; n < (1 << d_thresh); ++n) {
        s_hist[SH_idx(n, 0, lane)] = 0;
        s_hist[SH_idx(n, 1, lane)] = 0;
    }
    __syncwarp();

    int depth_offset[16];
    #pragma unroll
    for (int d = 0; d < 16; ++d) {
        if (d < max_depth) depth_offset[d] = (1 << d) - 1;
    }

    const uint16_t* LF_f  = LF + (size_t)f * Dm * N;
    const int       M     = cols_32M / LANES;
    const unsigned  fmask = 0xFFFFFFFFu;

    for (int j = 0; j < stride_per_block; ++j) {
        const int m = stride_per_block * bi + j;
        if (m >= M) break;

        const int base = m * LANES;
        const uint32_t xs_word = XS[(size_t)f * cols_32M + base + lane];

        if (__ballot_sync(fmask, xs_word != 0u) == 0u) continue;

        const int32_t y_val = (base + lane < N) ? (int32_t)Y[base + lane] : 0;
        const int n_valid = min(N - base, LANES);
        const bool full_tile = (n_valid == LANES);

        for (int d = 0; d < max_depth; ++d) {
            uint32_t lf_word = 0u;
            if (d > 0) {
                const int n_idx = base + lane;
                lf_word = (n_idx < N)
                    ? (uint32_t)LF_f[(size_t)(d - 1) * N + n_idx]
                    : 0u;
            }

            if (full_tile) {
                process_full_tile(
                    fmask, y_val, lf_word, xs_word, d,
                    depth_offset, d_thresh, nodes_tot, f, lane, s_hist, H);
            } else {
                process_tail_tile(
                    n_valid, fmask, y_val, lf_word, xs_word, d,
                    depth_offset, d_thresh, nodes_tot, f, lane, s_hist, H);
            }
        }
    }

    __syncwarp();
    for (int node = 0; node < smem_nodes && node < nodes_tot; ++node) {
        const int32_t ss = s_hist[SH_idx(node, 0, lane)];
        const int32_t sc = s_hist[SH_idx(node, 1, lane)];
        if (ss) atomicAdd_i64(Hptr(H, nodes_tot, f, node, 0, lane), (int64_t)ss);
        if (sc) atomicAdd_i64(Hptr(H, nodes_tot, f, node, 1, lane), (int64_t)sc);
    }
}

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

} // namespace

torch::Tensor h_sm(
    torch::Tensor XS,
    torch::Tensor Y,
    torch::Tensor LF,
    int max_depth)
{
    TORCH_CHECK(XS.is_cuda() && Y.is_cuda() && LF.is_cuda(),
                "h_sm: all inputs must be CUDA tensors");
    TORCH_CHECK(XS.dim() == 2, "XS must be 2-D [nf, 32*M]");
    TORCH_CHECK(Y.dim()  == 1, "Y must be 1-D [N]");
    TORCH_CHECK(LF.dim() == 3, "LF must be 3-D [nf, Dm, N]");
    TORCH_CHECK(max_depth > 0 && max_depth <= 16, "max_depth must be in [1, 16]");
    TORCH_CHECK(XS.scalar_type() == torch::kUInt32 ||
                XS.scalar_type() == torch::kInt32,  "XS must be uint32 or int32");
    TORCH_CHECK(Y.scalar_type()  == torch::kInt16,  "Y must be int16");
    TORCH_CHECK(LF.scalar_type() == torch::kUInt16, "LF must be uint16");

    auto XSc = XS.contiguous();
    auto Yc  = Y.contiguous();
    auto LFc = LF.contiguous();

    const int nf        = (int)XSc.size(0);
    const int cols_32M  = (int)XSc.size(1);
    const int N         = (int)Yc.size(0);
    const int Dm        = (int)LFc.size(1);
    const int nodes_tot = (1 << max_depth) - 1;
    const int M         = cols_32M / LANES;

    TORCH_CHECK(cols_32M % LANES == 0, "XS.size(1) must be divisible by 32");
    TORCH_CHECK((int)LFc.size(0) == nf, "LF.size(0) must equal nf");
    TORCH_CHECK(Dm == (max_depth > 1 ? max_depth - 1 : 0),
                "LF.size(1) must equal max_depth-1");
    TORCH_CHECK((int)LFc.size(2) == N, "LF.size(2) must equal N");

    auto H = torch::zeros(
        {nf, nodes_tot, 2, LANES},
        torch::TensorOptions().device(XS.device()).dtype(torch::kInt64));

    auto* prop = at::cuda::getCurrentDeviceProperties();
    size_t smem_cap = prop->sharedMemPerBlockOptin
        ? (size_t)prop->sharedMemPerBlockOptin
        : (size_t)prop->sharedMemPerBlock;

    int d_thresh = max_depth;
    while (d_thresh > 0 &&
           (size_t)(1 << d_thresh) * 2 * LANES * sizeof(int32_t) > smem_cap)
        --d_thresh;

    const size_t smem_bytes = (size_t)(1 << d_thresh) * 2 * LANES * sizeof(int32_t);

    const int SM = prop->multiProcessorCount;
    static constexpr int kTargetBlocksPerSM  = 32;
    static constexpr int kMinWorkloadPerBlock = 4;

    int strides = ceil_div(SM * kTargetBlocksPerSM, nf);
    strides = min(strides, max(1, ceil_div(M, kMinWorkloadPerBlock)));
    if (strides < 1) strides = 1;

    const int stride_per_block = max(1, ceil_div(M, strides));

    const dim3 block(LANES, 1, 1);
    const dim3 grid (nf, strides, 1);

    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        _h_sm, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    _h_sm<<<grid, block, smem_bytes, stream.stream()>>>(
        reinterpret_cast<const uint32_t*>(XSc.data_ptr()),
        Yc.data_ptr<int16_t>(),
        LFc.data_ptr<uint16_t>(),
        H.data_ptr<int64_t>(),
        nf, cols_32M, N, Dm, max_depth, nodes_tot, d_thresh,
        stride_per_block);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return H;
}