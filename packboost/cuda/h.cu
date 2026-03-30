#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// h_sm  —  per-feature-set split histogram
//
// Directly mirrors the structure of h0_sm_butterfly, with three changes:
//
//   1. grid.x = nf  (feature sets, not folds)
//
//   2. lane = feature index (0..31), not sample index.
//      One warp per block → smem is warp-private → plain += (no atomics) in smem.
//
//   3. Inner k-loop replaces h0's single-sample-per-lane body:
//        xs_word[lane]  = XS[f, m*32+lane]        coalesced uint32 load once per tile
//        y_val  [lane]  = Y  [m*32+lane]           coalesced int16  load once per tile
//        lf_word[lane]  = LF [f, d-1, m*32+lane]  coalesced uint16 load once per depth
//      Then k=0..31 unpacks 32 samples with free warp shuffles.
//
//   4. No butterfly needed — lane IS the output dimension (H[..,lane]).
//      Flush is a direct per-lane atomicAdd to H (coalesced across 32 lanes).
//
// Accumulation tiers (identical threshold logic to h0):
//   d < d_thresh  →  lane-private smem: int64 gain + int32 count, plain +=
//   d >= d_thresh →  global int64 atomicAdd  (coalesced across 32 lanes)
// ---------------------------------------------------------------------------

namespace {

static constexpr int LANES = 32;

// Shared-memory gain index: layout [nodes, LANES] int64
static __device__ __forceinline__ int SG_idx(int node, int lane) {
    return node * LANES + lane;
}
// Shared-memory count index: layout [nodes, LANES] int32
static __device__ __forceinline__ int SC_idx(int node, int lane) {
    return node * LANES + lane;
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

// XS : [nf, cols_32M] uint32   XS[f, m*32+l]: bit k = (sample m*32+k has feature l set)
// Y  : [N]            int16
// LF : [nf, Dm, N]    uint16   LF[f, d, n]  = (d+1)-bit tree-path prefix for sample n at depth d+1
// H  : [nf, nodes_tot, 2, LANES] int64   (zero-initialised by caller)
__global__ void _h_sm(
    const uint32_t* __restrict__ XS,
    const int16_t*  __restrict__ Y,
    const uint16_t* __restrict__ LF,
    int64_t*        __restrict__ H,
    int nf, int cols_32M, int N, int Dm,
    int max_depth, int nodes_tot, int d_thresh,
    int stride_per_block)
{
    const int f    = blockIdx.x;   // feature-set index
    const int bi   = blockIdx.y;   // sample-block tile index
    const int lane = threadIdx.x;  // 0..31  =  feature index within the set

    if (f >= nf || lane >= LANES) return;

    // ---- Shared memory: int64 gain + int32 count per node per lane ----
    // One warp per block → this memory is warp-private → plain += is safe.
    const int smem_alloc  = 1 << d_thresh;          // allocation slots (>= nodes)
    const int smem_nodes  = smem_alloc - 1;          // actual tree nodes 0..smem_nodes-1
    extern __shared__ char raw_smem[];
    int64_t* s_gain  = reinterpret_cast<int64_t*>(raw_smem);
    int32_t* s_count = reinterpret_cast<int32_t*>(raw_smem + (size_t)smem_alloc * LANES * sizeof(int64_t));

    for (int n = 0; n < smem_alloc; ++n) {
        s_gain [SG_idx(n, lane)] = 0;
        s_count[SC_idx(n, lane)] = 0;
    }
    __syncwarp();

    // Depth offsets: depth_offset[d] = (1<<d) - 1  (first node index at depth d)
    int depth_offset[16];
    for (int d = 0; d < max_depth; ++d) depth_offset[d] = (1 << d) - 1;

    const uint16_t* LF_f  = LF + (size_t)f * Dm * N;
    const int       M     = cols_32M / LANES;     // number of 32-sample blocks
    const unsigned  fmask = 0xFFFFFFFFu;

    // ---- Sample-tile loop (mirrors h0's j-loop) ----
    for (int j = 0; j < stride_per_block; ++j) {
        const int m = stride_per_block * bi + j;
        if (m >= M) break;

        const int base = m * LANES;

        // All 32 lanes load different elements — fully coalesced.
        // xs_word[lane]: bit k = (sample base+k has feature `lane` set)
        const uint32_t xs_word = XS[(size_t)f * cols_32M + base + lane];

        // Skip tile entirely if no feature in this set fires for any sample.
        // (ballot across lane dimension — identical to h0's implicit early exit)
        if (__ballot_sync(fmask, xs_word != 0u) == 0u) continue;

        // y_val[lane] = gradient of sample base+lane
        const int32_t y_val = (base + lane < N) ? (int32_t)Y[base + lane] : 0;

        // Valid sample count in this tile
        const int n_valid = min(N - base, LANES);

        // ---- Depth loop ----
        for (int d = 0; d < max_depth; ++d) {

            // Coalesced LF load: lane l reads LF[f, d-1, base+l]
            // All 32 lanes load their own element in a single transaction.
            uint32_t lf_word = 0u;
            if (d > 0) {
                const int n_idx = base + lane;
                lf_word = (n_idx < N)
                    ? (uint32_t)LF_f[(size_t)(d - 1) * N + n_idx] : 0u;
            }

            // ---- k-loop: process 32 samples in this tile ----
            // __shfl_sync calls MUST come before any v-based branching so that
            // all 32 lanes always participate (mask = 0xFFFFFFFF).
            for (int k = 0; k < n_valid; ++k) {
                // Broadcast gradient and path-prefix of sample base+k to all lanes.
                const int32_t yk = __shfl_sync(fmask, y_val,  k);
                const int32_t lk = (d == 0)
                    ? 0 : (int32_t)__shfl_sync(fmask, (int32_t)lf_word, k);

                const int     node = lk + depth_offset[d];
                const int32_t v    = (int32_t)((xs_word >> k) & 1u);

                if (d < d_thresh) {
                    // Lane-private smem — no conflict (1 warp/block) — plain +=.
                    // int64 gain avoids overflow; int32 count is always safe.
                    s_gain [SG_idx(node, lane)] += (int64_t)v * (int64_t)yk;
                    s_count[SC_idx(node, lane)] += v;
                } else if (v) {
                    // Global int64 atomic — all participating lanes write to
                    // H[f, node, chan, 0..31]: 32 consecutive int64 → coalesced.
                    // v-guard is after all shfl calls so divergence is safe here.
                    atomicAdd_i64(Hptr(H, nodes_tot, f, node, 0, lane), (int64_t)yk);
                    atomicAdd_i64(Hptr(H, nodes_tot, f, node, 1, lane), (int64_t)1);
                }
            }
        }
    }

    // ---- Flush smem → global H ----
    // No butterfly needed: lane IS the output dimension.
    // Each lane flushes its own accumulated value directly.
    // 32 lanes writing H[f, node, chan, 0..31] = one coalesced warp store per node.
    __syncwarp();
    for (int node = 0; node < smem_nodes && node < nodes_tot; ++node) {
        const int64_t ss = s_gain [SG_idx(node, lane)];
        const int32_t sc = s_count[SC_idx(node, lane)];
        if (ss) atomicAdd_i64(Hptr(H, nodes_tot, f, node, 0, lane), ss);
        if (sc) atomicAdd_i64(Hptr(H, nodes_tot, f, node, 1, lane), (int64_t)sc);
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers  (mirror h0_sm_butterfly's launcher logic)
// ---------------------------------------------------------------------------

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

} // namespace

// ---------------------------------------------------------------------------
// Public API — same signature and output shape as the previous kernel
// ---------------------------------------------------------------------------

torch::Tensor h_sm(
    torch::Tensor XS,   // [nf, 32*M]   uint32 / int32
    torch::Tensor Y,    // [N]          int16
    torch::Tensor LF,   // [nf, Dm, N]  uint16
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
    const int M         = cols_32M / LANES;   // number of 32-sample blocks

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

    // d_thresh: largest d such that (1<<d) * LANES * (int64 + int32) fits in smem.
    // Gain channel uses int64 to avoid overflow; count channel uses int32.
    int d_thresh = max_depth;
    while (d_thresh > 0 &&
           (size_t)(1 << d_thresh) * LANES * (sizeof(int64_t) + sizeof(int32_t)) > smem_cap)
        --d_thresh;

    const size_t smem_bytes = (size_t)(1 << d_thresh) * LANES * (sizeof(int64_t) + sizeof(int32_t));

    // Grid scheduling: mirrors h0's Murky heuristic, adapted for M sample-blocks.
    // One warp (32 threads) per block — smem is warp-private, enabling plain +=.
    const int SM = prop->multiProcessorCount;
    static constexpr int kTargetBlocksPerSM  = 32;
    static constexpr int kMinWorkloadPerBlock = 4;   // sample-blocks per block minimum

    int strides = ceil_div(SM * kTargetBlocksPerSM, nf);
    strides = min(strides, max(1, ceil_div(M, kMinWorkloadPerBlock)));
    if (strides < 1) strides = 1;

    const int stride_per_block = max(1, ceil_div(M, strides));

    const dim3 block(LANES, 1, 1);          // 1 warp per block
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
