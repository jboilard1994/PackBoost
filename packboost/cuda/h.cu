#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int LANES = 32;
constexpr int MAX_WARPS_PER_BLOCK = 4;
constexpr int TILE_NODES = 32;

__device__ __forceinline__ int sm_idx(int w, int t, int l) {
    return (w * TILE_NODES + t) * LANES + l;
}

// XS: [nf, M, 32] uint32 (bit-packed)
// Y : [N] int16
// LF: [nf, Dm, N] uint16
// H : [nf, nodes_tot, 2, 32] int64
//
// Shared memory layout (per block):
//   sm_gain:  [warps_per_block * TILE_NODES * LANES] int64
//   sm_count: [warps_per_block * TILE_NODES * LANES] int32
//
// grid.x = nf
// grid.y = ceil_div(nodes_tot, TILE_NODES)
// block  = (32, warps_per_block)
__global__ void _h_sm(
    const uint32_t* __restrict__ XS,
    const int16_t* __restrict__ Y,
    const uint16_t* __restrict__ LF,
    int64_t* __restrict__ H,
    int nf,
    int M,
    int N,
    int Dm,
    int max_depth,
    int nodes_tot,
    int warps_per_block
) {
    const int lane = threadIdx.x;  // 0..31
    const int warp = threadIdx.y;
    const unsigned full_mask = 0xFFFFFFFFu;
    const int f = blockIdx.x;
    const int node_base = blockIdx.y * TILE_NODES;

    if (f >= nf || node_base >= nodes_tot) {
        return;
    }

    // Shared memory: gain (int64) then count (int32)
    extern __shared__ char raw_smem[];
    const int tile_elems = warps_per_block * TILE_NODES * LANES;
    int64_t* sm_gain = reinterpret_cast<int64_t*>(raw_smem);
    int32_t* sm_count = reinterpret_cast<int32_t*>(raw_smem + tile_elems * sizeof(int64_t));

    for (int t = 0; t < TILE_NODES; ++t) {
        sm_gain[sm_idx(warp, t, lane)] = 0;
        sm_count[sm_idx(warp, t, lane)] = 0;
    }
    __syncthreads();

    // Precompute per-depth node offsets (only depend on d, not n)
    int64_t depth_offset[16];
    for (int d = 0; d < max_depth && d < 16; ++d) {
        depth_offset[d] = (1LL << d) - 1LL;
    }

    // Base pointer for this feature set's LF data
    const uint16_t* LF_f = LF + static_cast<size_t>(f) * Dm * N;

    // Each warp processes a subset of samples.
    // All lanes stay active to keep shuffles well-defined.
    for (int n = warp; n < N; n += warps_per_block) {
        const int m = n >> 5;  // n / 32
        const int b = n & 31;  // n % 32

        // XS[f, m, lane]
        const uint32_t word = XS[((f * M + m) * LANES) + lane];
        const int v = static_cast<int>((word >> b) & 1u);

        // Ballot: skip this sample entirely if no lane has v==1
        const unsigned vmask = __ballot_sync(full_mask, v);
        if (vmask == 0u) {
            continue;
        }

        // Lane 0 loads Y[n], broadcast to all lanes via shuffle
        const int16_t y_raw = (lane == 0) ? Y[n] : 0;
        const int64_t y = static_cast<int64_t>(__shfl_sync(full_mask, y_raw, 0));

        // Weighted values (0 when v==0, so inactive lanes contribute nothing)
        const int64_t vy = static_cast<int64_t>(v) * y;

        // All depths in one loop (depth 0: node = 0, no LF load needed)
        for (int d = 0; d < max_depth; ++d) {
            int64_t node;
            if (d == 0) {
                node = 0;
            } else {
                // Lane 0 loads LF prefix, broadcast to all lanes
                const uint16_t tk_raw = (lane == 0)
                    ? LF_f[static_cast<size_t>(d - 1) * N + n]
                    : 0;
                node = static_cast<int64_t>(
                    __shfl_sync(full_mask, tk_raw, 0)) + depth_offset[d];
            }

            if (node >= node_base && node < node_base + TILE_NODES) {
                const int t = static_cast<int>(node - node_base);
                sm_gain[sm_idx(warp, t, lane)] += vy;
                sm_count[sm_idx(warp, t, lane)] += v;
            }
        }
    }
    __syncthreads();

    // Reduce warps and flush to global H
    for (int t = warp; t < TILE_NODES; t += warps_per_block) {
        const int node = node_base + t;
        if (node < nodes_tot) {
            int64_t gain_sum = 0;
            int32_t count_sum = 0;

            #pragma unroll
            for (int w = 0; w < MAX_WARPS_PER_BLOCK; ++w) {
                if (w >= warps_per_block) {
                    break;
                }
                gain_sum += sm_gain[sm_idx(w, t, lane)];
                count_sum += sm_count[sm_idx(w, t, lane)];
            }

            const int64_t base =
                ((static_cast<int64_t>(f) * nodes_tot + node) * 2) * LANES + lane;

            H[base + 0 * LANES] += gain_sum;
            H[base + 1 * LANES] += static_cast<int64_t>(count_sum);
        }
    }
}

static inline size_t calc_shmem(int wpb) {
    const size_t tile_elems = static_cast<size_t>(wpb) * TILE_NODES * LANES;
    return tile_elems * sizeof(int64_t) + tile_elems * sizeof(int32_t);
}

static inline int choose_warps_per_block(size_t smem_cap) {
    const int candidates[3] = {4, 2, 1};
    int best_wpb = 1;
    int best_score = -1;

    for (int c = 0; c < 3; ++c) {
        const int wpb = candidates[c];
        const size_t shmem_bytes = calc_shmem(wpb);
        if (shmem_bytes > smem_cap) {
            continue;
        }

        int active_blocks = 0;
        const cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            _h_sm,
            wpb * LANES,
            static_cast<int>(shmem_bytes));
        if (occ_err != cudaSuccess) {
            continue;
        }

        const int score = active_blocks * wpb;
        if (score > best_score) {
            best_score = score;
            best_wpb = wpb;
        }
    }
    return best_wpb;
}

}  // namespace

torch::Tensor h_sm(
    torch::Tensor XS,   // [nf, 32*M], int32/int64/uint32
    torch::Tensor Y,    // [N], int16/int32/int64
    torch::Tensor LF,   // [nf, Dm, N], uint16/int32/int64
    int max_depth
) {
    TORCH_CHECK(XS.is_cuda() && Y.is_cuda() && LF.is_cuda(), "all inputs must be CUDA");
    TORCH_CHECK(XS.dim() == 2, "XS must be [nf, 32*M]");
    TORCH_CHECK(Y.dim() == 1, "Y must be [N]");
    TORCH_CHECK(LF.dim() == 3, "LF must be [nf, Dm, N]");
    TORCH_CHECK(max_depth > 0 && max_depth <= 16, "h_sm supports max_depth in [1, 16]");

    TORCH_CHECK(
        XS.scalar_type() == torch::kUInt32 || XS.scalar_type() == torch::kInt32,
        "XS must be uint32 or int32");
    TORCH_CHECK(Y.scalar_type() == torch::kInt16, "Y must be int16");
    TORCH_CHECK(LF.scalar_type() == torch::kUInt16, "LF must be uint16");

    auto XSc = XS.contiguous();
    auto Yc = Y.contiguous();
    auto LFc = LF.contiguous();

    const int nf = static_cast<int>(XSc.size(0));
    const int cols_32M = static_cast<int>(XSc.size(1));
    TORCH_CHECK(cols_32M % 32 == 0, "XS columns must be divisible by 32");

    const int M = cols_32M / 32;
    const int N = static_cast<int>(Yc.size(0));
    const int Dm = static_cast<int>(LFc.size(1));
    const int nodes_tot = (1 << max_depth) - 1;

    TORCH_CHECK(M * 32 >= N, "XS does not contain enough packed bits for N");
    TORCH_CHECK(LFc.size(0) == nf, "LF.size(0) must equal XS.size(0)");
    TORCH_CHECK(LFc.size(1) == max_depth - 1, "LF.size(1) must equal max_depth - 1");
    TORCH_CHECK(LFc.size(2) == N, "LF.size(2) must equal Y.size(0)");

    auto H = torch::zeros(
        {nf, nodes_tot, 2, LANES},
        torch::TensorOptions().device(XS.device()).dtype(torch::kInt64));

    auto* prop = at::cuda::getCurrentDeviceProperties();
    const size_t smem_cap = prop->sharedMemPerBlockOptin
        ? static_cast<size_t>(prop->sharedMemPerBlockOptin)
        : static_cast<size_t>(prop->sharedMemPerBlock);

    const int warps_per_block = choose_warps_per_block(smem_cap);

    const size_t shmem_bytes = calc_shmem(warps_per_block);

    const dim3 block(LANES, static_cast<unsigned int>(warps_per_block));
    const dim3 grid(nf, (nodes_tot + TILE_NODES - 1) / TILE_NODES);

    TORCH_CHECK(
        shmem_bytes <= smem_cap,
        "Required dynamic shared memory (", shmem_bytes,
        ") exceeds device limit (", smem_cap, ")");

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(
        _h_sm,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes));

    _h_sm<<<grid, block, shmem_bytes, stream.stream()>>>(
        reinterpret_cast<const uint32_t*>(XSc.data_ptr()),
        Yc.data_ptr<int16_t>(),
        LFc.data_ptr<uint16_t>(),
        H.data_ptr<int64_t>(),
        nf,
        M,
        N,
        Dm,
        max_depth,
        nodes_tot,
        warps_per_block);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return H;
}
