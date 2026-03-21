#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Forward declaration
torch::Tensor h0_sm_butterfly(torch::Tensor G, torch::Tensor L_old, int max_depth);

// Legacy entry point — delegates to the butterfly variant.
torch::Tensor h0_sm(
    torch::Tensor G,        // [N], int16, CUDA
    torch::Tensor L_old,    // [nfolds, Dm, N], uint8 branch bits, CUDA
    int max_depth
){
    return h0_sm_butterfly(G, L_old, max_depth);
}


__device__ __forceinline__ int SH_idx(int node, int ch, int lane) {
    // shared layout: [nodes, 2, 32] packed as ((node*2 + ch)*32 + lane)
    return (node * 2 + ch) * 32 + lane;
}

static __device__ __forceinline__ uint64_t pack_sc(int sum32, int cnt32) {
    return ( (uint64_t)(uint32_t)cnt32 << 32 ) | (uint64_t)(uint32_t)sum32;
}

static __device__ __forceinline__ uint64_t add_pack(uint64_t a, uint64_t b) {
    int sa = (int)(uint32_t)a;           // low 32 = sum
    int ca = (int)(uint32_t)(a >> 32);   // high 32 = count
    int sb = (int)(uint32_t)b;
    int cb = (int)(uint32_t)(b >> 32);
    return pack_sc(sa + sb, ca + cb);
}

// d_thresh: depth levels [0, d_thresh) use lane-private shared mem + butterfly,
//           levels [d_thresh, max_depth) fall back to global atomics.
__global__ void _h0_sm_butterfly(
    const int16_t* __restrict__ G,      // [N]
    const uint8_t* __restrict__ L_old,  // [nfolds, Dm, N] branch bits
    int64_t*       __restrict__ H0,     // [nfolds, 2^D, 2] int64
    const int N,
    const int nfolds,
    const int Dm,
    const int max_depth,
    const int stride_per_warp,
    const int d_thresh
){
    const int tree_set = blockIdx.x;   // 0..nfolds-1
    const int bi       = blockIdx.y;   // tile id along samples
    const int lane     = threadIdx.x;  // 0..31

    if (tree_set >= nfolds || blockDim.x != 32 || lane >= 32) return;

    const int nodes      = 1 << max_depth;
    const int smem_nodes = (1 << d_thresh) - 1;

    extern __shared__ int s_hist[]; // [1<<d_thresh, 2, 32] as int32

    const unsigned mask = __ballot_sync(__activemask(), true);
    // --- zero shared histogram (shallow levels only) ---
    for (int n = 0; n < (1 << d_thresh); ++n) {
        s_hist[SH_idx(n, 0, lane)] = 0;
        s_hist[SH_idx(n, 1, lane)] = 0;
    }
    __syncwarp();

    // Global base pointer for this fold
    long long* g_base = reinterpret_cast<long long*>(H0)
                      + (long long)tree_set * nodes * 2;

    // --- accumulate: shallow → lane-private shared, deep → global atomics ---
    for (int j = 0; j < stride_per_warp; ++j) {
        const int jj = 32 * (stride_per_warp * bi + j) + lane;
        if (jj >= N) break;

        const int g = (int)G[jj];

        int node_prefix = 0;
        for (int d = 0; d < max_depth; ++d) {
            const int to = (1 << d) - 1;
            const int node = to + node_prefix;

            if (d < d_thresh) {
                s_hist[SH_idx(node, 0, lane)] += g;
                s_hist[SH_idx(node, 1, lane)] += 1;
            } else {
                atomicAdd(
                    reinterpret_cast<unsigned long long*>(g_base + node * 2 + 0),
                    (unsigned long long)(long long)g);
                atomicAdd(
                    reinterpret_cast<unsigned long long*>(g_base + node * 2 + 1),
                    (unsigned long long)1LL);
            }

            if (d < Dm) {
                const size_t off = (((size_t)tree_set * (size_t)Dm) + (size_t)d) * (size_t)N + (size_t)jj;
                const int branch_bit = (int)L_old[off];
                node_prefix = (node_prefix << 1) | branch_bit;
            }
        }
    }

    // --- butterfly reduce-scatter for shallow nodes only ---
    for (int k0 = 0; k0 < smem_nodes; k0 += 32) {
        uint64_t P[32];
        int idx = (k0 * 2 + 0) * 32 + lane;
        for (int i = 0; i < 32; ++i, idx += 64) {
            const int node = k0 + i;
            int s = 0, c = 0;
            if (node < smem_nodes) {
                s = s_hist[idx];
                c = s_hist[idx + 32];
            }
            P[i] = pack_sc(s, c);
        }

        #pragma unroll
        for (int s = 0; s < 5; ++s) {
            const int ofs = 1 << s;
            for (int i = 0; i < 32; ++i) {
                const uint64_t partner = __shfl_xor_sync(mask, P[i], ofs, 32);
                P[i] = add_pack(P[i], partner);
            }
        }

        const int node_out = k0 + lane;
        if (node_out < smem_nodes) {
            const uint64_t pack = P[lane];
            const long long sum_ll = (long long)(int)(uint32_t)pack;
            const long long cnt_ll = (long long)(int)(uint32_t)(pack >> 32);

            if (sum_ll) {
                auto* p_sum = reinterpret_cast<unsigned long long*>(g_base + node_out * 2 + 0);
                atomicAdd(p_sum, (unsigned long long)sum_ll);
            }
            if (cnt_ll) {
                auto* p_cnt = reinterpret_cast<unsigned long long*>(g_base + node_out * 2 + 1);
                atomicAdd(p_cnt, (unsigned long long)cnt_ll);
            }
        }
    }

}



torch::Tensor h0_sm_butterfly(
    torch::Tensor G,        // [N], int16, CUDA
    torch::Tensor L_old,    // [nfolds, Dm, N], uint8 branch bits, CUDA
    int max_depth
){
    TORCH_CHECK(G.is_cuda() && L_old.is_cuda(), "G and L_old must be CUDA tensors.");
    TORCH_CHECK(G.dim() == 1 && L_old.dim() == 3, "G:[N], L_old:[nfolds,Dm,N].");
    TORCH_CHECK(G.scalar_type() == c10::ScalarType::Short, "G must be int16.");
    TORCH_CHECK(L_old.scalar_type() == c10::ScalarType::Byte, "L_old must be uint8.");
    TORCH_CHECK(max_depth > 0, "max_depth must be > 0.");

    const int64_t nfolds64 = L_old.size(0);
    const int64_t Dm64     = L_old.size(1);
    const int64_t N64      = L_old.size(2);
    TORCH_CHECK(G.size(0) == N64, "L_old.size(2) must equal G.size(0).");
    TORCH_CHECK(Dm64 == max_depth - 1, "L_old.size(1) must equal max_depth - 1.");
    TORCH_CHECK(nfolds64 <= std::numeric_limits<int>::max());
    TORCH_CHECK(N64      <= std::numeric_limits<int>::max());
    const int nfolds = (int)nfolds64;
    const int Dm     = (int)Dm64;
    const int N      = (int)N64;

    // Murky tiling
    static constexpr int lanes   = 32;
    int SM = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    int target_blocks_per_SM = 32;
    int min_total_blocks = SM * target_blocks_per_SM;
    int strides = (min_total_blocks + nfolds - 1) / nfolds;
    static constexpr int min_workload_per_thread = 128;
    int64_t max_strides = (N64 + (lanes * min_workload_per_thread) - 1) / (lanes * min_workload_per_thread);

    strides = std::min((int64_t)strides, max_strides);
    if (strides < 32) strides = 32;

    int stride = (N + (lanes * strides) - 1) / (lanes * strides);
    if (stride < 1) stride = 1;

    const dim3 block(lanes, 1, 1);
    const dim3 grid (nfolds, strides, 1);

    const int nodes = 1 << max_depth;
    auto H0 = torch::zeros({nfolds64, (int64_t)nodes, 2},
                           L_old.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous));

    // Depth threshold: fit as many levels as possible in shared memory
    auto* prop = at::cuda::getCurrentDeviceProperties();
    size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                   : (size_t)prop->sharedMemPerBlock;
    int d_thresh = max_depth;
    while (d_thresh > 0 && (size_t)(1 << d_thresh) * 2 * 32 * sizeof(int) > smem_cap)
        d_thresh--;

    const size_t smem_bytes = (size_t)(1 << d_thresh) * 2 * 32 * sizeof(int);

    auto stream = at::cuda::getCurrentCUDAStream();

    cudaFuncSetAttribute(_h0_sm_butterfly, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    _h0_sm_butterfly<<<grid, block, smem_bytes, stream.stream()>>>(
        G.data_ptr<int16_t>(), L_old.data_ptr<uint8_t>(), H0.data_ptr<int64_t>(),
        N, nfolds, Dm, max_depth, stride, d_thresh);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "H0 butterfly launch failed: ",
                cudaGetErrorString(cudaGetLastError()));
    return H0;
}