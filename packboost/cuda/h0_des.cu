#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

static __device__ __forceinline__ int SH_idx(int node, int ch, int lane) {
    // shared layout: [nodes, 2, 32] as ((node*2 + ch)*32 + lane)
    return (node * 2 + ch) * 32 + lane;
}

static __device__ __forceinline__ uint64_t pack_sc(int sum32, int cnt32) {
    return ((uint64_t)(uint32_t)cnt32 << 32) | (uint64_t)(uint32_t)sum32;
}

static __device__ __forceinline__ uint64_t add_pack(uint64_t a, uint64_t b) {
    int sa = (int)(uint32_t)a;
    int ca = (int)(uint32_t)(a >> 32);
    int sb = (int)(uint32_t)b;
    int cb = (int)(uint32_t)(b >> 32);
    return pack_sc(sa + sb, ca + cb);
}

template <typename T>
__global__ void _h0_des_butterfly(
    const int16_t* __restrict__ G,     // [N]
    const T*       __restrict__ LE,    // [nfolds, N]  (unsigned payload)
    const int32_t* __restrict__ era_ends, // [E] exclusive ends (sorted, era_ends[E-1] == N)
    int64_t*       __restrict__ H0,    // [nfolds, 2^D, E, 2] (int64)
    const int N,
    const int nfolds,
    const int max_depth,
    const int E,
    const int stride_per_warp
){
    const int fold = blockIdx.x;     // 0..nfolds-1
    const int bi   = blockIdx.y;     // tile id along samples
    const int lane = threadIdx.x;    // 0..31
    if (fold >= nfolds || blockDim.x != 32 || lane >= 32) return;

    const unsigned mask = __ballot_sync(__activemask(), true);

    const int nodes      = 1 << max_depth;   // includes the unused last row
    const int used_nodes = nodes - 1;        // 0..used_nodes-1

    // Shared histogram for *one era at a time* : [nodes, 2, 32] as int32
    extern __shared__ int s_hist[];
    // Base pointer for this fold in global
    long long* Hbase = reinterpret_cast<long long*>(H0) + ((long long)fold * nodes * E * 2);

    // Utility: reduce shared -> global for a given target era e_tgt
    auto flush_tile_to_global = [&](int e_tgt) {
        // each lane holds a column; do 32x32 butterfly in chunks of 32 nodes
        for (int k0 = 0; k0 < used_nodes; k0 += 32) {
            // load 32 rows (nodes) of this lane's column (both channels packed)
            uint64_t P[32];
            int idx = (k0 * 2 + 0) * 32 + lane;
            #pragma unroll
            for (int i = 0; i < 32; ++i, idx += 64) {
                const int node = k0 + i;
                int s = 0, c = 0;
                if (node < used_nodes) {
                    s = s_hist[idx + 0];
                    c = s_hist[idx + 32];
                }
                P[i] = pack_sc(s, c);
            }
            // butterfly reduce across lanes
            #pragma unroll
            for (int s = 0; s < 5; ++s) {
                const int ofs = 1 << s;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const uint64_t partner = __shfl_xor_sync(mask, P[i], ofs, 32);
                    P[i] = add_pack(P[i], partner);
                }
            }
            // lane ℓ writes node_out = k0 + ℓ
            const int node_out = k0 + lane;
            if (node_out < used_nodes) {
                const uint64_t pack = P[lane];
                const long long sum_ll = (long long)(int)(uint32_t)pack;
                const long long cnt_ll = (long long)(int)(uint32_t)(pack >> 32);

                // H0 layout: [fold, node, era, ch]
                long long* p_sum = Hbase + ((long long)node_out * E + e_tgt) * 2 + 0;
                long long* p_cnt = p_sum + 1;

                if (sum_ll) {
                    auto* u = reinterpret_cast<unsigned long long*>(p_sum);
                    atomicAdd(u, (unsigned long long)sum_ll);
                }
                if (cnt_ll) {
                    auto* u = reinterpret_cast<unsigned long long*>(p_cnt);
                    atomicAdd(u, (unsigned long long)cnt_ll);
                }
            }
        }
    };

    // Main loop over this warp's sample tiles
    for (int j = 0; j < stride_per_warp; ++j) {
        const int base = 32 * (stride_per_warp * bi + j);
        if (base >= N) break;
        const int col_lane = base + lane;

        // Preload lane's (G, LE) once for this tile
        int32_t g_lane = 0;
        uint32_t lk32  = 0;
        if (col_lane < N) {
            g_lane = (int32_t)G[col_lane];
            // LE indexed [nfolds, N]
            using U = typename std::make_unsigned<T>::type;
            U lk = (U)LE[(size_t)fold * (size_t)N + (size_t)col_lane];
            lk32 = (uint32_t)lk;  // max_depth ≤ 8–9 -> safe in 32 bits
        }

        // Determine era for each of the 32 columns in this tile: era_idx[k] for sample (base + k)
        __shared__ int era_idx[32];
        if (lane == 0) {
            int e = 0;
            // Find first era containing 'base'
            while (e < E && base >= era_ends[e]) ++e;
            for (int k = 0; k < 32; ++k) {
                const int pos = base + k;
                while (e < E && pos >= era_ends[e]) ++e;
                era_idx[k] = (e < E ? e : (E - 1));
            }
        }
        __syncwarp();

        // Work out which (up to 2) eras are present in this 32-wide tile
        int e0 = era_idx[0];
        int e1 = -1;
        #pragma unroll
        for (int k = 1; k < 32; ++k) {
            const int ek = era_idx[k];
            if (ek != e0) { e1 = ek; break; }
        }
        const int npresent = (e1 >= 0 ? 2 : 1);

        // We process one era at a time into shared, then flush to global.
        // This avoids doubling shared memory footprint.
        #pragma unroll 2
        for (int pass = 0; pass < npresent; ++pass) {
            const int e_tgt = (pass == 0 ? e0 : e1);

            // Zero shared histogram for this pass
            #pragma unroll
            for (int n = 0; n < nodes; ++n) {
                s_hist[SH_idx(n, 0, lane)] = 0;
                s_hist[SH_idx(n, 1, lane)] = 0;
            }
            __syncwarp();

            // Load this lane's 32-bit word of xfd (bitmask) and mask tail
            uint32_t xfd_local = 0u;
            if (base + lane < N) {
                // No XS input here; H0 depends only on (LE, G) — routing via LE
                // We still need a valid-mask for the tail tile to bound jj < N.
                xfd_local = 0xFFFFFFFFu;
            }
            const int rem = N - base;
            uint32_t valid_mask = (rem >= 32) ? 0xFFFFFFFFu : (rem > 0 ? ((1u << rem) - 1u) : 0u);
            xfd_local &= valid_mask;

            // Consume the 32 columns via shuffles; only accumulate for era == e_tgt
            for (int k = 0; k < 32; ++k) {
                const int pos_valid = (int)(xfd_local & 1u);
                xfd_local >>= 1;

                // Pull (g, lk) for column k from lane k
                const int32_t gk = __shfl_sync(mask, g_lane, k);
                uint32_t      lk = __shfl_sync(mask, lk32,  k);

                // Skip if this column's sample is out of range / not the target era
                const int match = (pos_valid && (era_idx[k] == e_tgt));
                if (!match) continue;

                // Walk depths exactly like the non-DES H0:
                // node = (2^d - 1) + (d==0 ? 0 : (lk & ((1<<d)-1))); then if (d>0) lk >>= d
                #pragma unroll
                for (int d = 0; d < 32; ++d) {
                    if (d >= max_depth) break;
                    const int to = (1 << d) - 1;
                    const int tk = (d == 0) ? 0 : (int)(lk & (uint32_t)to);
                    if (d > 0) lk >>= d;
                    const int node = to + tk; // 0..used_nodes-1
                    // sum(G), count
                    atomicAdd(&s_hist[SH_idx(node, 0, lane)], gk);
                    atomicAdd(&s_hist[SH_idx(node, 1, lane)], 1);
                }
            }

            // Reduce & flush this era tile to global
            flush_tile_to_global(e_tgt);
            __syncwarp();
        }
    }
}

torch::Tensor h0_des_butterfly(
    torch::Tensor G,          // [N], int16, CUDA
    torch::Tensor LE,         // [nfolds, N], (u)int16/32/64, CUDA
    torch::Tensor era_ends,   // [E], int32, CUDA or CPU (will be moved)
    int max_depth
){
    TORCH_CHECK(G.is_cuda() && LE.is_cuda(), "G and LE must be CUDA tensors.");
    TORCH_CHECK(G.dim() == 1 && LE.dim() == 2, "G:[N], LE:[nfolds,N].");
    TORCH_CHECK(G.scalar_type() == c10::ScalarType::Short, "G must be int16.");
    TORCH_CHECK(max_depth > 0, "max_depth must be > 0.");

    // We keep the ≤8 shared-mem variant (nodes<=256). Extend with node-tiling if you need D=9.
    const int nodes = 1 << max_depth;
    TORCH_CHECK(nodes <= 256, "h0_des_butterfly supports max_depth ≤ 8 (nodes ≤ 256). Got max_depth=", max_depth);

    const int64_t nfolds64 = LE.size(0);
    const int64_t N64      = LE.size(1);
    TORCH_CHECK(G.size(0) == N64, "LE.size(1) must equal G.size(0).");
    TORCH_CHECK(nfolds64 <= std::numeric_limits<int>::max());
    TORCH_CHECK(N64      <= std::numeric_limits<int>::max());
    const int nfolds = (int)nfolds64;
    const int N      = (int)N64;

    // Era ends → device int32
    TORCH_CHECK(era_ends.dim() == 1, "era_ends must be 1-D [E] of exclusive ends.");
    auto era_ends_i32 = era_ends.to(LE.device(), /*dtype=*/torch::kInt32, /*non_blocking=*/false, /*copy=*/true).contiguous();
    const int E = (int)era_ends_i32.size(0);
    TORCH_CHECK(E > 0, "era_ends must have at least one era.");
    TORCH_CHECK(era_ends_i32.dtype() == torch::kInt32, "era_ends must be int32.");
    TORCH_CHECK( (int)era_ends_i32.data_ptr<int32_t>()[E-1] == N,
        "era_ends[-1] must equal N (exclusive).");

    // Murky-like tiling: plenty of work in flight
    static constexpr int lanes = 32;
    int SM = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int target_blocks_per_SM = 32;
    int min_total_blocks = SM * target_blocks_per_SM;
    int strides = (min_total_blocks + nfolds - 1) / nfolds;
    static constexpr int min_workload_per_thread = 128;
    int64_t max_strides = (N64 + (lanes * (int64_t)min_workload_per_thread) - 1) / (lanes * (int64_t)min_workload_per_thread);

    strides = (int)std::min((int64_t)strides, max_strides);
    if (strides < 32) strides = 32;

    int stride = (N + (lanes * strides) - 1) / (lanes * strides);
    if (stride < 1) stride = 1;

    const dim3 block(lanes, 1, 1);
    const dim3 grid (nfolds, strides, 1);

    // Output: [nfolds, 2^D, E, 2] int64 (last row unused)
    auto H0 = torch::zeros({nfolds64, (int64_t)nodes, (int64_t)E, (int64_t)2},
                           LE.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous));

    // Dynamic SMEM: one era plane: [nodes, 2, 32] * sizeof(int)
    const size_t smem_bytes = (size_t)nodes * 2 * 32 * sizeof(int);

    auto* prop = at::cuda::getCurrentDeviceProperties();
    size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                   : (size_t)prop->sharedMemPerBlock;
    TORCH_CHECK(smem_bytes <= smem_cap,
        "H0-DES requires ", smem_bytes, "B shared memory, device allows ", smem_cap, "B. "
        "(max_depth too large for this variant)");

    auto stream = at::cuda::getCurrentCUDAStream();
    const auto dt = LE.scalar_type();

    if (dt == c10::ScalarType::UInt16) {
        cudaFuncSetAttribute(_h0_des_butterfly<uint16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_des_butterfly<uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<int16_t>(),
            LE.data_ptr<uint16_t>(),
            era_ends_i32.data_ptr<int32_t>(),
            H0.data_ptr<int64_t>(),
            N, nfolds, max_depth, E, stride);
    } else if (dt == c10::ScalarType::UInt32) {
        cudaFuncSetAttribute(_h0_des_butterfly<uint32_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_des_butterfly<uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<int16_t>(),
            LE.data_ptr<uint32_t>(),
            era_ends_i32.data_ptr<int32_t>(),
            H0.data_ptr<int64_t>(),
            N, nfolds, max_depth, E, stride);
    } else if (dt == c10::ScalarType::UInt64) {
        cudaFuncSetAttribute(_h0_des_butterfly<uint64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_des_butterfly<uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<int16_t>(),
            LE.data_ptr<uint64_t>(),
            era_ends_i32.data_ptr<int32_t>(),
            H0.data_ptr<int64_t>(),
            N, nfolds, max_depth, E, stride);
    } else {
        TORCH_CHECK(false, "LE must be one of: (u)int16, (u)int32, (u)int64.");
    }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "H0-DES launch failed: ",
                cudaGetErrorString(cudaGetLastError()));
    return H0;
}
