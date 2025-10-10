#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template <typename T>
__global__ void _h0_sm(
    const int16_t* G,
    const T* LE,
    int64_t* H0,
    const int N,
    const int nfolds,
    const int max_depth,
    const int stride
) {
    const int tree_set = blockIdx.x;
    const int ti = blockIdx.y;
    const int wi = threadIdx.x;

    if (tree_set >= nfolds || blockDim.x != 32) return;

    extern __shared__ int hist[];
    auto SH = [&](int node, int ch, int lane) -> int& {
        return hist[(node * 2 + ch) * 32 + lane];
    };

    for (int n = 0; n < 1<<max_depth; ++n) {
        SH(n, 0, wi) = 0;
        SH(n, 1, wi) = 0;
    }
    __syncthreads();

    for (int j = 0; j < stride; ++j) {
        const int jj = 32*(stride*ti +j) + wi;
        if (jj < N) {
            T lk = LE[(size_t)tree_set*N + jj];
            int16_t g = G[jj];

            for (int d = 0; d < max_depth; ++d) {
                const int to = (1<<d)-1;
                const int tk = (d == 0) ? 0 : (lk & to);
                if (d > 0) lk >>= d;
                SH(to + tk, 0, wi) += g;
                SH(to + tk, 1, wi) += 1;
            }
        }
    }

    const int used_nodes = (1<<max_depth) -1;
    for (int k = 0; k < used_nodes; ++k) {
        atomicAdd((unsigned long long int*)&H0[(size_t)tree_set*(used_nodes+1)*2 + (size_t)k*2 + 0], (unsigned long long int)SH(k, 0, wi));
        atomicAdd((unsigned long long int*)&H0[(size_t)tree_set*(used_nodes+1)*2 + (size_t)k*2 + 1], (unsigned long long int)SH(k, 1, wi));
    }

}


torch::Tensor h0_sm(
    torch::Tensor G,        // [N], int16, CUDA
    torch::Tensor LE,       // [nfolds, N], (u)int16/32/64, CUDA
    int max_depth
){

    const int64_t nfolds64 = LE.size(0);
    const int64_t N64      = LE.size(1);
    
    const int nfolds = static_cast<int>(nfolds64);
    const int N      = static_cast<int>(N64);

    // Murky config
    static constexpr int lanes   = 32;
    static constexpr int strides = 512;
    const dim3 block(lanes, 1, 1);
    const dim3 grid (nfolds, strides, 1);

    int stride = (N + lanes * strides - 1) / (lanes * strides);
    if (stride < 1) stride = 1;

    // Output: [nfolds, 2^D, 2] int64 (last row unused)
    const int nodes = 1 << max_depth;
    auto opts = LE.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous);
    auto H0   = torch::zeros({nfolds64, (int64_t)nodes, 2}, opts);

    // Dynamic SMEM requirement: [nodes, 2, 32] int
    const size_t smem_bytes = static_cast<size_t>(nodes) * 2 * 32 * sizeof(int);
    auto* prop = at::cuda::getCurrentDeviceProperties();
    size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                   : (size_t)prop->sharedMemPerBlock;

    auto stream = at::cuda::getCurrentCUDAStream();

    // Dispatch by LE dtype (unsigned first, then signed-as-unsigned)
    const auto dt = LE.scalar_type();
    if (dt == torch::kUInt16) {
        cudaFuncSetAttribute(_h0_sm<uint16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_sm<uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<int16_t>(),
            LE.data_ptr<uint16_t>(),
            H0.data_ptr<int64_t>(),
            N, nfolds, max_depth, stride
        );
    } else if (dt == torch::kUInt32) {
        cudaFuncSetAttribute(_h0_sm<uint32_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_sm<uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<int16_t>(),
            LE.data_ptr<uint32_t>(),
            H0.data_ptr<int64_t>(),
            N, nfolds, max_depth, stride
        );
    } else if (dt == torch::kUInt64) {
        cudaFuncSetAttribute(_h0_sm<uint64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_sm<uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<int16_t>(),
            LE.data_ptr<uint64_t>(),
            H0.data_ptr<int64_t>(),
            N, nfolds, max_depth, stride
        );
    } else {
        TORCH_CHECK(false, "LE must be one of: (u)int16, (u)int32, (u)int64.");
    }
    return H0;
}



__device__ __forceinline__ int SH_idx(int node, int ch, int lane) {
    // shared layout: [nodes, 2, 32] packed as ((node*2 + ch)*32 + lane)
    return (node * 2 + ch) * 32 + lane;
}

template <typename T>
__global__ void _h0_sm_butterfly(
    const int16_t* __restrict__ G,   // [N]
    const T*       __restrict__ LE,  // [nfolds, N], UNSIGNED in template
    int64_t*     __restrict__ H0,  // [nfolds, 2^D, 2] int64
    const int N,
    const int nfolds,
    const int max_depth,
    const int stride_per_warp
){
    const int tree_set = blockIdx.x;   // 0..nfolds-1
    const int bi       = blockIdx.y;   // tile id along samples
    const int lane     = threadIdx.x;  // 0..31

    if (tree_set >= nfolds || blockDim.x != 32 || lane >= 32) return;

    const int nodes      = 1 << max_depth;  // full Murky stride (last row unused)
    const int used_nodes = nodes - 1;

    extern __shared__ int s_hist[]; // [nodes, 2, 32] as int32

    // --- zero shared histogram ---
    for (int n = 0; n < nodes; ++n) {
        s_hist[SH_idx(n, 0, lane)] = 0;
        s_hist[SH_idx(n, 1, lane)] = 0;
    }
    __syncthreads();

    // --- accumulate lane-private columns in shared ---
    using U = typename std::make_unsigned<T>::type;
    for (int j = 0; j < stride_per_warp; ++j) {
        const int jj = 32 * (stride_per_warp * bi + j) + lane;
        if (jj >= N) break;

        U lk = static_cast<U>(LE[(size_t)tree_set * N + jj]);
        const int g = (int)G[jj];

        #pragma unroll
        for (int d = 0; d < 32; ++d) {
            if (d >= max_depth) break;
            const int to = (1 << d) - 1;
            const int tk = (d == 0) ? 0 : (int)(lk & (U)to);
            if (d > 0) lk >>= d;
            const int node = to + tk;  // 0..used_nodes-1

            s_hist[SH_idx(node, 0, lane)] += g;   // sum(G)
            s_hist[SH_idx(node, 1, lane)] += 1;   // count
        }
    }

    // --- butterfly transpose + reduce-scatter over lanes (32×32 tiles) ---
    const unsigned mask = __ballot_sync(__activemask(), true);
    long long* base = H0 + ((long long)tree_set * nodes * 2);

    for (int k0 = 0; k0 < used_nodes; k0 += 32) {
        const int node_out = k0 + lane;  // row this lane will own after transpose

        #pragma unroll
        for (int ch = 0; ch < 2; ++ch) {
            // Load this lane's column for the 32-row tile
            int A[32];
            for (int i = 0; i < 32; ++i) {
                const int node = k0 + i;
                A[i] = (node < used_nodes) ? s_hist[SH_idx(node, ch, lane)] : 0;
            }

            // Butterfly transpose (EXACT core you validated: U[i ^ ofs], gate on ((lane ^ i) & ofs))
            #pragma unroll
            for (int s = 0; s < 5; ++s) {
                const int ofs = 1 << s;
                int Ubuf[32];
                for (int i = 0; i < 32; ++i) Ubuf[i] = A[i];
                for (int i = 0; i < 32; ++i) {
                    const int partner = __shfl_xor_sync(mask, Ubuf[i ^ ofs], ofs, 32);
                    A[i] = (((lane ^ i) & ofs) ? partner : Ubuf[i]);
                }
            }

            // Now lane holds the full row (node_out) across the 32 lanes. Reduce locally.
            if (node_out < used_nodes) {
                long long acc = 0;
                for (int i = 0; i < 32; ++i) acc += (long long)A[i];

                // Single 64-bit atomic per node per block per channel
                auto* p = reinterpret_cast<unsigned long long*>(base + node_out * 2 + ch);
                if (acc) atomicAdd(p, (unsigned long long)acc);
            }
        }
    }
}



torch::Tensor h0_sm_butterfly(
    torch::Tensor G,        // [N], int16, CUDA
    torch::Tensor LE,       // [nfolds, N], (u)int16/32/64, CUDA
    int max_depth
){
    TORCH_CHECK(G.is_cuda() && LE.is_cuda(), "G and LE must be CUDA tensors.");
    TORCH_CHECK(G.dim() == 1 && LE.dim() == 2, "G:[N], LE:[nfolds,N].");
    TORCH_CHECK(G.scalar_type() == c10::ScalarType::Short, "G must be int16.");
    TORCH_CHECK(max_depth > 0, "max_depth must be > 0.");

    const int64_t nfolds64 = LE.size(0);
    const int64_t N64      = LE.size(1);
    TORCH_CHECK(G.size(0) == N64, "LE.size(1) must equal G.size(0).");
    TORCH_CHECK(nfolds64 <= std::numeric_limits<int>::max());
    TORCH_CHECK(N64      <= std::numeric_limits<int>::max());
    const int nfolds = (int)nfolds64;
    const int N      = (int)N64;

    // Murky tiling
    static constexpr int lanes   = 32;
    static constexpr int strides = 512;
    const dim3 block(lanes, 1, 1);
    const dim3 grid (nfolds, strides, 1);
    int stride = (N + lanes * strides - 1) / (lanes * strides);
    if (stride < 1) stride = 1;

    const int nodes = 1 << max_depth;
    auto H0 = torch::zeros({nfolds64, (int64_t)nodes, 2},
                           LE.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous));

    // Dynamic SMEM: [nodes, 2, 32] * sizeof(int)
    const size_t smem_bytes = (size_t)nodes * 2 * 32 * sizeof(int);
    auto* prop = at::cuda::getCurrentDeviceProperties();
    size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                   : (size_t)prop->sharedMemPerBlock;
    TORCH_CHECK(smem_bytes <= smem_cap,
        "H0 butterfly requires ", smem_bytes, "B shared memory, device allows ", smem_cap, "B.");

    auto stream = at::cuda::getCurrentCUDAStream();
    const auto dt = LE.scalar_type();

    if (dt == c10::ScalarType::UInt16) {
        cudaFuncSetAttribute(_h0_sm_butterfly<uint16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_sm_butterfly<uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<int16_t>(), LE.data_ptr<uint16_t>(), H0.data_ptr<int64_t>(),
            N, nfolds, max_depth, stride);
    } else if (dt == c10::ScalarType::UInt32) {
        cudaFuncSetAttribute(_h0_sm_butterfly<uint32_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_sm_butterfly<uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<int16_t>(), LE.data_ptr<uint32_t>(), H0.data_ptr<int64_t>(),
            N, nfolds, max_depth, stride);
    } else if (dt == c10::ScalarType::UInt64) {
        cudaFuncSetAttribute(_h0_sm_butterfly<uint64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h0_sm_butterfly<uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
            G.data_ptr<int16_t>(), LE.data_ptr<uint64_t>(), H0.data_ptr<int64_t>(),
            N, nfolds, max_depth, stride);
    }  else {
        TORCH_CHECK(false, "LE must be one of: (u)int16, (u)int32, (u)int64.");
    }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "H0 butterfly launch failed: ",
                cudaGetErrorString(cudaGetLastError()));
    return H0;
}