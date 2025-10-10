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
        atomicAdd((unsigned long long int)(H0 + (size_t)tree_set*(used_nodes+1)*2 + (size_t)k*2 + 0), (long long)SH(k, 0, wi));
        atomicAdd((unsigned long long int)(H0 + (size_t)tree_set*(used_nodes+1)*2 + (size_t)k*2 + 1), (long long)SH(k, 1, wi));
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