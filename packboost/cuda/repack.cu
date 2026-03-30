#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using at::Tensor;

// FST:   [nsets, nfeatsets, max_depth], uint8
// L_old: [nfolds, Dm, N],                uint8 branch bits
// LF:    [nfeatsets, Dm, N],             uint8 branch bits
// Logic: LF[fs, d, :] = L_old[FST[tree_set, fs, d+1], d, :]

__device__ __forceinline__ int fst_idx(int k, int d, int lane, int Dm) {
    // Index into shared FST: (k * Dm + d) * 32 + lane
    return (((k * Dm) + d) << 5) + lane; // <<5 == *32
}

__global__ void repack_trees_for_features_kernel(
    const uint8_t* __restrict__ FST,    // [nsets, nfeatsets, max_depth]
    const uint8_t* __restrict__ L_old,  // [nfolds, Dm, N]
    uint8_t* __restrict__ LF,           // [nfeatsets, Dm, N]
    int nsets, int nfeatsets, int max_depth, int nfolds, int Dm, int N,
    int tree_set, int stride
){
    const int fi = blockIdx.x;  // feature-set block
    const int ti = blockIdx.y;  // N block
    const int wi = threadIdx.x; // lane
    if (blockDim.x != 32) return;

    extern __shared__ unsigned char smem[];
    uint32_t* fst = reinterpret_cast<uint32_t*>(smem);
    const size_t fst_elems = static_cast<size_t>(8) * Dm * 32;
    uint8_t* branch_buf = reinterpret_cast<uint8_t*>(fst + fst_elems);

    // Load FST tile: fst[k, d, wi] = FST[tree_set, 8*fi + k, d+1]
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int fs = 8 * fi + k;
        if (fs < nfeatsets) {
            for (int d = 0; d < Dm; ++d) {
                const uint8_t v = FST[((tree_set * nfeatsets) + fs) * max_depth + (d + 1)];
                fst[fst_idx(k, d, wi, Dm)] = static_cast<uint32_t>(v);
            }
        }
    }
    __syncthreads();

    for (int i = 0; i < stride; ++i) {
        const int j = 32 * stride * ti + 32 * i + wi;
        if (j >= N) continue;

        // Process each depth dimension
        for (int d = 0; d < Dm; ++d) {
            // Load branch bits from all folds for this depth and sample
            // branch_buf[k] = L_old[k, d, j]
            for (int k = 0; k < nfolds; ++k) {
                branch_buf[k] = L_old[((k * Dm) + d) * N + j];
            }

            // Gather for each feature-set in this block
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int fs = 8 * fi + k;
                if (fs < nfeatsets) {
                    const uint32_t which_fold = fst[fst_idx(k, d, wi, Dm)];
                    const uint8_t branch_bit = branch_buf[which_fold];
                    LF[((fs * Dm) + d) * N + j] = branch_bit;
                }
            }
        }
    }
}


void repack_trees_for_features_cuda(
    const torch::Tensor& FST,      // uint8 [nsets, nfeatsets, max_depth]
    const torch::Tensor& L_old,    // uint8 [nfolds, Dm, N]
    torch::Tensor& LF,             // uint8 [nfeatsets, Dm, N] (output)
    int64_t tree_set               // which set (round)
) {
    const int nsets      = static_cast<int>(FST.size(0));
    const int nfeatsets  = static_cast<int>(FST.size(1));
    const int max_depth  = static_cast<int>(FST.size(2));
    const int nfolds     = static_cast<int>(L_old.size(0));
    const int Dm         = static_cast<int>(L_old.size(1));
    const int N          = static_cast<int>(L_old.size(2));
    const int grid_y     = 1024;

    int inner_stride = (N + (grid_y * 32) - 1) / (grid_y * 32);
    if (inner_stride < 1) inner_stride = 1;

    const dim3 block(32, 1, 1);
    const dim3 grid( (nfeatsets + 7) / 8, grid_y, 1 );

    // Shared memory bytes:
    // fst:        8 * Dm * 32 * sizeof(uint32_t)
    // branch_buf: nfolds       * sizeof(uint8_t)
    size_t fst_bytes = static_cast<size_t>(8) * Dm * 32 * sizeof(uint32_t);
    size_t branch_bytes = static_cast<size_t>(nfolds) * sizeof(uint8_t);
    size_t smem = fst_bytes + branch_bytes;

    auto stream = at::cuda::getCurrentCUDAStream();

    repack_trees_for_features_kernel
        <<<grid, block, smem, stream>>>(
            FST.data_ptr<uint8_t>(),
            L_old.data_ptr<uint8_t>(),
            LF.data_ptr<uint8_t>(),
            nsets, nfeatsets, max_depth, nfolds, Dm, N,
            static_cast<int>(tree_set),
            inner_stride
        );
}
