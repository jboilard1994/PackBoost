#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using at::Tensor;

// FST: [nsets, nfeatsets, max_depth],  uint8
// LE : [nfolds, N],                    packed (uint16/uint32/uint64)
// LF : [nfeatsets, N],                 packed (same as LE)
// Parity with Murky's Numba kernel:
//   - gridDim.x = ceil(nfeatsets/8)
//   - gridDim.y = strides
//   - blockDim.x = 32 (warp)
//   - kernel param 'stride' = ceil(N / (strides*32))
//   - loop i in [0,stride)
//   - j = 32*stride*ti + 32*i + wi
//   - shared 'fst' tile: (8, max_depth-1, 32) of uint32
//   - shared 'trees' : (nfolds, 32) of packed type
//   - v |= trees[ fst[k, d-1, wi], wi ] & mask(d)

__device__ __forceinline__ int fst_idx(int k, int d1, int lane, int mdm1) {
    // mdm1 = (max_depth - 1)
    // equivalent to ((k * mdm1 + d1) * 32 + lane)
    return (((k * mdm1) + d1) << 5) + lane; // <<5 == *32
}

template <typename PackedT>
__global__ void repack_trees_for_features_kernel(
    const uint8_t* __restrict__ FST,  // [nsets, nfeatsets, max_depth]
    const PackedT* __restrict__ LE,   // [nfolds, N]
    PackedT* __restrict__ LF,         // [nfeatsets, N]
    int nsets, int nfeatsets, int max_depth, int nfolds, int N,
    int tree_set, int stride
){
    const int fi = blockIdx.x;
    const int ti = blockIdx.y;
    const int wi = threadIdx.x; // lane
    if (blockDim.x != 32) return;

    extern __shared__ unsigned char smem[];
    uint32_t* fst = reinterpret_cast<uint32_t*>(smem);
    const size_t fst_elems = static_cast<size_t>(8) * (max_depth - 1) * 32;
    PackedT* trees = reinterpret_cast<PackedT*>(fst + fst_elems);

    const int mdm1 = max_depth - 1;   // precompute (max_depth - 1)

    // Load FST tile: fst[k, d-1, wi] = FST[tree_set, 8*fi + k, d]
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int fs = 8 * fi + k;
        if (fs < nfeatsets) {
            for (int d = 1; d < max_depth; ++d) {
                const uint8_t v = FST[((tree_set * nfeatsets) + fs) * max_depth + d];
                fst[fst_idx(k, d - 1, wi, mdm1)] = static_cast<uint32_t>(v);
            }
        }
    }
    __syncthreads();

    for (int i = 0; i < stride; ++i) {
        const int j = 32 * stride * ti + 32 * i + wi;
        if (j >= N) continue;

        // trees[k, wi] = LE[k, j]
        for (int k = 0; k < nfolds; ++k) {
            trees[k * 32 + wi] = LE[k * N + j];
        }

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int fs = 8 * fi + k;
            if (fs < nfeatsets) {
                PackedT v = PackedT(0);
                for (int d = 1; d < max_depth; ++d) {
                    const uint32_t which = fst[fst_idx(k, d - 1, wi, mdm1)];
                    const PackedT leafbits = trees[which * 32 + wi];
                    const PackedT mask =
                        (((PackedT(1) << d) - PackedT(1))
                         << ((d * (d - 1)) / 2));
                    v |= (leafbits & mask);
                }
                LF[fs * N + j] = v;
            }
        }
    }
}


void repack_trees_for_features_cuda(
    const torch::Tensor& FST,     // uint8  [nsets, nfeatsets, max_depth]
    const torch::Tensor& LE,      // u16/u32/u64 [nfolds, N]
    torch::Tensor& LF,            // u16/u32/u64 [nfeatsets, N] (output, in-place fill)
    int64_t tree_set      // which set (round)
) {
    

    const int nsets      = static_cast<int>(FST.size(0));
    const int nfeatsets  = static_cast<int>(FST.size(1));
    const int max_depth  = static_cast<int>(FST.size(2));
    const int nfolds     = static_cast<int>(LE.size(0));
    const int N          = static_cast<int>(LE.size(1));
    const int grid_y     = 1024;

    // Murky parity: inner 'stride' iters over N per (ti), warped by 32
    int inner_stride = (N + (grid_y * 32) - 1) / (grid_y * 32);
    if (inner_stride < 1) inner_stride = 1;

    const dim3 block(32, 1, 1);
    const dim3 grid( (nfeatsets + 7) / 8, grid_y, 1 );

    // Shared memory bytes:
    // fst:   8*(max_depth-1)*32 * sizeof(uint32_t)
    // trees: nfolds*32          * sizeof(PackedT)
    size_t fst_bytes = static_cast<size_t>(8) * (max_depth - 1) * 32 * sizeof(uint32_t);

    auto stream = at::cuda::getCurrentCUDAStream();

    switch (LE.scalar_type()) {
        case torch::kUInt16: { // uint16
            size_t trees_bytes = static_cast<size_t>(nfolds) * 32 * sizeof(uint16_t);
            size_t smem = fst_bytes + trees_bytes;
            repack_trees_for_features_kernel<uint16_t>
                <<<grid, block, smem, stream>>>(
                    FST.data_ptr<uint8_t>(),
                    LE.data_ptr<uint16_t>(),
                    LF.data_ptr<uint16_t>(),
                    nsets, nfeatsets, max_depth, nfolds, N,
                    static_cast<int>(tree_set),
                    inner_stride
                );
            break;
        }
        case at::kInt: // NOTE: if you use signed int32, change your tensor to kUInt32
        case at::kLong: // signed long not supported for bit packing here
            break;
        case torch::kUInt32: { // uint32
            size_t trees_bytes = static_cast<size_t>(nfolds) * 32 * sizeof(uint32_t);
            size_t smem = fst_bytes + trees_bytes;
            repack_trees_for_features_kernel<uint32_t>
                <<<grid, block, smem, stream>>>(
                    FST.data_ptr<uint8_t>(),
                    reinterpret_cast<const uint32_t*>(LE.data_ptr()),
                    reinterpret_cast<uint32_t*>(LF.data_ptr()),
                    nsets, nfeatsets, max_depth, nfolds, N,
                    static_cast<int>(tree_set),
                    inner_stride
                );
            break;
        }
        case torch::kUInt64: { // uint64
            size_t trees_bytes = static_cast<size_t>(nfolds) * 32 * sizeof(uint64_t);
            size_t smem = fst_bytes + trees_bytes;
            repack_trees_for_features_kernel<uint64_t>
                <<<grid, block, smem, stream>>>(
                    FST.data_ptr<uint8_t>(),
                    LE.data_ptr<uint64_t>(),
                    LF.data_ptr<uint64_t>(),
                    nsets, nfeatsets, max_depth, nfolds, N,
                    static_cast<int>(tree_set),
                    inner_stride
                );
            
            break;
        }
        default:
            TORCH_CHECK(false, "Unsupported packed dtype for LE/LF. Use uint16/uint32/uint64.");
    }
}
