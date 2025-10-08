#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


__device__ __forceinline__
uint32_t xpose_stage(uint32_t v, int idx, int lane, int s, unsigned mask){
    // swap when (lane_bit ^ idx_bit) == 1 at stage s
    uint32_t other = __shfl_xor_sync(mask, v, 1 << s);
    return (((lane >> s) ^ (idx >> s)) & 1) ? other : v;
}

// Shapes:
//   X    : [bF, M]          uint32
//   XS   : [nfeatsets, 32*M]uint32
//   Fsch : [rounds, 32*nfeatsets] uint16
// Mapping (Murky parity):
//   XS[f0, 32*(base + k) + wi] = X[ Fsch[round, 32*f0 + wi], base + k ]
// Coalesced-read strategy + SMEM 32x32 rotate to coalesced-write.

__global__ void _et_sample_1b_sm(
    const uint32_t* __restrict__ X,
    uint32_t* __restrict__ XS,
    const uint16_t* __restrict__ Fsch,
    int bF, int M, int nfeatsets, int round,
    int stride)
{
    const int f0 = blockIdx.x;   // feature-set index
    const int bi = blockIdx.y;   // tile group along M
    const int wi = threadIdx.x;  // lane 0..31
    if (f0 >= nfeatsets || wi >= 32) return;

    const size_t rowstride = (size_t)32 * (size_t)M;  // XS row stride per feature-set

    // Per-lane selected row index for this feature-set
    __shared__ uint32_t fs[32];
    fs[wi] = (uint32_t)Fsch[(size_t)round * (size_t)(32 * nfeatsets)
                           + (size_t)32 * (size_t)f0 + (size_t)wi];
    __syncwarp();

    // Padded to avoid bank conflicts
    __shared__ uint32_t tile[32][33];  // tile[row_k][lane]

    for (int i = 0; i < stride; ++i) {
        const int base = 32 * (stride * bi + i);  // first column in this tile
        if (base >= M) continue;
        const int col_lane = base + wi;           // column this lane loads each k

        // 1) Coalesced loads by row: fill SMEM tile[row_k][lane]
        for (int k = 0; k < 32; ++k) {
            uint32_t v = 0u;
            const uint32_t row_k = fs[k];
            if (col_lane < M && row_k < (uint32_t)bF) {
                v = X[(size_t)row_k * (size_t)M + (size_t)col_lane];
            }
            tile[k][wi] = v;
        }

        // 2) Coalesced writes: transpose read from tile[lane][k]
        for (int k = 0; k < 32; ++k) {
            const int col_out = base + k;
            if (col_out >= M) break;
            const size_t i_out = (size_t)32 * (size_t)col_out + (size_t)wi;
            XS[(size_t)f0 * rowstride + i_out] = tile[wi][k];
        }
    }
}


__global__ void _et_sample_1b_shfl(
    const uint32_t* __restrict__ X,    // [bF, M]
    uint32_t* __restrict__ XS,         // [nfeatsets, 32*M]
    const uint16_t* __restrict__ Fsch, // [rounds, 32*nfeatsets]
    int bF, int M, int nfeatsets, int round,
    int stride)
{
    const int f0 = blockIdx.x;                 // feature-set
    const int bi = blockIdx.y;                 // tile id along columns
    const int wi = threadIdx.x;                // lane 0..31
    if (f0 >= nfeatsets || wi >= 32) return;

    const size_t rowstride = (size_t)32 * (size_t)M;
    // per-lane selected row index for this feature-set
    const uint32_t fr_lane = (uint32_t)Fsch[
        (size_t)round * (size_t)(32 * nfeatsets) + (size_t)32 * (size_t)f0 + (size_t)wi
    ];

    for (int i = 0; i < stride; ++i) {
        const int base = 32 * (stride * bi + i);
        if (base >= M) continue;

        const int col_in = base + wi;
        const int K = min(32, M - base);

        // One consistent mask for all shuffles in this tile
        const unsigned mask = 0xFFFFFFFFu;

        // 1) Coalesced reads by row: build T[k] = X[ Fs[f0,k], base + wi ]
        uint32_t T[32];
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            uint32_t v = 0u;
            if (k < K) {
                const uint32_t row_k = __shfl_sync(mask, fr_lane, k);
                if (row_k < (uint32_t)bF) {
                    v = (col_in < M) ? X[(size_t)row_k * (size_t)M + (size_t)col_in] : 0u;
                }
            }
            T[k] = v;
        }

        // 2) Register 32x32 transpose (5 stages). Snapshot per stage.
        #pragma unroll
        for (int s = 0; s < 5; ++s) {
            uint32_t U[32];
            #pragma unroll
            for (int idx = 0; idx < 32; ++idx) U[idx] = T[idx];
            #pragma unroll
            for (int idx = 0; idx < 32; ++idx) {
                T[idx] = xpose_stage(U[idx], idx, wi, s, mask);
            }
        }

        // 3) Coalesced writes: XS[f0, 32*(base+k) + wi] = T[k]
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            if (k >= K) break;
            const size_t i_out = (size_t)32 * (size_t)(base + k) + (size_t)wi;
            XS[(size_t)f0 * rowstride + i_out] = T[k];
        }
    }
}

// Host launcher — signature unchanged

torch::Tensor et_sample_1b(torch::Tensor X,
                            torch::Tensor XS,
                            torch::Tensor Fsch,
                            int round) {
    const int bF = (int)X.size(0);
    const int M = (int)X.size(1);
    const int nfeatsets = (int)XS.size(0);

    const int strides = 256;
    int stride = (int)((M + 32 * strides - 1) / (32 * strides));
    if (stride < 1) stride = 1;

    dim3 grid(nfeatsets, strides, 1);
    dim3 block(32, 1, 1);

    auto stream = at::cuda::getCurrentCUDAStream();

    _et_sample_1b_shfl<<<grid, block, 0, stream.stream()>>>(
        X.data_ptr<uint32_t>(),
        XS.data_ptr<uint32_t>(),
        Fsch.data_ptr<uint16_t>(),
        bF, M, nfeatsets, round,
        stride);

    return XS;
}
