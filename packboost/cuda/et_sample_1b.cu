#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            uint32_t v = 0u;
            const uint32_t row_k = fs[k];
            if (col_lane < M && row_k < (uint32_t)bF) {
                v = X[(size_t)row_k * (size_t)M + (size_t)col_lane];
            }
            tile[k][wi] = v;
        }
        __syncwarp();

        // 2) Coalesced writes: transpose read from tile[lane][k]
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            const int col_out = base + k;
            if (col_out >= M) break;
            const size_t i_out = (size_t)32 * (size_t)col_out + (size_t)wi;
            XS[(size_t)f0 * rowstride + i_out] = tile[wi][k];
        }
        __syncwarp();
    }
}


__device__ __forceinline__ void warp_transpose32(uint32_t A[32], int lane, unsigned mask) {
    #pragma unroll
    for (int s = 0; s < 5; ++s) {              // s = 0..4  (ofs = 1,2,4,8,16)
        const int ofs = 1 << s;
        for (int i = 0; i < 32; ++i) {
            const uint32_t partner = __shfl_xor_sync(mask, A[i ^ ofs], ofs, 32);
            A[i] = (((lane ^ i) & ofs) ? partner : A[i]);
        }
    }
}

__global__ void _et_sample_1b_butterfly(
    const uint32_t* __restrict__ X,        // [bF, M]
    uint32_t* __restrict__ XS,             // [nfeatsets, 32*M]
    const uint16_t* __restrict__ Fsch,     // [rounds, 32*nfeatsets]
    int bF, int M, int nfeatsets, int round,
    int stride)
{
    const int f0   = blockIdx.x;      // feature-set index (0..nfeatsets-1)
    const int bi   = blockIdx.y;      // tile-group index along second dim (tiles)
    const int lane = threadIdx.x;     // warp lane 0..31

    if (f0 >= nfeatsets || blockDim.x != 32 || lane >= 32) return;

    // Stage the 32 feature rows for this feature-set
    const unsigned mask = __activemask();               // all 32 lanes participate

    __shared__ uint16_t fs[32];
    fs[lane] = Fsch[(size_t)round * (size_t)(32 * nfeatsets) + (size_t)(32 * f0 + lane)];
    __syncwarp();

    const size_t rowstride = (size_t)32 * (size_t)M; // XS stride per feature-set row

    // Iterate over tiles in *tile units*
    for (int i = 0; i < stride; ++i) {
        const int t = stride * bi + i;               // tile index (0..M-1)
        if (32*t >= M) break;                           // warp-uniform tail guard in tiles

        // 1) Build a 32×1 column in registers from packed column t of X
        //    (coalescing is not the goal here; correctness first)
        uint32_t T[32];
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            uint32_t v = 0u;
            const uint32_t row_k = (uint32_t)fs[k];
            if (row_k < (uint32_t)bF) {
                v = X[(size_t)row_k * (size_t)M + (size_t)t];
            }
            T[k] = v;
        }

        // 2) Transpose the 32×32 register tile across the warp
        warp_transpose32(T, lane, mask);

        // 3) Store out 32 expanded columns for tile t (coalesced across lanes)
        const size_t base_out = (size_t)32 * (size_t)t;
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            XS[(size_t)f0 * rowstride + (base_out + (size_t)(32 * k + lane))] = T[k];
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

    _et_sample_1b_butterfly<<<grid, block, 0, stream.stream()>>>(
        X.data_ptr<uint32_t>(),
        XS.data_ptr<uint32_t>(),
        Fsch.data_ptr<uint16_t>(),
        bF, M, nfeatsets, round,
        stride);

    return XS;
}
