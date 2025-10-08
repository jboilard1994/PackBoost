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

__global__ void _et_sample_1b_gather(
    const uint32_t* __restrict__ X,        // [bF, M]
    uint32_t* __restrict__ XS,             // [nfeatsets, 32*M]
    const uint16_t* __restrict__ Fsch,     // [rounds, 32*nfeatsets]
    int bF, int M, int nfeatsets, int round,
    int stride)
{
    const int f0   = blockIdx.x;     // feature-set index
    const int bi   = blockIdx.y;     // tile group along columns
    const int lane = threadIdx.x;    // warp lane 0..31

    if (f0 >= nfeatsets || blockDim.x != 32 || lane >= 32) return;

    // Stage 32 feature indices for this feature-set into shared (readable by all lanes)
    const unsigned mask = __ballot_sync(0xFFFFFFFFu, true);            // all lanes participate in shuffles
    __syncwarp();
    __shared__ uint16_t fs[32];
    fs[lane] = Fsch[(size_t)round * (size_t)(32 * nfeatsets) + (size_t)(32 * f0 + lane)];
    __syncwarp();

    const size_t rowstride = (size_t)32 * (size_t)M; // XS stride per feature-set row

    // Iterate over 'stride' tiles of width 32 columns each
    for (int i = 0; i < stride; ++i) {
        const int base_in = 32 * (stride * bi + i);   // first column of this tile
        if (base_in >= M) continue;                   // tile entirely OOB

        const int  col_in = base_in + lane;           // this lane's column within the tile
        const int  K      = min(32, M - base_in);     // valid columns in this tile
        

        // 1) Single scattered load per lane (guarded)
        uint32_t v_row = 0u;
        const uint32_t fr_lane = (uint32_t)fs[lane];
        if (col_in < M && fr_lane < (uint32_t)bF) {
            v_row = X[(size_t)fr_lane * (size_t)M + (size_t)col_in];
        }

        // 2) Emit K columns by pulling from source lane k (coalesced stores)
        const size_t base_out = (size_t)32 * (size_t)base_in;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            const uint32_t emit = __shfl_sync(mask, v_row, k, 32);
            XS[(size_t)f0 * rowstride + (base_out + (size_t)(32 * k + lane))] = emit;
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

    _et_sample_1b_gather<<<grid, block, 0, stream.stream()>>>(
        X.data_ptr<uint32_t>(),
        XS.data_ptr<uint32_t>(),
        Fsch.data_ptr<uint16_t>(),
        bF, M, nfeatsets, round,
        stride);

    return XS;
}
