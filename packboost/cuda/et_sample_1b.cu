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
    // A temporary register array for ping-ponging data between stages
    uint32_t B[32];

    #pragma unroll
    for (int s = 4; s >= 0; --s) { // Iterate through bit positions 4 down to 0
        const int ofs = 1 << s;    // Partner lanes differ by ofs (16, 8, 4, 2, 1)

        // Perform the butterfly exchange for all 32 elements in the register file
        #pragma unroll
        for (int idx = 0; idx < 32; ++idx) {
            // Get the value from the partner lane. This is a symmetric exchange.
            // Note: The original kernel's use of A[idx ^ ofs] was incorrect.
            const uint32_t partner_val = __shfl_xor_sync(mask, A[idx], ofs, 32);

            // The condition for swapping bits 's' of the row and column index.
            // If the s-th bit of the lane (original column) and the s-th bit
            // of the index (original row) are different, we take the partner's value.
            // The original kernel's condition ((lane >> s) & 1) was incomplete,
            // causing a one-way data transfer instead of a swap.
            if ((((unsigned)lane >> s) & 1) != ((idx >> s) & 1)) {
                B[idx] = partner_val;
            } else {
                B[idx] = A[idx];
            }
        }

        // Ping-pong: update A for the next stage
        #pragma unroll
        for (int idx = 0; idx < 32; ++idx) {
            A[idx] = B[idx];
        }
    }
}

extern "C" __global__ void _et_sample_1b_butterfly(
    const uint32_t* __restrict__ X,        // [bF, M]
    uint32_t* __restrict__ XS,             // [nfeatsets, 32*M]
    const uint16_t* __restrict__ Fsch,     // [rounds, 32*nfeatsets]
    int bF, int M, int nfeatsets, int round,
    int stride)
{
    const int f0   = blockIdx.x;      // feature-set index (0..nfeatsets-1)
    const int bi   = blockIdx.y;      // tile group along columns
    const int lane = threadIdx.x;     // warp lane 0..31

    if (f0 >= nfeatsets || blockDim.x != 32 || lane >= 32) return;

    // Stage 32 feature-row indices for this feature-set into shared
    __shared__ uint16_t fs[32];
    fs[lane] = Fsch[(size_t)round * (size_t)(32 * nfeatsets) + (size_t)(32 * f0 + lane)];
    __syncwarp();

    const size_t rowstride = (size_t)32 * (size_t)M; // XS stride per feature-set
    const unsigned mask = 0xFFFFFFFFu;               // all lanes participate in shuffles

    // Iterate over 'stride' tiles of width 32 columns each
    for (int i = 0; i < stride; ++i) {
        const int base = 32 * (stride * bi + i);      // first column in this tile
        if (base >= M) break;                         // warp-uniform tail guard
        const int  K      = (M - base >= 32) ? 32 : (M - base); // valid cols in tile
        const int  col_in = base + lane;
        const bool col_ok = (col_in < M);

        // 1) Build register tile (column-major per lane), coalesced loads per k
        uint32_t T[32];
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            uint32_t v = 0u;
            if (col_ok && k < K) {
                const uint32_t row_k = (uint32_t)fs[k];
                if (row_k < (uint32_t)bF) {
                    v = X[(size_t)row_k * (size_t)M + (size_t)col_in];
                }
            }
            T[k] = v;
        }

        // 2) Transpose 32×32 in registers via butterfly
        warp_transpose32(T, lane, mask);

        // 3) Coalesced store of K columns (row == lane after transpose)
        const size_t base_out = (size_t)32 * (size_t)base;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
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
