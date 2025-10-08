#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Shapes (internal):
//   dX  : [F, N] int8     (we transpose at the API boundary)
//   dXB : [4*F, M] uint32, where M = ceil_div(N, 32)
// Spec per feature f and word w (32 samples per word):
//   bit k in XB[4*f + t, w] is 1 iff  (uint8)dX[f, 32*w + k] > t,  for t in {0,1,2,3}.

__global__ void _encode_cuts_ballot(const int8_t* __restrict__ X, // [F,N]
                                    uint32_t* __restrict__ XB,    // [4*F,M]
                                    int F, int N, int tiles_per_block)
{
    const int f    = blockIdx.x;            // feature index
    const int tb   = blockIdx.y;            // tile block along words
    const int lane = threadIdx.x;           // 0..31
    if (f >= F || lane >= 32) return;

    const int M = (N + 31) >> 5;            // words (ceil_div)
    const unsigned FULL = 0xFFFFFFFFu;      // all 32 lanes active

    const int tile_base = tb * tiles_per_block; // starting word for this block.y

    #pragma unroll
    for (int t = 0; t < tiles_per_block; ++t) {
        const int w = tile_base + t;        // word index
        if (w >= M) break;

        const int i = (w << 5) + lane;      // sample index = 32*w + lane

        // Load sample as uint8 for logical comparisons
        uint32_t v = 0u;
        if (i < N) {
            v = (uint32_t)(uint8_t)X[(size_t)f * (size_t)N + (size_t)i];
        }

        // Four bit-planes via warp ballots
        uint32_t m0 = __ballot_sync(FULL, v > 0u);
        uint32_t m1 = __ballot_sync(FULL, v > 1u);
        uint32_t m2 = __ballot_sync(FULL, v > 2u);
        uint32_t m3 = __ballot_sync(FULL, v > 3u);

        // Store once per word; any single lane can do it (use lane 0)
        if (lane == 0) {
            const size_t base = (size_t)4 * (size_t)f * (size_t)M + (size_t)w;
            XB[base + 0*(size_t)M] = m0;
            XB[base + 1*(size_t)M] = m1;
            XB[base + 2*(size_t)M] = m2;
            XB[base + 3*(size_t)M] = m3;
        }
    }
}

// Host launcher: transpose X to [F,N], compute tiling along words, and launch.

torch::Tensor encode_cuts(torch::Tensor X /* [N,F] int8 */) {
    // Internal layout: [F,N]
    auto dX = X.transpose(0, 1).contiguous();
    const int F = (int)dX.size(0);
    const int N = (int)dX.size(1);
    const int M = (N + 31) >> 5;  // words

    auto dXB = torch::empty({(long long)4*F, (long long)M}, dX.options().dtype(torch::kUInt32));

    // Tile policy along words (no hard-coded 64/256)
    int tiles  = (M + 31) / 32;                    // 32 words per block.y step
    int by     = tiles > 0 ? min(tiles, 256) : 1;  // grid.y
    int stride = tiles > 0 ? (tiles + by - 1) / by : 1; // tiles_per_block

    dim3 block(32, 1, 1);
    dim3 grid(F, by, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    _encode_cuts_ballot<<<grid, block, 0, stream.stream()>>>(
        dX.data_ptr<int8_t>(), dXB.data_ptr<uint32_t>(), F, N, stride);

    return dXB;
}
