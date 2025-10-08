#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Murky-accurate encode_cuts with shared-memory rotate, strides fixed at 64.
// Shapes:
//   API X: [N, F] int8
//   Internal dX: [F, N] int8 (after transpose)
//   Output dXB: [4*F, M] uint32, where M = ceil_div(N, 32)
// Mapping:
//   For feature f and word w (covering samples i=32*w..32*w+31),
//   bit k in XB[4*f + t, w] = 1  iff  (uint8)X[f, 32*w + k] > t,  t∈{0,1,2,3}.

__global__ void _encode_cuts(
    const int8_t* __restrict__ X,  // [F, N]
    uint32_t* __restrict__ XB,     // [4*F, M]
    int F, int N, int stride)      // stride = ceil(N / (32*32*strides))
{
    const int f  = blockIdx.x;       // feature index
    const int bi = blockIdx.y;       // tile block along N (words grouped by 32)
    const int wi = threadIdx.x;      // lane 0..31

    if (f >= F || wi >= 32 || blockDim.x != 32) return;

    // Shared memory tile and rotate, as in Murky
    __shared__ uint32_t sm[32][32];  // tile[row, col]

    // Each block.y iterates over 'stride' tiles
    for (int i = 0; i < stride; ++i) {
        // Input/sample side indexing
        // i_in is the base column within the 32x32 tile for this lane
        const int i_in  = 32*32*stride*bi + 32*32*i + wi;   // sample-column base for this lane
        const int i_out =    32*stride*bi +    32*i + wi;   // output word index for this lane

        // Accumulators for 4 bit-planes
        uint32_t v0 = 0u, v1 = 0u, v2 = 0u, v3 = 0u;

        // 1) Load a 32x32 tile into shared with rotated column index (k+wi)%32
        //    Source: X[f, 32*k + i_in] if in range, else 0
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            const int col = 32*k + i_in;
            uint32_t v = 0u;
            if (col < N) {
                const size_t idx = (size_t)f * (size_t)N + (size_t)col;
                v = (uint32_t)(uint8_t)X[idx]; // unsigned semantics
            }
            sm[wi][(k + wi) & 31] = v;
        }
        __syncwarp();

        // 2) Read back rotated to realize the 32x32 transpose logic
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            const uint32_t v = sm[k][(k + wi) & 31];
            v0 |= ((v > 0u) ? 1u : 0u) << k;
            v1 |= ((v > 1u) ? 1u : 0u) << k;
            v2 |= ((v > 2u) ? 1u : 0u) << k;
            v3 |= ((v > 3u) ? 1u : 0u) << k;
        }

        // 3) Store the four bit-planes for this feature/word (guard tail)
        if (i_out < ( (N + 31) >> 5 )) { // M = ceil_div(N,32)
            const size_t M = (size_t)((N + 31) >> 5);
            const size_t base = (size_t)4 * (size_t)f * M + (size_t)i_out;
            XB[base + 0*M] = v0;
            XB[base + 1*M] = v1;
            XB[base + 2*M] = v2;
            XB[base + 3*M] = v3;
        }
        __syncwarp();
    }
}

// Host API: transpose X to [F,N], set strides=64, compute stride exactly like Murky, and launch.

torch::Tensor encode_cuts(torch::Tensor X /* [N,F] int8 */) {
    // Internal layout matches Murky: work on [F,N]
    auto dX = X.transpose(0, 1).contiguous();
    const int F = (int)dX.size(0);
    const int N = (int)dX.size(1);
    const int M = (N + 31) >> 5;  // words

    auto dXB = torch::empty({ (long long)4 * F, (long long)M },
                             dX.options().dtype(torch::kUInt32));

    // Keep strides fixed at 64 (as requested)
    const int strides = 64;                   // grid.y
    // Murky's stride: ceil( N / (32^2 * strides) )
    int stride = (N + (32*32*strides) - 1) / (32*32*strides);
    if (stride < 1) stride = 1;

    dim3 grid(F, strides, 1);
    dim3 block(32, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    _encode_cuts<<<grid, block, 0, stream.stream()>>>(
        dX.data_ptr<int8_t>(), dXB.data_ptr<uint32_t>(), F, N, stride);

    return dXB;
}
