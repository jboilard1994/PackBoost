#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void _encode_cuts(
    const int8_t* __restrict__ X,
    uint32_t* __restrict__ XB,
    int F, int N, int M,
    int stride_n,
    int stride_f)
{
    const int f_id   = blockIdx.x;
    const int word_id = blockIdx.y;
    const int bit_id  = threadIdx.x;   // 0..31

    const int sample = word_id * 32 + bit_id;

    bool b0 = false, b1 = false, b2 = false, b3 = false;

    if (sample < N && f_id < F) {
        uint32_t v = (uint32_t)(uint8_t)
            X[(size_t)sample * (size_t)stride_n + (size_t)f_id * (size_t)stride_f];

        b0 = (v > 0);
        b1 = (v > 1);
        b2 = (v > 2);
        b3 = (v > 3);
    }

    uint32_t m0 = __ballot_sync(0xffffffff, b0);
    uint32_t m1 = __ballot_sync(0xffffffff, b1);
    uint32_t m2 = __ballot_sync(0xffffffff, b2);
    uint32_t m3 = __ballot_sync(0xffffffff, b3);

    if (bit_id == 0) {
        size_t base = (size_t)4 * (size_t)f_id * (size_t)M + (size_t)word_id;
        XB[base + 0 * (size_t)M] = m0;
        XB[base + 1 * (size_t)M] = m1;
        XB[base + 2 * (size_t)M] = m2;
        XB[base + 3 * (size_t)M] = m3;
    }
}

torch::Tensor encode_cuts(torch::Tensor X) {
    const int N = (int)X.size(0);
    const int F = (int)X.size(1);
    const int M = (N + 31) >> 5;

    const int stride_n = (int)X.stride(0);
    const int stride_f = (int)X.stride(1);

    auto dXB = torch::empty(
        { (long long)4 * F, (long long)M },
        X.options().dtype(torch::kUInt32)
    );

    dim3 grid(F, M, 1);
    dim3 block(32, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    _encode_cuts<<<grid, block, 0, stream.stream()>>>(
        X.data_ptr<int8_t>(),
        dXB.data_ptr<uint32_t>(),
        F, N, M,
        stride_n, stride_f
    );

    return dXB;
}