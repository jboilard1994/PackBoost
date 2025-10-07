#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <torch/extension.h>

template <typename T>
__device__ __forceinline__ T butterfly_stage_xor(T val, int idx, int s) {
	unsigned mask = __activemask();
	T other = __shfl_xor_sync(mask, val, 1 << s);
	return ((idx >> s) & 1) ? other : val;
}

__global__ void _encode_cuts(const int8_t* __restrict__ X,
                             uint32_t* __restrict__ XB,
                             int F, int N, int stride)
{
    const int f  = blockIdx.x;
    const int bi = blockIdx.y;
    const int wi = threadIdx.x;   // 0..31
    const int M = (N + 31) >> 5;
    if (f >= F || blockDim.x != 32) return;

    for (int i = 0; i < stride; ++i) {
        const int tile_idx = stride * bi + i;   // which 32x32 tile
        const int base_col = 32*32 * tile_idx;   // 32*32
        const int i_out    = 32 * tile_idx + wi;

        uint32_t row[32];
        
	for (int k = 0; k < 32; ++k) {
            const int col = base_col + wi + 32 * k;
            uint32_t v = 0u;
            if (col < N) {
                size_t idx = size_t(f) * size_t(N) + size_t(col);
                v = uint32_t(uint8_t(X[idx]));   // unsigned semantics
            }
            row[k] = v;
        }

	#pragma unroll
	for (int s = 0; s < 5; ++s) {
            for (int k = 0; k < 32; ++k) {
                row[k] = butterfly_stage_xor<uint32_t>(row[k], k, s);
            }
        }
	
	uint32_t v[4] = {0u, 0u, 0u, 0u};

	for (int k = 0; k < 32; ++k) {
	  const uint32_t r = row[k], bit = 1u << k;
	  #pragma unroll 4
	  for (int t = 0; t < 4; ++t) {
	    const uint32_t pred = (r > (uint32_t)t);   // 0 or 1
	    v[t] |= (0u - pred) & bit;                 // 0xFFFFFFFF*pred & bit
	  }
	}
	
	if (i_out < M) {
	  #pragma unroll
	  for (int k = 0; k < 4; ++k) {
	  	XB[(size_t(4*f + k)*M) + i_out] = v[k];
	  }
	}
    }
}



torch::Tensor encode_cuts(torch::Tensor X){

	const int strides = 64;
	auto dX = X.transpose(0, 1).contiguous();
	const int F = (int)dX.size(0), N = (int)dX.size(1); //nfeats, nrows
	const int M = (N + 31) >> 5; // nwords
	auto dXB = torch::empty({4LL*F, (long long)M}, dX.options().dtype(torch::kUInt32));

	int stride = (int)((M + (32*strides) - 1) / (32*strides));
	if (stride < 1) stride = 1;

	dim3 block(32, 1, 1); // 32 threads_per_block: x-axis
	dim3 grid(F, strides, 1);

	auto s = at::cuda::getCurrentCUDAStream();
	_encode_cuts<<<grid, block, 0, s.stream()>>>(
			dX.data_ptr<int8_t>(),
			dXB.data_ptr<uint32_t>(),
			(int) F, (int) N, (int) stride
			);

	return dXB;
}
