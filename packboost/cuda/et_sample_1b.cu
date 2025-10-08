#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ uint32_t butterfly_stage_lane(uint32_t v, int idx, int lane, int s){
	unsigned mask = __activemask();
	uint32_t other = __shfl_xor_sync(mask, v, 1 << s);
	return (((lane >> s) ^ (idx >> s)) & 1) ? other : v;
}

__global__ void _et_sample_1b(
		const uint32_t* __restrict__ X,
		uint32_t* __restrict__ XS,
		const uint16_t* __restrict__ Fsch,
		int bF, int M, int nfeatsets, int round,
		int stride
		)
{
	const int f0 = blockIdx.x; // feat-set index
	const int bi = blockIdx.y; // tile group along M
	const int wi = threadIdx.x; // lane 0..31

	if (f0 >= nfeatsets || wi >= 32) return;

	const size_t rowstride = (size_t)32 * (size_t)M;

	const uint32_t fr_lane = (uint32_t)Fsch[
        (size_t)round * (size_t)(32 * nfeatsets) + (size_t)32 * (size_t)f0 + (size_t)wi
    ];

	for (int i = 0; i < stride; ++i) {
		const int base_in = 32*(stride*bi + i); // first word-column in this tile
		if (base_in >= M) continue;
		const int i_in = base_in + wi; // this lane's word-column
		const int K = min(32, M - base_in);
		const unsigned mask = __activemask();

		uint32_t T[32]; // T[k] = X[Fs[f0, k], base + wi]
		for (int k = 0; k < 32; ++k) {
			uint32_t v = 0u;
			if (k < K){
				const uint32_t rk = __shfl_sync(mask, fr_lane, k);
				if (i_in < M && rk < (uint32_t)bF) {
					v = X[(size_t)rk*(size_t)M+(size_t)i_in];
				}
			}
			T[k] = v;
		}

		

		#pragma unroll
		for (int s = 0; s < 5; ++s) {
			uint32_t U[32];
			for (int k = 0; k < 32; ++k) U[k] = T[k];
			for (int k = 0; k < 32; ++k) {
				T[k] = butterfly_stage_lane(U[k], k, wi, s);
			}
		}
			
		for (int k = 0; k < 32; ++k) {
			const int col_out = base_in + k;
			if (col_out >= M) break;
			const size_t i_out = (size_t)32 * (size_t)col_out + (size_t)wi;
			XS[(size_t)f0 * rowstride + i_out] = T[k];
		}		
	}
}

torch::Tensor et_sample_1b(torch::Tensor X, // [bF, M] uint32
          torch::Tensor XS, // [nfeatsets, 32*M], uint32
          torch::Tensor Fsch, // [rounds, 32*nfeatsets], uint16
          int round
          ){
	const int bF = (int)X.size(0), M = (int)X.size(1), nfeatsets = (int)XS.size(0);
	const int strides = 256;
	int stride = (int)((M + 32*strides - 1)/(32*strides));
	if (stride < 1) stride = 1;

	dim3 grid(nfeatsets, strides, 1);
	dim3 block(32, 1, 1);

	auto stream = at::cuda::getCurrentCUDAStream();

	_et_sample_1b<<<grid, block, 0, stream.stream()>>>(
			X.data_ptr<uint32_t>(),
			XS.data_ptr<uint32_t>(),
			Fsch.data_ptr<uint16_t>(),
			bF, M, nfeatsets, round,
			stride
			);

	return XS;
}
