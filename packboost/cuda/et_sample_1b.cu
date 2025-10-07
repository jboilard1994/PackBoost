#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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

	if (f0 >= nfeatsets || blockDim.x != 32) return;

	__shared__ uint16_t fs[32];

	fs[wi] = Fsch[size_t(round)*size_t(32*nfeatsets) + size_t(32*f0 + wi)];
	__syncwarp();

	for (int i = 0; i < stride; ++i) {
		const int base_in = 32*(stride*bi + i); // first word-column in this tile
		if (base_in >= M) continue;
		const int i_in = base_in + wi; // this lane's word-column
		
		uint32_t row[32];
		for (int r = 0; r < 32; ++r) {
			uint32_t v = 0u;
			const uint32_t fr = uint32_t(fs[r]);
			if (i_in < M && fr < uint32_t(bF)) {
				v = X[size_t(fr)*size_t(M) + size_t(i_in)];
			}
			row[r] = v;
		}
	
		// We'll emit: XS[f0, 32*(base_in + k) + wi] = A[wi][(k+wi)&31]
		const unsigned mask = __activemask();
		const size_t rowstride = size_t(32)*size_t(M);

		const uint32_t base_element = row[wi];

		for (int k = 0; k < 32; ++k) {
			const int kk = (k + wi) & 31; // rotated column index
			const size_t i_out = size_t(32)*size_t(base_in + k) + size_t(wi);
			if (i_out < rowstride) {
				// pull A[wi][kk] from src lane = kk
				uint32_t emit = __shfl_sync(mask, base_element, kk);
				XS[size_t(f0)*rowstride + i_out] = emit;
			}
		}
	}
}

void et_sample_1b(torch::Tensor X, // [bF, M] uint32
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
			F, M, nfeatsets, round,
			stride
			);
}
