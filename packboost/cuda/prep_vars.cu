#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <tuple>

__device__ __forceinline__ int16_t sat_i16_by_mbits(int x){
    const int lo = -((1<<15)); // grad_mbits = 15
    const int hi =  ((1<<15) - 1);
    x = (x < lo) ? lo : x;
    x = (x > hi) ? hi : x;
    return static_cast<int16_t>(x);
  }
  
template <typename LeafT, typename PackedT>
__global__ void _prep_vars(
  const LeafT* __restrict__ L,     // [K, Dm, N], N-contiguous
  PackedT* __restrict__ LE,        // [K, N]
      const int32_t* __restrict__ Y,   // [N] (Q30)
      const int32_t* __restrict__ P,   // [N] (Q30)
      int16_t* __restrict__ G,         // [N]
      int nfolds,                      // K
      int max_depth,                   // D = 4..11
      int N,
      int stride)                      // tiles per block
  {
    const int ti = blockIdx.x;
    const int wi = threadIdx.x;  
  
    for (int i = 0; i < stride; ++i) {
      const int j = 32 * (stride * ti + i) + wi;
      if (j >= N) break;
  
      for (int k = 0; k < nfolds; ++k) {
        uint64_t v = 0u;
        
        #pragma unroll
        for (int d = 1; d < max_depth; ++d) {
          unsigned bit = static_cast<unsigned>(
              L[((size_t)k * (size_t)(max_depth - 1) + (size_t)(d - 1)) * (size_t)N + (size_t)j]
          );
          bit &= ((1u << d) - 1u);
          v |= ((uint64_t)bit) << ((d*(d-1))/2);
        }
  
        LE[(long long)k * (long long)N + j] = static_cast<PackedT>(v);
      }
  
      int32_t g32 = (int32_t)((long long)(((long long)Y[j] - (long long)P[j])) >> 20); //32 - qgrad_mbits
      G[j] = sat_i16_by_mbits(g32);
    }
}
  
std::tuple<torch::Tensor, torch::Tensor>
prep_vars(torch::Tensor L,  // [K, Dm, N], uint8/uint16
            torch::Tensor Y,  // [N], int32 (Q30)
            torch::Tensor P)  // [N], int32 (Q30)
{
  
    const int64_t K   = L.size(0);
    const int64_t Dm  = L.size(1);            // max_depth - 1
    const int64_t N64 = L.size(2);
    
    const int  nfolds     = static_cast<int>(K);
    const int  max_depth  = static_cast<int>(Dm + 1);
    const int  N          = static_cast<int>(N64);
  
    static constexpr int lanes   = 32;
    static constexpr int strides = 512;
    const dim3 block(lanes, 1, 1);
    const dim3 grid (strides, 1, 1);
  
    int stride = (N + lanes * strides - 1) / (lanes * strides);
    if (stride < 1) stride = 1;
  
    auto opts_base = L.options().device(L.device()).memory_format(c10::MemoryFormat::Contiguous);
    auto LE = torch::empty({K, N64}, opts_base.dtype(torch::kUInt64));
    auto G  = torch::empty({N64},     opts_base.dtype(torch::kInt16));
  
    // Grab stream
    auto stream = at::cuda::getCurrentCUDAStream();
  
    const int32_t* Yp = Y.data_ptr<int32_t>();
    const int32_t* Pp = P.data_ptr<int32_t>();
    int16_t*       Gp = G.data_ptr<int16_t>();
  
    auto* LEp = LE.data_ptr<uint64_t>();
    if (L.scalar_type() == torch::kUInt8) {
      const uint8_t* Lp = L.data_ptr<uint8_t>();
      _prep_vars<uint8_t, uint64_t><<<grid, block, 0, stream.stream()>>>(
          Lp, LEp, Yp, Pp, Gp, nfolds, max_depth, N, stride);
    } else if (L.scalar_type() == torch::kUInt16) {
      const uint16_t* Lp = L.data_ptr<uint16_t>();
      _prep_vars<uint16_t, uint64_t><<<grid, block, 0, stream.stream()>>>(
          Lp, LEp, Yp, Pp, Gp, nfolds, max_depth, N, stride);
    } else {
      TORCH_CHECK(false, "L must be uint8 or uint16 (got ", L.scalar_type(), ")");
    }
  
    return {LE, G};
}
