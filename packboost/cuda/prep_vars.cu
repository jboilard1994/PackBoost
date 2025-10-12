#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <tuple>

__device__ __forceinline__ int16_t sat_i16_by_mbits(int x){
    const int lo = -((1<<15) - 1); // grad_mbits = 15
    const int hi =  ((1<<15) - 1);
    x = (x < lo) ? lo : x;
    x = (x > hi) ? hi : x;
    return static_cast<int16_t>(x);
  }
  
template <typename T>
__global__ void _prep_vars(
      const uint8_t* __restrict__ L,   // [K, Dm, N], N-contiguous
      T* __restrict__ LE,              // [K, N]
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
        T v = (T)0;
        int off = 0; 
  
        #pragma unroll
        for (int d = 1; d < max_depth; ++d) {
          // NEW: pack the full d-bit code (0 .. 2^d-1)
            unsigned code = (unsigned)L[ ((long long)k * (max_depth - 1) + (d - 1)) * (long long)N + j ];
            v |= (T)((T)code << off);
            off += d;

        }
  
        LE[(long long)k * (long long)N + j] = v;
      }
  
      int32_t g32 = (Y[j] - P[j]) >> 20; //32 - qgrad_mbits
      G[j] = sat_i16_by_mbits(g32);
    }
}
  
std::tuple<torch::Tensor, torch::Tensor>
prep_vars(torch::Tensor L,  // [K, Dm, N], uint8
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
  
    // Pick output dtype for LE from max_depth (store as signed types; bit pattern preserved)
    c10::ScalarType le_dtype;
    if      (max_depth > 8) le_dtype = torch::kUInt64; // stores packed 64-bit
    else if (max_depth > 6) le_dtype = torch::kUInt32; // stores packed 32-bit
    else                    le_dtype = torch::kUInt16; // stores packed 16-bit
  
    auto opts_base = L.options().device(L.device()).memory_format(c10::MemoryFormat::Contiguous);
    auto LE = torch::empty({K, N64}, opts_base.dtype(le_dtype));
    auto G  = torch::empty({N64},     opts_base.dtype(torch::kInt16));
  
    // Grab stream
    auto stream = at::cuda::getCurrentCUDAStream();
  
    // Launch with matching unsigned pointer type
    const uint8_t* Lp = L.data_ptr<uint8_t>();
    const int32_t* Yp = Y.data_ptr<int32_t>();
    const int32_t* Pp = P.data_ptr<int32_t>();
    int16_t*       Gp = G.data_ptr<int16_t>();
  
    if (max_depth > 8) {
      // LE dtype is kInt64 in Tensor; reinterpret as uint64_t* for kernel
      auto* LEp = reinterpret_cast<uint64_t*>(LE.data_ptr<uint64_t>());
      _prep_vars<uint64_t><<<grid, block, 0, stream.stream()>>>(Lp, LEp, Yp, Pp, Gp, nfolds, max_depth, N, stride);
    } else if (max_depth > 6) {
      auto* LEp = reinterpret_cast<uint32_t*>(LE.data_ptr<uint32_t>());
      _prep_vars<uint32_t><<<grid, block, 0, stream.stream()>>>(Lp, LEp, Yp, Pp, Gp, nfolds, max_depth, N, stride);
    } else {
      auto* LEp = reinterpret_cast<uint16_t*>(LE.data_ptr<uint16_t>());
      _prep_vars<uint16_t><<<grid, block, 0, stream.stream()>>>(Lp, LEp, Yp, Pp, Gp, nfolds, max_depth, N, stride);
    }
  
    return {LE, G};
}
