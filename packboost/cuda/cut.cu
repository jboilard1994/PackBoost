#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>          // <- add this

constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

#ifndef CUTCUDA_MAX_FOLDS
#define CUTCUDA_MAX_FOLDS 32
#endif

__device__ __forceinline__ int depth_from_leaf(int leaf) {
  return 31 - __clz(static_cast<unsigned>(leaf + 1));
}

extern "C" __global__ void cut_cuda_kernel(
    const uint16_t* __restrict__ F,      // [treesets, 32*K1]      (uint16)
    const uint8_t*  __restrict__ FST,    // [treesets, K1, D]      (uint8)
    const int64_t*  __restrict__ H,      // [K1, nodes, 2, 32]     (int64)
    const int64_t*  __restrict__ H0,     // [K0, nodes, 2]         (int64)
    int32_t*        __restrict__ V,      // [treesets, K0, 2*nodes](int32)
    uint16_t*       __restrict__ I,      // [treesets, K0, nodes]  (uint16)
    // dims
    int treesets, int K0, int K1, int nodes, int D,
    int tree_set,
    // hyperparams
    float L2, float lr, int qgrad_bits, int max_depth)
{
  const int leaf = blockIdx.x;
  const int wi   = threadIdx.x;
  if (blockDim.x != WARP_SIZE || wi >= WARP_SIZE) return;
  if (leaf >= nodes) return;
  if (tree_set < 0 || tree_set >= treesets) return;
  if (K0 > CUTCUDA_MAX_FOLDS) return;

  const int depth = depth_from_leaf(leaf);
  if (depth < 0 || depth >= D) return;

  int   mxs[CUTCUDA_MAX_FOLDS];
  int   vls[CUTCUDA_MAX_FOLDS];
  int   vrs[CUTCUDA_MAX_FOLDS];
  uint16_t fs[CUTCUDA_MAX_FOLDS];

  float g01s[CUTCUDA_MAX_FOLDS];
  float n01s[CUTCUDA_MAX_FOLDS];

  for (int k = 0; k < K0; ++k) {
    mxs[k] = -100000000;
    const size_t base = ((size_t)k * nodes + leaf) * 2u;
    g01s[k] = static_cast<float>(H0[base + 0]);
    n01s[k] = static_cast<float>(H0[base + 1]);
  }

  const float L2_eff = L2 * ldexpf(1.0f, 5 - depth);
  const int shift = 31 - qgrad_bits;
  const float pow2 = (float)(1u << shift);
  const float depth_scale = ldexpf(1.0f, -(max_depth - depth));
  const float qscale = lr * pow2 * depth_scale;

  for (int k = 0; k < K1; ++k) {
    const size_t fst_idx = (((size_t)tree_set * K1) + k) * (size_t)D + depth;
    const int tree_fold = (int)FST[fst_idx];
    if (tree_fold < 0 || tree_fold >= K0) continue;

    const size_t h_base = ((((size_t)k * nodes) + leaf) * 2u) * 32u + wi;
    const float G0 = static_cast<float>(H[h_base + 0 * 32u]);
    const float N0 = static_cast<float>(H[h_base + 1 * 32u]);

    const float G01 = g01s[tree_fold];
    const float N01 = n01s[tree_fold];

    const float G1 = G01 - G0;
    const float N1 = N01 - N0;

    const float V0f = G0 / (N0 + L2_eff);
    const float V1f = G1 / (N1 + L2_eff);

    const float S0 = G0 * V0f;
    const float S1 = G1 * V1f;

    const int S_bits = __float_as_int(S0 + S1);

    if (mxs[tree_fold] < S_bits) {
      mxs[tree_fold] = S_bits;
      vls[tree_fold] = (int)roundf(qscale * V0f);
      vrs[tree_fold] = (int)roundf(qscale * V1f);
      fs [tree_fold] = (uint16_t)k;
    }
  }

  __syncwarp(FULL_MASK);

  for (int fold = 0; fold < K0; ++fold) {
    int mx  = mxs[fold];
    int mxw = mx;

    #pragma unroll
    for (int p = 0; p < 5; ++p) {
      const int partner = __shfl_xor_sync(FULL_MASK, mxw, 1 << p, WARP_SIZE);
      mxw = (partner > mxw) ? partner : mxw;
    }

    const unsigned msk = __ballot_sync(FULL_MASK, mx == mxw);
    const bool is_max = (mx == mxw) && ((1u << wi) > (msk >> 1));

    if (is_max) {
      const size_t v_base = (((size_t)tree_set * K0) + fold) * (size_t)(2 * nodes) + (size_t)(2 * leaf);
      V[v_base + 0] = vls[fold];
      V[v_base + 1] = vrs[fold];

      const size_t f_idx = (size_t)tree_set * (size_t)(32 * K1) + (size_t)fs[fold] * 32u + wi;
      const size_t i_idx = (((size_t)tree_set * K0) + fold) * (size_t)nodes + (size_t)leaf;
      I[i_idx] = F[f_idx];
    }
  }
}

// -------- C++ launcher --------
void cut_cuda_launcher(
    torch::Tensor F,      // int16 (stores uint16)
    torch::Tensor FST,    // uint8
    torch::Tensor H,      // int64
    torch::Tensor H0,     // int64
    torch::Tensor V,      // int32
    torch::Tensor I,      // int16 (stores uint16)
    int tree_set, double L2, double lr, int qgrad_bits, int max_depth)
{

  const int treesets = F.size(0);
  const int K1       = H.size(0);
  const int nodes    = H.size(1);

  const int K0       = H0.size(0);
  const int D = FST.size(2);

  const dim3 grid(nodes, 1, 1);
  const dim3 block(WARP_SIZE, 1, 1);

  auto stream = at::cuda::getCurrentCUDAStream();

  cut_cuda_kernel<<<grid, block, 0, stream.stream()>>>(
      reinterpret_cast<const uint16_t*>(F.data_ptr<int16_t>()),
      FST.data_ptr<uint8_t>(),
      H.data_ptr<int64_t>(),
      H0.data_ptr<int64_t>(),
      V.data_ptr<int32_t>(),
      reinterpret_cast<uint16_t*>(I.data_ptr<int16_t>()),
      treesets, K0, K1, nodes, D, tree_set,
      static_cast<float>(L2),
      static_cast<float>(lr),
      qgrad_bits,
      max_depth);
}