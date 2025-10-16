#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
#include <math.h>

constexpr int  WARP_SIZE  = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

#ifndef CUTCUDA_MAX_FOLDS
#define CUTCUDA_MAX_FOLDS 32
#endif

__device__ __forceinline__ int depth_from_leaf(int leaf) {
  return 31 - __clz((unsigned)(leaf + 1));
}

// sign as float in {-1,0,+1}
__device__ __forceinline__ float fsign(float x) {
  return (x > 0.f) - (x < 0.f);
}

// Lexicographic compare: (dir_score, classic_sbits)
__device__ __forceinline__ bool better(float dir_a, int sbits_a,
                                       float dir_b, int sbits_b) {
  if (dir_a > dir_b) return true;
  if (dir_a < dir_b) return false;
  return sbits_a > sbits_b;
}

extern "C" __global__ void cut_des_cuda_kernel(
    // inputs
    const uint16_t* __restrict__ F,        // [treesets, 32*K1] (feature ids per lane)
    const uint8_t*  __restrict__ FST,      // [treesets, K1, D]  -> fold id per (tree_set,k,depth)
    const int64_t*  __restrict__ H,        // [K1, nE, nodes, 2, 32] (left stats per era)
    const int64_t*  __restrict__ H0,       // [K0, nE, nodes, 2]     (parent stats per era)
    // outputs
    int32_t*        __restrict__ V,        // [treesets, K0, 2*nodes] quantized leaf values
    uint16_t*       __restrict__ I,        // [treesets, K0, nodes]   chosen feature id
    // dims
    int treesets, int K0, int K1, int nE, int nodes, int D,
    int tree_set,
    // hyperparams (same policy as your classic kernel)
    float L2, float lr, int qgrad_bits, int max_depth)
{
  #pragma fp_contract(off)

  const int leaf = blockIdx.x;
  const int lane = threadIdx.x;

  if (blockDim.x != WARP_SIZE || lane >= WARP_SIZE) return;
  if (leaf >= nodes) return;
  if ((unsigned)tree_set >= (unsigned)treesets) return;
  if (K0 > CUTCUDA_MAX_FOLDS) return;

  const int depth = depth_from_leaf(leaf);
  if ((unsigned)depth >= (unsigned)D) return;

  // Depth-dependent regularization & q-scale (unchanged)
  const float L2_eff = L2 * powf(2.0f, 5.0f - (float)depth);
  const float qscale = lr * (float)(1u << (31 - qgrad_bits))
                       * powf(2.0f, -(float)(max_depth - depth));

  // Per-fold best (directional score + classic sbits)
  float    best_dir[CUTCUDA_MAX_FOLDS];
  int      best_sbits[CUTCUDA_MAX_FOLDS];
  int      best_vl[CUTCUDA_MAX_FOLDS];
  int      best_vr[CUTCUDA_MAX_FOLDS];
  uint16_t best_k [CUTCUDA_MAX_FOLDS];

  #pragma unroll
  for (int f = 0; f < CUTCUDA_MAX_FOLDS; ++f) {
    if (f >= K0) break;
    best_dir[f]   = -INFINITY;
    best_sbits[f] = INT_MIN;
    best_vl[f]    = 0;
    best_vr[f]    = 0;
    best_k [f]    = 0;
  }

  // Iterate candidates in the sampled set
  for (int k = 0; k < K1; ++k) {
    // Which fold does this candidate belong to for this tree-set at this depth?
    const size_t fst_idx = (((size_t)tree_set * (size_t)K1) + (size_t)k) * (size_t)D + (size_t)depth;
    const int tree_fold = (int)FST[fst_idx];
    if ((unsigned)tree_fold >= (unsigned)K0) continue;

    // Accumulate pooled totals for classic tie-break & leaf values
    float G0_tot = 0.f, N0_tot = 0.f;
    float GT_tot = 0.f, HT_tot = 0.f;

    // Directional agreement accumulator
    float sum_dir = 0.f;
    int   eras_used = 0;

    // parent per-era base for this (fold, leaf)
    const size_t base_H0_fold_leaf = (((size_t)tree_fold * (size_t)nE) * (size_t)nodes + (size_t)leaf) * 2u;

    // sweep eras
    for (int e = 0; e < nE; ++e) {
      // left per-era lane stats: [K1, nE, nodes, 2, 32]
      const size_t base_H = (((((size_t)k * (size_t)nE) + (size_t)e) * (size_t)nodes + (size_t)leaf) * 2u) * 32u + (size_t)lane;

      const float G0 = __ll2float_rn(H[base_H + 0 * 32u]);
      const float N0 = __ll2float_rn(H[base_H + 1 * 32u]);

      // parent per-era: [K0, nE, nodes, 2]
      const float GT = __ll2float_rn(H0[base_H0_fold_leaf + (size_t)e * 2u + 0]);
      const float HT = __ll2float_rn(H0[base_H0_fold_leaf + (size_t)e * 2u + 1]);

      const float G1 = GT - G0;
      const float N1 = HT - N0;

      // Direction sign only if both children are non-empty in this era.
      if (N0 > 0.f && N1 > 0.f) {
        const float V0 = G0 / (N0 + L2_eff);
        const float V1 = G1 / (N1 + L2_eff);
        // v = -G/(H+λ)  => sign(vL - vR) == sign(V1 - V0)
        sum_dir += fsign(V1 - V0);
        ++eras_used;
      }

      // pooled totals for classic tie-break & leaf values
      G0_tot += G0; N0_tot += N0;
      GT_tot += GT; HT_tot += HT;
    }

    // Directional agreement score in [0,1]; if no usable eras, treat as 0.
    const float dir_score = (eras_used > 0) ? fabsf(sum_dir) / (float)eras_used : 0.f;

    // Classic score (for tie-break): S_bits = as_int(S0+S1) using pooled totals
    const float G1_tot = GT_tot - G0_tot;
    const float N1_tot = HT_tot - N0_tot;
    const float V0f    = G0_tot / (N0_tot + L2_eff);
    const float V1f    = G1_tot / (N1_tot + L2_eff);

    const float S0 = G0_tot * V0f;
    const float S1 = G1_tot * V1f;
    const int   S_bits = __float_as_int(S0 + S1);

    // Update per-fold best in this lane using (dir, classic) lexicographic rule
    if (better(dir_score, S_bits, best_dir[tree_fold], best_sbits[tree_fold])) {
      best_dir  [tree_fold] = dir_score;
      best_sbits[tree_fold] = S_bits;
      best_vl   [tree_fold] = __float2int_rz(qscale * V0f);
      best_vr   [tree_fold] = __float2int_rz(qscale * V1f);
      best_k    [tree_fold] = (uint16_t)k;
    }
  }

  __syncwarp(FULL_MASK);

  // ---- warp-wide reduction per fold over (dir, classic) pairs ----
  for (int fold = 0; fold < K0; ++fold) {
    float dir_w = best_dir[fold];
    int   sb_w  = best_sbits[fold];

    #pragma unroll
    for (int p = 0; p < 5; ++p) {
      const int ofs = 1 << p;
      const float dir_p = __shfl_xor_sync(FULL_MASK, dir_w, ofs, WARP_SIZE);
      const int   sb_p  = __shfl_xor_sync(FULL_MASK, sb_w,  ofs, WARP_SIZE);
      if (better(dir_p, sb_p, dir_w, sb_w)) {
        dir_w = dir_p; sb_w = sb_p;
      }
    }

    if (!isfinite(dir_w)) continue; // shouldn't happen, but be safe

    // Identify the winner lane: both dir AND classic must match the reduced pair
    const unsigned m_dir = __ballot_sync(FULL_MASK, best_dir[fold]   == dir_w);
    const unsigned m_s   = __ballot_sync(FULL_MASK, best_sbits[fold] == sb_w);
    const unsigned msk   = m_dir & m_s;
    if (msk == 0u) continue;

    const int winner_lane = 31 - __clz(msk);
    if (lane == winner_lane) {
      // Write leaf values and feature id
      const size_t v_base = (((size_t)tree_set * (size_t)K0) + (size_t)fold)
                          * (size_t)(2 * nodes) + (size_t)(2 * leaf);
      V[v_base + 0] = best_vl[fold];
      V[v_base + 1] = best_vr[fold];

      const size_t f_idx = (size_t)tree_set * (size_t)(32 * K1)
                         + (size_t)best_k[fold] * 32u + (size_t)winner_lane;
      const size_t i_idx = (((size_t)tree_set * (size_t)K0) + (size_t)fold)
                         * (size_t)nodes + (size_t)leaf;
      I[i_idx] = F[f_idx];
    }
  }
}

// -------- Launcher (directional DES only) --------
//
// Shapes:
//   F   : [treesets, 32*K1] (UInt16)
//   FST : [treesets, K1, D] (UInt8)
//   H   : [K1, nE, nodes, 2, 32] (Int64)    <-- per-era left stats
//   H0  : [K0, nE, nodes, 2]     (Int64)    <-- per-era parent stats
//   V   : [treesets, K0, 2*nodes] (Int32)
//   I   : [treesets, K0, nodes]   (UInt16)
//
void cut_des_cuda_launcher(
    torch::Tensor F,
    torch::Tensor FST,
    torch::Tensor H,
    torch::Tensor H0,
    torch::Tensor V,
    torch::Tensor I,
    int tree_set,
    double L2, double lr, int qgrad_bits, int max_depth)
{
  const int treesets = F.size(0);
  const int K1       = H.size(0);
  const int nE       = H.size(1);
  const int nodes    = H.size(2);
  const int K0       = H0.size(0);
  const int D        = FST.size(2);

  TORCH_CHECK(H.dim()==5 && H.size(3)==2 && H.size(4)==32, "H must be [K1,nE,nodes,2,32]");
  TORCH_CHECK(H0.dim()==4 && H0.size(2)==nodes && H0.size(3)==2, "H0 must be [K0,nE,nodes,2]");
  TORCH_CHECK(K0 <= CUTCUDA_MAX_FOLDS, "K0 exceeds CUTCUDA_MAX_FOLDS");

  const dim3 grid((1<<max_depth) - 1, 1, 1);
  const dim3 block(WARP_SIZE, 1, 1);

  auto stream = at::cuda::getCurrentCUDAStream();

  cut_des_cuda_kernel<<<grid, block, 0, stream.stream()>>>(
      reinterpret_cast<const uint16_t*>(F.data_ptr<uint16_t>()),
      FST.data_ptr<uint8_t>(),
      H.data_ptr<int64_t>(),
      H0.data_ptr<int64_t>(),
      V.data_ptr<int32_t>(),
      reinterpret_cast<uint16_t*>(I.data_ptr<uint16_t>()),
      treesets, K0, K1, nE, nodes, D, tree_set,
      (float)L2, (float)lr, qgrad_bits, max_depth);

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "cut_des_cuda_kernel (directional) launch failed: ",
              cudaGetErrorString(err));
}
