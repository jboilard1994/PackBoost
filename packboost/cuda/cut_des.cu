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
  // node 0 -> depth 0, nodes 1,2 -> depth 1, etc.
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
    const uint8_t* __restrict__ FST,      // [treesets, K1, D]  -> fold id per (tree_set,k,depth)
    const int64_t* __restrict__ H,        // [K1, nE, nodes, 2, 32] (left stats per era)
    const int64_t* __restrict__ H0,       // [K0, nE, nodes, 2]     (parent stats per era)
    // outputs
    int32_t* __restrict__ V,        // [treesets, K0, 2*nodes] quantized leaf values
    uint16_t* __restrict__ I,        // [treesets, K0, nodes]   chosen feature id
    // dims
    int treesets, int K0, int K1, int nE, int nodes, int D,
    int tree_set,
    // hyperparams
    float L2, float lr, int qgrad_bits, int max_depth)
{
  #pragma fp_contract(off)

  const int leaf = blockIdx.x;
  const int lane = threadIdx.x;

  if (leaf >= (nodes-1)) return; // Only process internal nodes
  if (lane >= WARP_SIZE) return;
  if ((unsigned)tree_set >= (unsigned)treesets) return;
  if (K0 > CUTCUDA_MAX_FOLDS) return;

  const int depth = depth_from_leaf(leaf);
  if ((unsigned)depth >= (unsigned)D) return;

  const float L2_eff = L2 * powf(2.0f, 5.0f - (float)depth);
  const float qscale = lr * (float)(1u << (31 - qgrad_bits))
                       * powf(2.0f, -(float)(max_depth - depth));

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

  for (int k = 0; k < K1; ++k) {
    const size_t fst_idx = (((size_t)tree_set * K1) + k) * D + depth;
    const int tree_fold = (int)FST[fst_idx];
    if ((unsigned)tree_fold >= (unsigned)K0) continue;

    float G0_tot = 0.f, N0_tot = 0.f;
    float GT_tot = 0.f, HT_tot = 0.f;
    float sum_dir = 0.f;
    int   eras_used = 0;

    for (int e = 0; e < nE; ++e) {
      // H: [K1, nE, nodes, 2, 32]
      const size_t base_H = (((((size_t)k * nE) + e) * nodes + leaf) * 2u) * 32u + lane;
      const float G0 = __ll2float_rn(H[base_H + 0 * 32u]);
      const float N0 = __ll2float_rn(H[base_H + 1 * 32u]);

      // **FIX**: Correct indexing for H0: [K0, nE, nodes, 2]
      const size_t idx_H0 = ((((size_t)tree_fold * nE) + e) * nodes + leaf) * 2u;
      const float GT = __ll2float_rn(H0[idx_H0 + 0]);
      const float HT = __ll2float_rn(H0[idx_H0 + 1]);

      const float G1 = GT - G0;
      const float N1 = HT - N0;

      if (N0 > 0.f && N1 > 0.f) {
        const float V0 = G0 / (N0 + L2_eff);
        const float V1 = G1 / (N1 + L2_eff);
        sum_dir += fsign(V1 - V0);
        ++eras_used;
      }

      G0_tot += G0; N0_tot += N0;
      GT_tot += GT; HT_tot += HT;
    }

    const float dir_score = (eras_used > 0) ? fabsf(sum_dir) / (float)eras_used : 0.f;

    const float G1_tot = GT_tot - G0_tot;
    const float N1_tot = HT_tot - N0_tot;
    const float V0f    = (N0_tot > 0.f) ? (G0_tot / (N0_tot + L2_eff)) : 0.f;
    const float V1f    = (N1_tot > 0.f) ? (G1_tot / (N1_tot + L2_eff)) : 0.f;

    const float S0 = G0_tot * V0f;
    const float S1 = G1_tot * V1f;
    const int   S_bits = __float_as_int(S0 + S1);

    if (better(dir_score, S_bits, best_dir[tree_fold], best_sbits[tree_fold])) {
      best_dir  [tree_fold] = dir_score;
      best_sbits[tree_fold] = S_bits;
      best_vl   [tree_fold] = __float2int_rz(qscale * V0f);
      best_vr   [tree_fold] = __float2int_rz(qscale * V1f);
      best_k    [tree_fold] = (uint16_t)k;
    }
  }

  __syncwarp(FULL_MASK);

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

    if (!isfinite(dir_w)) continue;

    const unsigned m_dir = __ballot_sync(FULL_MASK, best_dir[fold] == dir_w);
    const unsigned m_s   = __ballot_sync(FULL_MASK, best_sbits[fold] == sb_w);
    const unsigned msk   = m_dir & m_s;
    if (msk == 0u) continue;

    const int winner_lane = 31 - __clz(msk);
    if (lane == winner_lane) {
      const size_t v_base = (((size_t)tree_set * K0) + fold)
                          * (size_t)(2 * nodes) + (size_t)(2 * leaf);
      V[v_base + 0] = best_vl[fold];
      V[v_base + 1] = best_vr[fold];

      const size_t f_idx = (size_t)tree_set * (size_t)(32 * K1)
                         + (size_t)best_k[fold] * 32u + (size_t)winner_lane;
      const size_t i_idx = (((size_t)tree_set * K0) + fold)
                         * (size_t)nodes + (size_t)leaf;
      I[i_idx] = F[f_idx];
    }
  }
}

// -------- Launcher --------
void cut_des_cuda_launcher(
    torch::Tensor F, torch::Tensor FST, torch::Tensor H, torch::Tensor H0,
    torch::Tensor V, torch::Tensor I,
    int tree_set,
    double L2, double lr, int qgrad_bits, int max_depth)
{
  const int treesets = F.size(0);
  const int K1       = H.size(0);
  const int nE       = H.size(1);
  const int nodes_dim= H.size(2); // This is 2^max_depth
  const int K0       = H0.size(0);
  const int D        = FST.size(2);

  TORCH_CHECK(H.dim()==5 && H.size(3)==2 && H.size(4)==32, "H must be [K1,nE,nodes,2,32]");
  TORCH_CHECK(H0.dim()==4 && H0.size(2)==nodes_dim && H0.size(3)==2, "H0 must be [K0,nE,nodes,2]");
  TORCH_CHECK(K0 <= CUTCUDA_MAX_FOLDS, "K0 exceeds CUTCUDA_MAX_FOLDS");
  TORCH_CHECK(max_depth == D, "max_depth must match FST dimension D");

  // Grid is launched over internal nodes, which are 0 to (2^D - 2)
  const int internal_nodes = (1 << max_depth) - 1;
  const dim3 grid(internal_nodes, 1, 1);
  const dim3 block(WARP_SIZE, 1, 1);
  auto stream = at::cuda::getCurrentCUDAStream();

  cut_des_cuda_kernel<<<grid, block, 0, stream.stream()>>>(
      F.data_ptr<uint16_t>(), FST.data_ptr<uint8_t>(),
      H.data_ptr<int64_t>(), H0.data_ptr<int64_t>(),
      V.data_ptr<int32_t>(), I.data_ptr<uint16_t>(),
      treesets, K0, K1, nE, nodes_dim, D, tree_set,
      (float)L2, (float)lr, qgrad_bits, max_depth);

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "cut_des_cuda_kernel launch failed: ", cudaGetErrorString(err));
}