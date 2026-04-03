#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------- Helper Functions ----------------

static __device__ __forceinline__ unsigned long long pack_sc(int sum32, int cnt32) {
  return ( (unsigned long long)(unsigned int)cnt32 << 32 ) |
           (unsigned long long)(unsigned int)sum32;
}
static __device__ __forceinline__ unsigned long long add_pack(unsigned long long a, unsigned long long b) {
  int sa = (int)(unsigned int)a;
  int ca = (int)(unsigned int)(a >> 32);
  int sb = (int)(unsigned int)b;
  int cb = (int)(unsigned int)(b >> 32);
  return pack_sc(sa + sb, ca + cb);
}

// H layout helper: [K1, nE, nodes, 2, 32]
static __device__ __forceinline__ unsigned long long* HptrE(
    int64_t* H, int nodes, int nE,
    int feat, int era, int node, int chan, int lane)
{
  const size_t idx =
      (((((size_t)feat * (size_t)nE + (size_t)era) * (size_t)nodes + (size_t)node) * 2u
        + (size_t)chan) * 32u + (size_t)lane);
  return (unsigned long long*)(H + idx);
}

// REFACTORED HELPER: Flushes register-accumulated packed values to global memory
static __device__ __forceinline__ void flush_node_reg(
    int node, unsigned long long P,
    int64_t* H, int nodes_total, int E,
    int feat_set, int era, int lane)
{
  const long long sum_ll = (long long)(int)(uint32_t)P;
  const long long cnt_ll = (long long)(int)(uint32_t)(P >> 32);
  if (sum_ll | cnt_ll) {
    unsigned long long* ps = HptrE(H, nodes_total, E, feat_set, era, node, 0, lane);
    unsigned long long* pc = HptrE(H, nodes_total, E, feat_set, era, node, 1, lane);
    if (sum_ll) atomicAdd(ps, (unsigned long long)sum_ll);
    if (cnt_ll) atomicAdd(pc, (unsigned long long)cnt_ll);
  }
}


// ---------------- Kernel (uint64 packed LF) ----------------
// XS: [K1, cols_32M] (cols_32M = 32*M, padded), each entry is a 32-bit "lane" word
// Y : [N] int16 (grad)
// LF: [K1, N] (uint64) packed leaf codes
// era_ends: [E] int32 (exclusive ends), era_ends[E-1]==N
// H : [K1, E, nodes, 2, 32] int64  (sum, count) per lane
template <typename LF_T>
__global__ void _h_des_sm(
    const uint32_t* __restrict__ XS,
    const int16_t* __restrict__ Y,
    const LF_T* __restrict__ LF,
    const int32_t* __restrict__ era_ends,
    int64_t* __restrict__ H,
    int K1, int cols_32M, int N, int E, int max_depth,
    int warps_per_block, int stride, int nodes_total,
    int sh_cap)
{
  const int feat_set   = blockIdx.x;
  const int warp_in_blk= threadIdx.x >> 5;     // 0..(warps_per_block-1)
  const int lane       = threadIdx.x & 31;     // 0..31
  const int gwarp      = warps_per_block * blockIdx.y + warp_in_blk;

  if (feat_set >= K1) return;

  // Shared histogram for depths >=3 (capped at sh_cap; overflow goes to global)
  extern __shared__ int sh_high[];

  const unsigned mask = __ballot_sync(__activemask(), true);

  // Process 'stride' tiles of 32 columns per warp
  for (int j = 0; j < stride; ++j) {
    const int base = 32 * (stride * gwarp + j);
    if (base >= cols_32M) break;

    // Lane-local loads for this tile
    const int jj_lane = base + lane;
    int32_t  y_lane = 0;
    uint64_t l64    = 0;
    if (jj_lane < N) {
      y_lane = (int32_t)Y[jj_lane];
      l64    = (uint64_t)((LF_T)LF[(size_t)feat_set * (size_t)N + (size_t)jj_lane]);
    }
    const uint32_t l64_lo = static_cast<uint32_t>(l64);
    const uint32_t l64_hi = static_cast<uint32_t>(l64 >> 32);
    // 32-bit feature word for this lane
    uint32_t xfd_local = 0u;
    if (jj_lane < cols_32M) {
      xfd_local = XS[(size_t)feat_set * (size_t)cols_32M + (size_t)jj_lane];
    }
    // Mask tail beyond N
    const int rem = N - base;
    const uint32_t valid_mask = (rem >= 32 ? 0xFFFFFFFFu : (rem > 0 ? ((1u << rem) - 1u) : 0u));
    xfd_local &= valid_mask;

    // ---- Build era id for each of the 32 columns and compress to segments ----
    __shared__ int s_era_k[32];
    __shared__ int s_seg_start[33];
    __shared__ int s_seg_end  [33];
    __shared__ int s_seg_era  [33];
    __shared__ int s_nsegs;

    if (lane == 0) {
      int e = 0;
      while (e < E && era_ends[e] <= base) ++e;
      for (int k = 0; k < 32; ++k) {
        const int pos = base + k;
        while (e < E && era_ends[e] <= pos) ++e;
        s_era_k[k] = (e < E ? e : (E - 1));
      }
      // compress consecutive equal-eras into segments [k0,k1)
      int s = 0;
      const int k_valid = (rem >= 32 ? 32 : (rem > 0 ? rem : 0));
      if (k_valid > 0) {
        int cur = s_era_k[0];
        int k0  = 0;
        for (int k = 1; k < k_valid; ++k) {
          const int ek = s_era_k[k];
          if (ek != cur) {
            s_seg_start[s] = k0; s_seg_end[s] = k; s_seg_era[s] = cur; ++s;
            cur = ek; k0 = k;
          }
        }
        s_seg_start[s] = k0; s_seg_end[s] = k_valid; s_seg_era[s] = cur; ++s;
      }
      s_nsegs = s;
    }
    __syncwarp();

    // iterate segments (usually 1–2)
    for (int seg = 0; seg < s_nsegs; ++seg) {
      const int era = s_seg_era[seg];
      const int k0  = s_seg_start[seg];
      const int k1  = s_seg_end[seg];

      // zero shared high-depth histogram (this segment only)
      for (int r = 0; r < sh_cap; ++r) {
        sh_high[(r * 2 + 0) * 32 + lane] = 0; // sum
        sh_high[(r * 2 + 1) * 32 + lane] = 0; // cnt
      }
      // FIX 1: Use __syncthreads for block-level shared memory.
      __syncthreads();

      // low-depth packed (registers) for this segment
      unsigned long long p0  = 0ull;                   // node 0
      unsigned long long p10 = 0ull, p11 = 0ull;       // nodes 1..2
      unsigned long long p20 = 0ull, p21 = 0ull,
                         p22 = 0ull, p23 = 0ull;       // nodes 3..6

      // scan k within [k0,k1)
      uint32_t bits = xfd_local >> k0; // consume only needed part
      for (int k = k0; k < k1; ++k) {
        const int v = (int)(bits & 1u);
        bits >>= 1;

        // broadcast label & leaf-code of the kth column
        const int src_lane = k;                 // lane id == column id
        const int32_t yk = __shfl_sync(mask, y_lane, src_lane);
        // 64-bit warp shuffle via two 32-bit shuffles (supports depth > 8)
        uint64_t      lk = ((uint64_t)__shfl_sync(mask, l64_hi, src_lane) << 32)
                          | (uint64_t)__shfl_sync(mask, l64_lo, src_lane);

        // d = 0
        if (v) p0 = add_pack(p0, pack_sc((int)yk, 1));

        // d = 1
        const unsigned tk1 = (lk & 1u); lk >>= 1;
        if (v) {
          (tk1 == 0u ? p10 : p11) = add_pack((tk1 == 0u ? p10 : p11), pack_sc((int)yk, 1));
        }

        // d = 2
        const unsigned tk2 = (lk & 3u); lk >>= 2;
        if (v) {
          if      (tk2 == 0u) p20 = add_pack(p20, pack_sc((int)yk, 1));
          else if (tk2 == 1u) p21 = add_pack(p21, pack_sc((int)yk, 1));
          else if (tk2 == 2u) p22 = add_pack(p22, pack_sc((int)yk, 1));
          else                p23 = add_pack(p23, pack_sc((int)yk, 1));
        }

        // d >= 3: shared if idx < sh_cap, else global atomic spillover
        #pragma unroll
        for (int d = 3; d < 32; ++d) {
          if (d >= max_depth) break;
          const unsigned to  = (1u << d) - 1u;
          const unsigned tkd = (static_cast<unsigned>(lk) & to); lk >>= d;
          const int node = (int)to + (int)tkd;   // 7..nodes_total-1
          const int idx  = node - 7;
          if (v) {
            if (idx < sh_cap) {
              atomicAdd(&sh_high[(idx * 2 + 0) * 32 + lane], (int)yk); // sum
              atomicAdd(&sh_high[(idx * 2 + 1) * 32 + lane], 1);       // cnt
            } else {
              atomicAdd(HptrE(H, nodes_total, E, feat_set, era, node, 0, lane),
                        static_cast<unsigned long long>(static_cast<int64_t>((int)yk)));
              atomicAdd(HptrE(H, nodes_total, E, feat_set, era, node, 1, lane),
                        static_cast<unsigned long long>(static_cast<int64_t>(1)));
            }
          }
        }
      } // k

      // flush low depths (nodes 0..6) to global for this ERA
      flush_node_reg(0, p0,  H, nodes_total, E, feat_set, era, lane);
      flush_node_reg(1, p10, H, nodes_total, E, feat_set, era, lane);
      flush_node_reg(2, p11, H, nodes_total, E, feat_set, era, lane);
      flush_node_reg(3, p20, H, nodes_total, E, feat_set, era, lane);
      flush_node_reg(4, p21, H, nodes_total, E, feat_set, era, lane);
      flush_node_reg(5, p22, H, nodes_total, E, feat_set, era, lane);
      flush_node_reg(6, p23, H, nodes_total, E, feat_set, era, lane);

      // FIX 3: Sync all threads in the block before reading from shared memory.
      __syncthreads();

      // drain shared high-depth rows to global for this ERA
      const int rows_per_warp = (sh_cap + warps_per_block - 1) / warps_per_block;
      for (int rr = 0; rr < rows_per_warp; ++rr) {
        const int idx = rows_per_warp * warp_in_blk + rr;   // 0..sh_cap-1
        const int node = 7 + idx;
        if (idx < sh_cap && node < nodes_total) {
          const int base = (idx * 2) * 32 + lane;
          const int64_t fsum = (int64_t)sh_high[base + 0];
          const int64_t csum = (int64_t)sh_high[base + 32];
          if (fsum | csum) {
            atomicAdd(HptrE(H, nodes_total, E, feat_set, era, node, 0, lane), (unsigned long long)fsum);
            atomicAdd(HptrE(H, nodes_total, E, feat_set, era, node, 1, lane), (unsigned long long)csum);
          }
        }
      }
      // FIX 4: Sync after draining to prevent a race on the next segment's zeroing phase.
      __syncthreads();
    } // segments
  } // tiles
}

// ---------------- Launch plumbing ----------------
static inline int ceil_div_int(int a, int b){ return (a + b - 1) / b; }

static inline int choose_warps_that_fit(size_t smem_high, size_t smem_cap) {
  int wpb = 16;
  while (wpb > 1) {
    if (smem_high <= smem_cap) break;
    wpb >>= 1;
  }
  return (wpb < 1) ? 1 : wpb;
}

static inline void infer_grid_stride_des(
    int K1, int cols_32M, int warps_per_block,
    int& blocks_per_feat_out, int& stride_out)
{
  // Reuse Murky heuristic
  const int A100_SCHED = 64 * 103;
  int blocks_per_feat = ceil_div_int(A100_SCHED, warps_per_block);
  blocks_per_feat = ceil_div_int(blocks_per_feat, K1);
  if (blocks_per_feat < 1) blocks_per_feat = 1;

  const int total_warps = blocks_per_feat * warps_per_block;
  int stride = ceil_div_int(cols_32M, total_warps * 32);
  if (stride < 1) stride = 1;

  blocks_per_feat_out = blocks_per_feat;
  stride_out = stride;
}

torch::Tensor h_des(
    torch::Tensor XS,        // [K1, cols_32M] (uint32/int32)
    torch::Tensor Y,         // [N] int16
    torch::Tensor LF,        // [K1, N] uint64
    torch::Tensor era_ends,  // [E] int32 (exclusive ends)
    int max_depth)
{
  TORCH_CHECK(XS.is_cuda() && Y.is_cuda() && LF.is_cuda(),
              "XS, Y, LF must be CUDA tensors.");
  TORCH_CHECK(Y.scalar_type()==torch::kInt16, "Y must be int16.");
  TORCH_CHECK(XS.scalar_type()==torch::kUInt32 || XS.scalar_type()==torch::kInt32,
              "XS must be uint32/int32.");
  TORCH_CHECK(LF.dim()==2 && XS.dim()==2 && Y.dim()==1, "Shapes: XS[K1,cols], Y[N], LF[K1,N].");
  TORCH_CHECK(max_depth>0,
              "max_depth must be > 0.");

  const int K1       = (int)XS.size(0);
  const int cols_32M = (int)XS.size(1);
  const int N        = (int)Y.size(0);

  auto era_i32 = era_ends.to(XS.device(), torch::kInt32, /*non_blocking=*/false, /*copy=*/true)
                         .contiguous();
  TORCH_CHECK(era_i32.dim()==1, "era_ends must be 1-D.");
  const int E = (int)era_i32.size(0);
  TORCH_CHECK(E >= 1, "era_ends must have at least one item (N).");

  // Check on host to avoid device-side assert
  int last_era_end_host;
  cudaMemcpy(&last_era_end_host, era_i32.data_ptr<int32_t>() + (E - 1), sizeof(int32_t), cudaMemcpyDeviceToHost);
  TORCH_CHECK(last_era_end_host == N, "era_ends[-1] must equal N.");


  const int nodes_total = (1 << max_depth) - 1;

  auto H = torch::zeros({ (long long)K1, (long long)E,
                          (long long)nodes_total, 2LL, 32LL },
                        XS.options().dtype(torch::kLong)
                           .memory_format(c10::MemoryFormat::Contiguous));

  int n_ge3 = nodes_total - 7; if (n_ge3 < 1) n_ge3 = 1;

  auto* prop = at::cuda::getCurrentDeviceProperties();
  size_t smem_cap = prop->sharedMemPerBlockOptin ?
                      (size_t)prop->sharedMemPerBlockOptin :
                      (size_t)prop->sharedMemPerBlock;

  // Cap shared-memory histogram to what fits; overflow nodes use global atomics
  int sh_cap = std::min(n_ge3, (int)(smem_cap / (2 * 32 * sizeof(int))));
  if (sh_cap < 0) sh_cap = 0;
  const size_t smem_bytes = (size_t)sh_cap * 2 * 32 * sizeof(int);

  const int warps_per_block = ((size_t)n_ge3 * 2 * 32 * sizeof(int) <= smem_cap) ?
                              choose_warps_that_fit((size_t)n_ge3 * 2 * 32 * sizeof(int), smem_cap) : 1;
  int blocks_per_feat = 0, stride = 0;
  infer_grid_stride_des(K1, cols_32M, warps_per_block, blocks_per_feat, stride);

  dim3 grid (K1, blocks_per_feat, 1);
  dim3 block(warps_per_block * 32, 1, 1);

  auto stream = at::cuda::getCurrentCUDAStream();
  const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());

  TORCH_CHECK(LF.scalar_type() == torch::kUInt64, "LF must be uint64.");
  cudaFuncSetAttribute(_h_des_sm<uint64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)smem_bytes);
  _h_des_sm<uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
    XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint64_t>(),
    era_i32.data_ptr<int32_t>(), H.data_ptr<int64_t>(),
    K1, cols_32M, N, E, max_depth,
    warps_per_block, stride, nodes_total, sh_cap);

  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "h_des launch failed: ",
              cudaGetErrorString(cudaGetLastError()));
  return H;
}