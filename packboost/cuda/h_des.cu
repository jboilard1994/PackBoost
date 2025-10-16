#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// -------- helpers (packed (sum,count) in one 64) --------
static __device__ __forceinline__ uint64_t pack_sc(int sum32, int cnt32) {
    return ( (uint64_t)(uint32_t)cnt32 << 32 ) | (uint64_t)(uint32_t)sum32;
}
static __device__ __forceinline__ uint64_t add_pack(uint64_t a, uint64_t b) {
    int sa = (int)(uint32_t)a;
    int ca = (int)(uint32_t)(a >> 32);
    int sb = (int)(uint32_t)b;
    int cb = (int)(uint32_t)(b >> 32);
    return pack_sc(sa + sb, ca + cb);
}

// H layout: [nfeatsets, nodes, E, 2, 32] int64
static inline __device__ unsigned long long* Hptr_des(
    int64_t* H,
    int nodes, int E,
    int feat, int node, int era, int chan, int lane)
{
  const size_t idx =
      (((((size_t)feat * (size_t)nodes + (size_t)node) * (size_t)E + (size_t)era) * 2u
        + (size_t)chan) * 32u + (size_t)lane);
  return (unsigned long long*)(H + idx);
}

// -------- kernel: per-ERA H for DES (packed low-depth accumulation) --------
template <typename LF_T>
__global__ void _h_des_packed(
    const uint32_t* __restrict__ XS,       // [nfeatsets, N]
    const int16_t*  __restrict__ Y,        // [N]
    const LF_T*     __restrict__ LF,       // [nfeatsets, N] (u16/u32/u64)
    const int32_t*  __restrict__ era_ends, // [E] exclusive ends
    int64_t*        __restrict__ H,        // [nfeatsets, nodes, E, 2, 32]
    int nfeatsets, int N, int E, int max_depth,
    int stride)
{
  const int feat_set = blockIdx.x;
  const int bi       = blockIdx.y;
  const int lane     = threadIdx.x; // 0..31
  if (feat_set >= nfeatsets || blockDim.x != 32 || lane >= 32) return;

  const unsigned mask = __ballot_sync(__activemask(), true);
  const int nodes_total = (1 << max_depth) - 1;

  // shared small scratch for era segmentation within a 32-sample tile
  __shared__ int s_era_k[32];
  __shared__ int s_seg_start[33];
  __shared__ int s_seg_end  [33];
  __shared__ int s_seg_era  [33];
  __shared__ int s_nsegs;

  // process 'stride' tiles of 32 samples each
  for (int i = 0; i < stride; ++i) {
    const int tile_base = 32 * (stride * bi + i);
    if (tile_base >= N) break;

    // lane-local loads for this tile
    const int jj_lane = tile_base + lane;
    int32_t  y_lane = 0;
    uint32_t l32    = 0;

    if (jj_lane < N) {
      y_lane = (int32_t)Y[jj_lane];
      l32    = (uint32_t)((LF_T)LF[(size_t)feat_set * (size_t)N + jj_lane]);
    }

    const uint32_t xbits = XS[(size_t)feat_set * (size_t)N + (size_t)jj_lane];

    // valid k in this tile
    const int rem = N - tile_base;
    const int k_valid = (rem >= 32 ? 32 : (rem > 0 ? rem : 0));

    // build per-k era id and compress into segments (lane0)
    if (lane == 0) {
      int e = 0;
      while (e < E && era_ends[e] <= tile_base) ++e;
      for (int k = 0; k < 32; ++k) {
        const int idx = tile_base + k;
        while (e < E && era_ends[e] <= idx) ++e;
        s_era_k[k] = (e < E ? e : (E > 0 ? E - 1 : 0));
      }
      int s = 0;
      if (k_valid > 0) {
        int cur_era = s_era_k[0];
        int start   = 0;
        for (int k = 1; k < k_valid; ++k) {
          const int ek = s_era_k[k];
          if (ek != cur_era) {
            s_seg_start[s] = start;
            s_seg_end[s]   = k;
            s_seg_era[s]   = cur_era;
            ++s;
            cur_era = ek;
            start   = k;
          }
        }
        s_seg_start[s] = start;
        s_seg_end[s]   = k_valid;
        s_seg_era[s]   = cur_era;
        s_nsegs = s + 1;
      } else {
        s_nsegs = 0;
      }
    }
    __syncwarp();

    // iterate segments (few in practice)
    for (int seg = 0; seg < s_nsegs; ++seg) {
      const int era = s_seg_era[seg];
      const int k0  = s_seg_start[seg];
      const int k1  = s_seg_end[seg];

      // low-depth packed accumulators (per lane, per era-segment)
      uint64_t p0   = 0;      // node 0
      uint64_t p10  = 0, p11  = 0;               // nodes 1..2
      uint64_t p20  = 0, p21  = 0, p22  = 0, p23 = 0; // nodes 3..6

      // scan k in segment
      #pragma unroll
      for (int k = 0; k < 32; ++k) {
        if (k < k0 || k >= k1) continue;

        // bit for this sample (don’t destroy xbits)
        const int v = (int)((xbits >> k) & 1u);
        if (!v) continue;

        // broadcast y and leafcode for sample k
        const int32_t yk = __shfl_sync(mask, y_lane, k);
        uint32_t lk     = __shfl_sync(mask, l32,    k);

        // d = 0 (node 0)
        p0 = add_pack(p0, pack_sc((int)yk, 1));

        // d = 1 -> nodes 1..2
        unsigned tk = (lk & 1u); lk >>= 1;
        if (tk == 0u) p10 = add_pack(p10, pack_sc((int)yk, 1));
        else          p11 = add_pack(p11, pack_sc((int)yk, 1));

        // d = 2 -> nodes 3..6
        tk = (lk & 3u); lk >>= 2;
        if      (tk == 0u) p20 = add_pack(p20, pack_sc((int)yk, 1));
        else if (tk == 1u) p21 = add_pack(p21, pack_sc((int)yk, 1));
        else if (tk == 2u) p22 = add_pack(p22, pack_sc((int)yk, 1));
        else               p23 = add_pack(p23, pack_sc((int)yk, 1));

        // d >= 3 -> direct atomics (rare shared contention; era split keeps it small)
        #pragma unroll
        for (int d = 3; d < 32; ++d) {
          if (d >= max_depth) break;
          const unsigned to  = (1u << d) - 1u;
          const unsigned tkd = (lk & to); lk >>= d;
          const int node = (int)to + (int)tkd;
          unsigned long long* ps = Hptr_des(H, (1<<max_depth)-1, E, feat_set, node, era, 0, lane);
          unsigned long long* pc = Hptr_des(H, (1<<max_depth)-1, E, feat_set, node, era, 1, lane);
          atomicAdd(ps, (unsigned long long)( (long long)yk ));
          atomicAdd(pc, (unsigned long long)1ull);
        }
      } // k

      // flush low-depth packs to global (unpack -> two atomics)
      auto flush_node = [&](int node, uint64_t p) {
        const long long sum_ll = (long long)(int)(uint32_t)p;
        const long long cnt_ll = (long long)(int)(uint32_t)(p >> 32);
        if (sum_ll | cnt_ll) {
          unsigned long long* ps = Hptr_des(H, (1<<max_depth)-1, E, feat_set, node, era, 0, lane);
          unsigned long long* pc = Hptr_des(H, (1<<max_depth)-1, E, feat_set, node, era, 1, lane);
          if (sum_ll) atomicAdd(ps, (unsigned long long)sum_ll);
          if (cnt_ll) atomicAdd(pc, (unsigned long long)cnt_ll);
        }
      };
      flush_node(0, p0);
      flush_node(1, p10); flush_node(2, p11);
      flush_node(3, p20); flush_node(4, p21);
      flush_node(5, p22); flush_node(6, p23);
    } // segments
  } // tiles
}

// -------- host launcher --------
static inline int ceil_div_int(int a, int b){ return (a + b - 1) / b; }

torch::Tensor h_des(
    torch::Tensor XS,        // [nfeatsets, N] uint32/int32
    torch::Tensor Y,         // [N]            int16
    torch::Tensor LF,        // [nfeatsets, N] uint16/uint32/uint64
    torch::Tensor era_ends,  // [E]            int32 (exclusive ends)
    int max_depth)
{
  TORCH_CHECK(XS.is_cuda() && Y.is_cuda() && LF.is_cuda() && era_ends.is_cuda(),
              "XS, Y, LF, era_ends must be CUDA tensors.");
  TORCH_CHECK(XS.dim()==2 && Y.dim()==1 && LF.dim()==2, "XS:[K1,N], Y:[N], LF:[K1,N].");
  TORCH_CHECK(era_ends.dim()==1, "era_ends must be [E].");
  TORCH_CHECK(Y.scalar_type()==torch::kInt16, "Y must be int16.");
  TORCH_CHECK(XS.scalar_type()==torch::kUInt32 || XS.scalar_type()==torch::kInt32,
              "XS must be uint32/int32.");
  TORCH_CHECK(era_ends.scalar_type()==torch::kInt32, "era_ends must be int32.");
  TORCH_CHECK(max_depth>0 && max_depth<=32, "max_depth must be in 1..32.");

  const int nfeatsets = (int)XS.size(0);
  const int N         = (int)XS.size(1);
  TORCH_CHECK((int)LF.size(0)==nfeatsets && (int)LF.size(1)==N, "LF shape mismatch with XS.");
  const int E         = (int)era_ends.size(0);
  TORCH_CHECK(E >= 1, "era_ends must contain at least one era (e.g., [N] for single era).");

  const int nodes_total = (1 << max_depth) - 1;

  auto H = torch::zeros({ (long long)nfeatsets,
                          (long long)nodes_total,
                          (long long)E,
                          2LL, 32LL },
                        XS.options().dtype(torch::kLong)
                           .memory_format(c10::MemoryFormat::Contiguous));

  // schedule: many y-tiles, one warp per block
  const int strides = 1024;
  const int stride  = ceil_div_int(N, strides * 32);
  dim3 grid(nfeatsets, strides, 1);
  dim3 block(32, 1, 1);
  auto stream = at::cuda::getCurrentCUDAStream();

  const auto lf_dt = LF.scalar_type();
  const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());

  if (lf_dt == torch::kUInt16) {
    _h_des_packed<uint16_t><<<grid, block, 0, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint16_t>(),
      era_ends.data_ptr<int32_t>(), H.data_ptr<int64_t>(),
      nfeatsets, N, E, max_depth, stride);
  } else if (lf_dt == torch::kUInt32) {
    _h_des_packed<uint32_t><<<grid, block, 0, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint32_t>(),
      era_ends.data_ptr<int32_t>(), H.data_ptr<int64_t>(),
      nfeatsets, N, E, max_depth, stride);
  } else if (lf_dt == torch::kUInt64) {
    _h_des_packed<uint64_t><<<grid, block, 0, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int16_t>(), LF.data_ptr<uint64_t>(),
      era_ends.data_ptr<int32_t>(), H.data_ptr<int64_t>(),
      nfeatsets, N, E, max_depth, stride);
  } else {
    TORCH_CHECK(false, "LF must be one of: uint16, uint32, uint64.");
  }

  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "h_des launch failed: ",
              cudaGetErrorString(cudaGetLastError()));
  return H;
}
