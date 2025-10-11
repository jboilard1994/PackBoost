#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>


// H layout helper: [nfeatsets, nodes, 2, 32]
static inline __device__ unsigned long long int* Hptr(int64_t* H, int nodes,
                                         int feat, int node, int chan, int lane) {
  size_t idx = (((static_cast<size_t>(feat) * nodes + node) * 2 + chan) * 32u + lane);
  return (unsigned long long int*)(H + idx);
}

// ---------------- Kernel (templated on LF dtype) ----------------
template <typename LF_T>
__global__ void _h_sm(
    const uint32_t* __restrict__ XS,  // [nfeatsets, 32*M]
    const int32_t*  __restrict__ Y,   // [N]
    const LF_T*     __restrict__ LF,  // [nfeatsets, N] (u16/u32/u64)
    int64_t*      __restrict__ H,   // [nfeatsets, nodes, 2, 32] (int64)
    int nfeatsets,
    int cols_32M,     // XS.shape[1] == 32*M
    int N,            // Y.shape[0]
    int max_depth,
    int warps_per_block,
    int stride,       // tiles per warp along columns
    int nodes_total   // (1<<max_depth)-1
){
  const int feat_set   = blockIdx.x;
  const int block_warp = threadIdx.x >> 5;    // 0..(warps_per_block-1)
  const int lane       = threadIdx.x & 31;    // 0..31
  const int gwarp      = warps_per_block * blockIdx.y + block_warp;

  // Registers for depths 0..2
  int64_t hf0=0,  hw0=0;
  int64_t hf10=0, hf11=0, hw10=0, hw11=0;
  int64_t hf20=0, hf21=0, hf22=0, hf23=0;
  int64_t hw20=0, hw21=0, hw22=0, hw23=0;

  // Shared histogram for depths >=3 : shape [(2^D - 8), 2, 32] int32
  int n_ge3 = (1 << max_depth) - 8;
  if (n_ge3 < 1) n_ge3 = 1;

  extern __shared__ int shmem[];
  // Zero this lane’s column for both channels
  for (int i = 0; i < n_ge3; ++i) {
    shmem[(i * 2 + 0) * 32 + lane] = 0; // sum(y)
    shmem[(i * 2 + 1) * 32 + lane] = 0; // count
  }
  __syncthreads();

  const unsigned mask = __ballot_sync(__activemask(), true);

  // Each warp processes 'stride' tiles of 32 columns
  for (int j = 0; j < stride; ++j) {
    const int base = 32 * (stride * gwarp + j); // column start
    if (base < cols_32M) {
      // Load lane’s locals
      const int jj_lane = base + lane;

      int32_t  y_lane = 0;
      uint32_t l32    = 0;
      if (jj_lane < N) {
        y_lane = Y[jj_lane];
        // LF indexed [nfeatsets, N]
        LF_T lval = LF[static_cast<size_t>(feat_set) * static_cast<size_t>(N) + jj_lane];
        // Cast to 32-bit for warp shuffle (max_depth<=7 => safe)
        l32 = static_cast<uint32_t>(lval);
      }

      // XS row value for this lane (bits consumed across k)
      uint32_t xfd = XS[static_cast<size_t>(feat_set) * static_cast<size_t>(cols_32M)
                      + static_cast<size_t>(base + lane)];

      #pragma unroll
      for (int k = 0; k < 32; ++k) {
        const int jj_k = base + k;
        if (jj_k < N) {
          const int v = static_cast<int>(xfd & 1u);
          xfd >>= 1;

          const int32_t  yk = __shfl_sync(mask, y_lane, k);
          uint32_t       lk = __shfl_sync(mask, l32,    k);

          // d = 0
          hf0 += static_cast<int64_t>(v) * static_cast<int64_t>(yk);
          hw0 += v;

          // d = 1
          unsigned tk = lk & 1u;
          if (tk == 0u) { hf10 += v * (int64_t)yk; hw10 += v; }
          else          { hf11 += v * (int64_t)yk; hw11 += v; }
          lk >>= 1;

          // d = 2
          tk = lk & 3u;
          if      (tk == 0u) { hf20 += v * (int64_t)yk; hw20 += v; }
          else if (tk == 1u) { hf21 += v * (int64_t)yk; hw21 += v; }
          else if (tk == 2u) { hf22 += v * (int64_t)yk; hw22 += v; }
          else               { hf23 += v * (int64_t)yk; hw23 += v; }
          lk >>= 2;

          // d >= 3
          for (int d = 3; d < max_depth; ++d) {
            const unsigned to  = (1u << d) - 1u;
            const unsigned tkd = lk & to;
            lk >>= d;
            const int idx = static_cast<int>(to + tkd) - 7; // shift after first 7 nodes
            atomicAdd(&shmem[(idx * 2 + 0) * 32 + lane], v * yk);
            atomicAdd(&shmem[(idx * 2 + 1) * 32 + lane], v);
          }
        }
      }
    }
  }

  // Write back depths 0..2 (per lane)
  atomicAdd(Hptr(H, nodes_total, feat_set, 0, 0, lane), (unsigned long long int)hf0);
  atomicAdd(Hptr(H, nodes_total, feat_set, 0, 1, lane), (unsigned long long int)hw0);

  atomicAdd(Hptr(H, nodes_total, feat_set, 1, 0, lane), (unsigned long long int)hf10);
  atomicAdd(Hptr(H, nodes_total, feat_set, 2, 0, lane), (unsigned long long int)hf11);
  atomicAdd(Hptr(H, nodes_total, feat_set, 1, 1, lane), (unsigned long long int)hw10);
  atomicAdd(Hptr(H, nodes_total, feat_set, 2, 1, lane), (unsigned long long int)hw11);

  atomicAdd(Hptr(H, nodes_total, feat_set, 3, 0, lane), (unsigned long long int)hf20);
  atomicAdd(Hptr(H, nodes_total, feat_set, 4, 0, lane), (unsigned long long int)hf21);
  atomicAdd(Hptr(H, nodes_total, feat_set, 5, 0, lane), (unsigned long long int)hf22);
  atomicAdd(Hptr(H, nodes_total, feat_set, 6, 0, lane), (unsigned long long int)hf23);
  atomicAdd(Hptr(H, nodes_total, feat_set, 3, 1, lane), (unsigned long long int)hw20);
  atomicAdd(Hptr(H, nodes_total, feat_set, 4, 1, lane), (unsigned long long int)hw21);
  atomicAdd(Hptr(H, nodes_total, feat_set, 5, 1, lane), (unsigned long long int)hw22);
  atomicAdd(Hptr(H, nodes_total, feat_set, 6, 1, lane), (unsigned long long int)hw23);

  __syncthreads();

  // Drain shared histogram (d >= 3)
  const int rows_per_warp = (n_ge3 + warps_per_block - 1) / warps_per_block;
  for (int k = 0; k < rows_per_warp; ++k) {
    const int node = 7 + rows_per_warp * block_warp + k;
    if (node < nodes_total) {
      const int base = ((node - 7) * 2) * 32 + lane;
      const int64_t fsum = static_cast<int64_t>(shmem[base + 0]);
      const int64_t csum = static_cast<int64_t>(shmem[base + 32]);
      atomicAdd(Hptr(H, nodes_total, feat_set, node, 0, lane), (unsigned long long int)fsum);
      atomicAdd(Hptr(H, nodes_total, feat_set, node, 1, lane), (unsigned long long int)csum);
    }
  }
}

// ---------------- Host launcher that RETURNS H (Murky-style) ----------------

static inline int ceil_div_int(int a, int b){ return (a + b - 1) / b; }

// Murky A100 heuristic for warps-per-block
static inline int infer_hist_warps_per_block(int max_depth){
  // residual_sm_a100 = 15 - 2 - 6 - max_depth
  const int residual = 15 - 2 - 6 - max_depth; // = 7 - max_depth
  TORCH_CHECK(residual >= 0, "residual_sm_a100 < 0; max_depth too large for this variant");
  int lg2 = 4 - max(0, residual);  // 4,3,2,1,0 -> 16,8,4,2,1
  if (lg2 < 0) lg2 = 0;
  return 1 << lg2; // warps_per_block in {1,2,4,8,16}
}

// blocks_per_feat & stride per Murky
static inline void infer_grid_stride(
    int nfeatsets, int cols_32M,
    int warps_per_block,
    int& blocks_per_feat_out, int& stride_out)
{
  // 64*103 from Murky scheduling (A100 warp/slot heuristic)
  const int A100_SCHED = 64 * 103;
  int blocks_per_feat = ceil_div_int(A100_SCHED, warps_per_block);
  blocks_per_feat = ceil_div_int(blocks_per_feat, nfeatsets);
  if (blocks_per_feat < 1) blocks_per_feat = 1;

  const int total_warps = blocks_per_feat * warps_per_block;
  int stride = ceil_div_int(cols_32M, total_warps * 32);
  if (stride < 1) stride = 1;

  blocks_per_feat_out = blocks_per_feat;
  stride_out          = stride;
}

// API: returns H tensor; creates it inside like your h0_sm
// XS: [nfeatsets, 32*M] (torch.uint32)
// Y : [N] (torch.int32)
// LF: [nfeatsets, N] (torch.uint16/32/64)
// max_depth: <= 7 (this SMEM variant)
torch::Tensor h_sm(
    torch::Tensor XS,
    torch::Tensor Y,
    torch::Tensor LF,
    int max_depth)
{


  const int nfeatsets = static_cast<int>(XS.size(0));
  const int cols_32M  = static_cast<int>(XS.size(1));
  const int N         = static_cast<int>(Y.size(0));
  const int nodes_tot = (1 << max_depth) - 1;

  // Output H: [nfeatsets, (1<<D)-1, 2, 32] int64
  auto opts = XS.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous);
  auto H    = torch::zeros({XS.size(0), nodes_tot, 2, 32}, opts);

  // Infer launch params (Murky defaults)
  const int warps_per_block = infer_hist_warps_per_block(max_depth);
  int blocks_per_feat = 0, stride = 0;
  infer_grid_stride(nfeatsets, cols_32M, warps_per_block, blocks_per_feat, stride);

  dim3 grid(nfeatsets, blocks_per_feat, 1);
  dim3 block(warps_per_block * 32, 1, 1);

  // Dynamic shared memory for (2^D - 8, 2, 32) ints
  const int n_ge3 = std::max((1 << max_depth) - 8, 1);
  const size_t smem_bytes = static_cast<size_t>(n_ge3) * 2 * 32 * sizeof(int);

  auto* prop = at::cuda::getCurrentDeviceProperties();
  size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                 : (size_t)prop->sharedMemPerBlock;
  TORCH_CHECK(smem_bytes <= smem_cap,
              "Required dynamic shared memory (", smem_bytes,
              ") exceeds device limit (", smem_cap, ")");

  auto stream = at::cuda::getCurrentCUDAStream();

  // Dtypes: XS must be (u)int32; Y must be int32; LF in {uint16,uint32,uint64}
  TORCH_CHECK(Y.scalar_type() == torch::kInt32,
              "Y must be int32 (got ", Y.scalar_type(), ")");

  // We accept XS either as uint32 or int32 (bit-identical)
  TORCH_CHECK(
    XS.scalar_type() == torch::kUInt32 || XS.scalar_type() == torch::kInt32,
    "XS must be uint32/int32 (got ", XS.scalar_type(), ")"
  );
  const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());

  // Dispatch LF dtype
  const auto lf_dt = LF.scalar_type();

  if (lf_dt == torch::kUInt16) {
    cudaFuncSetAttribute(_h_sm<uint16_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
      _h_sm<uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr,
      Y.data_ptr<int32_t>(),
      LF.data_ptr<uint16_t>(),
      H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth,
      warps_per_block, stride, nodes_tot
    );
  } else if (lf_dt == torch::kUInt32) {
    cudaFuncSetAttribute(_h_sm<uint32_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    _h_sm<uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr,
      Y.data_ptr<int32_t>(),
      LF.data_ptr<uint32_t>(),
      H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth,
      warps_per_block, stride, nodes_tot
    );
  } else if (lf_dt == torch::kUInt64) {
    cudaFuncSetAttribute(_h_sm<uint64_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    _h_sm<uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr,
      Y.data_ptr<int32_t>(),
      static_cast<uint64_t*>(LF.data_ptr()),
      H.data_ptr<int64_t>(),
      nfeatsets, cols_32M, N, max_depth,
      warps_per_block, stride, nodes_tot
    );
  } else {
    TORCH_CHECK(false, "LF must be one of: uint16, uint32, uint64 (got ", lf_dt, ")");
  }

  return H;
}


#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFFu

// ---------------- Warp reduce helpers
__device__ __forceinline__ long long warp_reduce_sum_ll(long long v, unsigned mask = FULL_MASK) {
    #pragma unroll
    for (int ofs = WARP_SIZE >> 1; ofs; ofs >>= 1)
        v += __shfl_down_sync(mask, v, ofs);   // works for 64-bit on recent toolchains
    return v;
}

__device__ __forceinline__ int warp_reduce_sum_int(int v, unsigned mask = FULL_MASK) {
    #pragma unroll
    for (int ofs = WARP_SIZE >> 1; ofs; ofs >>= 1)
        v += __shfl_down_sync(mask, v, ofs);
    return v;
}

// ---------------- Kernel
template <typename LF_T>
__global__ void _h_sm_hierarchical(
    const uint32_t* __restrict__ XS,   // [nfeatsets, 32*M]
    const int32_t*  __restrict__ Y,    // [N]
    const LF_T*     __restrict__ LF,   // [nfeatsets * N] (packed leaf indices)
    int64_t*      __restrict__ H,    // [nfeatsets, nodes_tot, 2, 32] (int64)
    int nfeatsets,
    int cols_32M,                      // 32 * ceil(N/32)
    int N,
    int max_depth,
    int warps_per_block,
    int stride,
    int nodes_tot)
{
    const int feat_set = blockIdx.x;
    const int warp_id  = threadIdx.x / WARP_SIZE;       // 0..warps_per_block-1
    const int lane     = threadIdx.x & (WARP_SIZE - 1); // 0..31
    const int gwarp    = warps_per_block * blockIdx.y + warp_id;

    // ---------- Shared memory layout ----------
    extern __shared__ unsigned char smem[];
    // Part 1: histogram planes for d>=3 (2 channels * 32 lanes per node)
    const int n_ge3 = max(1, (1 << max_depth) - 8);              // nodes from index 7..nodes_tot-1
    int* sh_hist = reinterpret_cast<int*>(smem);                  // size = n_ge3 * 2 * 32
    const size_t hist_elems = static_cast<size_t>(n_ge3) * 2u * 32u;

    // Part 2: inter-warp reduce buffer for 14 values (d=0..2 => 7 nodes × 2 chans)
    long long* s_warp = reinterpret_cast<long long*>(sh_hist + hist_elems); // size = 14 * warps_per_block

    // Zero-out our lane's column for d>=3 (both channels).
    for (int i = lane; i < n_ge3 * 32; i += WARP_SIZE) {
        // i encodes (node_offset*32 + lane); write both channels
        sh_hist[(i * 2) + 0] = 0;
        sh_hist[(i * 2) + 1] = 0;
    }
    __syncthreads();

    // ---------- Per-thread accumulators for d=0..2 ----------
    long long hf0=0, hw0=0;
    long long hf10=0, hf11=0, hw10=0, hw11=0;
    long long hf20=0, hf21=0, hf22=0, hf23=0;
    long long hw20=0, hw21=0, hw22=0, hw23=0;

    // ---------- Phase 1: tile loop ----------
    for (int j = 0; j < stride; ++j) {
        const int base = WARP_SIZE * (stride * gwarp + j);
        if (base >= cols_32M) break;

        const int jj_lane = base + lane;
        const bool inb    = (jj_lane < N);

        int32_t  y_lane = 0;
        uint32_t l32    = 0;
        if (inb) {
            y_lane = Y[jj_lane];
            // Only lower 32 bits are used (max_depth ≤ 7), so cast is fine.
            uint64_t lv64 = static_cast<uint64_t>(LF[(size_t)feat_set * N + jj_lane]);
            l32 = static_cast<uint32_t>(lv64);
        }

        // Bits (rows) this lane contributes for this column
        const uint32_t xfd = XS[(size_t)feat_set * cols_32M + base + lane];

        // Active lane mask for this warp (only lanes with in-bounds columns)
        const unsigned warp_mask = __ballot_sync(FULL_MASK, inb);
        if (!warp_mask) continue;

        // Iterate only over set bits in xfd (sparser → faster)
        for (uint32_t m = xfd; m; m &= (m - 1)) {
            const int k = __ffs(m) - 1;              // row index 0..31 where bit is 1
            // Broadcast Y and leaf index from lane k
            const int32_t yk = __shfl_sync(warp_mask, y_lane, k);
            uint32_t      lk = __shfl_sync(warp_mask, l32,    k);

            // d = 0
            hf0 += (long long)yk;  // v==1 guaranteed (bit set), so add y, count 1
            hw0 += 1;

            // d = 1
            unsigned t = lk & 1u; lk >>= 1;
            if (t == 0u) { hf10 += yk; hw10 += 1; } else { hf11 += yk; hw11 += 1; }

            // d = 2
            t = lk & 3u; lk >>= 2;
            if      (t == 0u) { hf20 += yk; hw20 += 1; }
            else if (t == 1u) { hf21 += yk; hw21 += 1; }
            else if (t == 2u) { hf22 += yk; hw22 += 1; }
            else              { hf23 += yk; hw23 += 1; }

            // d >= 3 — warp-aggregated updates into shared histogram
            #pragma unroll
            for (int d = 3; d < max_depth; ++d) {
                const unsigned to   = (1u << d) - 1u;     // node-code mask of width d
                const int      baseN = (int)to - 7;       // node offset for idx 7.. (d>=3)
                const unsigned tkd  = lk & to;
                lk >>= d;
                const int b = baseN + (int)tkd;           // bucket 0..n_ge3-1

                // Only the lanes with a contribution participate in this group
                // (here, every lane in m-set has v==1 already; restrict to warp_mask for safety)
                const unsigned vmask = warp_mask;

                // Group lanes that target the same bucket b
                const unsigned gmask = __match_any_sync(vmask, b);
                const int leader = __ffs(gmask) - 1;

                // Count: number of contributing lanes in gmask
                int c_sum = 0;
                if (lane == leader) c_sum = __popc(gmask);

                // Sum: reduce yk over gmask
                int f_sum = yk;
                for (int ofs = 16; ofs; ofs >>= 1)
                    f_sum += __shfl_down_sync(gmask, f_sum, ofs);

                // Single atomic per group to our lane’s column to preserve the 32-lane contract
                if (lane == leader) {
                    // sh_hist layout: [n_ge3, 2, 32] as ((b*2 + chan)*32 + lane)
                    atomicAdd(&sh_hist[(b * 2 + 0) * 32 + lane], f_sum);
                    atomicAdd(&sh_hist[(b * 2 + 1) * 32 + lane], c_sum);
                }
            } // d>=3
        } // set bits in xfd
    } // tiles j

    // ---------- Phase 2: inter-warp reduction for d=0..2 ----------
    // Reduce within warp
    const long long hf0_r  = warp_reduce_sum_ll(hf0);
    const long long hw0_r  = warp_reduce_sum_ll(hw0);
    const long long hf10_r = warp_reduce_sum_ll(hf10);
    const long long hw10_r = warp_reduce_sum_ll(hw10);
    const long long hf11_r = warp_reduce_sum_ll(hf11);
    const long long hw11_r = warp_reduce_sum_ll(hw11);
    const long long hf20_r = warp_reduce_sum_ll(hf20);
    const long long hw20_r = warp_reduce_sum_ll(hw20);
    const long long hf21_r = warp_reduce_sum_ll(hf21);
    const long long hw21_r = warp_reduce_sum_ll(hw21);
    const long long hf22_r = warp_reduce_sum_ll(hf22);
    const long long hw22_r = warp_reduce_sum_ll(hw22);
    const long long hf23_r = warp_reduce_sum_ll(hf23);
    const long long hw23_r = warp_reduce_sum_ll(hw23);

    // Lane-0 of each warp writes its 14 values into shared buffer
    if (lane == 0) {
        // s_warp is [14, warps_per_block] laid out row-major
        s_warp[0  * warps_per_block + warp_id] = hf0_r;
        s_warp[1  * warps_per_block + warp_id] = hw0_r;
        s_warp[2  * warps_per_block + warp_id] = hf10_r;
        s_warp[3  * warps_per_block + warp_id] = hw10_r;
        s_warp[4  * warps_per_block + warp_id] = hf11_r;
        s_warp[5  * warps_per_block + warp_id] = hw11_r;
        s_warp[6  * warps_per_block + warp_id] = hf20_r;
        s_warp[7  * warps_per_block + warp_id] = hw20_r;
        s_warp[8  * warps_per_block + warp_id] = hf21_r;
        s_warp[9  * warps_per_block + warp_id] = hw21_r;
        s_warp[10 * warps_per_block + warp_id] = hf22_r;
        s_warp[11 * warps_per_block + warp_id] = hw22_r;
        s_warp[12 * warps_per_block + warp_id] = hf23_r;
        s_warp[13 * warps_per_block + warp_id] = hw23_r;
    }
    __syncthreads();  // <<< fence before using sh_hist and s_warp further >>>

    // One warp (warp 0) reduces across warps and writes global
    if (warp_id == 0) {
        long long final_val = 0;
        if (lane < 14) {
            for (int w = 0; w < warps_per_block; ++w)
                final_val += s_warp[lane * warps_per_block + w];
        }
        // Map lanes 0..13 to (node,chan)
        if (lane == 0)  atomicAdd(Hptr(H, nodes_tot, feat_set, 0, 0, lane), (unsigned long long)final_val);
        if (lane == 1)  atomicAdd(Hptr(H, nodes_tot, feat_set, 0, 1, lane), (unsigned long long)final_val);
        if (lane == 2)  atomicAdd(Hptr(H, nodes_tot, feat_set, 1, 0, lane), (unsigned long long)final_val);
        if (lane == 3)  atomicAdd(Hptr(H, nodes_tot, feat_set, 1, 1, lane), (unsigned long long)final_val);
        if (lane == 4)  atomicAdd(Hptr(H, nodes_tot, feat_set, 2, 0, lane), (unsigned long long)final_val);
        if (lane == 5)  atomicAdd(Hptr(H, nodes_tot, feat_set, 2, 1, lane), (unsigned long long)final_val);
        if (lane == 6)  atomicAdd(Hptr(H, nodes_tot, feat_set, 3, 0, lane), (unsigned long long)final_val);
        if (lane == 7)  atomicAdd(Hptr(H, nodes_tot, feat_set, 3, 1, lane), (unsigned long long)final_val);
        if (lane == 8)  atomicAdd(Hptr(H, nodes_tot, feat_set, 4, 0, lane), (unsigned long long)final_val);
        if (lane == 9)  atomicAdd(Hptr(H, nodes_tot, feat_set, 4, 1, lane), (unsigned long long)final_val);
        if (lane == 10) atomicAdd(Hptr(H, nodes_tot, feat_set, 5, 0, lane), (unsigned long long)final_val);
        if (lane == 11) atomicAdd(Hptr(H, nodes_tot, feat_set, 5, 1, lane), (unsigned long long)final_val);
        if (lane == 12) atomicAdd(Hptr(H, nodes_tot, feat_set, 6, 0, lane), (unsigned long long)final_val);
        if (lane == 13) atomicAdd(Hptr(H, nodes_tot, feat_set, 6, 1, lane), (unsigned long long)final_val);
    }

    // ---------- Phase 3: drain d>=3 rows  ----------
    // Assign rows (nodes) to warps evenly
    const int rows_per_warp = (n_ge3 + warps_per_block - 1) / warps_per_block;
    for (int r = 0; r < rows_per_warp; ++r) {
        const int node = 7 + rows_per_warp * warp_id + r;   // absolute node id
        const int b    = node - 7;                          // 0..n_ge3-1
        if (node >= nodes_tot) break;

        // Load our lane’s column for both channels
        const int fcol = sh_hist[(b * 2 + 0) * 32 + lane];
        const int ccol = sh_hist[(b * 2 + 1) * 32 + lane];

        // Reduce across lanes (within this warp)
        const int fsum = warp_reduce_sum_int(fcol);
        const int csum = warp_reduce_sum_int(ccol);

        // Lane 0 writes one atomic per channel (per node)
        if (lane == 0) {
            if (fsum) atomicAdd(Hptr(H, nodes_tot, feat_set, node, 0, lane), (unsigned long long)fsum);
            if (csum) atomicAdd(Hptr(H, nodes_tot, feat_set, node, 1, lane), (unsigned long long)csum);
        }
    }
}

// ---------------- Launcher (PyTorch extension style)
torch::Tensor h_sm_hierarchical(
    torch::Tensor XS,  // uint32 [nfeatsets, 32*M]
    torch::Tensor Y,   //  int32 [N]
    torch::Tensor LF,  // uint{16,32,64} [nfeatsets * N]
    int max_depth)
{
    

    const int nfeatsets = XS.size(0);
    const int cols_32M  = XS.size(1);
    const int N         = Y.size(0);
    const int nodes_tot = (1 << max_depth) - 1;

    auto H = torch::zeros({nfeatsets, nodes_tot, 2, 32}, XS.options().dtype(torch::kLong));

    // Grid/block inference (use your existing helpers)
    const int warps_per_block = infer_hist_warps_per_block(max_depth); // e.g., {8,12,16} on A100
    int blocks_per_feat = 0, stride = 0;
    infer_grid_stride(nfeatsets, cols_32M, warps_per_block, blocks_per_feat, stride);

    dim3 grid(nfeatsets, blocks_per_feat, 1);
    dim3 block(warps_per_block * WARP_SIZE, 1, 1);

    // Dynamic shared memory size
    const int n_ge3 = std::max(1, (1 << max_depth) - 8);
    const size_t smem_hist_bytes   = static_cast<size_t>(n_ge3) * 2u * 32u * sizeof(int);
    const size_t smem_reduce_bytes = static_cast<size_t>(14) * static_cast<size_t>(warps_per_block) * sizeof(long long);
    const size_t smem_bytes        = smem_hist_bytes + smem_reduce_bytes;

    // Device limit guard
    auto* prop = at::cuda::getCurrentDeviceProperties();
    const size_t smem_cap =
        prop->sharedMemPerBlockOptin ? static_cast<size_t>(prop->sharedMemPerBlockOptin)
                                     : static_cast<size_t>(prop->sharedMemPerBlock);
    TORCH_CHECK(smem_bytes <= smem_cap,
        "Dynamic shared memory (", smem_bytes, " B) exceeds device limit (", smem_cap, " B).");

    auto stream = at::cuda::getCurrentCUDAStream();
    const uint32_t* XS_ptr = XS.data_ptr<uint32_t>();
    int64_t*      H_ptr  = H.data_ptr<int64_t>();

    // Dispatch by LF dtype
    const auto lf_dt = LF.scalar_type();
    if (lf_dt == torch::kUInt16) {
        cudaFuncSetAttribute(_h_sm_hierarchical<uint16_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h_sm_hierarchical<uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
            XS_ptr, Y.data_ptr<int32_t>(), LF.data_ptr<uint16_t>(), H_ptr,
            nfeatsets, cols_32M, N, max_depth, warps_per_block, stride, nodes_tot);
    } else if (lf_dt == torch::kUInt32) {
        cudaFuncSetAttribute(_h_sm_hierarchical<uint32_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h_sm_hierarchical<uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
            XS_ptr, Y.data_ptr<int32_t>(), LF.data_ptr<uint32_t>(), H_ptr,
            nfeatsets, cols_32M, N, max_depth, warps_per_block, stride, nodes_tot);
    } else if (lf_dt == torch::kUInt64) {
        cudaFuncSetAttribute(_h_sm_hierarchical<uint64_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        _h_sm_hierarchical<uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
            XS_ptr, Y.data_ptr<int32_t>(), LF.data_ptr<uint64_t>(), H_ptr,
            nfeatsets, cols_32M, N, max_depth, warps_per_block, stride, nodes_tot);
    } else {
        TORCH_CHECK(false, "Unsupported LF dtype: ", lf_dt);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return H;
}