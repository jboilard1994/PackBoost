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

__device__ __forceinline__ int idx_s02(int warp, int node, int ch, int lane) {
  // [warps, 7 nodes, 2 chans, 32 lanes]
  return (((warp * 7 + node) * 2 + ch) * 32 + lane);
}

__device__ __forceinline__ int idx_sGE3(int warp, int row, int ch, int lane, int rows_ge3) {
  // [warps, rows_ge3, 2 chans, 32 lanes]
  return (((warp * rows_ge3 + row) * 2 + ch) * 32 + lane);
}
  
  // ---------------- Kernel ----------------
template <typename LF_T>
__global__ void _h_multiwarp_generic(
      const uint32_t* __restrict__ XS,  // [nfeatsets, 32*M]
      const int32_t*  __restrict__ Y,   // [N]
      const LF_T*     __restrict__ LF,  // [nfeatsets, N] (u16/u32/u64)
      int64_t*        __restrict__ H,   // [nfeatsets, nodes, 2, 32] (int64)
      int nfeatsets,
      int cols_32M,
      int N,
      int max_depth,                    // supports 7..9 (use uint64 LF for D=9)
      int warps_per_block,
      int stride,                       // tiles per warp along columns
      int nodes_total                   // (1<<max_depth)-1
  ){
    const int feat_set = blockIdx.x;
    const int wid      = threadIdx.x >> 5;     // warp id in block
    const int lane     = threadIdx.x & 31;     // lane id
    const int gwarp    = warps_per_block * blockIdx.y + wid;
  
    // d<=2 accumulators (nodes 0..6)
    int64_t hfs02[7];  // sum
    int64_t hws02[7];  // count
    #pragma unroll
    for (int i=0;i<7;++i){ hfs02[i]=0; hws02[i]=0; }
  
    // Shared memory layout
    int rows_ge3 = (1 << max_depth) - 8;
    if (rows_ge3 < 1) rows_ge3 = 1;
  
    extern __shared__ int smem[]; // int32
    const int s02_perwarp_ints  = 7 * 2 * 32;
    const int sGE3_perwarp_ints = rows_ge3 * 2 * 32;
  
    int* s02  = smem;                                       // [W,7,2,32]
    int* sGE3 = s02 + warps_per_block * s02_perwarp_ints;   // [W,rows_ge3,2,32]
  
    // Zero this warp's slices
    #pragma unroll
    for (int n = 0; n < 7; ++n) {
      s02[idx_s02(wid, n, 0, lane)] = 0;
      s02[idx_s02(wid, n, 1, lane)] = 0;
    }
    for (int r = lane; r < rows_ge3 * 32; r += 32) {
      const int row  = r / 32;
      const int lcol = r % 32;
      sGE3[idx_sGE3(wid, row, 0, lcol, rows_ge3)] = 0;
      sGE3[idx_sGE3(wid, row, 1, lcol, rows_ge3)] = 0;
    }
    __syncthreads();
  
    const unsigned full_mask = __activemask();
  
    // Process tiles owned by this warp
    for (int j = 0; j < stride; ++j) {
      const int base = 32 * (stride * gwarp + j);
      if (base >= cols_32M) break;
  
      // Lane-local loads (safe-load XS)
      const int jj_lane = base + lane;
      int32_t  y_lane = 0;
      uint32_t l_lo   = 0;
      uint32_t l_hi   = 0;
      if (jj_lane < N) {
        y_lane = Y[jj_lane];
        const LF_T lval = LF[(size_t)feat_set * (size_t)N + jj_lane];
        if constexpr (sizeof(LF_T) >= 8) {
          const uint64_t v = static_cast<uint64_t>(lval);
          l_lo = (uint32_t)(v & 0xFFFFFFFFu);
          l_hi = (uint32_t)(v >> 32);
        } else {
          l_lo = (uint32_t)static_cast<uint64_t>(lval);
          l_hi = 0u;
        }
      }
      uint32_t xbits = 0;
      if (base + lane < cols_32M) {
        xbits = XS[(size_t)feat_set * (size_t)cols_32M + (base + lane)];
      }
  
      // Iterate columns in this 32-wide tile
      for (int k = 0; k < 32; ++k) {
        const int jj_k = base + k;
  
        // Per-k participation mask, as requested
        const unsigned mask_k = __ballot_sync(full_mask, jj_k < N);
        if (!mask_k) break;  // all lanes agree there is no more work in this tile
  
        // --- Do shuffles UNCONDITIONALLY (avoid deadlock) ---
        const int32_t  yk   = __shfl_sync(mask_k, y_lane, k);
        const uint32_t loK  = __shfl_sync(mask_k, l_lo,   k);
        const uint32_t hiK  = __shfl_sync(mask_k, l_hi,   k);
        uint64_t lk = (uint64_t)loK | ((uint64_t)hiK << 32);
  
        // Now gate by v
        const int v = (int)((xbits >> k) & 1u);
        if (!v) continue;
  
        // d=0 (node 0)
        hfs02[0] += (int64_t)yk;
        hws02[0] += 1;
  
        // d=1..max_depth-1 (generic variable-length decode)
        #pragma unroll
        for (int d = 1; d < 10; ++d) {     // unroll upper bound; guard at runtime
          if (d >= max_depth) break;
          const uint64_t maskd = (1ull << d) - 1ull;
          const uint64_t tkd   = lk & maskd;
          lk >>= d;
  
          const int node = ((1 << d) - 1) + (int)tkd;
          if (node < 7) {
            hfs02[node] += (int64_t)yk;
            hws02[node] += 1;
          } else {
            const int row = node - 7;  // 0..rows_ge3-1
            sGE3[idx_sGE3(wid, row, 0, lane, rows_ge3)] += (int)yk;
            sGE3[idx_sGE3(wid, row, 1, lane, rows_ge3)] += 1;
          }
        }
      } // k
    } // j
  
    // Spill d<=2 regs to s02 once
    #pragma unroll
    for (int n = 0; n < 7; ++n) {
      s02[idx_s02(wid, n, 0, lane)] += (int)hfs02[n];
      s02[idx_s02(wid, n, 1, lane)] += (int)hws02[n];
    }
    __syncthreads();
  
    // Inter-warp butterfly merge (O(log W)) on s02 + sGE3
    for (int s = 0; (1 << s) < warps_per_block; ++s) {
      const int partner = wid ^ (1 << s);
      __syncthreads();
      if ((wid & ((1 << (s + 1)) - 1)) == 0 && partner < warps_per_block) {
        // merge s02
        #pragma unroll
        for (int n = 0; n < 7; ++n) {
          const int self_f = idx_s02(wid,     n, 0, lane);
          const int self_c = idx_s02(wid,     n, 1, lane);
          const int part_f = idx_s02(partner, n, 0, lane);
          const int part_c = idx_s02(partner, n, 1, lane);
          s02[self_f] += s02[part_f];
          s02[self_c] += s02[part_c];
        }
        // merge sGE3 (stride lanes across rows)
        for (int r = lane; r < rows_ge3 * 32; r += 32) {
          const int row  = r / 32;
          const int lcol = r % 32;
          const int self_f = idx_sGE3(wid,     row, 0, lcol, rows_ge3);
          const int self_c = idx_sGE3(wid,     row, 1, lcol, rows_ge3);
          const int part_f = idx_sGE3(partner, row, 0, lcol, rows_ge3);
          const int part_c = idx_sGE3(partner, row, 1, lcol, rows_ge3);
          sGE3[self_f] += sGE3[part_f];
          sGE3[self_c] += sGE3[part_c];
        }
      }
    }
    __syncthreads();
  
    // Drain once per block (warp 0), preserving lane axis
    if ((threadIdx.x >> 5) == 0) {
      // nodes 0..6
      #pragma unroll
      for (int n = 0; n < 7; ++n) {
        const int base = idx_s02(0, n, 0, lane);
        const int64_t fsum = (int64_t)s02[base];
        const int64_t csum = (int64_t)s02[base + 32];
        if (fsum) atomicAdd(Hptr(H, nodes_total, feat_set, n, 0, lane), (unsigned long long)fsum);
        if (csum) atomicAdd(Hptr(H, nodes_total, feat_set, n, 1, lane), (unsigned long long)csum);
      }
      // d>=3 (nodes 7..)
      const int rows_ge3_local = rows_ge3;
      for (int r = lane; r < rows_ge3_local * 32; r += 32) {
        const int row  = r / 32;
        const int lcol = r % 32;
        const int node = 7 + row;
        const int64_t fsum = (int64_t)sGE3[idx_sGE3(0, row, 0, lcol, rows_ge3_local)];
        const int64_t csum = (int64_t)sGE3[idx_sGE3(0, row, 1, lcol, rows_ge3_local)];
        if (fsum) atomicAdd(Hptr(H, nodes_total, feat_set, node, 0, lcol), (unsigned long long)fsum);
        if (csum) atomicAdd(Hptr(H, nodes_total, feat_set, node, 1, lcol), (unsigned long long)csum);
      }
    }
}

// ---------------- Launcher ----------------

// Murky-like heuristic (we'll cap by SMEM)
static inline int heuristic_warps_per_block(int max_depth){
  const int residual = 15 - 2 - 6 - max_depth; // 7 - max_depth
  int lg2 = 4 - max(0, residual);              // -> {16,8,4,2,1}
  if (lg2 < 0) lg2 = 0;
  return 1 << lg2;
}


template <typename LF_T>
static void launch_h_multiwarp_generic_typed(
    const uint32_t* XS_ptr, const int32_t* Y_ptr, const LF_T* LF_ptr,
    int64_t* H_ptr,
    int nfeatsets, int cols_32M, int N, int max_depth,
    cudaStream_t stream)
{
  const int rows_ge3 = max((1 << max_depth) - 8, 1);

  auto* prop = at::cuda::getCurrentDeviceProperties();
  const size_t smem_cap = prop->sharedMemPerBlockOptin
                            ? (size_t)prop->sharedMemPerBlockOptin
                            : (size_t)prop->sharedMemPerBlock;

  const size_t s02_perwarp_bytes  = (size_t)7 * 2 * 32 * sizeof(int);
  const size_t sGE3_perwarp_bytes = (size_t)rows_ge3 * 2 * 32 * sizeof(int);
  const size_t perwarp_bytes      = s02_perwarp_bytes + sGE3_perwarp_bytes;

  // Cap warps by SMEM (no row-tiling in this clean version)
  const int max_warps_by_smem = (int)max((size_t)1, smem_cap / perwarp_bytes);
  int warps_per_block = max(1, min(heuristic_warps_per_block(max_depth), max_warps_by_smem));

  int blocks_per_feat = 0, stride = 0;
  infer_grid_stride(nfeatsets, cols_32M, warps_per_block, blocks_per_feat, stride);

  const dim3 grid(nfeatsets, blocks_per_feat, 1);
  const dim3 block(warps_per_block * 32, 1, 1);
  const size_t smem_bytes = (size_t)warps_per_block * perwarp_bytes;

  cudaFuncSetAttribute(_h_multiwarp_generic<LF_T>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)smem_bytes);

  _h_multiwarp_generic<LF_T><<<grid, block, smem_bytes, stream>>>(
    XS_ptr, Y_ptr, LF_ptr, H_ptr,
    nfeatsets, cols_32M, N, max_depth,
    warps_per_block, stride, (1 << max_depth) - 1
  );
}

torch::Tensor h_multiwarp_generic(
    torch::Tensor XS,  // [nfeatsets, 32*M] (uint32/int32)
    torch::Tensor Y,   // [N] (int32)
    torch::Tensor LF,  // [nfeatsets, N] (uint16/32/64) — use uint64 for D=9
    int max_depth)
{
  TORCH_CHECK(Y.scalar_type() == torch::kInt32, "Y must be int32");
  TORCH_CHECK(
    XS.scalar_type() == torch::kUInt32 || XS.scalar_type() == torch::kInt32,
    "XS must be uint32/int32"
  );

  const int bits_needed = (max_depth - 1) * max_depth / 2; // d=1..D-1
  const auto lf_dt = LF.scalar_type();
  TORCH_CHECK(
    (bits_needed <= 16 && (lf_dt == torch::kUInt16 || lf_dt == torch::kUInt32 || lf_dt == torch::kUInt64)) ||
    (bits_needed <= 32 && (lf_dt == torch::kUInt32 || lf_dt == torch::kUInt64)) ||
    (bits_needed <= 64 && (lf_dt == torch::kUInt64)),
    "LF dtype too small for max_depth=", max_depth, " (need ", bits_needed, " bits). "
    "Use uint64 for depth 9."
  );

  const int nfeatsets = (int)XS.size(0);
  const int cols_32M  = (int)XS.size(1);
  const int nodes_tot = (1 << max_depth) - 1;

  auto H = torch::zeros({XS.size(0), nodes_tot, 2, 32},
                        XS.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous));

  const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();

  if (lf_dt == torch::kUInt16) {
    launch_h_multiwarp_generic_typed<uint16_t>(
      XS_ptr, Y.data_ptr<int32_t>(), LF.data_ptr<uint16_t>(),
      H.data_ptr<int64_t>(), nfeatsets, cols_32M, (int)Y.size(0), max_depth, stream.stream());
  } else if (lf_dt == torch::kUInt32) {
    launch_h_multiwarp_generic_typed<uint32_t>(
      XS_ptr, Y.data_ptr<int32_t>(), LF.data_ptr<uint32_t>(),
      H.data_ptr<int64_t>(), nfeatsets, cols_32M, (int)Y.size(0), max_depth, stream.stream());
  } else if (lf_dt == torch::kUInt64) {
    launch_h_multiwarp_generic_typed<uint64_t>(
      XS_ptr, Y.data_ptr<int32_t>(), reinterpret_cast<const uint64_t*>(LF.data_ptr()),
      H.data_ptr<int64_t>(), nfeatsets, cols_32M, (int)Y.size(0), max_depth, stream.stream());
  } else {
    TORCH_CHECK(false, "LF must be one of: uint16, uint32, uint64");
  }

  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "h_multiwarp_generic launch failed: ", cudaGetErrorString(cudaGetLastError()));
  return H;
}


