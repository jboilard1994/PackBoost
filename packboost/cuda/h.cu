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

// H layout helper remains the same
static inline __device__ unsigned long long int* Hptr(long long* H, int nodes,
                                         int feat, int node, int chan, int lane) {
  size_t idx = (((static_cast<size_t>(feat) * nodes + node) * 2 + chan) * 32u + lane);
  return (unsigned long long int*)(H + idx);
}

// ---------------- Reduction Helpers ----------------
// Intra-warp butterfly reduction for a 64-bit long long
__device__ __forceinline__ long long warp_reduce_sum_ll(long long val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, offset, WARP_SIZE);
    }
    return val;
}

// Intra-warp butterfly reduction for a 32-bit int
__device__ __forceinline__ int warp_reduce_sum_int(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, offset, WARP_SIZE);
    }
    return val;
}


// ---------------- Hierarchical Kernel ----------------
template <typename LF_T>
__global__ void _h_sm_hierarchical(
    const uint32_t* __restrict__ XS,
    const int32_t* __restrict__ Y,
    const LF_T* __restrict__ LF,
    long long* __restrict__ H,
    int nfeatsets,
    int cols_32M,
    int N,
    int max_depth,
    int warps_per_block,
    int stride,
    int nodes_total
){
  const int feat_set   = blockIdx.x;
  const int warp_id    = threadIdx.x / WARP_SIZE; // 0..(warps_per_block-1)
  const int lane       = threadIdx.x % WARP_SIZE; // 0..31
  const int gwarp      = warps_per_block * blockIdx.y + warp_id;

  // --- Phase 1: Per-thread register accumulation (Unchanged) ---
  long long hf0=0,  hw0=0;
  long long hf10=0, hf11=0, hw10=0, hw11=0;
  long long hf20=0, hf21=0, hf22=0, hf23=0;
  long long hw20=0, hw21=0, hw22=0, hw23=0;

  int n_ge3 = (1 << max_depth) - 8;
  if (n_ge3 < 1) n_ge3 = 1;

  // Shared memory is now larger to hold inter-warp results
  extern __shared__ int shmem[];
  int* shmem_hist = shmem; // For d>=3 histogram
  long long (*s_warp_results)[16] = (long long (*)[16])&shmem[n_ge3 * 2 * 32];

  for (int i = 0; i < n_ge3; ++i) {
    shmem_hist[(i * 2 + 0) * 32 + lane] = 0;
    shmem_hist[(i * 2 + 1) * 32 + lane] = 0;
  }
  __syncthreads();

  for (int j = 0; j < stride; ++j) {
    const int base = WARP_SIZE * (stride * gwarp + j);
    if (base >= cols_32M) break;

    const int jj_lane = base + lane;
    int32_t  y_lane = 0;
    uint32_t l32    = 0;
    if (jj_lane < N) {
      y_lane = Y[jj_lane];
      LF_T lval = LF[(size_t)feat_set * N + jj_lane];
      l32 = static_cast<uint32_t>(lval);
    }

    uint32_t xfd = XS[(size_t)feat_set * cols_32M + base + lane];
    
    const unsigned mask = __ballot_sync(__activemask(), base + lane < N);
    
    for (int k = 0; k < WARP_SIZE; ++k) {
      if (!((mask >> k) & 1u)) continue;
      const int v = (int)(__shfl_sync(FULL_MASK, xfd, k) & 1u);
      
      const int32_t  yk = __shfl_sync(mask, y_lane, k);
      uint32_t       lk = __shfl_sync(mask, l32, k);

      // d = 0
      hf0 += (long long)v * yk; hw0 += v;
      // d = 1
      unsigned tk = lk & 1u;
      if (tk == 0u) { hf10 += v * (long long)yk; hw10 += v; } else { hf11 += v * (long long)yk; hw11 += v; }
      lk >>= 1;
      // d = 2
      tk = lk & 3u;
      if      (tk == 0u) { hf20 += v * (long long)yk; hw20 += v; }
      else if (tk == 1u) { hf21 += v * (long long)yk; hw21 += v; }
      else if (tk == 2u) { hf22 += v * (long long)yk; hw22 += v; }
      else               { hf23 += v * (long long)yk; hw23 += v; }
      lk >>= 2;
      // d >= 3
      for (int d = 3; d < max_depth; ++d) {
        const unsigned to  = (1u << d) - 1u;
        const unsigned tkd = lk & to;
        lk >>= d;
        const int idx = (int)(to + tkd) - 7;
        atomicAdd(&shmem_hist[(idx * 2 + 0) * 32 + lane], v * yk);
        atomicAdd(&shmem_hist[(idx * 2 + 1) * 32 + lane], v);
      }
    }
  }

  // --- Phase 2: Hierarchical Reduction for Depths 0-2 ---

  // Step 2.1: Intra-warp reduction. Result valid in lane 0.
  hf0 = warp_reduce_sum_ll(hf0);   hw0 = warp_reduce_sum_ll(hw0);
  hf10 = warp_reduce_sum_ll(hf10); hw10 = warp_reduce_sum_ll(hw10);
  hf11 = warp_reduce_sum_ll(hf11); hw11 = warp_reduce_sum_ll(hw11);
  hf20 = warp_reduce_sum_ll(hf20); hw20 = warp_reduce_sum_ll(hw20);
  hf21 = warp_reduce_sum_ll(hf21); hw21 = warp_reduce_sum_ll(hw21);
  hf22 = warp_reduce_sum_ll(hf22); hw22 = warp_reduce_sum_ll(hw22);
  hf23 = warp_reduce_sum_ll(hf23); hw23 = warp_reduce_sum_ll(hw23);

  // Step 2.2: Warp leaders write to shared memory
  if (lane == 0) {
    s_warp_results[0][warp_id] = hf0;  s_warp_results[1][warp_id] = hw0;
    s_warp_results[2][warp_id] = hf10; s_warp_results[3][warp_id] = hw10;
    s_warp_results[4][warp_id] = hf11; s_warp_results[5][warp_id] = hw11;
    s_warp_results[6][warp_id] = hf20; s_warp_results[7][warp_id] = hw20;
    s_warp_results[8][warp_id] = hf21; s_warp_results[9][warp_id] = hw21;
    s_warp_results[10][warp_id] = hf22; s_warp_results[11][warp_id] = hw22;
    s_warp_results[12][warp_id] = hf23; s_warp_results[13][warp_id] = hw23;
  }
  __syncthreads();

  // Step 2.3: One warp performs inter-warp reduction and final atomic write
  if (warp_id == 0) {
    // Each lane in warp 0 sums one value across all other warps
    // e.g. lane 0 sums hf0, lane 1 sums hw0, etc.
    long long final_val = 0;
    if (lane < 14) {
        for (int i = 0; i < warps_per_block; ++i) {
            final_val += s_warp_results[lane][i];
        }
    }
    
    // Final write-out. Each lane writes one value.
    if (lane == 0)  atomicAdd(Hptr(H, nodes_total, feat_set, 0, 0, threadIdx.x), (unsigned long long)final_val);
    if (lane == 1)  atomicAdd(Hptr(H, nodes_total, feat_set, 0, 1, threadIdx.x), (unsigned long long)final_val);
    if (lane == 2)  atomicAdd(Hptr(H, nodes_total, feat_set, 1, 0, threadIdx.x), (unsigned long long)final_val);
    if (lane == 3)  atomicAdd(Hptr(H, nodes_total, feat_set, 1, 1, threadIdx.x), (unsigned long long)final_val);
    if (lane == 4)  atomicAdd(Hptr(H, nodes_total, feat_set, 2, 0, threadIdx.x), (unsigned long long)final_val);
    if (lane == 5)  atomicAdd(Hptr(H, nodes_total, feat_set, 2, 1, threadIdx.x), (unsigned long long)final_val);
    if (lane == 6)  atomicAdd(Hptr(H, nodes_total, feat_set, 3, 0, threadIdx.x), (unsigned long long)final_val);
    if (lane == 7)  atomicAdd(Hptr(H, nodes_total, feat_set, 3, 1, threadIdx.x), (unsigned long long)final_val);
    if (lane == 8)  atomicAdd(Hptr(H, nodes_total, feat_set, 4, 0, threadIdx.x), (unsigned long long)final_val);
    if (lane == 9)  atomicAdd(Hptr(H, nodes_total, feat_set, 4, 1, threadIdx.x), (unsigned long long)final_val);
    if (lane == 10) atomicAdd(Hptr(H, nodes_total, feat_set, 5, 0, threadIdx.x), (unsigned long long)final_val);
    if (lane == 11) atomicAdd(Hptr(H, nodes_total, feat_set, 5, 1, threadIdx.x), (unsigned long long)final_val);
    if (lane == 12) atomicAdd(Hptr(H, nodes_total, feat_set, 6, 0, threadIdx.x), (unsigned long long)final_val);
    if (lane == 13) atomicAdd(Hptr(H, nodes_total, feat_set, 6, 1, threadIdx.x), (unsigned long long)final_val);
  }

  // --- Phase 3: Optimized Drain for Depths >= 3 ---
  const int rows_per_warp = (n_ge3 + warps_per_block - 1) / warps_per_block;
  for (int k = 0; k < rows_per_warp; ++k) {
    const int node = 7 + rows_per_warp * warp_id + k;
    if (node < nodes_total) {
      const int base = ((node - 7) * 2) * 32;
      // Each lane reads its value for this node
      int fval = shmem_hist[base + 0  + lane];
      int cval = shmem_hist[base + 32 + lane];

      // Intra-warp reduction
      fval = warp_reduce_sum_int(fval);
      cval = warp_reduce_sum_int(cval);

      // Lane 0 of this warp performs the single atomic write for this node
      if (lane == 0) {
        if (fval != 0) atomicAdd(Hptr(H, nodes_total, feat_set, node, 0, threadIdx.x), (unsigned long long int)fval);
        if (cval != 0) atomicAdd(Hptr(H, nodes_total, feat_set, node, 1, threadIdx.x), (unsigned long long int)cval);
      }
    }
  }
}

torch::Tensor h_sm_hierarchical(
    torch::Tensor XS,
    torch::Tensor Y,
    torch::Tensor LF,
    int max_depth)
{
  
  const int nfeatsets = XS.size(0);
  const int cols_32M  = XS.size(1);
  const int N         = Y.size(0);
  const int nodes_tot = (1 << max_depth) - 1;

  auto H = torch::zeros({nfeatsets, nodes_tot, 2, 32}, XS.options().dtype(torch::kLong));

  const int warps_per_block = infer_hist_warps_per_block(max_depth);
  int blocks_per_feat = 0, stride = 0;
  infer_grid_stride(nfeatsets, cols_32M, warps_per_block, blocks_per_feat, stride);

  dim3 grid(nfeatsets, blocks_per_feat, 1);
  dim3 block(warps_per_block * WARP_SIZE, 1, 1);

  // --- UPDATED SHARED MEMORY CALCULATION ---
  const int n_ge3 = std::max(1, (1 << max_depth) - 8);
  // Part 1: Histogram for depths >= 3
  const size_t smem_hist_bytes = (size_t)n_ge3 * 2 * 32 * sizeof(int);
  // Part 2: Buffer for inter-warp reduction (14 values, max 16 warps)
  const size_t smem_reduce_bytes = 14 * 16 * sizeof(long long);
  const size_t smem_bytes = smem_hist_bytes + smem_reduce_bytes;

  auto* prop = at::cuda::getCurrentDeviceProperties();
  size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin : (size_t)prop->sharedMemPerBlock;
  TORCH_CHECK(smem_bytes <= smem_cap, "Required dynamic shared memory (", smem_bytes, ") exceeds device limit (", smem_cap, ")");

  auto stream = at::cuda::getCurrentCUDAStream();
  const uint32_t* XS_ptr = reinterpret_cast<const uint32_t*>(XS.data_ptr());

  // Dispatch LF dtype (same as before)
  const auto lf_dt = LF.scalar_type();
  if (lf_dt == torch::kUInt16) {
    cudaFuncSetAttribute(_h_sm_hierarchical<uint16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    _h_sm_hierarchical<uint16_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int32_t>(), LF.data_ptr<uint16_t>(), H.data_ptr<long long>(),
      nfeatsets, cols_32M, N, max_depth, warps_per_block, stride, nodes_tot);
  } else if (lf_dt == torch::kUInt32) {
    cudaFuncSetAttribute(_h_sm_hierarchical<uint32_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    _h_sm_hierarchical<uint32_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int32_t>(), LF.data_ptr<uint32_t>(), H.data_ptr<long long>(),
      nfeatsets, cols_32M, N, max_depth, warps_per_block, stride, nodes_tot);
  } else if (lf_dt == torch::kUInt64) {
    cudaFuncSetAttribute(_h_sm_hierarchical<uint64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    _h_sm_hierarchical<uint64_t><<<grid, block, smem_bytes, stream.stream()>>>(
      XS_ptr, Y.data_ptr<int32_t>(), static_cast<uint64_t*>(LF.data_ptr()), H.data_ptr<long long>(),
      nfeatsets, cols_32M, N, max_depth, warps_per_block, stride, nodes_tot);
  } else {
    TORCH_CHECK(false, "Unsupported LF dtype: ", lf_dt);
  }
  return H;
}