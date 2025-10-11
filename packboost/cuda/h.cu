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


// ------------------------ Kernel A: build per-block partials ------------------------
// Shared histogram covers ALL nodes: [nodes_tot, 2, 32] int32
// Then each block writes its partial to: Scratch[F, nodes_tot, 2, 32, B] int64
template <typename LF_T>
__global__ void _h_partials(
    const uint32_t* __restrict__ XS,   // [F, 32*M]
    const int32_t*  __restrict__ Y,    // [N]
    const LF_T*     __restrict__ LF,   // [F, N] (u16/u32/u64)
    int64_t*        __restrict__ SCR,  // [F, nodes_tot, 2, 32, B] int64
    int F, int cols_32M, int N, int D,
    int warps_per_block, int stride_per_warp,
    int nodes_tot, int B                 // blocks_per_feat (last scratch dim)
){
    const int feat  = blockIdx.x;       // 0..F-1
    const int bidx  = blockIdx.y;       // 0..B-1 (this block's partial slot)
    if (feat >= F || bidx >= B) return;

    const int lane  = threadIdx.x & (32-1);
    const int wid   = threadIdx.x >> 5;               // 0..warps_per_block-1
    const int gwarp = warps_per_block * bidx + wid;   // global warp ordinal along samples

    // ----- shared histogram -----
    extern __shared__ int s_hist[];                   // [nodes_tot, 2, 32] int32
    const size_t hist_ints = (size_t)nodes_tot * 2 * 32;

    // helpers
    auto HIST = [&](int node, int ch, int l)->int& {
        return s_hist[( (size_t)node * 2 + ch ) * 32 + l];
    };

    // zero cooperatively
    for (size_t i = threadIdx.x; i < hist_ints; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    // iterate tiles owned by this warp
    for (int j = 0; j < stride_per_warp; ++j) {
        const int base = 32 * (gwarp * stride_per_warp + j);
        if (base >= cols_32M) break;

        // lane-local loads
        uint32_t xbits = 0u;
        const int col = base + lane;
        if (col < cols_32M)
            xbits = XS[(size_t)feat * cols_32M + col];

        const int jj_lane = base + lane;
        int32_t y_lane = 0;
        unsigned long long lf_lane = 0ULL; // 64b to support D up to 9
        if (jj_lane < N) {
            y_lane  = Y[jj_lane];
            lf_lane = (unsigned long long)(
                (typename std::make_unsigned<LF_T>::type)
                (LF[(size_t)feat * N + jj_lane]));
        }

        const unsigned full = __activemask();

        for (int k = 0; k < 32; ++k) {
            const int jj_k = base + k;
            const unsigned mask_k = __ballot_sync(full, jj_k < N);
            if (!mask_k) break;

            // always broadcast with the same mask
            const int  yk = __shfl_sync(mask_k, y_lane,  k);
            uint64_t   lk = (uint64_t)__shfl_sync(mask_k, l_lo, k)
                        | ((uint64_t)__shfl_sync(mask_k, l_hi, k) << 32);

            // lane’s bit for this column
            const int v = (xbits >> k) & 1u;
            if (!v) continue;


            // dynamic depths: node = ((1<<d)-1) + (lk & ((1<<d)-1)); lk >>= d
            for (int d = 0; d < D; ++d) {
                const unsigned to = (d==0) ? 0u : ((1u<<d) - 1u);
                const unsigned tk = (d==0) ? 0u : (unsigned)(lk & to);
                if (d > 0) lk >>= d;
                const int node = (int)(to + tk);     // 0..nodes_tot-1
                atomicAdd(&HIST(node, 0, lane), yk);
                atomicAdd(&HIST(node, 1, lane), 1);
            }
        }
    }
    __syncthreads();

    // drain shared histogram to this block's scratch slice (no atomics)
    // layout: SCR[feat, node, ch, lane, bidx]
    const size_t nodes_stride = (size_t)2 * 32 * B;
    const size_t ch_stride    = (size_t)32 * B;
    const size_t lane_stride  = (size_t)B;

    int64_t* base_ptr = SCR + ((size_t)feat * nodes_tot) * (2*32*B);

    for (int node = wid; node < nodes_tot; node += warps_per_block) {
        const size_t base_off = (size_t)node * nodes_stride + lane * lane_stride + bidx;
        const int fs = HIST(node, 0, lane);
        const int cs = HIST(node, 1, lane);
        base_ptr[base_off + 0*ch_stride] = (long long)fs;
        base_ptr[base_off + 1*ch_stride] = (long long)cs;
    }
}

// ------------------------ Kernel B: reduce partials to final H ------------------------
__global__ void _h_reduce(
    const int64_t* __restrict__ SCR,   // [F, nodes_tot, 2, 32, B]
    int64_t*       __restrict__ H,     // [F, nodes_tot, 2, 32]
    int F, int nodes_tot, int B
){
    const int feat = blockIdx.x;
    const int node = blockIdx.y;
    const int ch   = blockIdx.z;       // 0 or 1
    const int lane = threadIdx.x;      // 0..31
    if (feat >= F || node >= nodes_tot || ch >= 2 || lane >= 32) return;

    const size_t nodes_stride = (size_t)2 * 32 * B;
    const size_t ch_stride    = (size_t)32 * B;
    const size_t lane_stride  = (size_t)B;

    const int64_t* base_ptr = SCR + ((size_t)feat * nodes_tot + node) * (2*32*B);
    int64_t acc = 0;
    for (int b = 0; b < B; ++b) {
        acc += base_ptr[ (size_t)ch * ch_stride + (size_t)lane * lane_stride + b ];
    }

    // H[feat, node, ch, lane] = acc
    H[ (((size_t)feat * nodes_tot + node) * 2 + ch) * 32 + lane ] = acc;
}

// ---------------------------------- Launcher ----------------------------------
torch::Tensor h_two_pass(
    torch::Tensor XS,   // [F, 32*M], (u)int32
    torch::Tensor Y,    // [N], int32
    torch::Tensor LF,   // [F, N], (u)int16/32/64
    int max_depth       // supports 1..9 (SMEM bound)
){
    
    const int64_t F64    = XS.size(0);
    const int64_t cols64 = XS.size(1);
    const int64_t N64    = Y.size(0);
    

    const int F        = (int)F64;
    const int cols_32M = (int)cols64;
    const int N        = (int)N64;
    const int D        = max_depth;
    const int nodes_tot = (1<<D) - 1;

    auto* prop = at::cuda::getCurrentDeviceProperties();
    size_t smem_cap = prop->sharedMemPerBlockOptin ? (size_t)prop->sharedMemPerBlockOptin
                                                   : (size_t)prop->sharedMemPerBlock;

    // SMEM for full-node histogram (int32): nodes_tot * 2 * 32 * 4
    const size_t hist_bytes = (size_t)nodes_tot * 2 * 32 * sizeof(int);
    TORCH_CHECK(hist_bytes <= smem_cap,
        "Kernel A needs ", hist_bytes, "B SMEM; device allows ", smem_cap, "B. "
        "D=8 fits on 64KB, D=9 needs ~128KB (A100-class).");

    // Choose warps_per_block and B (blocks_per_feat) to keep GPU busy
    int warps_per_block = std::min(16, prop->maxThreadsPerBlock / 32);
    if (warps_per_block < 1) warps_per_block = 1;

    int SM = prop->multiProcessorCount;
    int target_blocks_per_SM = 24; // good default for two-pass
    int min_total_blocks = SM * target_blocks_per_SM;
    int B = (min_total_blocks + F - 1) / F;
    if (B < 16) B = 16; // ensure enough partials

    // per-warp tiles along samples
    int stride_per_warp = (N + (32 * B * warps_per_block) - 1) / (32 * B * warps_per_block);
    if (stride_per_warp < 1) stride_per_warp = 1;

    // Allocate scratch: [F, nodes_tot, 2, 32, B] int64
    auto SCR = torch::empty({F64, (int64_t)nodes_tot, 2, 32, (int64_t)B},
                            LF.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous));

    // Final output: [F, nodes_tot, 2, 32] int64
    auto H = torch::empty({F64, (int64_t)nodes_tot, 2, 32},
                          LF.options().dtype(torch::kLong).memory_format(c10::MemoryFormat::Contiguous));

    const dim3 gridA(F, B, 1);
    const dim3 blockA(warps_per_block * 32, 1, 1);
    const size_t smemA = hist_bytes;

    const dim3 gridB(F, (unsigned)nodes_tot, 2);
    const dim3 blockB(32, 1, 1);

    auto stream = at::cuda::getCurrentCUDAStream();
    const uint32_t* XS_u32 = reinterpret_cast<const uint32_t*>(XS.data_ptr());
    const auto dt = LF.scalar_type();

    // ---------------- Kernel A launch ----------------
    if (dt == torch::kUInt16) {
        cudaFuncSetAttribute(_h_partials<uint16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemA);
        _h_partials<uint16_t><<<gridA, blockA, smemA, stream.stream()>>>(
            XS_u32, Y.data_ptr<int32_t>(), LF.data_ptr<uint16_t>(),
            SCR.data_ptr<int64_t>(), F, cols_32M, N, D,
            warps_per_block, stride_per_warp, nodes_tot, B);
    } else if (dt == torch::kUInt32) {
        cudaFuncSetAttribute(_h_partials<uint32_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemA);
        _h_partials<uint32_t><<<gridA, blockA, smemA, stream.stream()>>>(
            XS_u32, Y.data_ptr<int32_t>(), LF.data_ptr<uint32_t>(),
            SCR.data_ptr<int64_t>(), F, cols_32M, N, D,
            warps_per_block, stride_per_warp, nodes_tot, B);
    } else if (dt == torch::kUInt64) {
        cudaFuncSetAttribute(_h_partials<uint64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemA);
        _h_partials<uint64_t><<<gridA, blockA, smemA, stream.stream()>>>(
            XS_u32, Y.data_ptr<int32_t>(), LF.data_ptr<uint64_t>(),
            SCR.data_ptr<int64_t>(), F, cols_32M, N, D,
            warps_per_block, stride_per_warp, nodes_tot, B);
    } else {
        TORCH_CHECK(false, "LF must be uint16/uint32/uint64.");
    }

    // ---------------- Kernel B launch ----------------
    _h_reduce<<<gridB, blockB, 0, stream.stream()>>>(
        SCR.data_ptr<int64_t>(), H.data_ptr<int64_t>(), F, nodes_tot, B);

    return H;
}