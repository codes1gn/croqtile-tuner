
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
// include the choreo header;
#include "choreo.h"
namespace cde = cuda::device::experimental;
#include <cooperative_groups.h>
using namespace choreo;

#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ 90

static inline void __choreo_check_cuda_environment__() {
  // ----------- ONE-TIME GUARD -----------
  static bool already_checked = false;
  if (already_checked) return;
  already_checked = true;
  // --------------------------------------

  auto decode_cuda_version =
   [](int v, int& major, int& minor, int& patch) {
    major = v / 1000;
    minor = (v % 1000) / 10;
    patch = v % 10;
  };

  // ----------- Runtime version check -----------
  int runtime_ver = 0;
  cudaError_t err = cudaRuntimeGetVersion(&runtime_ver);
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                "[choreo] CUDA runtime not available: %s\n",
                cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int driver_ver = 0;
  err = cudaDriverGetVersion(&driver_ver);
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                "[choreo] CUDA driver not available: %s\n",
                cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int rMaj, rMin, rPat;
  int dMaj, dMin, dPat;
  decode_cuda_version(runtime_ver, rMaj, rMin, rPat);
  decode_cuda_version(driver_ver, dMaj, dMin, dPat);

  int reqMaj, reqMin, reqPat;
  decode_cuda_version(CUDART_VERSION, reqMaj, reqMin, reqPat);

  if (runtime_ver < CUDART_VERSION) {
    std::fprintf(stderr,
       "[choreo] CUDA runtime too old:\n"
       "  found runtime %d.%d.%d (encoded=%d)\n"
       "  required      %d.%d.%d (encoded=%d)\n",
       rMaj, rMin, rPat, runtime_ver,
       reqMaj, reqMin, reqPat, CUDART_VERSION);
    std::exit(EXIT_FAILURE);
  }

  // Optional: check driver vs runtime mismatch
  if (driver_ver < runtime_ver) {
    std::fprintf(stderr,
       "[choreo] Warning: CUDA driver (%d.%d.%d, encoded=%d) is older than "
       "the CUDA runtime (%d.%d.%d, encoded=%d). This may cause issues.\n",
       dMaj, dMin, dPat, driver_ver,
       rMaj, rMin, rPat, runtime_ver);
  }

  // ----------- Device capability check -----------
  int device_count = 0;
  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::fprintf(stderr,
                "[choreo] No CUDA-capable devices found.\n");
    std::exit(EXIT_FAILURE);
  }

  // ----------- Device capability check (selected device) -----------
  int device_id = 0;
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                 "[choreo] cudaGetDeviceProperties failed: %s\n",
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int sm = prop.major * 10 + prop.minor;
  if (sm < __CHOREO_REQUIRED_GPU_DEVICE_SM__) {
    std::fprintf(stderr,
        "[choreo] Compute capability too low on device %d (%s):\n"
        "  found SM %d.%d (sm_%d)\n"
        "  required SM >= %d (sm_%d)\n",
        device_id, prop.name,
        prop.major, prop.minor, sm,
        __CHOREO_REQUIRED_GPU_DEVICE_SM__, __CHOREO_REQUIRED_GPU_DEVICE_SM__);
    std::exit(EXIT_FAILURE);
  }

#if 0
  // ----------- Optional success log -----------
  std::fprintf(stderr,
    "[choreo] CUDA environment OK\n"
    "  runtime %d.%d.%d (encoded=%d)\n"
    "  driver  %d.%d.%d (encoded=%d)\n"
    "  device  %d: %s, SM %d.%d (sm_%d)\n",
    rMaj, rMin, rPat, runtime_ver,
    dMaj, dMin, dPat, driver_ver,
    device_id, prop.name, prop.major, prop.minor, sm);
#endif
}

// Build & Run:
//   ./choreo -gs -t cute -arch=sm_90a --use-warpspec --stmatrix --hoist-offset --hoist-scale --ptx-barrier --tma-cluster-aware benchmark/performance/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter061_1p2c_so_wn160_kunroll.co -o /tmp/iter061.cute.result && bash /tmp/iter061.cute.result --execute
#include <cstring>
#include <cstdlib>

// iter066: iter062 + warpgroup_wait<1> (depth-1 WGMMA pipeline): allow K-tile[k] WGMMA to
//   execute while waiting for K-tile[k+1] data barrier. No extra accumulators needed since
//   hardware serializes on mc[] register dependencies. Overlap barrier latency with WGMMA execution.
// SMEM for WARP_N=256, STAGES=2, TILE_M=128:
//   rhs @ 0: 2*(256*64*2) = 65536B
//   lhs @ 65536: 2*(128*64*2) = 32768B (both 64-row halves per stage)
//   out1 @ 98304: 64*256*4 = 65536B (consumer 1 output)
//   out2 @ 163840: 64*256*4 = 65536B (consumer 2 output)
//   Total: 229376B (224KB)
#define H800_PCIE_PEAK_F16_TFLOPS 1513
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 256
#define MATMUL_TILE_M 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_SWIZ 128
#define MATMUL_STAGES 2

#define MATMUL_DEFAULT_M 16384
#define MATMUL_DEFAULT_N 16384
#define MATMUL_DEFAULT_K 16384

__global__ void __choreo_device_matmul(f16 * lhs, f16 * rhs, float * output, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_3_tensor_map) {
  extern __shared__ char __choreo_device_matmul__runtime_shared_buffer__raw[];
  auto __choreo_device_matmul__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<1024 * 8>(__choreo_device_matmul__runtime_shared_buffer__raw));
  { // parallel-by: .claude/skills/choreo-kernel-examples/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter061_1p2c_so_wn160_kunroll.co:20.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_0_barrier, 1);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_1_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_1_barrier, 1);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_1{&choreo_copy_atom_t_1_barrier};

  // SWIZZLE=16 grid decode: 1D blockIdx.x -> (tile_m, tile_n)
  const unsigned __N_tiles = (N + 255) / 256;
  const unsigned __SWIZZLE = 16;
  const unsigned __flat = blockIdx.x;
  const unsigned __super = __flat / (__SWIZZLE * __N_tiles);
  const unsigned __sub   = __flat % (__SWIZZLE * __N_tiles);
  const unsigned __tile_m = __super * __SWIZZLE + (__sub % __SWIZZLE);
  const unsigned __tile_n = __sub / __SWIZZLE;

  auto anon_3 = (unsigned char*)__choreo_device_matmul__runtime_shared_buffer__;
  __shared__ cuda::barrier<cuda::thread_scope_block> full[2]; // shared event barrier
  // initialize the event barrier - count=257: 1 producer arrive_tx + 2x128 consumer arrives
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&full[0], 257);
    init(&full[1], 257);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[2]; // shared event barrier
  // initialize the event barrier - count=257: 2x128 consumer arrives + 1 producer arrive
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&empty[0], 257);
    init(&empty[1], 257);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  // SMEM layout for WARP_N=256, STAGES=2, TILE_M=128:
  //   rhs_load_s @ 0:      2*(256*64*2) = 65536B
  //   lhs_load_s @ 65536:  2*(128*64*2) = 32768B
  //   output_s_1 @ 98304:  64*256*4 = 65536B (consumer 1: top 64 rows, FP32)
  //   output_s_2 @ 163840: 64*256*4 = 65536B (consumer 2: bot 64 rows, FP32)
  f16*   lhs_load_s = (f16*)(anon_3 + 65536);
  f16*   rhs_load_s = (f16*)(anon_3 + 0);
  float* output_s_1 = (float*)(anon_3 + 98304);
  float* output_s_2 = (float*)(anon_3 + 163840);
  auto __choreo_vg4id_x = threadIdx.x / 128;
  auto __choreo_vtid_x = threadIdx.x % 128;
  // inthreads: .claude/skills/choreo-kernel-examples/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter061_1p2c_so_wn160_kunroll.co:28.7
  if ((__choreo_vg4id_x == 0 && __choreo_vtid_x == 0)) {
    {
      int __iv_iv_k = 0;
      #pragma unroll 64
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 63) / 64); ++__iv_iv_k) {
        int stage = __iv_iv_k % 2;
        empty[stage].wait(empty[stage].arrive());
        const unsigned lhs_k_offset = (__iv_iv_k * 64);
        const unsigned lhs_m_offset = (__tile_m * 128);
        // lhs_load_s stage offset in f16 elements: 128*64 = 8192 f16 per stage = 16384B
        cde::cp_async_bulk_tensor_2d_global_to_shared((lhs_load_s + ((__iv_iv_k % 2 * 8192))), &__choreo_tma_0_tensor_map, lhs_k_offset, lhs_m_offset, full[stage]);
        const unsigned rhs_k_offset = (__iv_iv_k * 64);
        const unsigned rhs_n_offset = (__tile_n * 256);
        // rhs_load_s stage offset in f16 elements: 256*64 = 16384 f16 per stage = 32768B
        cde::cp_async_bulk_tensor_2d_global_to_shared((rhs_load_s + ((__iv_iv_k % 2 * 16384))), &__choreo_tma_1_tensor_map, rhs_k_offset, rhs_n_offset, full[stage]);
        // barrier_arrive_tx bytes: lhs=128*64*2=16384 + rhs=256*64*2=32768
        (void)cuda::device::barrier_arrive_tx(full[stage], 1, (16384) + (32768));
      } // iv_k
      __iv_iv_k = 0;
    }
  } // end inthreads
  // inthreads: .claude/skills/choreo-kernel-examples/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter061_1p2c_so_wn160_kunroll.co:38.7
  if ((__choreo_vg4id_x == 1)) {
    // Consumer 1: top 64 rows of the 128-row M-tile (F32 accumulator, WARP_N=256)
    // mc size = 128 floats for MMA_64x256x16_F32F16F16_SS (128 threads, 128 F32 accumulators/thread)
    float mc[128];
    for (int idx = 0; idx < 128; ++idx) mc[idx] = 0.0f;
    {
      int __iv_s = 0;
      for (__iv_s = 0; __iv_s < 2; ++__iv_s) {
        (void)empty[__iv_s].arrive();
      } // s
      __iv_s = 0;
    }
    {
      // Depth-1 WGMMA pipeline: issue WGMMA[k], then wait for barrier[k+1], then wait<1> to drain WGMMA[k]
      // This hides barrier latency inside WGMMA execution time
      const int N_K = (K + 63) / 64;
      int __iv_iv_k = 0;
      #pragma unroll 64
      for (__iv_iv_k = 0; __iv_iv_k < N_K; ++__iv_iv_k) {
        auto stage = __iv_iv_k % 2;
        auto base_lhs = __iv_iv_k % 2 * 8192;
        auto base_rhs = __iv_iv_k % 2 * 16384;
        full[stage].wait(full[stage].arrive());
        warpgroup_arrive();
        // Unrolled warp=0
        { f16* ma = (f16*)((0 * 16 + base_lhs + lhs_load_s)); f16* mb = (f16*)((0 * 16 + base_rhs + rhs_load_s));
          uint64_t da = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma);
          uint64_t db = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb);
          /*warpgroup_arrive();*/
          cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(da, db, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], mc[64], mc[65], mc[66], mc[67], mc[68], mc[69], mc[70], mc[71], mc[72], mc[73], mc[74], mc[75], mc[76], mc[77], mc[78], mc[79], mc[80], mc[81], mc[82], mc[83], mc[84], mc[85], mc[86], mc[87], mc[88], mc[89], mc[90], mc[91], mc[92], mc[93], mc[94], mc[95], mc[96], mc[97], mc[98], mc[99], mc[100], mc[101], mc[102], mc[103], mc[104], mc[105], mc[106], mc[107], mc[108], mc[109], mc[110], mc[111], mc[112], mc[113], mc[114], mc[115], mc[116], mc[117], mc[118], mc[119], mc[120], mc[121], mc[122], mc[123], mc[124], mc[125], mc[126], mc[127]); }
        // Unrolled warp=1
        { f16* ma = (f16*)((1 * 16 + base_lhs + lhs_load_s)); f16* mb = (f16*)((1 * 16 + base_rhs + rhs_load_s));
          uint64_t da = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma);
          uint64_t db = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb);
          /*warpgroup_arrive();*/
          cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(da, db, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], mc[64], mc[65], mc[66], mc[67], mc[68], mc[69], mc[70], mc[71], mc[72], mc[73], mc[74], mc[75], mc[76], mc[77], mc[78], mc[79], mc[80], mc[81], mc[82], mc[83], mc[84], mc[85], mc[86], mc[87], mc[88], mc[89], mc[90], mc[91], mc[92], mc[93], mc[94], mc[95], mc[96], mc[97], mc[98], mc[99], mc[100], mc[101], mc[102], mc[103], mc[104], mc[105], mc[106], mc[107], mc[108], mc[109], mc[110], mc[111], mc[112], mc[113], mc[114], mc[115], mc[116], mc[117], mc[118], mc[119], mc[120], mc[121], mc[122], mc[123], mc[124], mc[125], mc[126], mc[127]); }
        // Unrolled warp=2
        { f16* ma = (f16*)((2 * 16 + base_lhs + lhs_load_s)); f16* mb = (f16*)((2 * 16 + base_rhs + rhs_load_s));
          uint64_t da = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma);
          uint64_t db = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb);
          /*warpgroup_arrive();*/
          cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(da, db, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], mc[64], mc[65], mc[66], mc[67], mc[68], mc[69], mc[70], mc[71], mc[72], mc[73], mc[74], mc[75], mc[76], mc[77], mc[78], mc[79], mc[80], mc[81], mc[82], mc[83], mc[84], mc[85], mc[86], mc[87], mc[88], mc[89], mc[90], mc[91], mc[92], mc[93], mc[94], mc[95], mc[96], mc[97], mc[98], mc[99], mc[100], mc[101], mc[102], mc[103], mc[104], mc[105], mc[106], mc[107], mc[108], mc[109], mc[110], mc[111], mc[112], mc[113], mc[114], mc[115], mc[116], mc[117], mc[118], mc[119], mc[120], mc[121], mc[122], mc[123], mc[124], mc[125], mc[126], mc[127]); }
        // Unrolled warp=3
        { f16* ma = (f16*)((3 * 16 + base_lhs + lhs_load_s)); f16* mb = (f16*)((3 * 16 + base_rhs + rhs_load_s));
          uint64_t da = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma);
          uint64_t db = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb);
          /*warpgroup_arrive();*/
          cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(da, db, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], mc[64], mc[65], mc[66], mc[67], mc[68], mc[69], mc[70], mc[71], mc[72], mc[73], mc[74], mc[75], mc[76], mc[77], mc[78], mc[79], mc[80], mc[81], mc[82], mc[83], mc[84], mc[85], mc[86], mc[87], mc[88], mc[89], mc[90], mc[91], mc[92], mc[93], mc[94], mc[95], mc[96], mc[97], mc[98], mc[99], mc[100], mc[101], mc[102], mc[103], mc[104], mc[105], mc[106], mc[107], mc[108], mc[109], mc[110], mc[111], mc[112], mc[113], mc[114], mc[115], mc[116], mc[117], mc[118], mc[119], mc[120], mc[121], mc[122], mc[123], mc[124], mc[125], mc[126], mc[127]); }
        warpgroup_commit_batch();
        // Depth-1 pipeline: wait<1> allows 1 WGMMA group in flight
        // empty.arrive is delayed until AFTER we've started loading the next K-tile
        // This is safe because: we hold off releasing SMEM until WGMMA drains
        warpgroup_wait<4>();
        (void)empty[stage].arrive();
      } // iv_k
      // Drain the final in-flight WGMMA
      warpgroup_wait<0>();
      __iv_iv_k = 0;
    }
    // Store FP32 fragment to SMEM using store_fragment_d (not stmatrix, since FP32)
    auto __shape1_output_s_1 = cute::make_shape(cute::Int<64>{}, cute::Int<256>{});
    auto __stride1_output_s_1 = cute::make_stride(cute::Int<256>{}, cute::Int<1>{});
    auto __layout1_output_s_1 = cute::make_layout(__shape1_output_s_1, __stride1_output_s_1);
    auto __tensor1_output_s_1 = cute::make_tensor(cute::make_smem_ptr<float>((float*)output_s_1 + 0), __layout1_output_s_1);
    store_fragment_d<CUTE_WGMMA_M64K16, 256>(__tensor1_output_s_1, mc);
    future __choreo_anon_fut__0("", 55, 9);
    __choreo_anon_fut__0.is_tma = true;
    __choreo_anon_fut__0.set_atom(&choreo_copy_atom_t_0);
    cde::fence_proxy_async_shared_cta();
    if (__CHOREO_GROUPX4_SINGLE__) {
      cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_2_tensor_map, (__tile_n * 256), (__tile_m * 128), output_s_1);
      cde::cp_async_bulk_commit_group();
    }
  } // end inthreads
  // inthreads: .claude/skills/choreo-kernel-examples/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter061_1p2c_so_wn160_kunroll.co:58.7
  if ((__choreo_vg4id_x == 2)) {
    // Consumer 2: bottom 64 rows of the 128-row M-tile (F32 accumulator, WARP_N=256)
    float mc[128];
    for (int idx = 0; idx < 128; ++idx) mc[idx] = 0.0f;
    {
      int __iv_s = 0;
      for (__iv_s = 0; __iv_s < 2; ++__iv_s) {
        (void)empty[__iv_s].arrive();
      } // s
      __iv_s = 0;
    }
    {
      int __iv_iv_k = 0;
      #pragma unroll 64
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 63) / 64); ++__iv_iv_k) {
        auto stage = __iv_iv_k % 2;
        full[stage].wait(full[stage].arrive());
        auto base_lhs2 = __iv_iv_k % 2 * 8192;
        auto base_rhs2 = __iv_iv_k % 2 * 16384;
        warpgroup_arrive();
        // Unrolled warp=0 (bot)
        { f16* ma = (f16*)((0 * 16 + base_lhs2 + (lhs_load_s + 4096))); f16* mb = (f16*)((0 * 16 + base_rhs2 + rhs_load_s));
          uint64_t da = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma);
          uint64_t db = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb);
          /*warpgroup_arrive();*/
          cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(da, db, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], mc[64], mc[65], mc[66], mc[67], mc[68], mc[69], mc[70], mc[71], mc[72], mc[73], mc[74], mc[75], mc[76], mc[77], mc[78], mc[79], mc[80], mc[81], mc[82], mc[83], mc[84], mc[85], mc[86], mc[87], mc[88], mc[89], mc[90], mc[91], mc[92], mc[93], mc[94], mc[95], mc[96], mc[97], mc[98], mc[99], mc[100], mc[101], mc[102], mc[103], mc[104], mc[105], mc[106], mc[107], mc[108], mc[109], mc[110], mc[111], mc[112], mc[113], mc[114], mc[115], mc[116], mc[117], mc[118], mc[119], mc[120], mc[121], mc[122], mc[123], mc[124], mc[125], mc[126], mc[127]); }
        // Unrolled warp=1 (bot)
        { f16* ma = (f16*)((1 * 16 + base_lhs2 + (lhs_load_s + 4096))); f16* mb = (f16*)((1 * 16 + base_rhs2 + rhs_load_s));
          uint64_t da = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma);
          uint64_t db = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb);
          /*warpgroup_arrive();*/
          cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(da, db, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], mc[64], mc[65], mc[66], mc[67], mc[68], mc[69], mc[70], mc[71], mc[72], mc[73], mc[74], mc[75], mc[76], mc[77], mc[78], mc[79], mc[80], mc[81], mc[82], mc[83], mc[84], mc[85], mc[86], mc[87], mc[88], mc[89], mc[90], mc[91], mc[92], mc[93], mc[94], mc[95], mc[96], mc[97], mc[98], mc[99], mc[100], mc[101], mc[102], mc[103], mc[104], mc[105], mc[106], mc[107], mc[108], mc[109], mc[110], mc[111], mc[112], mc[113], mc[114], mc[115], mc[116], mc[117], mc[118], mc[119], mc[120], mc[121], mc[122], mc[123], mc[124], mc[125], mc[126], mc[127]); }
        // Unrolled warp=2 (bot)
        { f16* ma = (f16*)((2 * 16 + base_lhs2 + (lhs_load_s + 4096))); f16* mb = (f16*)((2 * 16 + base_rhs2 + rhs_load_s));
          uint64_t da = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma);
          uint64_t db = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb);
          /*warpgroup_arrive();*/
          cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(da, db, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], mc[64], mc[65], mc[66], mc[67], mc[68], mc[69], mc[70], mc[71], mc[72], mc[73], mc[74], mc[75], mc[76], mc[77], mc[78], mc[79], mc[80], mc[81], mc[82], mc[83], mc[84], mc[85], mc[86], mc[87], mc[88], mc[89], mc[90], mc[91], mc[92], mc[93], mc[94], mc[95], mc[96], mc[97], mc[98], mc[99], mc[100], mc[101], mc[102], mc[103], mc[104], mc[105], mc[106], mc[107], mc[108], mc[109], mc[110], mc[111], mc[112], mc[113], mc[114], mc[115], mc[116], mc[117], mc[118], mc[119], mc[120], mc[121], mc[122], mc[123], mc[124], mc[125], mc[126], mc[127]); }
        // Unrolled warp=3 (bot)
        { f16* ma = (f16*)((3 * 16 + base_lhs2 + (lhs_load_s + 4096))); f16* mb = (f16*)((3 * 16 + base_rhs2 + rhs_load_s));
          uint64_t da = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma);
          uint64_t db = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb);
          /*warpgroup_arrive();*/
          cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(da, db, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], mc[64], mc[65], mc[66], mc[67], mc[68], mc[69], mc[70], mc[71], mc[72], mc[73], mc[74], mc[75], mc[76], mc[77], mc[78], mc[79], mc[80], mc[81], mc[82], mc[83], mc[84], mc[85], mc[86], mc[87], mc[88], mc[89], mc[90], mc[91], mc[92], mc[93], mc[94], mc[95], mc[96], mc[97], mc[98], mc[99], mc[100], mc[101], mc[102], mc[103], mc[104], mc[105], mc[106], mc[107], mc[108], mc[109], mc[110], mc[111], mc[112], mc[113], mc[114], mc[115], mc[116], mc[117], mc[118], mc[119], mc[120], mc[121], mc[122], mc[123], mc[124], mc[125], mc[126], mc[127]); }
        warpgroup_commit_batch();
        warpgroup_wait<4>();
        (void)empty[stage].arrive();
      } // iv_k
      warpgroup_wait<0>();
      __iv_iv_k = 0;
    }
    auto __shape2_output_s_2 = cute::make_shape(cute::Int<64>{}, cute::Int<256>{});
    auto __stride2_output_s_2 = cute::make_stride(cute::Int<256>{}, cute::Int<1>{});
    auto __layout2_output_s_2 = cute::make_layout(__shape2_output_s_2, __stride2_output_s_2);
    auto __tensor2_output_s_2 = cute::make_tensor(cute::make_smem_ptr<float>((float*)output_s_2 + 0), __layout2_output_s_2);
    store_fragment_d<CUTE_WGMMA_M64K16, 256>(__tensor2_output_s_2, mc);
    future __choreo_anon_fut__1("", 75, 9);
    __choreo_anon_fut__1.is_tma = true;
    __choreo_anon_fut__1.set_atom(&choreo_copy_atom_t_1);
    cde::fence_proxy_async_shared_cta();
    if (__CHOREO_GROUPX4_SINGLE__) {
      cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_3_tensor_map, (__tile_n * 256), (__tile_m * 128 + 64), output_s_2);
      cde::cp_async_bulk_commit_group();
    }
  } // end inthreads
  } // end parallel-by
}

void matmul(const choreo::spanned_view<choreo::f16, 2> & lhs, const choreo::spanned_view<choreo::f16, 2> & rhs, const choreo::spanned_view<float, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &K = lhs.shape()[1];
  auto &M = lhs.shape()[0];
  auto &N = rhs.shape()[0];
  choreo::runtime_check(lhs.shape()[1] == rhs.shape()[1], "lhs K dim must match rhs K dim");
  choreo::runtime_check(lhs.shape()[0] == output.shape()[0], "lhs M dim must match output M dim");
  choreo::runtime_check(rhs.shape()[0] == output.shape()[1], "rhs N dim must match output N dim");

  // TMA map 0: lhs A-matrix FP16 [M, K], box [128, 64] (M-major with K as innermost)
  uint64_t __choreo_tma_0_shape[] = {K, M};
  uint64_t __choreo_tma_0_strides[] = {(K * 2)};
  uint32_t __choreo_tma_0_box_shape[] = {64, 128};
  uint32_t __choreo_tma_0_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_0_tensor_map{};
  CUresult __choreo_tma_0_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_0_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
          2,
          lhs.data(),
          __choreo_tma_0_shape,
          __choreo_tma_0_strides,
          __choreo_tma_0_box_shape,
          __choreo_tma_0_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_0_tensor_map_res != CUDA_SUCCESS);

  // TMA map 1: rhs B-matrix FP16 [N, K], box [256, 64]
  uint64_t __choreo_tma_1_shape[] = {K, N};
  uint64_t __choreo_tma_1_strides[] = {(K * 2)};
  uint32_t __choreo_tma_1_box_shape[] = {64, 256};
  uint32_t __choreo_tma_1_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_1_tensor_map{};
  CUresult __choreo_tma_1_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_1_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
          2,
          rhs.data(),
          __choreo_tma_1_shape,
          __choreo_tma_1_strides,
          __choreo_tma_1_box_shape,
          __choreo_tma_1_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_1_tensor_map_res != CUDA_SUCCESS);

  // TMA map 2: output C top half FP32 [M, N], box [256, 64] (consumer 1: rows tile_m*128 .. tile_m*128+63)
  uint64_t __choreo_tma_2_shape[] = {N, M};
  uint64_t __choreo_tma_2_strides[] = {(N * 4)};
  uint32_t __choreo_tma_2_box_shape[] = {256, 64};
  uint32_t __choreo_tma_2_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_2_tensor_map{};
  CUresult __choreo_tma_2_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_2_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
          2,
          output.data(),
          __choreo_tma_2_shape,
          __choreo_tma_2_strides,
          __choreo_tma_2_box_shape,
          __choreo_tma_2_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_2_tensor_map_res != CUDA_SUCCESS);

  // TMA map 3: output C bottom half FP32 [M, N], same map (consumer 2: rows tile_m*128+64 .. tile_m*128+127)
  uint64_t __choreo_tma_3_shape[] = {N, M};
  uint64_t __choreo_tma_3_strides[] = {(N * 4)};
  uint32_t __choreo_tma_3_box_shape[] = {256, 64};
  uint32_t __choreo_tma_3_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_3_tensor_map{};
  CUresult __choreo_tma_3_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_3_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
          2,
          output.data(),
          __choreo_tma_3_shape,
          __choreo_tma_3_strides,
          __choreo_tma_3_box_shape,
          __choreo_tma_3_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_3_tensor_map_res != CUDA_SUCCESS);

  // 1D grid with SWIZZLE=16: total CTAs = M_128_tiles * N_256_tiles
  const size_t M_128_tiles = (M + 127) / 128;
  const size_t N_256_tiles = (N + 255) / 256;
  const size_t total_ctas = M_128_tiles * N_256_tiles;
  dim3 __matmul_gdims0(total_ctas, 1, 1);
  dim3 __matmul_bdims0(384, 1, 1);
  // SMEM: 229376B = rhs(65536) + lhs(32768) + out1(65536) + out2(65536)
  cudaFuncSetAttribute(__choreo_device_matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, 229376 + (1024 - 1));
  __choreo_device_matmul<<<__matmul_gdims0, __matmul_bdims0, 229376 + (1024 - 1)>>>(lhs.data(), rhs.data(), output.data(), K, M, N, __choreo_tma_0_tensor_map, __choreo_tma_1_tensor_map, __choreo_tma_2_tensor_map, __choreo_tma_3_tensor_map);
  choreo::abend_true(cudaDeviceSynchronize());
}




int main(int argc, char** argv) {
  bool enable_timing = true;
  bool skip_verify = false;
  double user_flops = -1.0;
  auto is_disable_timing_arg = [](const char* s) { const char* t = "--disable-timing"; int i = 0; while (t[i] != '\0' && s[i] == t[i]) ++i; return t[i] == '\0' && s[i] == '\0'; };
  auto is_skip_verify_arg = [](const char* s) { const char* t = "--skip-verify"; int i = 0; while (t[i] != '\0' && s[i] == t[i]) ++i; return t[i] == '\0' && s[i] == '\0'; };
  for (int i = 1; i < argc; ++i) {
    if (is_disable_timing_arg(argv[i])) { enable_timing = false; continue; }
    if (is_skip_verify_arg(argv[i])) { skip_verify = true; }
    if (std::strncmp(argv[i], "--flops=", 8) == 0) { user_flops = std::atof(argv[i] + 8); continue; }
  }
  const char* timing_env = std::getenv("CHOREO_DISABLE_TIMING");
  if (timing_env && timing_env[0] == '1' && timing_env[1] == '\0') enable_timing = false;
  const char* skip_verify_env = std::getenv("CHOREO_SKIP_VERIFY");
  if (skip_verify_env && skip_verify_env[0] == '1' && skip_verify_env[1] == '\0') skip_verify = true;
  size_t M = MATMUL_DEFAULT_M, N = MATMUL_DEFAULT_N, K = MATMUL_DEFAULT_K;
  auto lhs_h = choreo::make_spandata<choreo::f16>(M, K);
  auto rhs_h = choreo::make_spandata<choreo::f16>(N, K);
  auto res_h = choreo::make_spandata<float>(M, N);
  lhs_h.fill_random(0, 2); rhs_h.fill_random(0, 2); res_h.fill(0.0f);
  half *a_d = nullptr, *b_d = nullptr;
  float *c_d = nullptr;
  choreo::abend_true(cudaMalloc(&a_d, M * K * sizeof(half)));
  choreo::abend_true(cudaMalloc(&b_d, N * K * sizeof(half)));
  choreo::abend_true(cudaMalloc(&c_d, M * N * sizeof(float)));
  choreo::abend_true(cudaMemcpy(a_d, lhs_h.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(b_d, rhs_h.data(), N * K * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(c_d, res_h.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());
  auto lhs_d = choreo::make_spanview<choreo::f16, 2>(a_d, {M, K});
  auto rhs_d = choreo::make_spanview<choreo::f16, 2>(b_d, {N, K});
  auto res_d = choreo::make_spanview<float, 2>(c_d, {M, N});
  if (enable_timing) {
    int warmup = 5, repeat = 20;
    const char* warmup_env = std::getenv("CHOREO_TIMING_WARMUP");
    const char* repeat_env = std::getenv("CHOREO_TIMING_REPEAT");
    if (warmup_env) { int v = std::atoi(warmup_env); if (v >= 0) warmup = v; }
    if (repeat_env) { int v = std::atoi(repeat_env); if (v > 0) repeat = v; }
    choreo::TimerOption topt; topt.warmup = warmup; topt.repeat = repeat;
    auto avg_ms = choreo::timing([&]() { matmul(lhs_d, rhs_d, res_d); cudaDeviceSynchronize(); }, topt);
    std::cout << "Timing avg ms: " << avg_ms << "\n";
    double flops = (user_flops > 0.0) ? user_flops : (2.0 * double(M) * double(N) * double(K));
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    std::cout << "TFLOPS: " << tflops << "\n";
    double eff = (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0;
    std::cout << "HW efficiency: " << eff << "%\n";
  } else { matmul(lhs_d, rhs_d, res_d); }
  choreo::abend_true(cudaMemcpy(res_h.data(), c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaDeviceSynchronize());
  if (skip_verify) { std::cout << "Test Passed (verify skipped)\n" << std::endl; return 0; }
  auto lhs_view = lhs_h.view(); auto rhs_view = rhs_h.view(); auto res_view = res_h.view();
  float tolerance = 0.05f;
  auto rel_error = [](float ref, float got) { float abs_ref = std::abs(ref); float denom = abs_ref > 1e-6f ? abs_ref : 1.0f; return std::abs(ref - got) / denom; };
  for (size_t i = 0; i < 128; ++i) {
    for (size_t j = 0; j < 128; ++j) {
      float ref = 0.0f;
      for (size_t k = 0; k < lhs_view.shape()[1]; ++k) ref += __half2float(lhs_view[i][k]) * __half2float(rhs_view[j][k]);
      float got = res_view[i][j];
      auto delta = rel_error(ref, got);
      if (delta >= tolerance) std::cout << "[" << i << ", " << j << "] " << ref << " <-> " << got << ", delta: " << delta * 100 << "%\n";
      choreo::choreo_assert((delta < tolerance), "values are not equal.");
    }
  }
  std::cout << "Test Passed\n" << std::endl;
}


