
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
// iter053: 1P2C TILE_M=128 WARP_N=256 STAGES=2 SWIZZLE=8 -Xptxas -O3
// 384 threads: warpgroup 0 = producer, warpgroups 1&2 = consumers (each 64-row slice of A).
// Producer loads A (both halves) and B (shared). 2 consumers compute in parallel.
// Consumer 0: A[tile_m*128:+64, :] × B → Output[tile_m*128:+64, tile_n*256:+256]
// Consumer 1: A[tile_m*128+64:+64, :] × B → Output[tile_m*128+64:+64, tile_n*256:+256]
// SMEM: 2*((128+256)*64*2) + (128*256+64*256)*4 = 2*49152*2 + 131072+65536 = 393216+196608=...
// Actually: 2*(128*64*2 + 256*64*2) + 64*256*4 + 64*256*4 = 2*49152 + 65536*2 = 98304+131072=229376B OK.
// SWIZZLE=8 pairs (each 128-row M pair = one super-tile level).
// Grid: (M/128 * N/256) CTAs with 384 threads each.
// Kernel uses flat blockIdx.x, decoded with SWIZZLE=8 for M/128 tile pairs.
// SMEM: 4*(64*64*2) + 4*(256*64*2) + 64*256*4 = 229376 bytes (224KB) < 228KB.
#include <cstring>
#include <cstdlib>

#define H800_PCIE_PEAK_F16_TFLOPS 1513

#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 256
#define MATMUL_TILE_M 64
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_SWIZ 128
#define MATMUL_STAGES 4

#define MATMUL_DEFAULT_M 16384
#define MATMUL_DEFAULT_N 16384
#define MATMUL_DEFAULT_K 16384

__global__ void __choreo_device_matmul(f16 * lhs, f16 * rhs, float * output, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_3_tensor_map) {
  // 1p2c: warpgroup 0 = producer, warpgroups 1&2 = consumer 0/1
  // TILE_M=128: CTA computes 128×256 output, each consumer does 64-row half
  // SMEM layout (STAGES=2):
  //   rhs[0..1]  @ 0:      2 * (256*64*2) = 65536B
  //   lhs_top[0..1] @ 65536: 2 * (64*64*2) = 16384B   (A rows 0-63)
  //   lhs_bot[0..1] @ 81920: 2 * (64*64*2) = 16384B   (A rows 64-127)
  //   out_top    @ 98304:  64*256*4 = 65536B
  //   out_bot    @ 163840: 64*256*4 = 65536B
  //   Total: 229376B (224KB) = exactly fits H800!
  extern __shared__ char __choreo_device_matmul__runtime_shared_buffer__raw[];
  auto __choreo_device_matmul__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<1024 * 8>(__choreo_device_matmul__runtime_shared_buffer__raw));
  auto anon_1 = (unsigned char*)__choreo_device_matmul__runtime_shared_buffer__;

  { // parallel-by: 1p2c TILE_M=128 with SWIZZLE=8 grid swizzle
  // Grid is (M/128 * N/256) CTAs. Swizzle remaps with 8 M128-super-tiles.
  const unsigned M128_tiles = (M + 127) / 128;
  const unsigned N_tiles = (N + 255) / 256;
  const unsigned SWIZZLE = 8;  // 8 M128-tiles per super-tile
  const unsigned flat = blockIdx.x;
  const unsigned super = flat / (SWIZZLE * N_tiles);
  const unsigned sub   = flat % (SWIZZLE * N_tiles);
  const unsigned local_m = sub % SWIZZLE;
  const unsigned tile_n  = sub / SWIZZLE;
  const unsigned tile_m128 = super * SWIZZLE + local_m;  // index into M128 tiles
  // tile_m128 → row offsets: row0 = tile_m128*128, row1 = tile_m128*128+64

  const int vg4id = threadIdx.x / 128;  // warpgroup id: 0=producer, 1=consumer0, 2=consumer1
  const int vtid  = threadIdx.x % 128;

  // SMEM pointers
  f16* rhs_load_s    = (f16*)(anon_1 + 0);         // 2 * 32768B = 65536B
  f16* lhs_top_load_s = (f16*)(anon_1 + 65536);    // A rows 0-63:  2 * 8192B = 16384B
  f16* lhs_bot_load_s = (f16*)(anon_1 + 81920);    // A rows 64-127: 2 * 8192B = 16384B
  float* output_top_s = (float*)(anon_1 + 98304);  // output 0-63: 65536B
  float* output_bot_s = (float*)(anon_1 + 163840); // output 64-127: 65536B

  // Full/empty barriers: shared between all 3 warpgroups
  // full[2]: producer signals when stage is filled; consumers wait
  // empty[2]: each consumer signals when stage can be reused; producer waits for both
  __shared__ cuda::barrier<cuda::thread_scope_block> full[2];
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[2];
  __shared__ cuda::barrier<cuda::thread_scope_block> output_barrier;  // for TMA output

  if (threadIdx.x == 0) {
    // full: producer signals once (1 thread), 2 consumers wait (total barrier count = 2+1=3)
    // empty: 2 consumers each signal, producer waits (total = 2 consumer signals + 1 producer wait)
    init(&full[0], 3);    // wait for 3: 1 producer arrive_tx + 2 consumer wait(...)
    init(&full[1], 3);
    init(&empty[0], 2+1); // 2 consumers arrive + 1 producer wait
    init(&empty[1], 2+1);
    init(&output_barrier, 1);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  // Producer warpgroup
  if (vg4id == 0 && vtid == 0) {
    // Prime: signal empty barriers so producer can start
    (void)empty[0].arrive(); (void)empty[0].arrive();  // 2 consumer primes
    (void)empty[1].arrive(); (void)empty[1].arrive();

    for (int k = 0; k < (int)((K + 63) / 64); ++k) {
      int s = k % 2;
      empty[s].wait(empty[s].arrive());  // wait for both consumers to release this stage

      const unsigned lhs_k_offset = k * 64;
      const unsigned rhs_k_offset = k * 64;
      const unsigned rhs_n_offset = tile_n * 256;
      // Load A top (rows 0-63)
      const unsigned lhs_top_m_offset = tile_m128 * 128;
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          lhs_top_load_s + s * 8192, &__choreo_tma_0_tensor_map,
          lhs_k_offset, lhs_top_m_offset, full[s]);
      // Load A bot (rows 64-127)
      const unsigned lhs_bot_m_offset = tile_m128 * 128 + 64;
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          lhs_bot_load_s + s * 8192, &__choreo_tma_0_tensor_map,
          lhs_k_offset, lhs_bot_m_offset, full[s]);
      // Load B
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          rhs_load_s + s * 32768, &__choreo_tma_1_tensor_map,
          rhs_k_offset, rhs_n_offset, full[s]);
      // Signal: loaded (8192 + 8192 + 32768 = 49152 bytes)
      (void)cuda::device::barrier_arrive_tx(full[s], 1, 49152);
    }
  }

  // Consumer warpgroup (both consumer 0 and consumer 1)
  if (vg4id == 1 || vg4id == 2) {
    float mc[128];
    for (int i = 0; i < 128; ++i) mc[i] = 0.0f;

    // Prime empty barriers
    (void)empty[0].arrive();
    (void)empty[1].arrive();

    f16* my_lhs = (vg4id == 1) ? lhs_top_load_s : lhs_bot_load_s;

    const int K_TILES = (K + 63) / 64;
    int __iv_iv_k = 0;
    if (K_TILES > 0) {
      full[0].wait(full[0].arrive());
      warpgroup_arrive();
      for (int w = 0; w < 4; ++w) {
        f16* ma = (f16*)(w * 16 + 0 + my_lhs);
        f16* mb = (f16*)(w * 16 + 0 + rhs_load_s);
        uint64_t dma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma);
        uint64_t dmb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb);
        cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(dma, dmb, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], mc[64], mc[65], mc[66], mc[67], mc[68], mc[69], mc[70], mc[71], mc[72], mc[73], mc[74], mc[75], mc[76], mc[77], mc[78], mc[79], mc[80], mc[81], mc[82], mc[83], mc[84], mc[85], mc[86], mc[87], mc[88], mc[89], mc[90], mc[91], mc[92], mc[93], mc[94], mc[95], mc[96], mc[97], mc[98], mc[99], mc[100], mc[101], mc[102], mc[103], mc[104], mc[105], mc[106], mc[107], mc[108], mc[109], mc[110], mc[111], mc[112], mc[113], mc[114], mc[115], mc[116], mc[117], mc[118], mc[119], mc[120], mc[121], mc[122], mc[123], mc[124], mc[125], mc[126], mc[127]);
      }
      warpgroup_commit_batch();
      __iv_iv_k = 1;
    }
    for (; __iv_iv_k < K_TILES; ++__iv_iv_k) {
      int s = __iv_iv_k % 2;
      int ps = (__iv_iv_k - 1) % 2;
      full[s].wait(full[s].arrive());
      warpgroup_wait<0>();
      (void)empty[ps].arrive();
      warpgroup_arrive();
      for (int w = 0; w < 4; ++w) {
        f16* ma = (f16*)(w * 16 + s * 8192 + my_lhs);
        f16* mb = (f16*)(w * 16 + s * 32768 + rhs_load_s);
        uint64_t dma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma);
        uint64_t dmb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb);
        cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(dma, dmb, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], mc[64], mc[65], mc[66], mc[67], mc[68], mc[69], mc[70], mc[71], mc[72], mc[73], mc[74], mc[75], mc[76], mc[77], mc[78], mc[79], mc[80], mc[81], mc[82], mc[83], mc[84], mc[85], mc[86], mc[87], mc[88], mc[89], mc[90], mc[91], mc[92], mc[93], mc[94], mc[95], mc[96], mc[97], mc[98], mc[99], mc[100], mc[101], mc[102], mc[103], mc[104], mc[105], mc[106], mc[107], mc[108], mc[109], mc[110], mc[111], mc[112], mc[113], mc[114], mc[115], mc[116], mc[117], mc[118], mc[119], mc[120], mc[121], mc[122], mc[123], mc[124], mc[125], mc[126], mc[127]);
      }
      warpgroup_commit_batch();
    }
    warpgroup_wait<0>();
    (void)empty[(__iv_iv_k - 1) % 2].arrive();

    // Store output to SMEM
    float* my_output_s = (vg4id == 1) ? output_top_s : output_bot_s;
    auto __shape_out = cute::make_shape(cute::Int<64>{}, cute::Int<256>{});
    auto __stride_out = cute::make_stride(cute::Int<256>{}, cute::Int<1>{});
    auto __layout_out = cute::make_layout(__shape_out, __stride_out);
    auto __tensor_out = cute::make_tensor(cute::make_smem_ptr<float>(my_output_s), __layout_out);
    store_fragment_d<CUTE_WGMMA_M64K16, 256>(__tensor_out, reinterpret_cast<float*>(mc));

    __syncwarp();
    // TMA store to global memory (one warpgroup at a time)
    cde::fence_proxy_async_shared_cta();
    if (vtid == 0) {
      const unsigned my_row = (vg4id == 1) ? tile_m128 * 128 : tile_m128 * 128 + 64;
      // Use different TMA maps for top and bot (different output offsets)
      if (vg4id == 1) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_2_tensor_map,
            tile_n * 256, my_row, output_top_s);
      } else {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_3_tensor_map,
            tile_n * 256, my_row, output_bot_s);
      }
      cde::cp_async_bulk_commit_group();
    }
  }
  } // end parallel-by
}

void matmul(const choreo::spanned_view<choreo::f16, 2> & lhs, const choreo::spanned_view<choreo::f16, 2> & rhs, const choreo::spanned_view<choreo::f32, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &K = lhs.shape()[1];
  auto &M = lhs.shape()[0];
  auto &N = rhs.shape()[0];
  choreo::runtime_check(lhs.shape()[1] == rhs.shape()[1], "The shapes of the 1st parameter (dim: 1) and the 2nd parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(lhs.shape()[0] == output.shape()[0], "The shapes of the 1st parameter (dim: 0) and the 3rd parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(rhs.shape()[0] == output.shape()[1], "The shapes of the 2nd parameter (dim: 0) and the 3rd parameter (dim: 1) are inconsistent.");

  choreo::runtime_check(((static_cast<long long>(M) + 63LL) / 64LL > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_16384x16384x16384/sonnet46/iter014_wn256_s4_async_pipeline.co:24.13");
  choreo::runtime_check(((static_cast<long long>(N) + 255LL) / 256LL > 0LL), "The 2nd bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_16384x16384x16384/sonnet46/iter014_wn256_s4_async_pipeline.co:24.22");
  choreo::runtime_check(((static_cast<long long>(K) + 63LL) / 64LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_16384x16384x16384/sonnet46/iter014_wn256_s4_async_pipeline.co:32.27");
  choreo::runtime_check(((static_cast<long long>(K) + 63LL) / 64LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_16384x16384x16384/sonnet46/iter014_wn256_s4_async_pipeline.co:46.27");
  uint64_t __choreo_tma_0_shape[] = {K, M};
  uint64_t __choreo_tma_0_strides[] = {(K * 2)};
  uint32_t __choreo_tma_0_box_shape[] = {64, 64};
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
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_0_tensor_map_res != CUDA_SUCCESS);
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
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_1_tensor_map_res != CUDA_SUCCESS);
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
  // TMA map 3: same as map 2 (output, for consumer 1 / bottom half)
  alignas(64) CUtensorMap __choreo_tma_3_tensor_map{};
  CUresult __choreo_tma_3_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_3_tensor_map,
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
  choreo::abend_true(__choreo_tma_3_tensor_map_res != CUDA_SUCCESS);
  // 1p2c: 384 threads per block, M128 * N_tiles CTAs
  const unsigned M128_tiles = (M + 127) / 128;
  const unsigned N_tiles2 = (N + 255) / 256;
  dim3 __matmul_gdims0(M128_tiles * N_tiles2, 1, 1);
  dim3 __matmul_bdims0(384, 1, 1);  // 3 warpgroups: 1P + 2C
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
  auto res_h = choreo::make_spandata<choreo::f32>(M, N);
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
  auto res_d = choreo::make_spanview<choreo::f32, 2>(c_d, {M, N});

  if (enable_timing) {
    int warmup = 10, repeat = 50;
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

  auto lhs_view = lhs_h.view();
  auto rhs_view = rhs_h.view();
  auto res_view = res_h.view();
  float tolerance = 0.01f;
  auto rel_error = [](float ref, float got) {
    float abs_ref = std::abs(ref);
    float denom = abs_ref > 1e-6f ? abs_ref : 1.0f;
    return std::abs(ref - got) / denom;
  };
  for (size_t i = 0; i < 128; ++i) {
    for (size_t j = 0; j < 128; ++j) {
      float ref = 0.0f;
      for (size_t k = 0; k < lhs_view.shape()[1]; ++k)
        ref += __half2float(lhs_view[i][k]) * __half2float(rhs_view[j][k]);
      float got = res_view[i][j];
      auto delta = rel_error(ref, got);
      if (delta >= tolerance)
        std::cout << "[" << i << ", " << j << "] " << ref << " <-> " << got << ", delta: " << delta * 100 << "%\n";
      choreo::choreo_assert((delta < tolerance), "values are not equal.");
    }
  }
  std::cout << "Test Passed\n" << std::endl;
}


