
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

  int major, minor, patch;
  decode_cuda_version(runtime_ver, major, minor, patch);

  int device_id = 0;
  err = cudaGetDevice(&device_id);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[choreo] cudaGetDevice failed: %s\n", cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int sm_major = 0, sm_minor = 0;
  err = cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device_id);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[choreo] cudaDeviceGetAttribute failed: %s\n", cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
  err = cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device_id);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[choreo] cudaDeviceGetAttribute failed: %s\n", cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int sm_version = sm_major * 10 + sm_minor;
  if (sm_version < __CHOREO_REQUIRED_GPU_DEVICE_SM__) {
    std::fprintf(stderr,
                "[choreo] GPU SM version %d.%d is too old. Required: SM %d+\n",
                sm_major, sm_minor, __CHOREO_REQUIRED_GPU_DEVICE_SM__);
    std::exit(EXIT_FAILURE);
  }
}

// Utility: align pointer up to alignment
template<int alignment>
inline __device__ __host__ char* aligned_up_ptr(char* ptr) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  addr = (addr + alignment - 1) & ~(uintptr_t)(alignment - 1);
  return reinterpret_cast<char*>(addr);
}

// Utility macros
#define __CHOREO_BLOCK_SINGLE__ (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
#define __CHOREO_GROUPX4_SINGLE__ (threadIdx.x % 128 == 0)

// iter059: 1P1C, WARP_N=128, TILE_K=32, STAGES=2, FP32 output, SWIZZLE=16 grid
// Idea: 1P1C with smaller WARP_N=128 allows 2 blocks/SM (96KB SMEM vs 228KB limit)
// doubling SM occupancy should improve SM throughput from 73% to closer to 100%
// SMEM for WARP_N=128, STAGES=2, TILE_K=32, TILE_M=128:
//   rhs @ 0:       2*(128*32*2) = 16384B
//   lhs @ 16384:   2*(128*32*2) = 16384B
//   out_top @ 32768: 64*128*4 = 32768B (consumer top 64 rows)
//   out_bot @ 65536: 64*128*4 = 32768B (consumer bottom 64 rows)
//   Total: 98304B = 96KB  (fits 2 blocks/SM: 2*96=192KB < 228KB H800 limit)
#define H800_PCIE_PEAK_F16_TFLOPS 1513
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_M 128
#define MATMUL_TILE_K 32
#define MATMUL_WARP_K 16
#define MATMUL_SWIZ 128
#define MATMUL_STAGES 2

#define MATMUL_DEFAULT_M 16384
#define MATMUL_DEFAULT_N 16384
#define MATMUL_DEFAULT_K 16384

// 1P1C: 2 warpgroups total, 256 threads per block
__global__ void __choreo_device_matmul(f16 * lhs, f16 * rhs, float * output, unsigned K, unsigned M, unsigned N,
    const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map,
    const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map,
    const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map,
    const __grid_constant__ CUtensorMap __choreo_tma_3_tensor_map) {
  extern __shared__ char __choreo_device_matmul__runtime_shared_buffer__raw[];
  auto __choreo_device_matmul__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<1024 * 8>(__choreo_device_matmul__runtime_shared_buffer__raw));
  {
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
  const unsigned __N_tiles = (N + 127) / 128;  // WARP_N=128
  const unsigned __SWIZZLE = 16;
  const unsigned __flat = blockIdx.x;
  const unsigned __super = __flat / (__SWIZZLE * __N_tiles);
  const unsigned __sub   = __flat % (__SWIZZLE * __N_tiles);
  const unsigned __tile_m = __super * __SWIZZLE + (__sub % __SWIZZLE);
  const unsigned __tile_n = __sub / __SWIZZLE;

  auto anon_3 = (unsigned char*)__choreo_device_matmul__runtime_shared_buffer__;

  // 1P1C: full[s] count = 1 (producer arrive_tx) + 128 (consumer arrive) = 129
  __shared__ cuda::barrier<cuda::thread_scope_block> full[2];
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&full[0], 129);
    init(&full[1], 129);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  // empty[s] count = 128 (consumer arrive) + 1 (producer arrive) = 129
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[2];
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&empty[0], 129);
    init(&empty[1], 129);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  // SMEM layout for WARP_N=128, STAGES=2, TILE_K=32, TILE_M=128:
  //   rhs_load_s @ 0:       2*(128*32*2) = 16384B
  //   lhs_load_s @ 16384:   2*(128*32*2) = 16384B
  //   output_s_top @ 32768: 64*128*4 = 32768B (consumer top 64 rows)
  //   output_s_bot @ 65536: 64*128*4 = 32768B (consumer bottom 64 rows)
  //   Total: 98304B = 96KB
  f16*   rhs_load_s   = (f16*)(anon_3 + 0);
  f16*   lhs_load_s   = (f16*)(anon_3 + 16384);
  float* output_s_top = (float*)(anon_3 + 32768);
  float* output_s_bot = (float*)(anon_3 + 65536);

  auto __choreo_vg4id_x = threadIdx.x / 128;
  auto __choreo_vtid_x = threadIdx.x % 128;

  // Producer warpgroup: vg4id=0, single thread does TMA loads
  if ((__choreo_vg4id_x == 0 && __choreo_vtid_x == 0)) {
    {
      int __iv_iv_k = 0;
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 31) / 32); ++__iv_iv_k) {
        int stage = __iv_iv_k % 2;
        empty[stage].wait(empty[stage].arrive());
        const unsigned lhs_k_offset = (__iv_iv_k * 32);
        const unsigned lhs_m_offset = (__tile_m * 128);
        // lhs stage offset: 128*32=4096 f16 per stage = 8192B
        cde::cp_async_bulk_tensor_2d_global_to_shared((lhs_load_s + (__iv_iv_k % 2 * 4096)), &__choreo_tma_0_tensor_map, lhs_k_offset, lhs_m_offset, full[stage]);
        const unsigned rhs_k_offset = (__iv_iv_k * 32);
        const unsigned rhs_n_offset = (__tile_n * 128);
        // rhs stage offset: 128*32=4096 f16 per stage = 8192B
        cde::cp_async_bulk_tensor_2d_global_to_shared((rhs_load_s + (__iv_iv_k % 2 * 4096)), &__choreo_tma_1_tensor_map, rhs_k_offset, rhs_n_offset, full[stage]);
        // barrier_arrive_tx bytes: lhs=128*32*2=8192 + rhs=128*32*2=8192
        (void)cuda::device::barrier_arrive_tx(full[stage], 1, (8192) + (8192));
      }
      __iv_iv_k = 0;
    }
  } // end producer

  // Consumer warpgroup: vg4id=1, does all 128 M rows for WARP_N=128
  if ((__choreo_vg4id_x == 1)) {
    // MMA_64x128x16_F32F16F16_SS: 128 threads, 64 F32 accumulators/thread
    float mc_top[64];  // top 64 M rows
    float mc_bot[64];  // bottom 64 M rows
    for (int idx = 0; idx < 64; ++idx) { mc_top[idx] = 0.0f; mc_bot[idx] = 0.0f; }

    // Prime empty barriers (consumer arrives for initial stages)
    {
      int __iv_s = 0;
      for (__iv_s = 0; __iv_s < 2; ++__iv_s) {
        (void)empty[__iv_s].arrive();
      }
      __iv_s = 0;
    }

    {
      int __iv_iv_k = 0;
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 31) / 32); ++__iv_iv_k) {
        auto stage = __iv_iv_k % 2;
        full[stage].wait(full[stage].arrive());

        // TILE_K=32 -> 2 WGMMA per tile (each WGMMA handles K=16)
        // Top 64 M rows:
        {
          int __iv_iv_warp = 0;
          for (__iv_iv_warp = 0; __iv_iv_warp < 2; ++__iv_iv_warp) {
            // lhs top half: lhs_load_s[stage], then +warp*16 for K-slice
            f16* ma_smem_ptr = (f16*)((__iv_iv_warp * 16 + __iv_iv_k % 2 * 4096 + lhs_load_s));
            uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma_smem_ptr);
            // rhs: same stage, K-slice
            f16* mb_smem_ptr = (f16*)((__iv_iv_warp * 16 + __iv_iv_k % 2 * 4096 + rhs_load_s));
            uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
            warpgroup_arrive();
            cute::SM90::GMMA::MMA_64x128x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(
              desc_ma, desc_mb,
              mc_top[0], mc_top[1], mc_top[2], mc_top[3], mc_top[4], mc_top[5], mc_top[6], mc_top[7],
              mc_top[8], mc_top[9], mc_top[10], mc_top[11], mc_top[12], mc_top[13], mc_top[14], mc_top[15],
              mc_top[16], mc_top[17], mc_top[18], mc_top[19], mc_top[20], mc_top[21], mc_top[22], mc_top[23],
              mc_top[24], mc_top[25], mc_top[26], mc_top[27], mc_top[28], mc_top[29], mc_top[30], mc_top[31],
              mc_top[32], mc_top[33], mc_top[34], mc_top[35], mc_top[36], mc_top[37], mc_top[38], mc_top[39],
              mc_top[40], mc_top[41], mc_top[42], mc_top[43], mc_top[44], mc_top[45], mc_top[46], mc_top[47],
              mc_top[48], mc_top[49], mc_top[50], mc_top[51], mc_top[52], mc_top[53], mc_top[54], mc_top[55],
              mc_top[56], mc_top[57], mc_top[58], mc_top[59], mc_top[60], mc_top[61], mc_top[62], mc_top[63]);
          }
          __iv_iv_warp = 0;
        }

        // Bottom 64 M rows (lhs offset by 2048 f16 = 4096B for bottom half of 128xK tile):
        {
          int __iv_iv_warp = 0;
          for (__iv_iv_warp = 0; __iv_iv_warp < 2; ++__iv_iv_warp) {
            // lhs bottom half: +2048 f16 = +4096B (64*32 f16 per half-stage)
            f16* ma_smem_ptr = (f16*)((__iv_iv_warp * 16 + __iv_iv_k % 2 * 4096 + (lhs_load_s + 2048)));
            uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma_smem_ptr);
            f16* mb_smem_ptr = (f16*)((__iv_iv_warp * 16 + __iv_iv_k % 2 * 4096 + rhs_load_s));
            uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
            warpgroup_arrive();
            cute::SM90::GMMA::MMA_64x128x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(
              desc_ma, desc_mb,
              mc_bot[0], mc_bot[1], mc_bot[2], mc_bot[3], mc_bot[4], mc_bot[5], mc_bot[6], mc_bot[7],
              mc_bot[8], mc_bot[9], mc_bot[10], mc_bot[11], mc_bot[12], mc_bot[13], mc_bot[14], mc_bot[15],
              mc_bot[16], mc_bot[17], mc_bot[18], mc_bot[19], mc_bot[20], mc_bot[21], mc_bot[22], mc_bot[23],
              mc_bot[24], mc_bot[25], mc_bot[26], mc_bot[27], mc_bot[28], mc_bot[29], mc_bot[30], mc_bot[31],
              mc_bot[32], mc_bot[33], mc_bot[34], mc_bot[35], mc_bot[36], mc_bot[37], mc_bot[38], mc_bot[39],
              mc_bot[40], mc_bot[41], mc_bot[42], mc_bot[43], mc_bot[44], mc_bot[45], mc_bot[46], mc_bot[47],
              mc_bot[48], mc_bot[49], mc_bot[50], mc_bot[51], mc_bot[52], mc_bot[53], mc_bot[54], mc_bot[55],
              mc_bot[56], mc_bot[57], mc_bot[58], mc_bot[59], mc_bot[60], mc_bot[61], mc_bot[62], mc_bot[63]);
          }
          __iv_iv_warp = 0;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        (void)empty[stage].arrive();
      }
      __iv_iv_k = 0;
    }

    // Store top 64 rows via SMEM -> TMA
    {
      auto __shape_top = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
      auto __stride_top = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
      auto __layout_top = cute::make_layout(__shape_top, __stride_top);
      auto __tensor_top = cute::make_tensor(cute::make_smem_ptr<float>(output_s_top), __layout_top);
      store_fragment_d<CUTE_WGMMA_M64K16, 128>(__tensor_top, mc_top);
    }
    future __choreo_anon_fut__0("", 55, 9);
    __choreo_anon_fut__0.is_tma = true;
    __choreo_anon_fut__0.set_atom(&choreo_copy_atom_t_0);
    cde::fence_proxy_async_shared_cta();
    if (__CHOREO_GROUPX4_SINGLE__) {
      cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_2_tensor_map, (__tile_n * 128), (__tile_m * 128), output_s_top);
      cde::cp_async_bulk_commit_group();
    }

    // Store bottom 64 rows via SMEM -> TMA
    {
      auto __shape_bot = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
      auto __stride_bot = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
      auto __layout_bot = cute::make_layout(__shape_bot, __stride_bot);
      auto __tensor_bot = cute::make_tensor(cute::make_smem_ptr<float>(output_s_bot), __layout_bot);
      store_fragment_d<CUTE_WGMMA_M64K16, 128>(__tensor_bot, mc_bot);
    }
    future __choreo_anon_fut__1("", 75, 9);
    __choreo_anon_fut__1.is_tma = true;
    __choreo_anon_fut__1.set_atom(&choreo_copy_atom_t_1);
    cde::fence_proxy_async_shared_cta();
    if (__CHOREO_GROUPX4_SINGLE__) {
      cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_3_tensor_map, (__tile_n * 128), (__tile_m * 128 + 64), output_s_bot);
      cde::cp_async_bulk_commit_group();
    }
  } // end consumer
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

  // TMA map 0: lhs A-matrix FP16 [M, K], box [32, 128] (TILE_K=32)
  uint64_t __choreo_tma_0_shape[] = {K, M};
  uint64_t __choreo_tma_0_strides[] = {(K * 2)};
  uint32_t __choreo_tma_0_box_shape[] = {32, 128};
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

  // TMA map 1: rhs B-matrix FP16 [N, K], box [32, 128] (WARP_N=128, TILE_K=32)
  uint64_t __choreo_tma_1_shape[] = {K, N};
  uint64_t __choreo_tma_1_strides[] = {(K * 2)};
  uint32_t __choreo_tma_1_box_shape[] = {32, 128};
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

  // TMA map 2: output C top half FP32 [M, N], box [128, 64] (consumer top: rows tile_m*128..+63)
  uint64_t __choreo_tma_2_shape[] = {N, M};
  uint64_t __choreo_tma_2_strides[] = {(N * 4)};
  uint32_t __choreo_tma_2_box_shape[] = {128, 64};
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

  // TMA map 3: output C bottom half FP32 [M, N], same map (consumer bot: rows tile_m*128+64..+127)
  uint64_t __choreo_tma_3_shape[] = {N, M};
  uint64_t __choreo_tma_3_strides[] = {(N * 4)};
  uint32_t __choreo_tma_3_box_shape[] = {128, 64};
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

  // 1D grid with SWIZZLE=16: total CTAs = M_128_tiles * N_128_tiles
  const size_t M_128_tiles = (M + 127) / 128;
  const size_t N_128_tiles = (N + 127) / 128;
  const size_t total_ctas = M_128_tiles * N_128_tiles;
  dim3 __matmul_gdims0(total_ctas, 1, 1);
  dim3 __matmul_bdims0(256, 1, 1);   // 1P1C: 256 threads (2 warpgroups)
  // SMEM: 98304B = rhs(16384) + lhs(16384) + out_top(32768) + out_bot(32768)
  cudaError_t attr_err = cudaFuncSetAttribute(__choreo_device_matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304 + (1024 - 1));
  if (attr_err != cudaSuccess) {
    std::fprintf(stderr, "[iter059] cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(attr_err));
    std::exit(EXIT_FAILURE);
  }
  __choreo_device_matmul<<<__matmul_gdims0, __matmul_bdims0, 98304 + (1024 - 1)>>>(
      lhs.data(), rhs.data(), output.data(), K, M, N,
      __choreo_tma_0_tensor_map, __choreo_tma_1_tensor_map,
      __choreo_tma_2_tensor_map, __choreo_tma_3_tensor_map);
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    std::fprintf(stderr, "[iter059] Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
    std::exit(EXIT_FAILURE);
  }
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


