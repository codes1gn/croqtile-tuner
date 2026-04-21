
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

// iter007: 1p2c warp specialization - 1 producer + 2 consumers
// TILE_M=128 (2x64), each consumer handles one 64xN slice

#include <cstring>
#include <cstdlib>

#define H800_PCIE_PEAK_F8_TFLOPS 3026

#define MATMUL_WARP_M 64
#define MATMUL_TILE_N 128
#define MATMUL_WARP_N 128
#define MATMUL_TILE_M 128    // 2 * WARP_M for 2 consumers
#define MATMUL_TILE_K 128
#define MATMUL_WARP_K 32
#define MATMUL_SWIZ 128
#define MATMUL_STAGES 4

#if MATMUL_WARP_K != 32
#error "MATMUL_WARP_K must be 32 for e4m3 WGMMA"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for e4m3 WGMMA"
#endif

#if MATMUL_TILE_K != 128
#error "MATMUL_TILE_K must be 128 for blockscale"
#endif

#if MATMUL_SWIZ != MATMUL_TILE_K
#error "MATMUL_SWIZ must equal MATMUL_TILE_K"
#endif

#define MATMUL_DEFAULT_M 8192
#define MATMUL_DEFAULT_N 8192
#define MATMUL_DEFAULT_K 4096

__global__ __launch_bounds__(384) void __choreo_device_blockscale_gemm(f8_e4m3 * lhs, float * scale_lhs, f8_e4m3 * rhs, float * scale_rhs, f16 * output, unsigned DIV_BLK_K, unsigned DIV_BLK_N, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map) {
  // Block ID swizzling for better L2 cache locality
  constexpr int SWIZ_GROUP = 16;
  const int grid_m = gridDim.x;
  const int grid_n = gridDim.y;
  const int linear_block_id = blockIdx.x + blockIdx.y * grid_m;
  const int group_id = linear_block_id / (SWIZ_GROUP * grid_m);
  const int local_id = linear_block_id % (SWIZ_GROUP * grid_m);
  const int swiz_block_m = local_id / SWIZ_GROUP;
  const int swiz_block_n = group_id * SWIZ_GROUP + local_id % SWIZ_GROUP;
  
  extern __shared__ char __choreo_device_blockscale_gemm__runtime_shared_buffer__raw[];
  auto __choreo_device_blockscale_gemm__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<1024 * 8>(__choreo_device_blockscale_gemm__runtime_shared_buffer__raw));
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:45.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_0_barrier, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  auto anon_4 = (unsigned char*)__choreo_device_blockscale_gemm__runtime_shared_buffer__;
  __shared__ cuda::barrier<cuda::thread_scope_block> full[4]; // shared event barrier
  // initialize the event barrier
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&full[0], 257);
    init(&full[1], 257);
    init(&full[2], 257);
    init(&full[3], 257);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[4]; // shared event barrier
  // initialize the event barrier
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&empty[0], 257);
    init(&empty[1], 257);
    init(&empty[2], 257);
    init(&empty[3], 257);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  f8_e4m3* lhs_load_s = (f8_e4m3*)(anon_4 + 65536);
  f8_e4m3* rhs_load_s = (f8_e4m3*)(anon_4 + 0);
  f16* output_s = (f16*)(anon_4 + 131072);
  auto __choreo_vg4id_x = threadIdx.x / 128;
  auto __choreo_vtid_x = threadIdx.x % 128;
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:53.7
  if ((__choreo_vg4id_x == 0 && __choreo_vtid_x == 0)) {
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:54.9
    {
      int __iv_iv_k = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:54.9
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 127) / 128); ++__iv_iv_k) {
        int stage = __iv_iv_k % 4;
        // wait event(barrier)  (empty elemof stage) 
        empty[stage].wait(empty[stage].arrive());
        cde::cp_async_bulk_tensor_2d_global_to_shared((lhs_load_s + ((__iv_iv_k % 4 * 16384))), &__choreo_tma_0_tensor_map, (__iv_iv_k * 128), (swiz_block_m * 128), full[stage]);
        cde::cp_async_bulk_tensor_2d_global_to_shared((rhs_load_s + ((__iv_iv_k % 4 * 16384))), &__choreo_tma_1_tensor_map, (K / ((K + 127) / 128) * __iv_iv_k), (N / ((N + 127) / 128) * swiz_block_n), full[stage]);
        // trigger event(barrier)  (full elemof stage) 
        (void)cuda::device::barrier_arrive_tx(full[stage], 1, (16384) + (16384));
      } // iv_k
      __iv_iv_k = 0;
    }
  } // end inthreads
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:64.7
  if ((__choreo_vg4id_x > 0)) {
    auto anon_3 = __choreo_vg4id_x - 1;
    float mc[64];
    float __frag_init_val0 = static_cast<float>(0.000000);
    for (int idx = 0; idx < 64; ++idx)
      mc[idx] = __frag_init_val0;
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:66.9
    {
      int __iv_s = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:66.9
      for (__iv_s = 0; __iv_s < 4; ++__iv_s) {
        // trigger event(barrier)  (empty elemof s) 
        (void)empty[__iv_s].arrive();
      } // s
      __iv_s = 0;
    }
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:69.9
    {
      int __iv_iv_k = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:69.9
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 127) / 128); ++__iv_iv_k) {
        float mc_scale_frag[64];
        memset(mc_scale_frag, 0, sizeof(mc_scale_frag));
        auto anon_2 = (swiz_block_m * 2 + (__choreo_vg4id_x - 1));
        auto stage = __iv_iv_k % 4;
        // wait event(barrier)  (full elemof stage) 
        full[stage].wait(full[stage].arrive());
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:72.11
        // Call warpgroup_arrive() once before all 4 WGMMAs in this K iteration
        // Per PTX docs: multiple WGMMA instructions sharing same accumulator shape can batch without fences
        warpgroup_arrive();
        {
          int __iv_iv_warp = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:72.11
          for (__iv_iv_warp = 0; __iv_iv_warp < 4; ++__iv_iv_warp) {
            auto anon_1 = __choreo_vg4id_x - 1;
            f8_e4m3* ma_smem_ptr = (f8_e4m3*)((__iv_iv_warp * 32 + ((__choreo_vg4id_x - 1) * 8192 + __iv_iv_k % 4 * 16384) + lhs_load_s));
            uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma_smem_ptr);
            f8_e4m3* mb_smem_ptr = (f8_e4m3*)((__iv_iv_warp * 32 + __iv_iv_k % 4 * 16384 + rhs_load_s));
            uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
            // Note: warpgroup_arrive() called once above, shared by all 4 WGMMAs
            cute::SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<>::fma(desc_ma, desc_mb, mc_scale_frag[0], mc_scale_frag[1], mc_scale_frag[2], mc_scale_frag[3], mc_scale_frag[4], mc_scale_frag[5], mc_scale_frag[6], mc_scale_frag[7], mc_scale_frag[8], mc_scale_frag[9], mc_scale_frag[10], mc_scale_frag[11], mc_scale_frag[12], mc_scale_frag[13], mc_scale_frag[14], mc_scale_frag[15], mc_scale_frag[16], mc_scale_frag[17], mc_scale_frag[18], mc_scale_frag[19], mc_scale_frag[20], mc_scale_frag[21], mc_scale_frag[22], mc_scale_frag[23], mc_scale_frag[24], mc_scale_frag[25], mc_scale_frag[26], mc_scale_frag[27], mc_scale_frag[28], mc_scale_frag[29], mc_scale_frag[30], mc_scale_frag[31], mc_scale_frag[32], mc_scale_frag[33], mc_scale_frag[34], mc_scale_frag[35], mc_scale_frag[36], mc_scale_frag[37], mc_scale_frag[38], mc_scale_frag[39], mc_scale_frag[40], mc_scale_frag[41], mc_scale_frag[42], mc_scale_frag[43], mc_scale_frag[44], mc_scale_frag[45], mc_scale_frag[46], mc_scale_frag[47], mc_scale_frag[48], mc_scale_frag[49], mc_scale_frag[50], mc_scale_frag[51], mc_scale_frag[52], mc_scale_frag[53], mc_scale_frag[54], mc_scale_frag[55], mc_scale_frag[56], mc_scale_frag[57], mc_scale_frag[58], mc_scale_frag[59], mc_scale_frag[60], mc_scale_frag[61], mc_scale_frag[62], mc_scale_frag[63]);
          } // iv_warp
          __iv_iv_warp = 0;
        }
        float* mc_scale_a_ptr = (float*)((DIV_BLK_K * anon_2 * 64 + __iv_iv_k + scale_lhs));
        float mc_scale_b_val = static_cast<float>(*((float*)scale_rhs + (DIV_BLK_K * swiz_block_n) + __iv_iv_k));
        scale_accumulator<float, float, 128>(reinterpret_cast<float*>(mc), reinterpret_cast<float*>(mc_scale_frag), mc_scale_a_ptr, DIV_BLK_K, 64, mc_scale_b_val);
        // Finalize WGMMA operations
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        // trigger event(barrier)  (empty elemof stage) 
        (void)empty[stage].arrive();
      } // iv_k
      __iv_iv_k = 0;
    }
    auto __shape1_output_s = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
    auto __stride1_output_s = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
    auto __layout1_output_s = cute::make_layout(__shape1_output_s, __stride1_output_s);
    auto __tensor1_output_s = cute::make_tensor(cute::make_smem_ptr<f16>((f16*)output_s + ((__choreo_vg4id_x - 1) * 8192)), __layout1_output_s);
    store_fragment_d<CUTE_WGMMA_M64K32, 128>(__tensor1_output_s, reinterpret_cast<float*>(mc));
  } // end inthreads
  future __choreo_anon_fut__0("", 88, 5);
  __choreo_anon_fut__0.is_tma = true;
  __choreo_anon_fut__0.set_atom(&choreo_copy_atom_t_0);
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  if (__CHOREO_BLOCK_SINGLE__) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_2_tensor_map, (swiz_block_n * 128), (swiz_block_m * 128), output_s);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
  }
  } // end parallel-by
}

void blockscale_gemm(const choreo::spanned_view<choreo::f8_e4m3, 2> & lhs, const choreo::spanned_view<choreo::f32, 2> & scale_lhs, const choreo::spanned_view<choreo::f8_e4m3, 2> & rhs, const choreo::spanned_view<choreo::f32, 2> & scale_rhs, const choreo::spanned_view<choreo::f16, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &DIV_BLK_K = scale_lhs.shape()[1];
  auto &DIV_BLK_N = scale_rhs.shape()[0];
  auto &K = lhs.shape()[1];
  auto &M = lhs.shape()[0];
  auto &N = rhs.shape()[0];
  choreo::runtime_check(scale_lhs.shape()[1] == scale_rhs.shape()[1], "The shapes of the 2nd parameter (dim: 1) and the 4th parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(lhs.shape()[1] == rhs.shape()[1], "The shapes of the 1st parameter (dim: 1) and the 3rd parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(lhs.shape()[0] == scale_lhs.shape()[0], "The shapes of the 1st parameter (dim: 0) and the 2nd parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(scale_lhs.shape()[0] == output.shape()[0], "The shapes of the 2nd parameter (dim: 0) and the 5th parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(rhs.shape()[0] == output.shape()[1], "The shapes of the 3rd parameter (dim: 0) and the 5th parameter (dim: 1) are inconsistent.");

  choreo::runtime_check(((static_cast<long long>(M) + 127LL) / 128LL > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:45.13");
  choreo::runtime_check(((static_cast<long long>(N) + 127LL) / 128LL > 0LL), "The 2nd bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:45.22");
  choreo::runtime_check(((static_cast<long long>(K) + 127LL) / 128LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:54.27");
  choreo::runtime_check((static_cast<long long>(K) / ((static_cast<long long>(K) + 127LL) / 128LL) * (static_cast<long long>(N) / ((static_cast<long long>(N) + 127LL) / 128LL)) <= 16384LL), "DMA to-buffer is too small (((::blockscale_gemm::K / ((::blockscale_gemm::K + 127) / 128)) * (::blockscale_gemm::N / ((::blockscale_gemm::N + 127) / 128))) > 16384), tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:58.11");
  choreo::runtime_check(((static_cast<long long>(K) + 127LL) / 128LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:69.27");
  choreo::runtime_check(((static_cast<long long>(N) + 127LL) / 128LL - 1 < static_cast<long long>(DIV_BLK_N)), "The 1st index `block_n` of element access 'scale_rhs' should be less than ::blockscale_gemm::DIV_BLK_N, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_e4m3fp16_8192x8192x4096/opus-4/iter012_swizzle.co:79.97");
  uint64_t __choreo_tma_0_shape[] = {K, M};
  uint64_t __choreo_tma_0_strides[] = {K};
  uint32_t __choreo_tma_0_box_shape[] = {128, 128};
  uint32_t __choreo_tma_0_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_0_tensor_map{};
  CUresult __choreo_tma_0_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_0_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
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
  uint64_t __choreo_tma_1_shape[] = {K, N};
  uint64_t __choreo_tma_1_strides[] = {K};
  uint32_t __choreo_tma_1_box_shape[] = {(K / ((K + 127) / 128)), (N / ((N + 127) / 128))};
  uint32_t __choreo_tma_1_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_1_tensor_map{};
  CUresult __choreo_tma_1_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_1_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
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
  uint64_t __choreo_tma_2_shape[] = {N, M};
  uint64_t __choreo_tma_2_strides[] = {(N * 2)};
  uint32_t __choreo_tma_2_box_shape[] = {128, 128};
  uint32_t __choreo_tma_2_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_2_tensor_map{};
  CUresult __choreo_tma_2_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_2_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
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
  dim3 __blockscale_gemm_gdims0(((M + 127) / 128), ((N + 127) / 128), 1);
  dim3 __blockscale_gemm_bdims0(384, 1, 1);
  cudaFuncSetAttribute(__choreo_device_blockscale_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840 + (1024 - 1));
  __choreo_device_blockscale_gemm<<<__blockscale_gemm_gdims0, __blockscale_gemm_bdims0, 163840 + (1024 - 1)>>>(lhs.data(), scale_lhs.data(), rhs.data(), scale_rhs.data(), output.data(), DIV_BLK_K, DIV_BLK_N, K, M, N, __choreo_tma_0_tensor_map, __choreo_tma_1_tensor_map, __choreo_tma_2_tensor_map);
  choreo::abend_true(cudaDeviceSynchronize());
}




int main(int argc, char** argv) {
  bool enable_timing = true;
  bool skip_verify = false;
  double user_flops = -1.0;
  auto is_disable_timing_arg = [](const char* s) {
    const char* t = "--disable-timing"; int i = 0; while (t[i] != '\0' && s[i] == t[i]) ++i; return t[i] == '\0' && s[i] == '\0';
  };
  auto is_skip_verify_arg = [](const char* s) {
    const char* t = "--skip-verify"; int i = 0; while (t[i] != '\0' && s[i] == t[i]) ++i; return t[i] == '\0' && s[i] == '\0';
  };
  for (int i = 1; i < argc; ++i) {
    if (is_disable_timing_arg(argv[i])) { enable_timing = false; continue; }
    if (is_skip_verify_arg(argv[i])) { skip_verify = true; continue; }
    if (std::strncmp(argv[i], "--flops=", 8) == 0) { user_flops = std::atof(argv[i] + 8); continue; }
  }

  const char* skip_verify_env = std::getenv("CHOREO_SKIP_VERIFY");
  if (skip_verify_env && skip_verify_env[0] == '1' && skip_verify_env[1] == '\0') skip_verify = true;

  size_t M = MATMUL_DEFAULT_M;
  size_t N = MATMUL_DEFAULT_N;
  size_t K = MATMUL_DEFAULT_K;

  auto lhs_h = choreo::make_spandata<choreo::f8_e4m3>(M, K);
  auto scale_lhs_h = choreo::make_spandata<choreo::f32>(M, K / MATMUL_TILE_K);
  auto rhs_h = choreo::make_spandata<choreo::f8_e4m3>(N, K);
  auto scale_rhs_h = choreo::make_spandata<choreo::f32>(N / MATMUL_WARP_N, K / MATMUL_TILE_K);
  auto res_h = choreo::make_spandata<choreo::f16>(M, N);

  lhs_h.fill_random(0, 2);
  rhs_h.fill_random(0, 2);
  scale_lhs_h.fill_random(1, 3);
  scale_rhs_h.fill_random(1, 3);
  res_h.fill(0.0f);

  __nv_fp8_e4m3 *lhs_d = nullptr;
  __nv_fp8_e4m3 *rhs_d = nullptr;
  float *s_lhs_d = nullptr;
  float *s_rhs_d = nullptr;
  half *res_d = nullptr;

  choreo::abend_true(cudaMalloc(&lhs_d, M * K * sizeof(__nv_fp8_e4m3)));
  choreo::abend_true(cudaMalloc(&s_lhs_d, M * (K / MATMUL_TILE_K) * sizeof(float)));
  choreo::abend_true(cudaMalloc(&rhs_d, N * K * sizeof(__nv_fp8_e4m3)));
  choreo::abend_true(cudaMalloc(&s_rhs_d, (N / MATMUL_WARP_N) * (K / MATMUL_TILE_K) * sizeof(float)));
  choreo::abend_true(cudaMalloc(&res_d, M * N * sizeof(half)));

  choreo::abend_true(cudaMemcpy(lhs_d, lhs_h.data(), M * K * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(s_lhs_d, scale_lhs_h.data(), M * (K / MATMUL_TILE_K) * sizeof(float), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(rhs_d, rhs_h.data(), N * K * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(s_rhs_d, scale_rhs_h.data(), (N / MATMUL_WARP_N) * (K / MATMUL_TILE_K) * sizeof(float), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(res_d, res_h.data(), M * N * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());

  auto lhs_d_view = choreo::make_spanview<choreo::f8_e4m3, 2>(lhs_d, {M, K});
  auto s_lhs_view = choreo::make_spanview<choreo::f32, 2>(s_lhs_d, {M, K / MATMUL_TILE_K});
  auto rhs_d_view = choreo::make_spanview<choreo::f8_e4m3, 2>(rhs_d, {N, K});
  auto s_rhs_view = choreo::make_spanview<choreo::f32, 2>(s_rhs_d, {N / MATMUL_WARP_N, K / MATMUL_TILE_K});
  auto out_d = choreo::make_spanview<choreo::f16, 2>(res_d, {M, N});

  if (enable_timing) {
    choreo::TimerOption topt; topt.warmup = 10; topt.repeat = 50;
    auto avg_ms = choreo::timing([&](){ blockscale_gemm(lhs_d_view, s_lhs_view, rhs_d_view, s_rhs_view, out_d); cudaDeviceSynchronize(); }, topt);
    std::cout << "Timing avg ms: " << avg_ms << "\n";
    double flops = (user_flops > 0.0) ? user_flops : (2.0 * double(M) * double(N) * double(K));
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    std::cout << "TFLOPS: " << tflops << "\n";
    double eff = (tflops / H800_PCIE_PEAK_F8_TFLOPS) * 100.0;
    std::cout << "HW efficiency: " << eff << "%\n";
  } else {
    blockscale_gemm(lhs_d_view, s_lhs_view, rhs_d_view, s_rhs_view, out_d);
  }
  cudaMemcpy(res_h.data(), res_d, M*N*sizeof(half), cudaMemcpyDeviceToHost);
  choreo::abend_true(cudaDeviceSynchronize());

  if (skip_verify) {
    std::cout << "Test Passed (verify skipped)\n";
    return 0;
  }

  auto lhs_view = lhs_h.view();
  auto rhs_view = rhs_h.view();
  auto scale_lhs_view = scale_lhs_h.view();
  auto scale_rhs_view = scale_rhs_h.view();
  auto res_view = res_h.view();

  float base_tol = 0.5f;
  float rel_tol = 0.01f;

  size_t verify_m = (M < 128) ? M : 128;
  size_t verify_n = (N < 256) ? N : 256;
  for (size_t i = 0; i < verify_m; ++i) {
    for (size_t j = 0; j < verify_n; ++j) {
      float ref = 0.0f;
      for (size_t kb = 0; kb < (size_t)(K / MATMUL_TILE_K); ++kb) {
        float sa = static_cast<float>(scale_lhs_view[i][kb]);
        float sb = static_cast<float>(scale_rhs_view[j / MATMUL_WARP_N][kb]);
        for (size_t kk = 0; kk < (size_t)MATMUL_TILE_K; ++kk) {
          size_t k = kb * MATMUL_TILE_K + kk;
          ref += static_cast<float>(lhs_view[i][k]) * sa * static_cast<float>(rhs_view[j][k]) * sb;
        }
      }
      float got = __half2float(res_view[i][j]);
      float tol = base_tol + rel_tol * std::abs(ref);
      if (std::abs(got - ref) > tol) {
        std::cout << "[" << i << ", " << j << "] " << ref << " <-> " << got << ", tol: " << tol << "\n";
      }
      choreo::choreo_assert(std::abs(got - ref) <= tol, "values are not equal.");
    }
  }

  std::cout << "Test Passed\n" << std::endl;
}


