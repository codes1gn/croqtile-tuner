
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

#include <cstring>
#include <cstdlib>

#define H800_PCIE_PEAK_F16_TFLOPS 1513
#define H800_PCIE_PEAK_F8_TFLOPS 3026

#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_M 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_SWIZ 128
#define MATMUL_STAGES 3

#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_SWIZ != 32 && MATMUL_SWIZ != 64 && MATMUL_SWIZ != 128
#error "MATMUL_SWIZ must be one of 32, 64, 128"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for f16 WGMMA constraints"
#endif

#if MATMUL_WARP_K != 16
#error "MATMUL_WARP_K must be 16 for f16 WGMMA constraints"
#endif

#if MATMUL_TILE_M % MATMUL_WARP_M != 0
#error "MATMUL_TILE_M must be divisible by MATMUL_WARP_M"
#endif

#if MATMUL_WARP_N < 8 || MATMUL_WARP_N > 256 || (MATMUL_WARP_N % 8) != 0
#error "MATMUL_WARP_N must be in [8,256] and divisible by 8 for SM90 WGMMA f16"
#endif

#define MATMUL_DEFAULT_M 16384
#define MATMUL_DEFAULT_N 16384
#define MATMUL_DEFAULT_K 512

__global__ void __choreo_device_matmul(f16 * lhs, f16 * rhs, float * output, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map) {
  extern __shared__ char __choreo_device_matmul__runtime_shared_buffer__raw[];
  auto __choreo_device_matmul__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<1024 * 8>(__choreo_device_matmul__runtime_shared_buffer__raw));
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:44.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_0_barrier, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  auto anon_3 = (unsigned char*)__choreo_device_matmul__runtime_shared_buffer__;
  __shared__ cuda::barrier<cuda::thread_scope_block> full[3]; // shared event barrier
  // initialize the event barrier
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&full[0], 257);
    init(&full[1], 257);
    init(&full[2], 257);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[3]; // shared event barrier
  // initialize the event barrier
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&empty[0], 257);
    init(&empty[1], 257);
    init(&empty[2], 257);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  f16* lhs_load_s = (f16*)(anon_3 + 65536);
  f16* rhs_load_s = (f16*)(anon_3 + 114688);
  float* output_s = (float*)(anon_3 + 0);
  auto __choreo_vg4id_x = threadIdx.x / 128;
  auto __choreo_vtid_x = threadIdx.x % 128;
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:51.7
  if ((__choreo_vg4id_x == 0 && __choreo_vtid_x == 0)) {
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:52.9
    {
      int __iv_iv_k = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:52.9
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 63) / 64); ++__iv_iv_k) {
        int stage = __iv_iv_k % 3;
        // wait event(barrier)  (empty elemof stage) 
        empty[stage].wait(empty[stage].arrive());
        cde::cp_async_bulk_tensor_2d_global_to_shared((lhs_load_s + ((__iv_iv_k % 3 * 8192))), &__choreo_tma_0_tensor_map, (__iv_iv_k * 64), (blockIdx.x * 128), full[stage]);
        cde::cp_async_bulk_tensor_2d_global_to_shared((rhs_load_s + ((__iv_iv_k % 3 * 8192))), &__choreo_tma_1_tensor_map, (K / ((K + 63) / 64) * __iv_iv_k), (N / ((N + 127) / 128) * blockIdx.y), full[stage]);
        // trigger event(barrier)  (full elemof stage) 
        (void)cuda::device::barrier_arrive_tx(full[stage], 1, (16384) + (16384));
      } // iv_k
      __iv_iv_k = 0;
    }
  } // end inthreads
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:61.7
  if ((__choreo_vg4id_x > 0)) {
    auto anon_2 = __choreo_vg4id_x - 1;
    float mc[64];
    float __frag_init_val0 = 0.000000f;
    for (int idx = 0; idx < 64; ++idx)
      mc[idx] = __frag_init_val0;
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:63.9
    {
      int __iv_s = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:63.9
      for (__iv_s = 0; __iv_s < 3; ++__iv_s) {
        // trigger event(barrier)  (empty elemof s) 
        (void)empty[__iv_s].arrive();
      } // s
      __iv_s = 0;
    }
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:66.9
    {
      int __iv_iv_k = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:66.9
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 63) / 64); ++__iv_iv_k) {
        auto stage = __iv_iv_k % 3;
        // wait event(barrier)  (full elemof stage) 
        full[stage].wait(full[stage].arrive());
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:69.11
        {
          int __iv_iv_warp = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:69.11
          for (__iv_iv_warp = 0; __iv_iv_warp < 4; ++__iv_iv_warp) {
            auto anon_1 = __choreo_vg4id_x - 1;
            f16* ma_smem_ptr = (f16*)((__iv_iv_warp * 16 + ((__choreo_vg4id_x - 1) * 4096 + __iv_iv_k % 3 * 8192) + lhs_load_s));
            uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma_smem_ptr);
            f16* mb_smem_ptr = (f16*)((__iv_iv_warp * 16 + __iv_iv_k % 3 * 8192 + rhs_load_s));
            uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
            warpgroup_arrive();
            // Note: warpgroup_arrive() should be called once before first WGMMA
            // and warpgroup_wait() should be called once after all WGMMAs
            cute::SM90::GMMA::MMA_64x128x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_ma, desc_mb, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63]);
          } // iv_warp
          __iv_iv_warp = 0;
        }
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
    auto __tensor1_output_s = cute::make_tensor(cute::make_smem_ptr<float>((float*)output_s + ((__choreo_vg4id_x - 1) * 8192)), __layout1_output_s);
    store_fragment_d<CUTE_WGMMA_M64K16, 128>(__tensor1_output_s, reinterpret_cast<float*>(mc));
  } // end inthreads
  future __choreo_anon_fut__0("", 81, 5);
  __choreo_anon_fut__0.is_tma = true;
  __choreo_anon_fut__0.set_atom(&choreo_copy_atom_t_0);
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  if (__CHOREO_BLOCK_SINGLE__) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_2_tensor_map, (blockIdx.y * 128), (blockIdx.x * 128), output_s);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
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

  choreo::runtime_check(((static_cast<long long>(M) + 127LL) / 128LL > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:44.13");
  choreo::runtime_check(((static_cast<long long>(N) + 127LL) / 128LL > 0LL), "The 2nd bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:44.22");
  choreo::runtime_check(((static_cast<long long>(K) + 63LL) / 64LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:52.27");
  choreo::runtime_check((static_cast<long long>(K) / ((static_cast<long long>(K) + 63LL) / 64LL) * (static_cast<long long>(N) / ((static_cast<long long>(N) + 127LL) / 128LL)) <= 8192LL), "DMA to-buffer is too small (((::matmul::K / ((::matmul::K + 63) / 64)) * (::matmul::N / ((::matmul::N + 127) / 128))) > 8192), tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:56.11");
  choreo::runtime_check(((static_cast<long long>(K) + 63LL) / 64LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x16384x512/claude-4-5-opus-high/iter001_warpspec_1p2c.co:66.27");
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
  uint64_t __choreo_tma_1_shape[] = {K, N};
  uint64_t __choreo_tma_1_strides[] = {(K * 2)};
  uint32_t __choreo_tma_1_box_shape[] = {(K / ((K + 63) / 64)), (N / ((N + 127) / 128))};
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
  uint64_t __choreo_tma_2_shape[] = {N, M};
  uint64_t __choreo_tma_2_strides[] = {(N * 4)};
  uint32_t __choreo_tma_2_box_shape[] = {128, 128};
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
  dim3 __matmul_gdims0(((M + 127) / 128), ((N + 127) / 128), 1);
  dim3 __matmul_bdims0(384, 1, 1);
  cudaFuncSetAttribute(__choreo_device_matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840 + (1024 - 1));
  __choreo_device_matmul<<<__matmul_gdims0, __matmul_bdims0, 163840 + (1024 - 1)>>>(lhs.data(), rhs.data(), output.data(), K, M, N, __choreo_tma_0_tensor_map, __choreo_tma_1_tensor_map, __choreo_tma_2_tensor_map);
  choreo::abend_true(cudaDeviceSynchronize());
}




int main(int argc, char** argv) {
  bool enable_timing = true;
  bool skip_verify = false;
  double user_flops = -1.0;
  auto is_disable_timing_arg = [](const char* s) {
    const char* t = "--disable-timing";
    int i = 0;
    while (t[i] != '\0' && s[i] == t[i]) ++i;
    return t[i] == '\0' && s[i] == '\0';
  };
  auto is_skip_verify_arg = [](const char* s) {
    const char* t = "--skip-verify";
    int i = 0;
    while (t[i] != '\0' && s[i] == t[i]) ++i;
    return t[i] == '\0' && s[i] == '\0';
  };
  for (int i = 1; i < argc; ++i) {
    if (is_disable_timing_arg(argv[i])) {
      enable_timing = false;
      continue;
    }
    if (is_skip_verify_arg(argv[i])) {
      skip_verify = true;
    }
    if (std::strncmp(argv[i], "--flops=", 8) == 0) {
      user_flops = std::atof(argv[i] + 8);
      continue;
    }
  }

  const char* timing_env = std::getenv("CHOREO_DISABLE_TIMING");
  if (timing_env && timing_env[0] == '1' && timing_env[1] == '\0') {
    enable_timing = false;
  }

  const char* skip_verify_env = std::getenv("CHOREO_SKIP_VERIFY");
  if (skip_verify_env && skip_verify_env[0] == '1' && skip_verify_env[1] == '\0') {
    skip_verify = true;
  }
  
  size_t M = MATMUL_DEFAULT_M;
  size_t N = MATMUL_DEFAULT_N;
  size_t K = MATMUL_DEFAULT_K;

  auto lhs_h = choreo::make_spandata<choreo::f16>(M, K);
  auto rhs_h = choreo::make_spandata<choreo::f16>(N, K);
  auto res_h = choreo::make_spandata<choreo::f32>(M, N);
  lhs_h.fill_random(0, 2);
  rhs_h.fill_random(0, 2);
  res_h.fill(0.0f);

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
    int warmup = 10;
    int repeat = 50;
    const char* warmup_env = std::getenv("CHOREO_TIMING_WARMUP");
    const char* repeat_env = std::getenv("CHOREO_TIMING_REPEAT");
    if (warmup_env) {
      int value = std::atoi(warmup_env);
      if (value >= 0) warmup = value;
    }
    if (repeat_env) {
      int value = std::atoi(repeat_env);
      if (value > 0) repeat = value;
    }
    choreo::TimerOption topt;
    topt.warmup = warmup;
    topt.repeat = repeat;
    auto avg_ms = choreo::timing([&]() { matmul(lhs_d, rhs_d, res_d); cudaDeviceSynchronize(); }, topt);
    std::cout << "Timing avg ms: " << avg_ms << "\n";

    double flops = (user_flops > 0.0) ? user_flops : (2.0 * double(M) * double(N) * double(K));
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    std::cout << "TFLOPS: " << tflops << "\n";

    double eff = (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0;
    std::cout << "HW efficiency: " << eff << "%\n";
  } else {
    matmul(lhs_d, rhs_d, res_d);
  }

  choreo::abend_true(cudaMemcpy(res_h.data(), c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaDeviceSynchronize());

  if (skip_verify) {
    std::cout << "Test Passed (verify skipped)\n" << std::endl;
    return 0;
  }

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
    for (size_t j = 0; j < 256; ++j) {
      float ref = 0.0f;
      for (size_t k = 0; k < lhs_view.shape()[1]; ++k)
        ref += __half2float(lhs_view[i][k] * rhs_view[j][k]);
      float got = res_view[i][j];
      auto delta = rel_error(ref, got);
      if (delta >= tolerance) {
        std::cout << "[" << i << ", " << j << "] " << ref << " <-> " << got << ", delta: " << delta * 100 << "%\n";
      }
      choreo::choreo_assert((delta < tolerance), "values are not equal.");
    }
  }

  std::cout << "Test Passed\n" << std::endl;
}


