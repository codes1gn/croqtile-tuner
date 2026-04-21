
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

// matmul_fp16fp32_2048x2048x16384 iter050: Non-persistent WARP_N=160, TILE_K=128, 2 stages
// Hypothesis: Larger TILE_K halves K loop iterations, reducing loop overhead
// REQUIRES: TARGET-SM_90
// RUN: choreo -gs -t cute -arch=sm_90a --use-warpspec %s -o %s.cute.result && bash %s.cute.result --execute

#include <cstring>
#include <cstdlib>
#include <chrono>

#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 160
#define MATMUL_TILE_M 128
#define MATMUL_TILE_K 128
#define MATMUL_WARP_K 16
#define MATMUL_SWIZ 128
#define MATMUL_STAGES 2
#define NUM_CONSUMERS 2

#define MATMUL_DEFAULT_M 2048
#define MATMUL_DEFAULT_N 2048
#define MATMUL_DEFAULT_K 16384

__global__ void __choreo_device_matmul_fp16fp32_np_wn160_tk128(f16 * lhs, f16 * rhs, float * output, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map) {
  extern __shared__ char __choreo_device_matmul_fp16fp32_np_wn160_tk128__runtime_shared_buffer__raw[];
  auto __choreo_device_matmul_fp16fp32_np_wn160_tk128__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<1024 * 8>(__choreo_device_matmul_fp16fp32_np_wn160_tk128__runtime_shared_buffer__raw));
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:24.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_0_barrier, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  auto anon_3 = (unsigned char*)__choreo_device_matmul_fp16fp32_np_wn160_tk128__runtime_shared_buffer__;
  __shared__ cuda::barrier<cuda::thread_scope_block> full[2]; // shared event barrier
  // initialize the event barrier
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&full[0], 257);
    init(&full[1], 257);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[2]; // shared event barrier
  // initialize the event barrier
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&empty[0], 257);
    init(&empty[1], 257);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  f16* lhs_load_s = (f16*)(anon_3 + 163840);
  f16* rhs_load_s = (f16*)(anon_3 + 81920);
  float* output_s = (float*)(anon_3 + 0);
  auto __choreo_vg4id_x = threadIdx.x / 128;
  auto __choreo_vtid_x = threadIdx.x % 128;
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:31.7
  if ((__choreo_vg4id_x == 0 && __choreo_vtid_x == 0)) {
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:32.9
    {
      int __iv_iv_k = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:32.9
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 127) / 128); ++__iv_iv_k) {
        int stage = __iv_iv_k % 2;
        // wait event(barrier)  (empty elemof stage) 
        empty[stage].wait(empty[stage].arrive());
        cde::cp_async_bulk_tensor_2d_global_to_shared((lhs_load_s + ((__iv_iv_k % 2 * 16384))), &__choreo_tma_0_tensor_map, (__iv_iv_k * 128), (blockIdx.x * 128), full[stage]);
        cde::cp_async_bulk_tensor_2d_global_to_shared((rhs_load_s + ((__iv_iv_k % 2 * 20480))), &__choreo_tma_1_tensor_map, (__iv_iv_k * 128), (blockIdx.y * 160), full[stage]);
        // trigger event(barrier)  (full elemof stage) 
        (void)cuda::device::barrier_arrive_tx(full[stage], 1, (32768) + (40960));
      } // iv_k
      __iv_iv_k = 0;
    }
  } // end inthreads
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:40.7
  if ((__choreo_vg4id_x > 0)) {
    auto anon_2 = __choreo_vg4id_x - 1;
    float mc[80];
    float __frag_init_val0 = static_cast<float>(0.000000);
    for (int idx = 0; idx < 80; ++idx)
      mc[idx] = __frag_init_val0;
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:42.9
    {
      int __iv_s = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:42.9
      for (__iv_s = 0; __iv_s < 2; ++__iv_s) {
        // trigger event(barrier)  (empty elemof s) 
        (void)empty[__iv_s].arrive();
      } // s
      __iv_s = 0;
    }
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:45.9
    {
      int __iv_iv_k = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:45.9
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 127) / 128); ++__iv_iv_k) {
        auto stage = __iv_iv_k % 2;
        // wait event(barrier)  (full elemof stage) 
        full[stage].wait(full[stage].arrive());
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:48.11
        {
          int __iv_iv_warp = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:48.11
          for (__iv_iv_warp = 0; __iv_iv_warp < 8; ++__iv_iv_warp) {
            auto anon_1 = __choreo_vg4id_x - 1;
            f16* ma_smem_ptr = (f16*)((__iv_iv_warp * 16 + ((__choreo_vg4id_x - 1) * 8192 + __iv_iv_k % 2 * 16384) + lhs_load_s));
            uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma_smem_ptr);
            f16* mb_smem_ptr = (f16*)((__iv_iv_warp * 16 + __iv_iv_k % 2 * 20480 + rhs_load_s));
            uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
            warpgroup_arrive();
            // Note: warpgroup_arrive() should be called once before first WGMMA
            // and warpgroup_wait() should be called once after all WGMMAs
            cute::SM90::GMMA::MMA_64x160x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_ma, desc_mb, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], mc[64], mc[65], mc[66], mc[67], mc[68], mc[69], mc[70], mc[71], mc[72], mc[73], mc[74], mc[75], mc[76], mc[77], mc[78], mc[79]);
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
    auto __shape1_output_s = cute::make_shape(cute::Int<64>{}, cute::Int<160>{});
    auto __stride1_output_s = cute::make_stride(cute::Int<160>{}, cute::Int<1>{});
    auto __layout1_output_s = cute::make_layout(__shape1_output_s, __stride1_output_s);
    auto __tensor1_output_s = cute::make_tensor(cute::make_smem_ptr<float>((float*)output_s + ((__choreo_vg4id_x - 1) * 10240)), __layout1_output_s);
    store_fragment_d<CUTE_WGMMA_M64K16, 160>(__tensor1_output_s, reinterpret_cast<float*>(mc));
  } // end inthreads
  future __choreo_anon_fut__0("", 60, 5);
  __choreo_anon_fut__0.is_tma = true;
  __choreo_anon_fut__0.set_atom(&choreo_copy_atom_t_0);
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  if (__CHOREO_BLOCK_SINGLE__) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_2_tensor_map, (blockIdx.y * 160), (blockIdx.x * 128), output_s);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
  }
  } // end parallel-by
}

void matmul_fp16fp32_np_wn160_tk128(const choreo::spanned_view<choreo::f16, 2> & lhs, const choreo::spanned_view<choreo::f16, 2> & rhs, const choreo::spanned_view<choreo::f32, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &K = lhs.shape()[1];
  auto &M = lhs.shape()[0];
  auto &N = rhs.shape()[0];
  choreo::runtime_check(lhs.shape()[1] == rhs.shape()[1], "The shapes of the 1st parameter (dim: 1) and the 2nd parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(lhs.shape()[0] == output.shape()[0], "The shapes of the 1st parameter (dim: 0) and the 3rd parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(rhs.shape()[0] == output.shape()[1], "The shapes of the 2nd parameter (dim: 0) and the 3rd parameter (dim: 1) are inconsistent.");

  choreo::runtime_check(((static_cast<long long>(M) + 127LL) / 128LL > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:24.13");
  choreo::runtime_check(((static_cast<long long>(N) + 159LL) / 160LL > 0LL), "The 2nd bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:24.22");
  choreo::runtime_check(((static_cast<long long>(K) + 127LL) / 128LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:32.27");
  choreo::runtime_check(((static_cast<long long>(K) + 127LL) / 128LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/opus-4/iter050_np_wn160_tk128_s2.co:45.27");
  uint64_t __choreo_tma_0_shape[] = {K, M};
  uint64_t __choreo_tma_0_strides[] = {(K * 2)};
  uint32_t __choreo_tma_0_box_shape[] = {128, 128};
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
  uint32_t __choreo_tma_1_box_shape[] = {128, 160};
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
  uint32_t __choreo_tma_2_box_shape[] = {160, 128};
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
  dim3 __matmul_fp16fp32_np_wn160_tk128_gdims0(((M + 127) / 128), ((N + 159) / 160), 1);
  dim3 __matmul_fp16fp32_np_wn160_tk128_bdims0(384, 1, 1);
  cudaFuncSetAttribute(__choreo_device_matmul_fp16fp32_np_wn160_tk128, cudaFuncAttributeMaxDynamicSharedMemorySize, 229376 + (1024 - 1));
  __choreo_device_matmul_fp16fp32_np_wn160_tk128<<<__matmul_fp16fp32_np_wn160_tk128_gdims0, __matmul_fp16fp32_np_wn160_tk128_bdims0, 229376 + (1024 - 1)>>>(lhs.data(), rhs.data(), output.data(), K, M, N, __choreo_tma_0_tensor_map, __choreo_tma_1_tensor_map, __choreo_tma_2_tensor_map);
  choreo::abend_true(cudaDeviceSynchronize());
}




int main(int argc, char** argv) {
  bool skip_verify = false;
  int warmup_iters = 10;
  int timed_iters = 50;

  size_t M = MATMUL_DEFAULT_M, N = MATMUL_DEFAULT_N, K = MATMUL_DEFAULT_K;

  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--skip-verify", 13) == 0) skip_verify = true;
    if (std::strncmp(argv[i], "--warmup=", 9) == 0) warmup_iters = std::atoi(argv[i] + 9);
    if (std::strncmp(argv[i], "--iters=", 8) == 0) timed_iters = std::atoi(argv[i] + 8);
    if (std::strncmp(argv[i], "--m=", 4) == 0) M = std::atol(argv[i] + 4);
    if (std::strncmp(argv[i], "--n=", 4) == 0) N = std::atol(argv[i] + 4);
    if (std::strncmp(argv[i], "--k=", 4) == 0) K = std::atol(argv[i] + 4);
  }
  const char* env = std::getenv("CHOREO_SKIP_VERIFY");
  if (env && env[0] == '1') skip_verify = true;

  auto lhs_h = choreo::make_spandata<choreo::f16>(M, K);
  auto rhs_h = choreo::make_spandata<choreo::f16>(N, K);
  auto res_h = choreo::make_spandata<choreo::f32>(M, N);
  lhs_h.fill_random(-1.0f, 1.0f);
  rhs_h.fill_random(-1.0f, 1.0f);
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

  matmul_fp16fp32_np_wn160_tk128(choreo::make_spanview<choreo::f16, 2>(a_d, {M, K}),
                                  choreo::make_spanview<choreo::f16, 2>(b_d, {N, K}),
                                  choreo::make_spanview<choreo::f32, 2>(c_d, {M, N}));
  choreo::abend_true(cudaDeviceSynchronize());

  if (!skip_verify) {
    choreo::abend_true(cudaMemcpy(res_h.data(), c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    auto lhs_view = lhs_h.view();
    auto rhs_view = rhs_h.view();
    auto res_view = res_h.view();
    float base_tol = 1.0f, rel_tol = 0.01f;
    size_t sample_step = std::max(1UL, M * N / 1000);
    for (size_t idx = 0; idx < M * N; idx += sample_step) {
      size_t i = idx / N, j = idx % N;
      float ref = 0.0f;
      for (size_t k = 0; k < K; ++k)
        ref += __half2float(lhs_view[i][k]) * __half2float(rhs_view[j][k]);
      float got = res_view[i][j];
      float tol = base_tol + rel_tol * std::abs(ref);
      if (std::abs(got - ref) > tol) {
        std::cout << "[" << i << ", " << j << "] ref=" << ref << " got=" << got << "\n";
        choreo::choreo_assert(false, "Verification failed");
      }
    }
    std::cout << "Verification passed (sampled " << (M * N / sample_step) << " elements)\n";
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < warmup_iters; ++i) {
    matmul_fp16fp32_np_wn160_tk128(choreo::make_spanview<choreo::f16, 2>(a_d, {M, K}),
                                    choreo::make_spanview<choreo::f16, 2>(b_d, {N, K}),
                                    choreo::make_spanview<choreo::f32, 2>(c_d, {M, N}));
  }
  choreo::abend_true(cudaDeviceSynchronize());

  cudaEventRecord(start);
  for (int i = 0; i < timed_iters; ++i) {
    matmul_fp16fp32_np_wn160_tk128(choreo::make_spanview<choreo::f16, 2>(a_d, {M, K}),
                                    choreo::make_spanview<choreo::f16, 2>(b_d, {N, K}),
                                    choreo::make_spanview<choreo::f32, 2>(c_d, {M, N}));
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  double avg_ms = elapsed_ms / timed_iters;
  double flops = 2.0 * M * N * K;
  double tflops = (flops / (avg_ms * 1e-3)) / 1e12;

  std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";
  std::cout << "Warmup: " << warmup_iters << ", Timed: " << timed_iters << "\n";
  std::cout << "Avg time: " << avg_ms << " ms\n";
  std::cout << "TFLOPS: " << tflops << "\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  return 0;
}


