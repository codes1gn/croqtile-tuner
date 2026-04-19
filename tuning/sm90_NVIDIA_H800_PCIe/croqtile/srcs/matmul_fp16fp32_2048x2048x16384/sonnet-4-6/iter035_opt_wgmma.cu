
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

// matmul_fp16fp32_2048x2048x16384 iter034: cuBLAS-like structure
// Matching cuBLAS parameters:
// - 384 threads (3 warp-groups: 1 producer + 2 consumers)
// - TILE_M=256, TILE_N=128 (each consumer handles 128 rows = 2×64)
// - TILE_K=64, 2 stages (limited by SMEM)
// - 168+ registers per thread
//
// Grid: ceil(2048/256) × ceil(2048/128) = 8 × 16 = 128 CTAs on 114 SMs (1.12 waves)
// SMEM: 2 stages × (256×64 + 128×64) × 2B = 2 × 24576 × 2 = 98304 B = 96 KB
//       + output_s: 256 × 128 × 4 = 131072 B = 128 KB
//       Total: 229376 B ≈ 224 KB (fits in 228 KB!)
// REQUIRES: TARGET-SM_90
// RUN: choreo -gs -t cute -arch=sm_90a --use-warpspec %s -o %s.cute.result && bash %s.cute.result --execute

#include <cstring>
#include <cstdlib>

#define H800_PCIE_PEAK_F16_TFLOPS 1513

#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_M 256
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_SWIZ 128
#define MATMUL_STAGES 2

#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for WGMMA SM90a"
#endif

#define MATMUL_DEFAULT_M 2048
#define MATMUL_DEFAULT_N 2048
#define MATMUL_DEFAULT_K 16384

__global__ void __choreo_device_matmul(f16 * lhs, f16 * rhs, float * output, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map) {
  extern __shared__ char __choreo_device_matmul__runtime_shared_buffer__raw[];
  auto __choreo_device_matmul__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<1024 * 8>(__choreo_device_matmul__runtime_shared_buffer__raw));
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:41.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_0_barrier, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  auto anon_5 = (unsigned char*)__choreo_device_matmul__runtime_shared_buffer__;
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
  f16* lhs_load_s = (f16*)(anon_5 + 131072);
  f16* rhs_load_s = (f16*)(anon_5 + 196608);
  float* output_s = (float*)(anon_5 + 0);
  auto __choreo_vg4id_x = threadIdx.x / 128;
  auto __choreo_vtid_x = threadIdx.x % 128;
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:50.7
  if ((__choreo_vg4id_x == 0 && __choreo_vtid_x == 0)) {
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:51.9
    {
      int __iv_iv_k = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:51.9
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 63) / 64); ++__iv_iv_k) {
        int stage = __iv_iv_k % 2;
        // wait event(barrier)  (empty elemof stage) 
        empty[stage].wait(empty[stage].arrive());
        cde::cp_async_bulk_tensor_2d_global_to_shared((lhs_load_s + ((__iv_iv_k % 2 * 16384))), &__choreo_tma_0_tensor_map, (__iv_iv_k * 64), (blockIdx.x * 256), full[stage]);
        cde::cp_async_bulk_tensor_2d_global_to_shared((rhs_load_s + ((__iv_iv_k % 2 * 8192))), &__choreo_tma_1_tensor_map, (__iv_iv_k * 64), (blockIdx.y * 128), full[stage]);
        // trigger event(barrier)  (full elemof stage) 
        (void)cuda::device::barrier_arrive_tx(full[stage], 1, (32768) + (16384));
      } // iv_k
      __iv_iv_k = 0;
    }
  } // end inthreads
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:61.7
  if ((__choreo_vg4id_x == 1)) {
    float mc0[64];
    float __frag_init_val0 = static_cast<float>(0.000000);
    for (int idx = 0; idx < 64; ++idx)
      mc0[idx] = __frag_init_val0;
    float mc1[64];
    float __frag_init_val1 = static_cast<float>(0.000000);
    for (int idx = 0; idx < 64; ++idx)
      mc1[idx] = __frag_init_val1;
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:65.9
    {
      int __iv_s = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:65.9
      for (__iv_s = 0; __iv_s < 2; ++__iv_s) {
        // trigger event(barrier)  (empty elemof s) 
        (void)empty[__iv_s].arrive();
      } // s
      __iv_s = 0;
    }
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:69.9
    {
      int __iv_iv_k = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:69.9
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 63) / 64); ++__iv_iv_k) {
        auto stage = __iv_iv_k % 2;
        // wait event(barrier)  (full elemof stage) 
        full[stage].wait(full[stage].arrive());
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:73.11
        {
          int __iv_iv_warp = 0;
          warpgroup_arrive();
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:73.11
          for (__iv_iv_warp = 0; __iv_iv_warp < 4; ++__iv_iv_warp) {
            auto anon_1 = stage * 4 + 0;
            auto anon_2 = stage * 4 + 1;
            f16* mb_smem_ptr = (f16*)((__iv_iv_warp * 16 + __iv_iv_k % 2 * 8192 + rhs_load_s));
            uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
            f16* ma0_smem_ptr = (f16*)((__iv_iv_warp * 16 + __iv_iv_k % 2 * 16384 + lhs_load_s));
            uint64_t desc_ma0 = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma0_smem_ptr);
            // Note: warpgroup_arrive() should be called once before first WGMMA
            // and warpgroup_wait() should be called once after all WGMMAs
            cute::SM90::GMMA::MMA_64x128x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_ma0, desc_mb, mc0[0], mc0[1], mc0[2], mc0[3], mc0[4], mc0[5], mc0[6], mc0[7], mc0[8], mc0[9], mc0[10], mc0[11], mc0[12], mc0[13], mc0[14], mc0[15], mc0[16], mc0[17], mc0[18], mc0[19], mc0[20], mc0[21], mc0[22], mc0[23], mc0[24], mc0[25], mc0[26], mc0[27], mc0[28], mc0[29], mc0[30], mc0[31], mc0[32], mc0[33], mc0[34], mc0[35], mc0[36], mc0[37], mc0[38], mc0[39], mc0[40], mc0[41], mc0[42], mc0[43], mc0[44], mc0[45], mc0[46], mc0[47], mc0[48], mc0[49], mc0[50], mc0[51], mc0[52], mc0[53], mc0[54], mc0[55], mc0[56], mc0[57], mc0[58], mc0[59], mc0[60], mc0[61], mc0[62], mc0[63]);
            f16* ma1_smem_ptr = (f16*)((__iv_iv_warp * 16 + (__iv_iv_k % 2 * 4 + 1) * 4096 + lhs_load_s));
            uint64_t desc_ma1 = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma1_smem_ptr);
            // Note: warpgroup_arrive() should be called once before first WGMMA
            // and warpgroup_wait() should be called once after all WGMMAs
            cute::SM90::GMMA::MMA_64x128x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_ma1, desc_mb, mc1[0], mc1[1], mc1[2], mc1[3], mc1[4], mc1[5], mc1[6], mc1[7], mc1[8], mc1[9], mc1[10], mc1[11], mc1[12], mc1[13], mc1[14], mc1[15], mc1[16], mc1[17], mc1[18], mc1[19], mc1[20], mc1[21], mc1[22], mc1[23], mc1[24], mc1[25], mc1[26], mc1[27], mc1[28], mc1[29], mc1[30], mc1[31], mc1[32], mc1[33], mc1[34], mc1[35], mc1[36], mc1[37], mc1[38], mc1[39], mc1[40], mc1[41], mc1[42], mc1[43], mc1[44], mc1[45], mc1[46], mc1[47], mc1[48], mc1[49], mc1[50], mc1[51], mc1[52], mc1[53], mc1[54], mc1[55], mc1[56], mc1[57], mc1[58], mc1[59], mc1[60], mc1[61], mc1[62], mc1[63]);
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
    auto __tensor1_output_s = cute::make_tensor(cute::make_smem_ptr<float>((float*)output_s + 0), __layout1_output_s);
    store_fragment_d<CUTE_WGMMA_M64K16, 128>(__tensor1_output_s, reinterpret_cast<float*>(mc0));
    auto __shape2_output_s = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
    auto __stride2_output_s = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
    auto __layout2_output_s = cute::make_layout(__shape2_output_s, __stride2_output_s);
    auto __tensor2_output_s = cute::make_tensor(cute::make_smem_ptr<float>((float*)output_s + 8192), __layout2_output_s);
    store_fragment_d<CUTE_WGMMA_M64K16, 128>(__tensor2_output_s, reinterpret_cast<float*>(mc1));
  } // end inthreads
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:96.7
  if ((__choreo_vg4id_x == 2)) {
    float mc2[64];
    float __frag_init_val2 = static_cast<float>(0.000000);
    for (int idx = 0; idx < 64; ++idx)
      mc2[idx] = __frag_init_val2;
    float mc3[64];
    float __frag_init_val3 = static_cast<float>(0.000000);
    for (int idx = 0; idx < 64; ++idx)
      mc3[idx] = __frag_init_val3;
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:100.9
    {
      int __iv_s = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:100.9
      for (__iv_s = 0; __iv_s < 2; ++__iv_s) {
        // trigger event(barrier)  (empty elemof s) 
        (void)empty[__iv_s].arrive();
      } // s
      __iv_s = 0;
    }
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:104.9
    {
      int __iv_iv_k = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:104.9
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 63) / 64); ++__iv_iv_k) {
        auto stage = __iv_iv_k % 2;
        // wait event(barrier)  (full elemof stage) 
        full[stage].wait(full[stage].arrive());
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:108.11
        {
          int __iv_iv_warp = 0;
          warpgroup_arrive();
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:108.11
          for (__iv_iv_warp = 0; __iv_iv_warp < 4; ++__iv_iv_warp) {
            auto anon_3 = stage * 4 + 2;
            auto anon_4 = stage * 4 + 3;
            f16* mb_smem_ptr = (f16*)((__iv_iv_warp * 16 + __iv_iv_k % 2 * 8192 + rhs_load_s));
            uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
            f16* ma2_smem_ptr = (f16*)((__iv_iv_warp * 16 + (__iv_iv_k % 2 * 4 + 2) * 4096 + lhs_load_s));
            uint64_t desc_ma2 = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma2_smem_ptr);
            // Note: warpgroup_arrive() should be called once before first WGMMA
            // and warpgroup_wait() should be called once after all WGMMAs
            cute::SM90::GMMA::MMA_64x128x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_ma2, desc_mb, mc2[0], mc2[1], mc2[2], mc2[3], mc2[4], mc2[5], mc2[6], mc2[7], mc2[8], mc2[9], mc2[10], mc2[11], mc2[12], mc2[13], mc2[14], mc2[15], mc2[16], mc2[17], mc2[18], mc2[19], mc2[20], mc2[21], mc2[22], mc2[23], mc2[24], mc2[25], mc2[26], mc2[27], mc2[28], mc2[29], mc2[30], mc2[31], mc2[32], mc2[33], mc2[34], mc2[35], mc2[36], mc2[37], mc2[38], mc2[39], mc2[40], mc2[41], mc2[42], mc2[43], mc2[44], mc2[45], mc2[46], mc2[47], mc2[48], mc2[49], mc2[50], mc2[51], mc2[52], mc2[53], mc2[54], mc2[55], mc2[56], mc2[57], mc2[58], mc2[59], mc2[60], mc2[61], mc2[62], mc2[63]);
            f16* ma3_smem_ptr = (f16*)((__iv_iv_warp * 16 + (__iv_iv_k % 2 * 4 + 3) * 4096 + lhs_load_s));
            uint64_t desc_ma3 = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma3_smem_ptr);
            // Note: warpgroup_arrive() should be called once before first WGMMA
            // and warpgroup_wait() should be called once after all WGMMAs
            cute::SM90::GMMA::MMA_64x128x16_F32F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_ma3, desc_mb, mc3[0], mc3[1], mc3[2], mc3[3], mc3[4], mc3[5], mc3[6], mc3[7], mc3[8], mc3[9], mc3[10], mc3[11], mc3[12], mc3[13], mc3[14], mc3[15], mc3[16], mc3[17], mc3[18], mc3[19], mc3[20], mc3[21], mc3[22], mc3[23], mc3[24], mc3[25], mc3[26], mc3[27], mc3[28], mc3[29], mc3[30], mc3[31], mc3[32], mc3[33], mc3[34], mc3[35], mc3[36], mc3[37], mc3[38], mc3[39], mc3[40], mc3[41], mc3[42], mc3[43], mc3[44], mc3[45], mc3[46], mc3[47], mc3[48], mc3[49], mc3[50], mc3[51], mc3[52], mc3[53], mc3[54], mc3[55], mc3[56], mc3[57], mc3[58], mc3[59], mc3[60], mc3[61], mc3[62], mc3[63]);
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
    auto __shape3_output_s = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
    auto __stride3_output_s = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
    auto __layout3_output_s = cute::make_layout(__shape3_output_s, __stride3_output_s);
    auto __tensor3_output_s = cute::make_tensor(cute::make_smem_ptr<float>((float*)output_s + 16384), __layout3_output_s);
    store_fragment_d<CUTE_WGMMA_M64K16, 128>(__tensor3_output_s, reinterpret_cast<float*>(mc2));
    auto __shape4_output_s = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
    auto __stride4_output_s = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
    auto __layout4_output_s = cute::make_layout(__shape4_output_s, __stride4_output_s);
    auto __tensor4_output_s = cute::make_tensor(cute::make_smem_ptr<float>((float*)output_s + 24576), __layout4_output_s);
    store_fragment_d<CUTE_WGMMA_M64K16, 128>(__tensor4_output_s, reinterpret_cast<float*>(mc3));
  } // end inthreads
  future __choreo_anon_fut__0("", 131, 5);
  __choreo_anon_fut__0.is_tma = true;
  __choreo_anon_fut__0.set_atom(&choreo_copy_atom_t_0);
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  if (__CHOREO_BLOCK_SINGLE__) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_2_tensor_map, (blockIdx.y * 128), (blockIdx.x * 256), output_s);
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

  choreo::runtime_check(((static_cast<long long>(M) + 255LL) / 256LL > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:41.13");
  choreo::runtime_check(((static_cast<long long>(N) + 127LL) / 128LL > 0LL), "The 2nd bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:41.22");
  choreo::runtime_check(((static_cast<long long>(K) + 63LL) / 64LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:51.27");
  choreo::runtime_check(((static_cast<long long>(K) + 63LL) / 64LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:69.27");
  choreo::runtime_check(((static_cast<long long>(K) + 63LL) / 64LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_fp16fp32_2048x2048x16384/sonnet-4-6/iter034_cublas_struct.co:104.27");
  uint64_t __choreo_tma_0_shape[] = {K, M};
  uint64_t __choreo_tma_0_strides[] = {(K * 2)};
  uint32_t __choreo_tma_0_box_shape[] = {64, 256};
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
  uint32_t __choreo_tma_1_box_shape[] = {64, 128};
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
  uint32_t __choreo_tma_2_box_shape[] = {128, 256};
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
  dim3 __matmul_gdims0(((M + 255) / 256), ((N + 127) / 128), 1);
  dim3 __matmul_bdims0(384, 1, 1);
  cudaFuncSetAttribute(__choreo_device_matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, 229376 + (1024 - 1));
  __choreo_device_matmul<<<__matmul_gdims0, __matmul_bdims0, 229376 + (1024 - 1)>>>(lhs.data(), rhs.data(), output.data(), K, M, N, __choreo_tma_0_tensor_map, __choreo_tma_1_tensor_map, __choreo_tma_2_tensor_map);
  choreo::abend_true(cudaDeviceSynchronize());
}




int main(int argc, char** argv) {
  bool enable_timing = true;
  bool skip_verify = false;
  double user_flops = -1.0;

  size_t M = MATMUL_DEFAULT_M;
  size_t N = MATMUL_DEFAULT_N;
  size_t K = MATMUL_DEFAULT_K;

  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--disable-timing", 16) == 0) { enable_timing = false; continue; }
    if (std::strncmp(argv[i], "--skip-verify", 13) == 0)    { skip_verify = true; continue; }
    if (std::strncmp(argv[i], "--m=", 4) == 0) { M = std::atol(argv[i] + 4); continue; }
    if (std::strncmp(argv[i], "--n=", 4) == 0) { N = std::atol(argv[i] + 4); continue; }
    if (std::strncmp(argv[i], "--k=", 4) == 0) { K = std::atol(argv[i] + 4); continue; }
    if (std::strncmp(argv[i], "--flops=", 8) == 0) { user_flops = std::atof(argv[i] + 8); continue; }
  }

  const char* te = std::getenv("CHOREO_DISABLE_TIMING");
  if (te && te[0] == '1' && te[1] == '\0') enable_timing = false;
  const char* sv = std::getenv("CHOREO_SKIP_VERIFY");
  if (sv && sv[0] == '1' && sv[1] == '\0') skip_verify = true;

  auto lhs_h = choreo::make_spandata<choreo::f16>(M, K);
  auto rhs_h = choreo::make_spandata<choreo::f16>(N, K);
  auto res_h = choreo::make_spandata<choreo::f32>(M, N);
  lhs_h.fill_random(-1.0f, 1.0f);
  rhs_h.fill_random(-1.0f, 1.0f);
  res_h.fill(0.0f);

  half  *a_d = nullptr, *b_d = nullptr;
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

  matmul(lhs_d, rhs_d, res_d);
  choreo::abend_true(cudaDeviceSynchronize());

  if (!skip_verify) {
    choreo::abend_true(cudaMemcpy(res_h.data(), c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    auto lhs_view = lhs_h.view();
    auto rhs_view = rhs_h.view();
    auto res_view = res_h.view();
    float base_tol = 1.0f, rel_tol = 0.01f;
    size_t sample_step = std::max(1UL, M * N / 2000UL);
    for (size_t idx = 0; idx < M * N; idx += sample_step) {
      size_t i = idx / N, j = idx % N;
      float ref = 0.0f;
      for (size_t k = 0; k < K; ++k)
        ref += __half2float(lhs_view[i][k]) * __half2float(rhs_view[j][k]);
      float got = res_view[i][j];
      float tol = base_tol + rel_tol * std::abs(ref);
      if (std::abs(got - ref) > tol) {
        std::cout << "[" << i << "," << j << "] ref=" << ref << " got=" << got << "\n";
        choreo::choreo_assert(false, "Verification failed");
      }
    }
    std::cout << "Verification passed (sampled " << (M * N / sample_step) << " elements)\n";
  }

  if (enable_timing) {
    int warmup = 10, repeat = 50;
    const char* we = std::getenv("CHOREO_TIMING_WARMUP");
    const char* re = std::getenv("CHOREO_TIMING_REPEAT");
    if (we) { int v = std::atoi(we); if (v >= 0) warmup = v; }
    if (re) { int v = std::atoi(re); if (v >  0) repeat = v; }

    choreo::TimerOption topt;
    topt.warmup = warmup;
    topt.repeat = repeat;
    auto avg_ms = choreo::timing([&]() { matmul(lhs_d, rhs_d, res_d); cudaDeviceSynchronize(); }, topt);
    double flops = (user_flops > 0.0) ? user_flops : (2.0 * double(M) * double(N) * double(K));
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    double eff = tflops / H800_PCIE_PEAK_F16_TFLOPS * 100.0;

    std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "Timing avg ms: " << avg_ms << "\n";
    std::cout << "TFLOPS: " << tflops << "\n";
    std::cout << "HW efficiency: " << eff << "%\n";
  }

  choreo::abend_true(cudaFree(a_d));
  choreo::abend_true(cudaFree(b_d));
  choreo::abend_true(cudaFree(c_d));
  return 0;
}


