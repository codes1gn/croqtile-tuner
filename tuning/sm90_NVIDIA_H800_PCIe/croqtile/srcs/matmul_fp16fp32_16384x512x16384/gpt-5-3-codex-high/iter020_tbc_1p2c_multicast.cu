
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

// iter020: Persistent 1p2c warpspec + TBC (Thread Block Cluster) with RHS multicast.
// CLUSTER_M=2: pairs of CTAs share the same block_n → TMA multicast halves B bandwidth.
// With 114 SMs, there are 57 clusters of 2 CTAs each.
// Consecutive tile_ids (k and k+1) differ by 1 in block_m but share block_n when within
// the same tiles_m=128 group → multicast always valid for this problem.
// SMEM: 4*(2*WARP_M + WARP_N)*TILE_K*2 + 2*WARP_M*WARP_N*4
//     = 4*(128+128)*64*2 + 2*64*128*4 = 131072 + 65536 = 196608 = 192KB
// Still 1 CTA/SM, but B bandwidth halved → ~2× effective bandwidth utilization.
// Expected: 350-430T (approaching cuBLAS).

#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_M 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_SWIZ 128
#define MATMUL_STAGES 4
#define MATMUL_NUM_CONSUMERS 2
// 57 clusters × 2 CTAs per cluster = 114 CTAs total.
#define CLUSTER_M 2
#define NUM_CLUSTERS 57

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
#define MATMUL_DEFAULT_N 512
#define MATMUL_DEFAULT_K 16384

// cluster parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:63.12
__cluster_dims__(2, 1, 1) __global__ void __choreo_device_matmul(f16 * lhs, f16 * rhs, float * output, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_3_tensor_map) {
  extern __shared__ char __choreo_device_matmul__runtime_shared_buffer__raw[];
  auto __choreo_device_matmul__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<1024 * 8>(__choreo_device_matmul__runtime_shared_buffer__raw));
  choreo::tma_cluster_sync();
auto anon_4 = (unsigned char*)__choreo_device_matmul__runtime_shared_buffer__;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:64.14
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ __align__(8) uint64_t choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    choreo::tma_mbarrier_init(&choreo_copy_atom_t_0_barrier, 1);
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{};
  choreo_copy_atom_t_0.EnablePTXMBarrier(&choreo_copy_atom_t_0_barrier);

  __shared__ __align__(8) uint64_t choreo_copy_atom_t_1_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    choreo::tma_mbarrier_init(&choreo_copy_atom_t_1_barrier, 1);
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_1{};
  choreo_copy_atom_t_1.EnablePTXMBarrier(&choreo_copy_atom_t_1_barrier);

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
  __shared__ __align__(8) uint64_t empty[4]; // shared event mbarrier (cluster-scope)
  int empty__phase[4];
  // initialize the event barrier
  if (__CHOREO_BLOCK_SINGLE__) {
    choreo::tma_mbarrier_init(&empty[0], (blockDim.x / 128 - 1) * choreo::tma_cluster_dim());
    choreo::tma_mbarrier_init(&empty[1], (blockDim.x / 128 - 1) * choreo::tma_cluster_dim());
    choreo::tma_mbarrier_init(&empty[2], (blockDim.x / 128 - 1) * choreo::tma_cluster_dim());
    choreo::tma_mbarrier_init(&empty[3], (blockDim.x / 128 - 1) * choreo::tma_cluster_dim());
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  empty__phase[0] = 0;
  empty__phase[1] = 0;
  empty__phase[2] = 0;
  empty__phase[3] = 0;
  f16* lhs_load_s = (f16*)(anon_4 + 65536);
  f16* rhs_load_s = (f16*)(anon_4 + 0);
  float* output_s_1 = (float*)(anon_4 + 163840);
  float* output_s_2 = (float*)(anon_4 + 131072);
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:71.7
  {
    int __iv_tile_iter = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:71.7
    for (__iv_tile_iter = 0; __iv_tile_iter < (((M + 127) / 128 * ((N + 127) / 128) + 113) / 114); ++__iv_tile_iter) {
      // cluster_id = blockIdx.x / 2 (each cluster has 2 physical blocks)
      // tile_id = cluster_id * 2 + tma_cluster_rank() + iter * 114
      int tile_id = ((blockIdx.x / 2) * 2 + tma_cluster_rank()) + __iv_tile_iter * 114;
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:76.9
      if ((tile_id < (M + 127) / 128 * ((N + 127) / 128))) {
        int block_n = (tile_id / ((M + 127) / 128));
        int block_m = (tile_id % ((M + 127) / 128));
        auto __choreo_vg4id_x = threadIdx.x / 128;
        auto __choreo_vtid_x = threadIdx.x % 128;
        // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:82.13
        if ((__choreo_vg4id_x == 0 && __choreo_vtid_x == 0)) {
          // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:83.15
          {
            int __iv_iv_k = 0;
            // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:83.15
            for (__iv_iv_k = 0; __iv_iv_k < ((K + 63) / 64); ++__iv_iv_k) {
              int stage = __iv_iv_k % 4;
              // wait event(mbarrier)  (empty elemof stage)  [cluster-scope]
              choreo::tma_mbarrier_wait_parity(&empty[stage], empty__phase[stage]);
              empty__phase[stage] ^= 1;
              choreo::tma_load_2d_shared_cluster_global_mbarrier((void*)(lhs_load_s + ((__iv_iv_k % 4 * 8192))), (const void*)&__choreo_tma_0_tensor_map, (uint64_t*)&(full[stage]), (__iv_iv_k * 64), (block_m * 128));
              if (choreo::tma_cluster_rank() == 0) {
                choreo::tma_load_2d_shared_cluster_global_mbarrier_multicast((void*)(rhs_load_s + ((__iv_iv_k % 4 * 8192))), (const void*)&__choreo_tma_1_tensor_map, (uint64_t*)&(full[stage]), (__iv_iv_k * 64), (block_n * 128), static_cast<uint16_t>((1u << 2) - 1u));
              }
              // trigger event(barrier)  (full elemof stage) 
              (void)cuda::device::barrier_arrive_tx(full[stage], 1, (16384) + (16384));
            } // iv_k
            __iv_iv_k = 0;
          }
        } // end inthreads
        // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:96.13
        if ((__choreo_vg4id_x > 0)) {
          float mc[64];
          float __frag_init_val0 = 0.000000f;
          for (int idx = 0; idx < 64; ++idx)
            mc[idx] = __frag_init_val0;
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:98.15
          if ((__iv_tile_iter == 0)) {
            // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:99.17
            {
              int __iv_s = 0;
              // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:99.17
              for (__iv_s = 0; __iv_s < 4; ++__iv_s) {
                // trigger event(barrier)  (empty elemof s)  [cluster-scope]
                if (__CHOREO_GROUPX4_SINGLE__) {
                  for (uint32_t __cta = 0; __cta < choreo::tma_cluster_dim(); ++__cta) {
                    choreo::tma_mbarrier_arrive_cluster(&empty[__iv_s], __cta);
                  }
                }
              } // s
              __iv_s = 0;
            }
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:98.15
          // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:103.15
          {
            int __iv_iv_k = 0;
            // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:103.15
            for (__iv_iv_k = 0; __iv_iv_k < ((K + 63) / 64); ++__iv_iv_k) {
              auto stage = __iv_iv_k % 4;
              // wait event(barrier)  (full elemof stage) 
              full[stage].wait(full[stage].arrive());
              // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:106.17
              {
                int __iv_iv_warp = 0;
                // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:106.17
                for (__iv_iv_warp = 0; __iv_iv_warp < 4; ++__iv_iv_warp) {
                  auto anon_1 = __choreo_vg4id_x - 1;
                  f16* ma_smem_ptr = (f16*)((__iv_iv_warp * 16 + ((__choreo_vg4id_x - 1) * 4096 + __iv_iv_k % 4 * 8192) + lhs_load_s));
                  uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma_smem_ptr);
                  f16* mb_smem_ptr = (f16*)((__iv_iv_warp * 16 + __iv_iv_k % 4 * 8192 + rhs_load_s));
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
              // trigger event(barrier)  (empty elemof stage)  [cluster-scope]
              if (__CHOREO_GROUPX4_SINGLE__) {
                for (uint32_t __cta = 0; __cta < choreo::tma_cluster_dim(); ++__cta) {
                  choreo::tma_mbarrier_arrive_cluster(&empty[stage], __cta);
                }
              }
            } // iv_k
            __iv_iv_k = 0;
          }
          // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:114.15
          if ((__choreo_vg4id_x == 1)) {
            auto __shape1_output_s_1 = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
            auto __stride1_output_s_1 = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
            auto __layout1_output_s_1 = cute::make_layout(__shape1_output_s_1, __stride1_output_s_1);
            auto __tensor1_output_s_1 = cute::make_tensor(cute::make_smem_ptr<float>((float*)output_s_1 + 0), __layout1_output_s_1);
            store_fragment_d<CUTE_WGMMA_M64K16, 128>(__tensor1_output_s_1, reinterpret_cast<float*>(mc));
            auto anon_2 = (block_m * 2) + (__choreo_vg4id_x - 1);
            future __choreo_anon_fut__0("", 116, 17);
            __choreo_anon_fut__0.is_tma = true;
            __choreo_anon_fut__0.set_atom(&choreo_copy_atom_t_0);
            cde::fence_proxy_async_shared_cta();
            if (__CHOREO_GROUPX4_SINGLE__) {
              cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_2_tensor_map, (block_n * 128), (anon_2 * 64), output_s_1);
              cde::cp_async_bulk_commit_group();
            }
          } // end inthreads
          // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:118.15
          if ((__choreo_vg4id_x == 2)) {
            auto __shape2_output_s_2 = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
            auto __stride2_output_s_2 = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
            auto __layout2_output_s_2 = cute::make_layout(__shape2_output_s_2, __stride2_output_s_2);
            auto __tensor2_output_s_2 = cute::make_tensor(cute::make_smem_ptr<float>((float*)output_s_2 + 0), __layout2_output_s_2);
            store_fragment_d<CUTE_WGMMA_M64K16, 128>(__tensor2_output_s_2, reinterpret_cast<float*>(mc));
            auto anon_3 = (block_m * 2) + (__choreo_vg4id_x - 1);
            future __choreo_anon_fut__1("", 120, 17);
            __choreo_anon_fut__1.is_tma = true;
            __choreo_anon_fut__1.set_atom(&choreo_copy_atom_t_1);
            cde::fence_proxy_async_shared_cta();
            if (__CHOREO_GROUPX4_SINGLE__) {
              cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_3_tensor_map, (block_n * 128), (anon_3 * 64), output_s_2);
              cde::cp_async_bulk_commit_group();
            }
          } // end inthreads
        } // end inthreads
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:76.9
    } // tile_iter
    __iv_tile_iter = 0;
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

  choreo::runtime_check((((static_cast<long long>(M) + 127LL) / 128LL * ((static_cast<long long>(N) + 127LL) / 128LL) + 113LL) / 114LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:71.30");
  choreo::runtime_check(((static_cast<long long>(K) + 63LL) / 64LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:83.33");
  choreo::runtime_check(((static_cast<long long>(K) + 63LL) / 64LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x512x16384/gpt-5-3-codex-high/iter020_tbc_1p2c_multicast.co:103.33");
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
  if (__choreo_tma_2_tensor_map_res != CUDA_SUCCESS) fprintf(stderr, "[iter020] TMA2 create error: %d\n", (int)__choreo_tma_2_tensor_map_res);
  choreo::abend_true(__choreo_tma_2_tensor_map_res != CUDA_SUCCESS);
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
  // Cluster with cluster_dim.x=2 requires gridDim.x to be a multiple of 2.
  // 57 clusters × 2 CTAs per cluster = 114 CTAs total. Grid = (114, 1, 1) with cluster (2,1,1).
  dim3 __matmul_gdims0(114, 1, 1);
  dim3 __matmul_bdims0(384, 1, 1);
  {
    cudaError_t e0 = cudaFuncSetAttribute(__choreo_device_matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, 196608 + (1024 - 1));
    cudaError_t e1 = cudaFuncSetAttribute(__choreo_device_matmul, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = __matmul_gdims0;
    config.blockDim = __matmul_bdims0;
    config.dynamicSmemBytes = 196608 + (1024 - 1);
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    auto __lhs_ptr = lhs.data();
    auto __rhs_ptr = rhs.data();
    auto __out_ptr = output.data();
    void* kargs[] = {(void*)&__lhs_ptr, (void*)&__rhs_ptr, (void*)&__out_ptr, (void*)&K, (void*)&M, (void*)&N, (void*)&__choreo_tma_0_tensor_map, (void*)&__choreo_tma_1_tensor_map, (void*)&__choreo_tma_2_tensor_map, (void*)&__choreo_tma_3_tensor_map};
    cudaError_t e2 = cudaLaunchKernelExC(&config, (void*)__choreo_device_matmul, kargs);
  }
  cudaError_t e3 = cudaDeviceSynchronize();
  if (e3 != cudaSuccess) { fprintf(stderr, "[iter020] Sync error: %s\n", cudaGetErrorString(e3)); abort(); }
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
    for (size_t j = 0; j < 128; ++j) {
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


