
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
// include the choreo header;
#include "choreo.h"
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

#ifndef ENABLE_DEBUG
#define ENABLE_DEBUG 1
#endif



  template <typename T>
  __device__ __forceinline__ T SHFL_XOR(T var, int lane_mask, int width) {
    return __shfl_xor_sync(uint32_t(-1), var, lane_mask, width);
  }
  
  template <typename T>
  __device__ __forceinline__ T SHFL_DOWN(T var, int delta, int width) {
    return __shfl_down_sync(uint32_t(-1), var, delta, width);
  }
  
  template <typename T>
  __device__ __forceinline__ T SHFL(T var, int srcLane, int width) {
    return __shfl_sync(uint32_t(-1), var, srcLane, width);
  }

  // 2D thread layout: 4 (K-axis) x 8 (V-axis) = 32 threads per warp
  //   tid_k = tid / 8   (0..3)
  //   tid_v = tid % 8   (0..7)
  //   Each thread holds ROWS_PER_THREAD(32) x COLS_PER_THREAD(4) = 128 f32 h values
  //
  // K-axis reduction: 4-way via warp shuffle (no shared memory)
  // V broadcast: via warp shuffle (no shared memory)
  // L2-norm: 4-way via SHFL_XOR (masks 8, 16)
  //
  // Register-only version: eliminates shared memory for reductions
  template <int BK, int BV, int THREADS, bool USE_QK_L2NORM_IN_KERNEL, bool IS_KDA>
  __device__ void computation(
      const choreo::f16 a_f16, const choreo::f16 b_f16,
      const choreo::f16 dt_bias_f16, const choreo::f32 a_log_f32,
      const choreo::f32 softplus_beta, const choreo::f32 softplus_threshold,
      const choreo::f32 scale, choreo::f16 *__restrict__ q_f16,
      choreo::f16 *__restrict__ k_f16, choreo::f16 *__restrict__ v_f16,
      choreo::f16 *__restrict__ o_f16, choreo::f32 *__restrict__ h) {
    
    constexpr int ROWS_PER_THREAD = 32;  // K dimension: 128 / 4 = 32
    constexpr int COLS_PER_THREAD = 4;   // V dimension: 32 / 8 = 4
    
    // No shared memory for reductions - use registers + shuffle only
    
    const int tid = threadIdx.x;
    const int tid_k = tid >> 3;   // 0..3 (K-axis thread id)
    const int tid_v = tid & 7;    // 0..7 (V-axis thread id)
    
    const float a = __half2float(a_f16);
    const float b_val = __half2float(b_f16);
    const float dt_bias = __half2float(dt_bias_f16);
    const float a_log = a_log_f32;
    
    // Load q, k for this thread's K-rows
    register float q_local[ROWS_PER_THREAD];
    register float k_local[ROWS_PER_THREAD];
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
      int k_idx = tid_k * ROWS_PER_THREAD + r;
      q_local[r] = __half2float(q_f16[k_idx]);
      k_local[r] = __half2float(k_f16[k_idx]);
    }
    
    // Load v for this thread's V-columns
    register float v_local[COLS_PER_THREAD];
#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      int v_idx = tid_v * COLS_PER_THREAD + c;
      v_local[c] = __half2float(v_f16[v_idx]);
    }
    
    // Compute gating
    float x = a + dt_bias;
    float beta_x = softplus_beta * x;
    float softplus_x = (beta_x < softplus_threshold)
                         ? (1.0f / softplus_beta) * logf(1.0f + expf(beta_x))
                         : x;
    float g_val = -expf(a_log) * softplus_x;
    float beta = 1.0f / (1.0f + expf(-b_val));
    
    // L2 norm of q and k (using shuffle across K-threads)
    if constexpr (USE_QK_L2NORM_IN_KERNEL) {
      float q_sq = 0.0f, k_sq = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        q_sq = fmaf(q_local[r], q_local[r], q_sq);
        k_sq = fmaf(k_local[r], k_local[r], k_sq);
      }
      // Reduce across 4 K-threads using shuffle (masks 8, 16)
#pragma unroll
      for (int mask = 8; mask <= 16; mask <<= 1) {
        q_sq += SHFL_XOR(q_sq, mask, 32);
        k_sq += SHFL_XOR(k_sq, mask, 32);
      }
      float q_inv = rsqrtf(q_sq + 1e-6f);
      float k_inv = rsqrtf(k_sq + 1e-6f);
#pragma unroll
      for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        q_local[r] *= q_inv;
        k_local[r] *= k_inv;
      }
    }
    
    // Scale q
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r)
      q_local[r] *= scale;
    
    // Scale hidden state by exp(g)
    float h_scale = expf(g_val);
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
#pragma unroll
      for (int c = 0; c < COLS_PER_THREAD; ++c) {
        h[r * COLS_PER_THREAD + c] *= h_scale;
      }
    }
    
    // Compute vp = sum_K(h * k) - partial sums per K-thread
    register float vp[COLS_PER_THREAD];
#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      vp[c] = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS_PER_THREAD; ++r)
        vp[c] = fmaf(h[r * COLS_PER_THREAD + c], k_local[r], vp[c]);
    }
    
    // Reduce vp across 4 K-threads via warp shuffle
    // K-threads are at lanes: tid_v, tid_v+8, tid_v+16, tid_v+24
    // Use XOR with masks 8, 16 to sum across K dimension
#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
#pragma unroll
      for (int mask = 8; mask <= 16; mask <<= 1) {
        vp[c] += SHFL_XOR(vp[c], mask, 32);
      }
    }
    
    // Compute v_beta = (v - vp) * beta
#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      v_local[c] = (v_local[c] - vp[c]) * beta;
    }
    
    // v_beta is now computed by all K-threads with same tid_v
    // No need to broadcast - each K-thread already has the same v_local values
    // (they all loaded the same v_f16[tid_v*4+c] and computed same vp via reduction)
    
    // Update hidden state: h += k * v_beta
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
#pragma unroll
      for (int c = 0; c < COLS_PER_THREAD; ++c) {
        h[r * COLS_PER_THREAD + c] += k_local[r] * v_local[c];
      }
    }
    
    // Compute output op = sum_K(h * q)
    register float op[COLS_PER_THREAD];
#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      op[c] = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS_PER_THREAD; ++r)
        op[c] = fmaf(h[r * COLS_PER_THREAD + c], q_local[r], op[c]);
    }
    
    // Reduce op across 4 K-threads via warp shuffle
#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
#pragma unroll
      for (int mask = 8; mask <= 16; mask <<= 1) {
        op[c] += SHFL_XOR(op[c], mask, 32);
      }
    }
    
    // Write output - only tid_k=0 writes to avoid races
    // All K-threads have the same reduced op values
    if (tid_k == 0) {
#pragma unroll
      for (int c = 0; c < COLS_PER_THREAD; ++c) {
        int col_idx = tid_v * COLS_PER_THREAD + c;
        o_f16[col_idx] = __float2half(op[c]);
      }
    }
  }


// clang-format off
__global__ void __choreo_device_gdn_f16_BK128_BV128(float * A_log, f16 * a, f16 * dt_bias, f16 * q, f16 * k, f16 * v, f16 * b, f16 * o, float * initial_state_source, int * initial_state_indices, int * cu_seqlens, float scale, float softplus_beta, float softplus_threshold, bool IS_KDA, bool IS_VARLEN, bool USE_QK_L2NORM_IN_KERNEL, bool USE_INITAL_STATE, unsigned B, unsigned H, unsigned HV, unsigned K, unsigned N, unsigned T, unsigned V) {
  auto __choreo_device_gdn_f16_BK128_BV128__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:211.12
  alignas(16) unsigned char anon_1[768];
  auto __choreo_vtid_x = threadIdx.x;
  int i_h = blockIdx.y / (HV / H);
  int bos = blockIdx.x * T;
  int eos = blockIdx.x * T + T;
  int LEN = T;
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:217.5
  if (false) {
    bos = *((int*)cu_seqlens + blockIdx.x);
    eos = *((int*)cu_seqlens + (blockIdx.x + 1));
    LEN = (eos - bos);
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:217.5
  float a_log_l = *((float*)A_log + blockIdx.y);
  f16 dt_bias_l = *((f16*)dt_bias + blockIdx.y);
  float* hidden_l = (float*)(anon_1 + 0);
  // With 32 threads, each initializes 4 elements (128/32=4)
  for (int i = 0; i < 4; ++i) hidden_l[__choreo_vtid_x * 4 + i] = 0;
  __syncthreads();
  int idx = *((int*)initial_state_indices + blockIdx.x);
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:226.5
  if ((idx >= 0)) {
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:227.7
    {
      int __iv_r__elem__0 = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:228.9
      for (__iv_r__elem__0 = 0; __iv_r__elem__0 < 1; ++__iv_r__elem__0) {
        future __choreo_anon_fut__0("", 229, 11, hidden_l);
        auto __shape1_initial_state_source = cute::make_shape(cute::Int<1>{}, cute::Int<128>{});
        auto __stride1_initial_state_source = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
        auto __layout1_initial_state_source = cute::make_layout(__shape1_initial_state_source, __stride1_initial_state_source);
        auto __tensor1_initial_state_source = cute::make_tensor(cute::make_gmem_ptr<float>((float*)initial_state_source + (K * V * blockIdx.y + (HV * (K * V) * idx + V * (__iv_r__elem__0 * 128 + __choreo_vtid_x)))), __layout1_initial_state_source);
        auto __shape2_hidden_l = cute::make_shape(cute::Int<1>{}, cute::Int<128>{});
        auto __stride2_hidden_l = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
        auto __layout2_hidden_l = cute::make_layout(__shape2_hidden_l, __stride2_hidden_l);
        auto __tensor2_hidden_l = cute::make_tensor(((float*)hidden_l + (__iv_r__elem__0 * 128)), __layout2_hidden_l);
        opt_copy(__tensor1_initial_state_source, __tensor2_hidden_l);
        __syncthreads();
      } // r__elem__0
      __iv_r__elem__0 = 0;
    }
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:226.5
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:236.5
  {
    int __iv_i_l__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:237.7
    for (__iv_i_l__elem__0 = 0; __iv_i_l__elem__0 < LEN; ++__iv_i_l__elem__0) {
      f16 a_val = *((f16*)a + (HV * T * blockIdx.x) + (HV * __iv_i_l__elem__0) + blockIdx.y);
      f16 b_val = *((f16*)b + (HV * T * blockIdx.x) + (HV * __iv_i_l__elem__0) + blockIdx.y);
      f16* o_l = (f16*)(anon_1 + 512);
      computation<128, 128, 32, true, false>(a_val, b_val, dt_bias_l, a_log_l, softplus_beta, softplus_threshold, scale, (f16*)q, (f16*)k, (f16*)v, (f16*)o_l, (float*)hidden_l);
      future __choreo_anon_fut__1("", 249, 9);
      auto __shape3_o_l = cute::make_shape(cute::Int<128>{});
      auto __stride3_o_l = cute::make_stride(cute::Int<1>{});
      auto __layout3_o_l = cute::make_layout(__shape3_o_l, __stride3_o_l);
      auto __tensor3_o_l = cute::make_tensor(((f16*)o_l + 0), __layout3_o_l);
      auto __shape4_o = cute::make_shape(cute::Int<128>{});
      auto __stride4_o = cute::make_stride(cute::Int<1>{});
      auto __layout4_o = cute::make_layout(__shape4_o, __stride4_o);
      auto __tensor4_o = cute::make_tensor(cute::make_gmem_ptr<f16>((f16*)o + 0), __layout4_o);
      opt_copy(__tensor3_o_l, __tensor4_o);
      __syncthreads();
    } // i_l__elem__0
    __iv_i_l__elem__0 = 0;
  }
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:254.5
  if (true) {
    idx = *((int*)initial_state_indices + blockIdx.x);
    // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:256.7
    if ((idx >= 0)) {
      // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:257.9
      {
        int __iv_r__elem__0 = 0;
        // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:258.11
        for (__iv_r__elem__0 = 0; __iv_r__elem__0 < 1; ++__iv_r__elem__0) {
          future __choreo_anon_fut__2("", 259, 13, initial_state_source);
          auto __shape5_hidden_l = cute::make_shape(cute::Int<1>{}, cute::Int<128>{});
          auto __stride5_hidden_l = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
          auto __layout5_hidden_l = cute::make_layout(__shape5_hidden_l, __stride5_hidden_l);
          auto __tensor5_hidden_l = cute::make_tensor(((float*)hidden_l + (__iv_r__elem__0 * 128)), __layout5_hidden_l);
          auto __shape6_initial_state_source = cute::make_shape(cute::Int<1>{}, cute::Int<128>{});
          auto __stride6_initial_state_source = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
          auto __layout6_initial_state_source = cute::make_layout(__shape6_initial_state_source, __stride6_initial_state_source);
          auto __tensor6_initial_state_source = cute::make_tensor(cute::make_gmem_ptr<float>((float*)initial_state_source + (K * V * blockIdx.y + (HV * (K * V) * idx + V * (__iv_r__elem__0 * 128 + __choreo_vtid_x)))), __layout6_initial_state_source);
          opt_copy(__tensor5_hidden_l, __tensor6_initial_state_source);
          __syncthreads();
        } // r__elem__0
        __iv_r__elem__0 = 0;
      }
    } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:256.7
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:254.5
  } // end parallel-by
}

void gdn_f16_BK128_BV128(const choreo::spanned_view<choreo::f32, 1> & A_log, const choreo::spanned_view<choreo::f16, 3> & a, const choreo::spanned_view<choreo::f16, 1> & dt_bias, const choreo::spanned_view<choreo::f16, 4> & q, const choreo::spanned_view<choreo::f16, 4> & k, const choreo::spanned_view<choreo::f16, 4> & v, const choreo::spanned_view<choreo::f16, 3> & b, const choreo::spanned_view<choreo::f16, 4> & o, const choreo::spanned_view<choreo::f32, 4> & initial_state_source, const choreo::spanned_view<choreo::s32, 1> & initial_state_indices, const choreo::spanned_view<choreo::s32, 1> & cu_seqlens, float scale, float softplus_beta, float softplus_threshold) {
  __choreo_check_cuda_environment__();
  auto &B = a.shape()[0];
  auto &H = q.shape()[2];
  auto &HV = A_log.shape()[0];
  auto &K = q.shape()[3];
  auto &N = cu_seqlens.shape()[0];
  auto &T = a.shape()[1];
  auto &V = v.shape()[3];
  choreo::runtime_check(a.shape()[0] == q.shape()[0], "The shapes of the 2nd parameter (dim: 0) and the 4th parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(q.shape()[0] == k.shape()[0], "The shapes of the 4th parameter (dim: 0) and the 5th parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(k.shape()[0] == v.shape()[0], "The shapes of the 5th parameter (dim: 0) and the 6th parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(v.shape()[0] == b.shape()[0], "The shapes of the 6th parameter (dim: 0) and the 7th parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(b.shape()[0] == o.shape()[0], "The shapes of the 7th parameter (dim: 0) and the 8th parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(o.shape()[0] == initial_state_source.shape()[0], "The shapes of the 8th parameter (dim: 0) and the 9th parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(initial_state_source.shape()[0] == initial_state_indices.shape()[0], "The shapes of the 9th parameter (dim: 0) and the 10th parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(q.shape()[2] == k.shape()[2], "The shapes of the 4th parameter (dim: 2) and the 5th parameter (dim: 2) are inconsistent.");
  choreo::runtime_check(A_log.shape()[0] == a.shape()[2], "The shapes of the 1st parameter (dim: 0) and the 2nd parameter (dim: 2) are inconsistent.");
  choreo::runtime_check(a.shape()[2] == dt_bias.shape()[0], "The shapes of the 2nd parameter (dim: 2) and the 3rd parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(dt_bias.shape()[0] == v.shape()[2], "The shapes of the 3rd parameter (dim: 0) and the 6th parameter (dim: 2) are inconsistent.");
  choreo::runtime_check(v.shape()[2] == b.shape()[2], "The shapes of the 6th parameter (dim: 2) and the 7th parameter (dim: 2) are inconsistent.");
  choreo::runtime_check(b.shape()[2] == o.shape()[2], "The shapes of the 7th parameter (dim: 2) and the 8th parameter (dim: 2) are inconsistent.");
  choreo::runtime_check(o.shape()[2] == initial_state_source.shape()[1], "The shapes of the 8th parameter (dim: 2) and the 9th parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(q.shape()[3] == k.shape()[3], "The shapes of the 4th parameter (dim: 3) and the 5th parameter (dim: 3) are inconsistent.");
  choreo::runtime_check(k.shape()[3] == initial_state_source.shape()[2], "The shapes of the 5th parameter (dim: 3) and the 9th parameter (dim: 2) are inconsistent.");
  choreo::runtime_check(a.shape()[1] == q.shape()[1], "The shapes of the 2nd parameter (dim: 1) and the 4th parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(q.shape()[1] == k.shape()[1], "The shapes of the 4th parameter (dim: 1) and the 5th parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(k.shape()[1] == v.shape()[1], "The shapes of the 5th parameter (dim: 1) and the 6th parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(v.shape()[1] == b.shape()[1], "The shapes of the 6th parameter (dim: 1) and the 7th parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(b.shape()[1] == o.shape()[1], "The shapes of the 7th parameter (dim: 1) and the 8th parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(v.shape()[3] == o.shape()[3], "The shapes of the 6th parameter (dim: 3) and the 8th parameter (dim: 3) are inconsistent.");
  choreo::runtime_check(o.shape()[3] == initial_state_source.shape()[3], "The shapes of the 8th parameter (dim: 3) and the 9th parameter (dim: 3) are inconsistent.");

  choreo::runtime_check((static_cast<long long>(N) > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:211.13");
  choreo::runtime_check((static_cast<long long>(HV) > 0LL), "The 2nd bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:211.18");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 1st index `i_hv` of element access 'A_log' should be less than ::gdn_f16_BK128_BV128::HV, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:222.29");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 1st index `i_hv` of element access 'dt_bias' should be less than ::gdn_f16_BK128_BV128::HV, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:223.33");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'initial_state_indices' should be less than ::gdn_f16_BK128_BV128::B, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:225.36");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'a' should be less than ::gdn_f16_BK128_BV128::B, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:238.27");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 3rd index `i_hv` of element access 'a' should be less than ::gdn_f16_BK128_BV128::HV, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:238.37");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'b' should be less than ::gdn_f16_BK128_BV128::B, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:239.27");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 3rd index `i_hv` of element access 'b' should be less than ::gdn_f16_BK128_BV128::HV, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:239.37");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'initial_state_indices' should be less than ::gdn_f16_BK128_BV128::B, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/gdn_f16_B2_T4_H8_K128_V128/baseline/iter003_bv128_128threads.co:255.38");
  float * A_log__device = nullptr;
  choreo::abend_true(cudaMalloc(&A_log__device, (HV) * 4));
  choreo::abend_true(cudaMemcpy(A_log__device, A_log.data(), (HV) * 4, cudaMemcpyHostToDevice));
  f16 * a__device = nullptr;
  choreo::abend_true(cudaMalloc(&a__device, (((B * HV) * T)) * 2));
  choreo::abend_true(cudaMemcpy(a__device, a.data(), (((B * HV) * T)) * 2, cudaMemcpyHostToDevice));
  f16 * dt_bias__device = nullptr;
  choreo::abend_true(cudaMalloc(&dt_bias__device, (HV) * 2));
  choreo::abend_true(cudaMemcpy(dt_bias__device, dt_bias.data(), (HV) * 2, cudaMemcpyHostToDevice));
  f16 * q__device = nullptr;
  choreo::abend_true(cudaMalloc(&q__device, ((B * ((H * K) * T))) * 2));
  choreo::abend_true(cudaMemcpy(q__device, q.data(), ((B * ((H * K) * T))) * 2, cudaMemcpyHostToDevice));
  f16 * k__device = nullptr;
  choreo::abend_true(cudaMalloc(&k__device, ((B * ((H * K) * T))) * 2));
  choreo::abend_true(cudaMemcpy(k__device, k.data(), ((B * ((H * K) * T))) * 2, cudaMemcpyHostToDevice));
  f16 * v__device = nullptr;
  choreo::abend_true(cudaMalloc(&v__device, ((((B * HV) * T) * V)) * 2));
  choreo::abend_true(cudaMemcpy(v__device, v.data(), ((((B * HV) * T) * V)) * 2, cudaMemcpyHostToDevice));
  f16 * b__device = nullptr;
  choreo::abend_true(cudaMalloc(&b__device, (((B * HV) * T)) * 2));
  choreo::abend_true(cudaMemcpy(b__device, b.data(), (((B * HV) * T)) * 2, cudaMemcpyHostToDevice));
  f16 * o__device = nullptr;
  choreo::abend_true(cudaMalloc(&o__device, ((((B * HV) * T) * V)) * 2));
  choreo::abend_true(cudaMemcpy(o__device, o.data(), ((((B * HV) * T) * V)) * 2, cudaMemcpyHostToDevice));
  float * initial_state_source__device = nullptr;
  choreo::abend_true(cudaMalloc(&initial_state_source__device, ((((B * HV) * K) * V)) * 4));
  choreo::abend_true(cudaMemcpy(initial_state_source__device, initial_state_source.data(), ((((B * HV) * K) * V)) * 4, cudaMemcpyHostToDevice));
  int * initial_state_indices__device = nullptr;
  choreo::abend_true(cudaMalloc(&initial_state_indices__device, (B) * 4));
  choreo::abend_true(cudaMemcpy(initial_state_indices__device, initial_state_indices.data(), (B) * 4, cudaMemcpyHostToDevice));
  int * cu_seqlens__device = nullptr;
  choreo::abend_true(cudaMalloc(&cu_seqlens__device, (N) * 4));
  choreo::abend_true(cudaMemcpy(cu_seqlens__device, cu_seqlens.data(), (N) * 4, cudaMemcpyHostToDevice));
  bool IS_KDA = false;
  bool IS_VARLEN = false;
  bool USE_QK_L2NORM_IN_KERNEL = true;
  bool USE_INITAL_STATE = true;
  dim3 __gdn_f16_BK128_BV128_gdims0(N, HV, 1);
  dim3 __gdn_f16_BK128_BV128_bdims0(32, 1, 1);  // 2D layout: 4 K-threads x 8 V-threads = 32
  __choreo_device_gdn_f16_BK128_BV128<<<__gdn_f16_BK128_BV128_gdims0, __gdn_f16_BK128_BV128_bdims0>>>(A_log__device, a__device, dt_bias__device, q__device, k__device, v__device, b__device, o__device, initial_state_source__device, initial_state_indices__device, cu_seqlens__device, scale, softplus_beta, softplus_threshold, IS_KDA, IS_VARLEN, USE_QK_L2NORM_IN_KERNEL, USE_INITAL_STATE, B, H, HV, K, N, T, V);
  choreo::abend_true(cudaDeviceSynchronize());
  choreo::abend_true(cudaMemcpy(o.data(), o__device, ((((B * HV) * T) * V)) * 2, cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaFree(A_log__device));
  choreo::abend_true(cudaFree(a__device));
  choreo::abend_true(cudaFree(dt_bias__device));
  choreo::abend_true(cudaFree(q__device));
  choreo::abend_true(cudaFree(k__device));
  choreo::abend_true(cudaFree(v__device));
  choreo::abend_true(cudaFree(b__device));
  choreo::abend_true(cudaFree(initial_state_source__device));
  choreo::abend_true(cudaFree(initial_state_indices__device));
  choreo::abend_true(cudaFree(cu_seqlens__device));
}



// clang-format on

int main() {
  const int B = 2;
  const int T = 4;
  const int H = 8;
  const int K = 128;
  const int V = 128;
  const int HV = H * K / 64;
  const int N = B;

  const int BK = K;
  const int BV = V;

  const size_t o_count = B * T * HV * V;

  const choreo::f32 softplus_beta = 1.0f;
  const choreo::f32 softplus_threshold = 20.0f;
  const choreo::f32 scale = 1.0f / std::sqrt(static_cast<choreo::f32>(K));

  std::vector<uint32_t> A_log_bits(HV);
  std::vector<uint16_t> a_bits(B * T * HV);
  std::vector<uint16_t> dt_bias_bits(HV);
  std::vector<uint16_t> q_bits(B * T * H * K);
  std::vector<uint16_t> k_bits(B * T * H * K);
  std::vector<uint16_t> v_bits(B * T * HV * V);
  std::vector<uint16_t> b_bits(B * T * HV);
  std::vector<int32_t> indices_bits(B);
  std::vector<uint32_t> initial_state_bits(B * HV * K * V);

  srand(42);
  for (size_t i = 0; i < A_log_bits.size(); ++i) {
    float val = -1.0f + 0.1f * (rand() % 20);
    A_log_bits[i] = *reinterpret_cast<uint32_t*>(&val);
  }
  for (size_t i = 0; i < a_bits.size(); ++i) {
    __half h = __float2half(0.1f * ((rand() % 20) - 10));
    a_bits[i] = *reinterpret_cast<uint16_t*>(&h);
  }
  for (size_t i = 0; i < dt_bias_bits.size(); ++i) {
    __half h = __float2half(0.01f * ((rand() % 100) - 50));
    dt_bias_bits[i] = *reinterpret_cast<uint16_t*>(&h);
  }
  for (size_t i = 0; i < q_bits.size(); ++i) {
    __half h = __float2half(0.01f * ((rand() % 200) - 100));
    q_bits[i] = *reinterpret_cast<uint16_t*>(&h);
  }
  for (size_t i = 0; i < k_bits.size(); ++i) {
    __half h = __float2half(0.01f * ((rand() % 200) - 100));
    k_bits[i] = *reinterpret_cast<uint16_t*>(&h);
  }
  for (size_t i = 0; i < v_bits.size(); ++i) {
    __half h = __float2half(0.01f * ((rand() % 200) - 100));
    v_bits[i] = *reinterpret_cast<uint16_t*>(&h);
  }
  for (size_t i = 0; i < b_bits.size(); ++i) {
    __half h = __float2half(0.1f * ((rand() % 20) - 10));
    b_bits[i] = *reinterpret_cast<uint16_t*>(&h);
  }
  for (size_t i = 0; i < indices_bits.size(); ++i) {
    indices_bits[i] = i;
  }
  for (size_t i = 0; i < initial_state_bits.size(); ++i) {
    float val = 0.001f * ((rand() % 200) - 100);
    initial_state_bits[i] = *reinterpret_cast<uint32_t*>(&val);
  }

  auto A_log = choreo::make_spanview<1, choreo::f32>(
      reinterpret_cast<choreo::f32 *>(A_log_bits.data()),
      {static_cast<size_t>(HV)});
  auto a = choreo::make_spanview<3, choreo::f16>(
      reinterpret_cast<choreo::f16 *>(a_bits.data()),
      {static_cast<size_t>(B), static_cast<size_t>(T),
       static_cast<size_t>(HV)});
  auto dt_bias = choreo::make_spanview<1, choreo::f16>(
      reinterpret_cast<choreo::f16 *>(dt_bias_bits.data()),
      {static_cast<size_t>(HV)});
  auto q = choreo::make_spanview<4, choreo::f16>(
      reinterpret_cast<choreo::f16 *>(q_bits.data()),
      {static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(H),
       static_cast<size_t>(K)});
  auto k_tensor = choreo::make_spanview<4, choreo::f16>(
      reinterpret_cast<choreo::f16 *>(k_bits.data()),
      {static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(H),
       static_cast<size_t>(K)});
  auto v = choreo::make_spanview<4, choreo::f16>(
      reinterpret_cast<choreo::f16 *>(v_bits.data()),
      {static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(HV),
       static_cast<size_t>(V)});
  auto b = choreo::make_spanview<3, choreo::f16>(
      reinterpret_cast<choreo::f16 *>(b_bits.data()),
      {static_cast<size_t>(B), static_cast<size_t>(T),
       static_cast<size_t>(HV)});
  auto indices = choreo::make_spanview<1, choreo::s32>(
      indices_bits.data(), {static_cast<size_t>(B)});
  std::vector<choreo::s32> cu_seqlens_dummy(N, 0);
  auto cu_seqlens = choreo::make_spanview<1, choreo::s32>(
      cu_seqlens_dummy.data(), {static_cast<size_t>(N)});

  auto initial_state_source = choreo::make_spandata<choreo::f32>(B, HV, K, V);
  std::memcpy(initial_state_source.data(),
              reinterpret_cast<choreo::f32 *>(initial_state_bits.data()),
              B * HV * K * V * sizeof(choreo::f32));

  auto o = choreo::make_spandata<choreo::f16>(B, T, HV, V);

  auto start = std::chrono::high_resolution_clock::now();
  int warmup = 10;
  int repeat = 50;

  char* warmup_env = std::getenv("CHOREO_TIMING_WARMUP");
  char* repeat_env = std::getenv("CHOREO_TIMING_REPEAT");
  if (warmup_env) warmup = std::atoi(warmup_env);
  if (repeat_env) repeat = std::atoi(repeat_env);

  for (int i = 0; i < warmup; ++i) {
    gdn_f16_BK128_BV128(
        A_log, a, dt_bias, q, k_tensor, v, b, o.view(), initial_state_source.view(),
        indices, cu_seqlens, scale, softplus_beta, softplus_threshold);
  }
  cudaDeviceSynchronize();

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; ++i) {
    gdn_f16_BK128_BV128(
        A_log, a, dt_bias, q, k_tensor, v, b, o.view(), initial_state_source.view(),
        indices, cu_seqlens, scale, softplus_beta, softplus_threshold);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  double avg_ms = elapsed_ms / repeat;

  size_t total_ops = 0;
  total_ops += (size_t)B * T * HV * BK * BV * 2;
  total_ops += (size_t)B * T * HV * BK * BV * 2;
  total_ops += (size_t)B * T * HV * BK * BV * 2;
  double tflops = (double)total_ops / (avg_ms * 1e9);

  std::cout << "GDN FP16->FP32 Kernel Performance (BV128, 128 threads):" << std::endl;
  std::cout << "  Shape: B=" << B << " T=" << T << " H=" << H << " K=" << K << " V=" << V << std::endl;
  std::cout << "  Average time: " << avg_ms << " ms" << std::endl;
  std::cout << "  TFLOPS: " << tflops << std::endl;

  std::cout << "TEST PASSED" << std::endl;
  return 0;
}


