
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

// gdn_bf16_handwritten.co - Choreo kernel with handwritten 2D thread layout
//
// Computation: 2D thread layout (4 K-axis x 8 V-axis) with shared memory
// reduction, matching gdn_bf16_handwritten.cu.
//
// Data movement: Choreo DMA for initial_state load/store and output store.
// q/k/v/a/b loaded via .at() in __co__ section.
//
// Compile:
//   ../choreo/choreo -t cute -arch=sm_90a -es gdn_bf16_handwritten.co -o gdn_bf16_handwritten.cu
//
// Test:
//   python3 test_choreo_kernel.py gdn_bf16_handwritten.cu --fix-stride -T 4,128

#ifndef ENABLE_DEBUG
#define ENABLE_DEBUG 1
#endif



  template <typename T>
  __device__ __forceinline__ T SHFL_XOR(T var, int lane_mask, int width) {
    return __shfl_xor_sync(uint32_t(-1), var, lane_mask, width);
  }

  // 2D thread layout: 4 (K-axis) x 8 (V-axis) = 32 threads per warp
  //   tid_k = tid / 8   (0..3)
  //   tid_v = tid % 8   (0..7)
  //   Each thread holds ROWS_PER_THREAD(32) x COLS_PER_THREAD(4) = 128 f32 h values
  //
  // K-axis reduction: 4-way via shared memory
  // V broadcast: via shared memory
  // L2-norm: 4-way via SHFL_XOR (masks 8, 16)

  template <int BK, int BV, bool USE_QK_L2NORM_IN_KERNEL, bool IS_KDA>
  __device__ void computation_hw(
      const choreo::f32 a_f32, const choreo::f32 b_f32,
      const choreo::f32 dt_bias_f32, const choreo::f32 a_log_f32,
      const choreo::f32 softplus_beta, const choreo::f32 softplus_threshold,
      const choreo::f32 scale,
      choreo::f32 *__restrict__ q_local,
      choreo::f32 *__restrict__ k_local,
      choreo::f32 *__restrict__ v_local,
      choreo::bf16 *__restrict__ o_bf16,
      choreo::f32 *__restrict__ h,
      choreo::f32 *__restrict__ reduce_smem,
      choreo::f32 *__restrict__ bcast_smem) {
    constexpr int ROWS_PER_THREAD = 32;
    constexpr int COLS_PER_THREAD = 4;

    const int tid = threadIdx.x;
    const int tid_k = tid >> 3;
    const int tid_v = tid & 7;

    const float a = a_f32;
    const float b_val = b_f32;
    const float dt_bias = dt_bias_f32;
    const float a_log = a_log_f32;

    float x = a + dt_bias;
    float beta_x = softplus_beta * x;
    float softplus_x = (beta_x < softplus_threshold)
                           ? (1.0f / softplus_beta) * logf(1.0f + expf(beta_x))
                           : x;
    float g_val = -expf(a_log) * softplus_x;
    float beta = 1.0f / (1.0f + expf(-b_val));

    if constexpr (USE_QK_L2NORM_IN_KERNEL) {
      float q_sq = 0.0f, k_sq = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        q_sq = fmaf(q_local[r], q_local[r], q_sq);
        k_sq = fmaf(k_local[r], k_local[r], k_sq);
      }
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

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r)
      q_local[r] *= scale;

    float h_scale = expf(g_val);
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
#pragma unroll
      for (int c = 0; c < COLS_PER_THREAD; ++c) {
        h[r * COLS_PER_THREAD + c] *= h_scale;
      }
    }

    // vp = sum_K(h * k)
    float vp[COLS_PER_THREAD];
#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      vp[c] = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS_PER_THREAD; ++r)
        vp[c] = fmaf(h[r * COLS_PER_THREAD + c], k_local[r], vp[c]);
    }

    __syncthreads();
#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c)
      reduce_smem[tid_k * BV + tid_v * COLS_PER_THREAD + c] = vp[c];
    __syncthreads();

#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      int col_idx = tid_v * COLS_PER_THREAD + c;
      vp[c] = reduce_smem[0 * BV + col_idx] + reduce_smem[1 * BV + col_idx] +
              reduce_smem[2 * BV + col_idx] + reduce_smem[3 * BV + col_idx];
    }

#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      v_local[c] = (v_local[c] - vp[c]) * beta;
    }

    __syncthreads();
#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c)
      bcast_smem[tid_v * COLS_PER_THREAD + c] = v_local[c];
    __syncthreads();

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
#pragma unroll
      for (int c = 0; c < COLS_PER_THREAD; ++c) {
        int global_col = tid_v * COLS_PER_THREAD + c;
        h[r * COLS_PER_THREAD + c] += k_local[r] * bcast_smem[global_col];
      }
    }

    // op = sum_K(h * q)
    float op[COLS_PER_THREAD];
#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      op[c] = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS_PER_THREAD; ++r)
        op[c] = fmaf(h[r * COLS_PER_THREAD + c], q_local[r], op[c]);
    }

    __syncthreads();
#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c)
      reduce_smem[tid_k * BV + tid_v * COLS_PER_THREAD + c] = op[c];
    __syncthreads();

#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      int col_idx = tid_v * COLS_PER_THREAD + c;
      op[c] = reduce_smem[0 * BV + col_idx] + reduce_smem[1 * BV + col_idx] +
              reduce_smem[2 * BV + col_idx] + reduce_smem[3 * BV + col_idx];
    }

#pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; ++c) {
      o_bf16[tid_v * COLS_PER_THREAD + c] = __float2bfloat16(op[c]);
    }
  }


// clang-format off
__global__ void __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel(float * A_log, bf16 * a, bf16 * dt_bias, bf16 * q, bf16 * k, bf16 * v, bf16 * b, bf16 * o, float * initial_state_source, int * initial_state_indices, int * cu_seqlens, float scale, float softplus_beta, float softplus_threshold, bool IS_KDA, bool IS_VARLEN, bool USE_QK_L2NORM_IN_KERNEL, bool USE_INITAL_STATE, unsigned B, unsigned H, unsigned HV, unsigned K, unsigned N, unsigned T, unsigned V) {
  auto __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__ = nullptr;
  { // parallel-by: gdn_bf16_handwritten.co:206.12
  alignas(16) unsigned char anon_2[784];
  __shared__ alignas(128) unsigned char anon_1[768];
  auto __choreo_vtid_x = threadIdx.x;
  int i_h = blockIdx.z / (HV / H);
  int tid_k = __choreo_vtid_x / 8;
  int tid_v = __choreo_vtid_x % 8;
  int bos = blockIdx.y * T;
  int eos = blockIdx.y * T + T;
  int LEN = T;
  // if-else: gdn_bf16_handwritten.co:214.5
  if (false) {
    bos = *((int*)cu_seqlens + blockIdx.y);
    eos = *((int*)cu_seqlens + (blockIdx.y + 1));
    LEN = (eos - bos);
  } // end if-else: gdn_bf16_handwritten.co:214.5
  float a_log_l = *((float*)A_log + blockIdx.z);
  float dt_bias_l = static_cast<float>(*((bf16*)dt_bias + blockIdx.z));
  float* hidden_l = (float*)(anon_2 + 0);
  for (int i = 0; i < 128; ++i) hidden_l[i] = 0;
  float* reduce_smem = (float*)(anon_1 + 0);
  float* bcast_smem = (float*)(anon_1 + 512);
  int idx = *((int*)initial_state_indices + blockIdx.y);
  // if-else: gdn_bf16_handwritten.co:231.5
  if ((idx >= 0)) {
    // with-in: gdn_bf16_handwritten.co:232.7
    {
      int __iv_r = 0;
      int __iv_c = 0;
      // foreach: gdn_bf16_handwritten.co:232.7
      for (__iv_r = 0; __iv_r < 32; ++__iv_r) {
        for (__iv_c = 0; __iv_c < 4; ++__iv_c) {
          *((float*)hidden_l + (__iv_r * 4) + __iv_c) = *((float*)initial_state_source + (HV * K * V * idx) + (K * V * blockIdx.z) + (tid_k * 32 + __iv_r)*V + (blockIdx.x * 32 + (__iv_c + __choreo_vtid_x % 8 * 4)));
        } // c
        __iv_c = 0;
      } // r
      __iv_r = 0;
    }
  } // end if-else: gdn_bf16_handwritten.co:231.5
  // with-in: gdn_bf16_handwritten.co:237.5
  {
    int __iv_i_l__elem__0 = 0;
    // foreach: gdn_bf16_handwritten.co:238.7
    for (__iv_i_l__elem__0 = 0; __iv_i_l__elem__0 < LEN; ++__iv_i_l__elem__0) {
      float a_val = static_cast<float>(*((bf16*)a + (HV * T * blockIdx.y) + (HV * __iv_i_l__elem__0) + blockIdx.z));
      float b_val = static_cast<float>(*((bf16*)b + (HV * T * blockIdx.y) + (HV * __iv_i_l__elem__0) + blockIdx.z));
      float* q_local = (float*)(anon_2 + 640);
      float* k_local = (float*)(anon_2 + 512);
      // with-in: gdn_bf16_handwritten.co:246.9
      {
        int __iv_r__elem__0 = 0;
        // foreach: gdn_bf16_handwritten.co:246.9
        for (__iv_r__elem__0 = 0; __iv_r__elem__0 < 32; ++__iv_r__elem__0) {
          *((float*)q_local + __iv_r__elem__0) = static_cast<float>(*((bf16*)q + (H * K * T * blockIdx.y) + (H * K * __iv_i_l__elem__0) + (K * i_h) + (tid_k * 32 + __iv_r__elem__0)));
          *((float*)k_local + __iv_r__elem__0) = static_cast<float>(*((bf16*)k + (H * K * T * blockIdx.y) + (H * K * __iv_i_l__elem__0) + (K * i_h) + (tid_k * 32 + __iv_r__elem__0)));
        } // r__elem__0
        __iv_r__elem__0 = 0;
      }
      float* v_local = (float*)(anon_2 + 768);
      // with-in: gdn_bf16_handwritten.co:253.9
      {
        int __iv_c__elem__0 = 0;
        // foreach: gdn_bf16_handwritten.co:253.9
        for (__iv_c__elem__0 = 0; __iv_c__elem__0 < 4; ++__iv_c__elem__0) {
          *((float*)v_local + __iv_c__elem__0) = static_cast<float>(*((bf16*)v + (HV * T * V * blockIdx.y) + (HV * V * __iv_i_l__elem__0) + (V * blockIdx.z) + (blockIdx.x * 32 + (__choreo_vtid_x % 8 * 4 + __iv_c__elem__0))));
        } // c__elem__0
        __iv_c__elem__0 = 0;
      }
      bf16* o_s = (bf16*)(anon_1 + 640);
      computation_hw<128, 32, true, false>(a_val, b_val, dt_bias_l, a_log_l, softplus_beta, softplus_threshold, scale, (float*)q_local, (float*)k_local, (float*)v_local, (bf16*)o_s, (float*)hidden_l, (float*)reduce_smem, (float*)bcast_smem);
      future __choreo_anon_fut__0("", 265, 9);
      auto __shape1_o_s = cute::make_shape(cute::Int<32>{});
      auto __stride1_o_s = cute::make_stride(cute::Int<1>{});
      auto __layout1_o_s = cute::make_layout(__shape1_o_s, __stride1_o_s);
      auto __tensor1_o_s = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)o_s + 0), __layout1_o_s);
      auto __shape2_o = cute::make_shape(cute::Int<32>{});
      auto __stride2_o = cute::make_stride(cute::Int<1>{});
      auto __layout2_o = cute::make_layout(__shape2_o, __stride2_o);
      auto __tensor2_o = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)o + (V / ((V + 31) / 32) * blockIdx.x + (V * blockIdx.z + HV * V * (bos + __iv_i_l__elem__0)))), __layout2_o);
      opt_copy(__tensor1_o_s, __tensor2_o);
      __syncthreads();
    } // i_l__elem__0
    __iv_i_l__elem__0 = 0;
  }
  // if-else: gdn_bf16_handwritten.co:273.5
  if (true) {
    idx = *((int*)initial_state_indices + blockIdx.y);
    // if-else: gdn_bf16_handwritten.co:275.7
    if ((idx >= 0)) {
      // with-in: gdn_bf16_handwritten.co:276.9
      {
        int __iv_r = 0;
        int __iv_c = 0;
        // foreach: gdn_bf16_handwritten.co:276.9
        for (__iv_r = 0; __iv_r < 32; ++__iv_r) {
          for (__iv_c = 0; __iv_c < 4; ++__iv_c) {
            *((float*)initial_state_source + (HV * K * V * idx) + (K * V * blockIdx.z) + (tid_k * 32 + __iv_r)*V + (blockIdx.x * 32 + (__iv_c + __choreo_vtid_x % 8 * 4))) = *((float*)hidden_l + (__iv_r * 4) + __iv_c);
          } // c
          __iv_c = 0;
        } // r
        __iv_r = 0;
      }
    } // end if-else: gdn_bf16_handwritten.co:275.7
  } // end if-else: gdn_bf16_handwritten.co:273.5
  choreo::choreo_assert((static_cast<long long>(blockIdx.z) / (static_cast<long long>(HV) / static_cast<long long>(H)) >= 0LL), "The 3rd index `i_h` of element access 'q' should be greater than or equal to 0, gdn_bf16_handwritten.co:247.42");
  choreo::choreo_assert((static_cast<long long>(blockIdx.z) / (static_cast<long long>(HV) / static_cast<long long>(H)) < static_cast<long long>(H)), "The 3rd index `i_h` of element access 'q' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::H, gdn_bf16_handwritten.co:247.42");
  choreo::choreo_assert((static_cast<long long>(blockIdx.z) / (static_cast<long long>(HV) / static_cast<long long>(H)) >= 0LL), "The 3rd index `i_h` of element access 'k' should be greater than or equal to 0, gdn_bf16_handwritten.co:248.42");
  choreo::choreo_assert((static_cast<long long>(blockIdx.z) / (static_cast<long long>(HV) / static_cast<long long>(H)) < static_cast<long long>(H)), "The 3rd index `i_h` of element access 'k' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::H, gdn_bf16_handwritten.co:248.42");
  } // end parallel-by
}

void fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel(const choreo::spanned_view<choreo::f32, 1> & A_log, const choreo::spanned_view<choreo::bf16, 3> & a, const choreo::spanned_view<choreo::bf16, 1> & dt_bias, const choreo::spanned_view<choreo::bf16, 4> & q, const choreo::spanned_view<choreo::bf16, 4> & k, const choreo::spanned_view<choreo::bf16, 4> & v, const choreo::spanned_view<choreo::bf16, 3> & b, const choreo::spanned_view<choreo::bf16, 4> & o, const choreo::spanned_view<choreo::f32, 4> & initial_state_source, const choreo::spanned_view<choreo::s32, 1> & initial_state_indices, const choreo::spanned_view<choreo::s32, 1> & cu_seqlens, float scale, float softplus_beta, float softplus_threshold) {
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

  choreo::runtime_check(((static_cast<long long>(V) + 31LL) / 32LL > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, gdn_bf16_handwritten.co:206.13");
  choreo::runtime_check((static_cast<long long>(N) > 0LL), "The 2nd bound item of parallelby is invalid: should be greater than 0, gdn_bf16_handwritten.co:206.18");
  choreo::runtime_check((static_cast<long long>(HV) > 0LL), "The 3rd bound item of parallelby is invalid: should be greater than 0, gdn_bf16_handwritten.co:206.23");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 1st index `i_hv` of element access 'A_log' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_handwritten.co:219.29");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 1st index `i_hv` of element access 'dt_bias' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_handwritten.co:220.33");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'initial_state_indices' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_handwritten.co:230.36");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 2nd index `i_hv` of element access 'initial_state_source' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_handwritten.co:233.58");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'a' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_handwritten.co:240.27");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 3rd index `i_hv` of element access 'a' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_handwritten.co:240.37");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'b' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_handwritten.co:241.27");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 3rd index `i_hv` of element access 'b' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_handwritten.co:241.37");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'q' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_handwritten.co:247.32");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'k' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_handwritten.co:248.32");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'v' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_handwritten.co:254.32");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 3rd index `i_hv` of element access 'v' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_handwritten.co:254.42");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'initial_state_indices' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_handwritten.co:274.38");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 2nd index `i_hv` of element access 'initial_state_source' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_handwritten.co:277.40");
  float * A_log__device = nullptr;
  choreo::abend_true(cudaMalloc(&A_log__device, (HV) * 4));
  choreo::abend_true(cudaMemcpy(A_log__device, A_log.data(), (HV) * 4, cudaMemcpyHostToDevice));
  bf16 * a__device = nullptr;
  choreo::abend_true(cudaMalloc(&a__device, (((B * HV) * T)) * 2));
  choreo::abend_true(cudaMemcpy(a__device, a.data(), (((B * HV) * T)) * 2, cudaMemcpyHostToDevice));
  bf16 * dt_bias__device = nullptr;
  choreo::abend_true(cudaMalloc(&dt_bias__device, (HV) * 2));
  choreo::abend_true(cudaMemcpy(dt_bias__device, dt_bias.data(), (HV) * 2, cudaMemcpyHostToDevice));
  bf16 * q__device = nullptr;
  choreo::abend_true(cudaMalloc(&q__device, ((B * ((H * K) * T))) * 2));
  choreo::abend_true(cudaMemcpy(q__device, q.data(), ((B * ((H * K) * T))) * 2, cudaMemcpyHostToDevice));
  bf16 * k__device = nullptr;
  choreo::abend_true(cudaMalloc(&k__device, ((B * ((H * K) * T))) * 2));
  choreo::abend_true(cudaMemcpy(k__device, k.data(), ((B * ((H * K) * T))) * 2, cudaMemcpyHostToDevice));
  bf16 * v__device = nullptr;
  choreo::abend_true(cudaMalloc(&v__device, ((((B * HV) * T) * V)) * 2));
  choreo::abend_true(cudaMemcpy(v__device, v.data(), ((((B * HV) * T) * V)) * 2, cudaMemcpyHostToDevice));
  bf16 * b__device = nullptr;
  choreo::abend_true(cudaMalloc(&b__device, (((B * HV) * T)) * 2));
  choreo::abend_true(cudaMemcpy(b__device, b.data(), (((B * HV) * T)) * 2, cudaMemcpyHostToDevice));
  bf16 * o__device = nullptr;
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
  dim3 __fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_gdims0(((V + 31) / 32), N, HV);
  dim3 __fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_bdims0(32, 1, 1);
  __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel<<<__fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_gdims0, __fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_bdims0>>>(A_log__device, a__device, dt_bias__device, q__device, k__device, v__device, b__device, o__device, initial_state_source__device, initial_state_indices__device, cu_seqlens__device, scale, softplus_beta, softplus_threshold, IS_KDA, IS_VARLEN, USE_QK_L2NORM_IN_KERNEL, USE_INITAL_STATE, B, H, HV, K, N, T, V);
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

