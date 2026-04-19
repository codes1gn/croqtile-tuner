
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
  template <int BK, int BV, bool USE_QK_L2NORM_IN_KERNEL, bool IS_KDA>
  __device__ __forceinline__ void computation(
      const choreo::bf16 a_bf16, const choreo::bf16 b_bf16,
      const choreo::bf16 dt_bias_bf16, const choreo::f32 a_log_f32,
      const choreo::f32 softplus_beta, const choreo::f32 softplus_threshold,
      const choreo::f32 scale, choreo::bf16 *__restrict__ q_bf16,
      choreo::bf16 *__restrict__ k_bf16, choreo::bf16 *__restrict__ v_bf16,
      choreo::bf16 *__restrict__ o_bf16, choreo::f32 *__restrict__ h) {
    constexpr int ROWS = BK / 32;
    int tid = threadIdx.x;

    float h_reg[ROWS * BV];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      int i = tid + r * 32;
#pragma unroll
      for (int j = 0; j < BV; ++j) {
        h_reg[r * BV + j] = h[i * BV + j];
      }
    }

    const float a = __bfloat162float(a_bf16);
    const float b = __bfloat162float(b_bf16);
    const float dt_bias = __bfloat162float(dt_bias_bf16);
    const float a_log = a_log_f32;
    float q[ROWS], k_local[ROWS];

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      q[r] = __bfloat162float(q_bf16[tid + r * 32]);
      k_local[r] = __bfloat162float(k_bf16[tid + r * 32]);
    }
    float v_local = __bfloat162float(v_bf16[tid]);

    float x = a + dt_bias;
    float beta_x = softplus_beta * x;
    float softplus_x = beta_x < softplus_threshold
                           ? (1.0f / softplus_beta) * logf(1.0f + expf(beta_x))
                           : x;
    float g = -expf(a_log) * softplus_x;
    float beta_val = 1.0f / (1.0f + expf(-b));

    if constexpr (USE_QK_L2NORM_IN_KERNEL) {
      float q_sum = 0.0f, k_sum = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        q_sum = fmaf(q[r], q[r], q_sum);
        k_sum = fmaf(k_local[r], k_local[r], k_sum);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += SHFL_XOR(q_sum, offset, 32);
        k_sum += SHFL_XOR(k_sum, offset, 32);
      }
      float q_div = rsqrtf(q_sum + 1e-6f);
      float k_div = rsqrtf(k_sum + 1e-6f);
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        q[r] *= q_div * scale;
        k_local[r] *= k_div;
      }
    } else {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) q[r] *= scale;
    }

    float h_scale = expf(g);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
      for (int j = 0; j < BV; ++j) {
        h_reg[r * BV + j] *= h_scale;
      }
    }

#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float v_sum = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        v_sum = fmaf(h_reg[r * BV + j], k_local[r], v_sum);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        v_sum += SHFL_XOR(v_sum, offset, 32);
      }
      if (j == tid) v_local -= v_sum;
    }

    v_local *= beta_val;

#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float vj = __shfl_sync(uint32_t(-1), v_local, j, 32);
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        h_reg[r * BV + j] += k_local[r] * vj;
      }
    }

#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float o_sum = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        o_sum = fmaf(h_reg[r * BV + j], q[r], o_sum);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        o_sum += SHFL_XOR(o_sum, offset, 32);
      }
      o_bf16[j] = __float2bfloat16(o_sum);
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      int i = tid + r * 32;
#pragma unroll
      for (int j = 0; j < BV; ++j) {
        h[i * BV + j] = h_reg[r * BV + j];
      }
    }
  }


/*
 * [###BALDLEE###]cu_seqlens is None: False
 * [###BALDLEE###]is_kda: False
 * [###BALDLEE###]q.type torch.bfloat16
 * [###BALDLEE###]k.type torch.bfloat16
 * [###BALDLEE###]v.type torch.bfloat16
 * [###BALDLEE###]a.type torch.bfloat16
 * [###BALDLEE###]b.type torch.bfloat16
 * [###BALDLEE###]A_log.type torch.bfloat16
 * [###BALDLEE###]dt_bias.type torch.bfloat16
 * [###BALDLEE###]initial_state_indices.type torch.int32
 * [###BALDLEE###]initial_state_indices.shape torch.Size([1])
 * [###BALDLEE###]initial_state_source.type torch.float32
 * [###BALDLEE###]cu_seqlens.type torch.int32
 * [###BALDLEE###]use_qk_l2norm_in_kernel: True
 * [###BALDLEE###]initial_state_source is None: False
 * [###BALDLEE###]B = 1, T = 1
 * [###BALDLEE###]BK = 128, BV = 32
 * [###BALDLEE###]cu_seqlens is None: False
 * [###BALDLEE###]is_kda: False
 * [###BALDLEE###]o.shape: torch.Size([1, 1, 1, 48, 128]) first dim is NK and it
 * is elimited here
 * [###BALDLEE###]o.dtype: torch.bfloat16
 */
// clang-format off
__global__ __launch_bounds__(32, 1) void __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel(float * A_log, bf16 * a, bf16 * dt_bias, bf16 * q, bf16 * k, bf16 * v, bf16 * b, bf16 * o, float * initial_state_source, int * initial_state_indices, int * cu_seqlens, float scale, float softplus_beta, float softplus_threshold, bool IS_KDA, bool IS_VARLEN, bool USE_QK_L2NORM_IN_KERNEL, bool USE_INITAL_STATE, unsigned B, unsigned H, unsigned HV, unsigned K, unsigned N, unsigned T, unsigned V) {
  constexpr int BK = 128;
  constexpr int BV = 32;
  constexpr int ROWS = BK / 32;
  int tid = threadIdx.x;

  int i_v = blockIdx.x;
  int i_n = blockIdx.y;
  int i_hv = blockIdx.z;
  int i_h = i_hv / (HV / H);
  int bos = i_n * T;

  float a_log_l = A_log[i_hv];
  float dt_bias_l = __bfloat162float(dt_bias[i_hv]);

  float h_reg[ROWS * BV];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
#pragma unroll
    for (int j = 0; j < BV; ++j) {
      h_reg[r * BV + j] = 0.0f;
    }
  }

  int idx = initial_state_indices[i_n];
  if (idx >= 0) {
    const float *h0 = initial_state_source + (size_t)idx * HV * K * V
                     + (size_t)i_hv * K * V
                     + (size_t)i_v * BV;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      int row = tid + r * 32;
#pragma unroll
      for (int j = 0; j < BV; ++j) {
        h_reg[r * BV + j] = h0[row * V + j];
      }
    }
  }

  const bf16 *p_a = a + (size_t)bos * HV + i_hv;
  const bf16 *p_b = b + (size_t)bos * HV + i_hv;
  const bf16 *p_q = q + ((size_t)bos * H + i_h) * K;
  const bf16 *p_k = k + ((size_t)bos * H + i_h) * K;
  const bf16 *p_v = v + ((size_t)bos * HV + i_hv) * V + i_v * BV;
  bf16 *p_o = o + ((size_t)bos * HV + i_hv) * V + i_v * BV;

  for (unsigned t = 0; t < T; ++t) {
    float b_a = __bfloat162float(*p_a);
    float b_b = __bfloat162float(*p_b);

    float q_local[ROWS], k_local[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      q_local[r] = __bfloat162float(p_q[tid + r * 32]);
      k_local[r] = __bfloat162float(p_k[tid + r * 32]);
    }
    float v_local = __bfloat162float(p_v[tid]);

    float x = b_a + dt_bias_l;
    float beta_x = softplus_beta * x;
    float softplus_x = beta_x < softplus_threshold
                           ? (1.0f / softplus_beta) * logf(1.0f + expf(beta_x))
                           : x;
    float g = -expf(a_log_l) * softplus_x;
    float beta_val = 1.0f / (1.0f + expf(-b_b));

    {
      float q_sum = 0.0f, k_sum = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        q_sum = fmaf(q_local[r], q_local[r], q_sum);
        k_sum = fmaf(k_local[r], k_local[r], k_sum);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += SHFL_XOR(q_sum, offset, 32);
        k_sum += SHFL_XOR(k_sum, offset, 32);
      }
      float q_div = rsqrtf(q_sum + 1e-6f);
      float k_div = rsqrtf(k_sum + 1e-6f);
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        q_local[r] *= q_div * scale;
        k_local[r] *= k_div;
      }
    }

    float h_scale = expf(g);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
      for (int j = 0; j < BV; ++j) {
        h_reg[r * BV + j] *= h_scale;
      }
    }

#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float v_sum = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        v_sum = fmaf(h_reg[r * BV + j], k_local[r], v_sum);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        v_sum += SHFL_XOR(v_sum, offset, 32);
      }
      if (j == tid) v_local -= v_sum;
    }

    v_local *= beta_val;

#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float vj = __shfl_sync(uint32_t(-1), v_local, j, 32);
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        h_reg[r * BV + j] += k_local[r] * vj;
      }
    }

#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float o_sum = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        o_sum = fmaf(h_reg[r * BV + j], q_local[r], o_sum);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        o_sum += SHFL_XOR(o_sum, offset, 32);
      }
      if (tid == j) p_o[j] = __float2bfloat16(o_sum);
    }

    p_a += HV;
    p_b += HV;
    p_q += H * K;
    p_k += H * K;
    p_v += HV * V;
    p_o += HV * V;
  }

  if (idx >= 0) {
    float *h0 = initial_state_source + (size_t)idx * HV * K * V
               + (size_t)i_hv * K * V
               + (size_t)i_v * BV;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      int row = tid + r * 32;
#pragma unroll
      for (int j = 0; j < BV; ++j) {
        h0[row * V + j] = h_reg[r * BV + j];
      }
    }
  }
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

  choreo::runtime_check(((static_cast<long long>(V) + 31LL) / 32LL > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, gdn_bf16_simple.co:180.13");
  choreo::runtime_check((static_cast<long long>(N) > 0LL), "The 2nd bound item of parallelby is invalid: should be greater than 0, gdn_bf16_simple.co:180.18");
  choreo::runtime_check((static_cast<long long>(HV) > 0LL), "The 3rd bound item of parallelby is invalid: should be greater than 0, gdn_bf16_simple.co:180.23");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 1st index `i_hv` of element access 'A_log' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_simple.co:191.29");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 1st index `i_hv` of element access 'dt_bias' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_simple.co:192.34");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'initial_state_indices' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_simple.co:194.36");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'initial_state_indices' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_simple.co:230.38");
  choreo::runtime_check((static_cast<long long>(K) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::K < 16777216., gdn_bf16_simple.co:207.30");
  choreo::runtime_check((static_cast<long long>(K) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::K < 16777216., gdn_bf16_simple.co:208.73");
  choreo::runtime_check((static_cast<long long>(K) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::K < 16777216., gdn_bf16_simple.co:209.30");
  choreo::runtime_check((static_cast<long long>(K) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::K < 16777216., gdn_bf16_simple.co:210.73");
  choreo::runtime_check((static_cast<long long>(V) / ((static_cast<long long>(V) + 31LL) / 32LL) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim (::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::V / ((::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::V + 31) / 32)) < 16777216., gdn_bf16_simple.co:211.30");
  choreo::runtime_check((static_cast<long long>(V) / ((static_cast<long long>(V) + 31LL) / 32LL) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim (::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::V / ((::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::V + 31) / 32)) < 16777216., gdn_bf16_simple.co:213.54");
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
  dim3 gdims(((V + 31) / 32), N, HV);
  dim3 bdims(32, 1, 1);
  cudaEvent_t __kstart, __kstop;
  cudaEventCreate(&__kstart);
  cudaEventCreate(&__kstop);
  cudaEventRecord(__kstart);
  __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel<<<gdims, bdims>>>(A_log__device, a__device, dt_bias__device, q__device, k__device, v__device, b__device, o__device, initial_state_source__device, initial_state_indices__device, cu_seqlens__device, scale, softplus_beta, softplus_threshold, IS_KDA, IS_VARLEN, USE_QK_L2NORM_IN_KERNEL, USE_INITAL_STATE, B, H, HV, K, N, T, V);
  cudaEventRecord(__kstop);
  cudaEventSynchronize(__kstop);
  float __kms = 0;
  cudaEventElapsedTime(&__kms, __kstart, __kstop);
  fprintf(stderr, "Pure kernel time: %.4f ms\n", __kms);
  cudaEventDestroy(__kstart);
  cudaEventDestroy(__kstop);
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

template <typename T>
void read_bin(const std::string &path, std::vector<T> &data) {
  std::ifstream file(path, std::ios::binary);
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  size_t count = size / sizeof(T);
  data.resize(count);
  file.read(reinterpret_cast<char *>(data.data()), size);
  file.close();
}

int main() {
  const unsigned B = 2;
  const unsigned T = 128;
  const unsigned H = 8;
  const unsigned K = 128;
  const unsigned V = 128;
  const unsigned HV = H * K / 64;
  const unsigned N = B;

  const size_t o_count = B * T * HV * V;

  const float softplus_beta = 1.0f;
  const float softplus_threshold = 20.0f;
  const float scale = 1.0f / std::sqrt(static_cast<float>(K));

  std::string ref_dir = "/home/baldlee/workspace/choreo-attn/gdn/reference/";

  std::vector<uint32_t> A_log_bits(HV);
  std::vector<uint16_t> a_bits(B * T * HV);
  std::vector<uint16_t> dt_bias_bits(HV);
  std::vector<uint16_t> q_bits(B * T * H * K);
  std::vector<uint16_t> k_bits(B * T * H * K);
  std::vector<uint16_t> v_bits(B * T * HV * V);
  std::vector<uint16_t> b_bits(B * T * HV);
  std::vector<int32_t> indices_bits(B);
  std::vector<uint16_t> o_expected(o_count);
  std::vector<uint32_t> initial_state_bits(B * HV * K * V);

  read_bin(ref_dir + "A_log.bin", A_log_bits);
  read_bin(ref_dir + "a.bin", a_bits);
  read_bin(ref_dir + "dt_bias.bin", dt_bias_bits);
  read_bin(ref_dir + "q.bin", q_bits);
  read_bin(ref_dir + "k.bin", k_bits);
  read_bin(ref_dir + "v.bin", v_bits);
  read_bin(ref_dir + "b.bin", b_bits);
  read_bin(ref_dir + "initial_state_indices.bin", indices_bits);
  read_bin(ref_dir + "o_expected.bin", o_expected);
  read_bin(ref_dir + "initial_state_source_in.bin", initial_state_bits);

  float *A_log_d, *iss_d;
  bf16 *a_d, *dt_bias_d, *q_d, *k_d, *v_d, *b_d, *o_d;
  int *indices_d, *cu_seqlens_d;

  cudaMalloc(&A_log_d, HV * 4);
  cudaMalloc(&a_d, B * T * HV * 2);
  cudaMalloc(&dt_bias_d, HV * 2);
  cudaMalloc(&q_d, (size_t)B * T * H * K * 2);
  cudaMalloc(&k_d, (size_t)B * T * H * K * 2);
  cudaMalloc(&v_d, (size_t)B * T * HV * V * 2);
  cudaMalloc(&b_d, B * T * HV * 2);
  cudaMalloc(&o_d, (size_t)B * T * HV * V * 2);
  cudaMalloc(&iss_d, (size_t)B * HV * K * V * 4);
  cudaMalloc(&indices_d, B * 4);
  cudaMalloc(&cu_seqlens_d, N * 4);

  cudaMemcpy(A_log_d, A_log_bits.data(), HV * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(a_d, a_bits.data(), B * T * HV * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dt_bias_d, dt_bias_bits.data(), HV * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(q_d, q_bits.data(), (size_t)B * T * H * K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(k_d, k_bits.data(), (size_t)B * T * H * K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(v_d, v_bits.data(), (size_t)B * T * HV * V * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_bits.data(), B * T * HV * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(iss_d, initial_state_bits.data(), (size_t)B * HV * K * V * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(indices_d, indices_bits.data(), B * 4, cudaMemcpyHostToDevice);
  std::vector<int> cu_dummy(N, 0);
  cudaMemcpy(cu_seqlens_d, cu_dummy.data(), N * 4, cudaMemcpyHostToDevice);

  float *iss_backup;
  cudaMalloc(&iss_backup, (size_t)B * HV * K * V * 4);
  cudaMemcpy(iss_backup, iss_d, (size_t)B * HV * K * V * 4, cudaMemcpyDeviceToDevice);

  int NV = (V + 31) / 32;
  dim3 grid(NV, N, HV);
  dim3 block(32);

  for (int w = 0; w < 3; w++) {
    cudaMemcpy(iss_d, iss_backup, (size_t)B * HV * K * V * 4, cudaMemcpyDeviceToDevice);
    __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel<<<grid, block>>>(
        A_log_d, a_d, dt_bias_d, q_d, k_d, v_d, b_d, o_d, iss_d,
        indices_d, cu_seqlens_d, scale, softplus_beta, softplus_threshold,
        false, false, true, true, B, H, HV, K, N, T, V);
    cudaDeviceSynchronize();
  }

  cudaEvent_t start_ev, stop_ev;
  cudaEventCreate(&start_ev);
  cudaEventCreate(&stop_ev);

  std::cout << "=== Choreo-based kernel (T=" << T << ", grid=(" << NV << "," << N << "," << HV << ")) ===" << std::endl;
  for (int run = 0; run < 10; run++) {
    cudaMemcpy(iss_d, iss_backup, (size_t)B * HV * K * V * 4, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(start_ev);
    __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel<<<grid, block>>>(
        A_log_d, a_d, dt_bias_d, q_d, k_d, v_d, b_d, o_d, iss_d,
        indices_d, cu_seqlens_d, scale, softplus_beta, softplus_threshold,
        false, false, true, true, B, H, HV, K, N, T, V);
    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);
    float ms = 0;
    cudaEventElapsedTime(&ms, start_ev, stop_ev);
    std::cout << "  Run " << run << ": " << ms << " ms" << std::endl;
  }
  cudaEventDestroy(start_ev);
  cudaEventDestroy(stop_ev);

  cudaMemcpy(iss_d, iss_backup, (size_t)B * HV * K * V * 4, cudaMemcpyDeviceToDevice);
  __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel<<<grid, block>>>(
      A_log_d, a_d, dt_bias_d, q_d, k_d, v_d, b_d, o_d, iss_d,
      indices_d, cu_seqlens_d, scale, softplus_beta, softplus_threshold,
      false, false, true, true, B, H, HV, K, N, T, V);
  cudaDeviceSynchronize();

  std::vector<uint16_t> o_host(o_count);
  cudaMemcpy(o_host.data(), o_d, o_count * 2, cudaMemcpyDeviceToHost);

  const int TOLERANCE = 32;
  size_t o_mismatch = 0, o_small_diff = 0;
  int max_diff = 0;
  for (size_t i = 0; i < o_count; ++i) {
    if (o_host[i] != o_expected[i]) {
      int diff = std::abs((int)o_host[i] - (int)o_expected[i]);
      if (diff > max_diff) max_diff = diff;
      if (diff <= TOLERANCE) {
        o_small_diff++;
      } else {
        o_mismatch++;
        if (o_mismatch <= 10)
          std::cout << "o[" << i << "]: 0x" << std::hex << o_host[i]
                    << " vs 0x" << o_expected[i] << std::dec
                    << " (diff=" << diff << ")" << std::endl;
      }
    }
  }

  std::cout << "o mismatches (>" << TOLERANCE << " ULP): " << o_mismatch << " / " << o_count << std::endl;
  std::cout << "o small diffs (1-" << TOLERANCE << " ULP): " << o_small_diff << " / " << o_count << std::endl;
  std::cout << "o max ULP diff: " << max_diff << std::endl;
  std::cout << (o_mismatch == 0 ? "TEST PASSED" : "TEST FAILED") << std::endl;

  cudaFree(A_log_d); cudaFree(a_d); cudaFree(dt_bias_d);
  cudaFree(q_d); cudaFree(k_d); cudaFree(v_d); cudaFree(b_d);
  cudaFree(o_d); cudaFree(iss_d); cudaFree(indices_d);
  cudaFree(cu_seqlens_d); cudaFree(iss_backup);
  return 0;
}

