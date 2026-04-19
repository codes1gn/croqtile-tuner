
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
  __device__ void computation(
      const choreo::f32 a_f32, const choreo::f32 b_f32,
      const choreo::f32 dt_bias_f32, const choreo::f32 a_log_f32,
      const choreo::f32 softplus_beta, const choreo::f32 softplus_threshold,
      const choreo::f32 scale, choreo::bf16 *__restrict__ q_bf16,
      choreo::bf16 *__restrict__ k_bf16, choreo::bf16 *__restrict__ v_bf16,
      choreo::bf16 *__restrict__ o_bf16, choreo::f32 *__restrict__ h) {
    int tid = threadIdx.x;
    constexpr int ROWS = BK / 32;
    const float a = a_f32;
    const float b = b_f32;
    const float dt_bias = dt_bias_f32;
    const float a_log = a_log_f32;
    float q[ROWS];
    float k[ROWS];
#pragma unroll
    for (int ii = 0; ii < ROWS; ++ii) {
      q[ii] = __bfloat162float(q_bf16[tid + ii * 32]);
      k[ii] = __bfloat162float(k_bf16[tid + ii * 32]);
    }
    float v_local = __bfloat162float(v_bf16[tid]);
    float x = a + dt_bias;
    float beta_x = softplus_beta * x;
    float softplus_x = beta_x < softplus_threshold
                           ? (1.0 / softplus_beta) * logf(1.0 + expf(beta_x))
                           : x;
    float g = -expf(a_log) * softplus_x;
    float beta = 1.0 / (1.0 + expf(-b));
    if constexpr (USE_QK_L2NORM_IN_KERNEL) {
      float q_sum = 0.0, k_sum = 0.0;
#pragma unroll
      for (int ii = 0; ii < ROWS; ++ii) {
        q_sum = fmaf(q[ii], q[ii], q_sum);
        k_sum = fmaf(k[ii], k[ii], k_sum);
      }
#pragma unroll
      for (int offset = 32 / 2; offset > 0; offset /= 2) {
        q_sum += SHFL_XOR(q_sum, offset, 32);
        k_sum += SHFL_XOR(k_sum, offset, 32);
      }
      float q_div = rsqrtf(q_sum + 1e-6);
      float k_div = rsqrtf(k_sum + 1e-6);
#pragma unroll
      for (int ii = 0; ii < ROWS; ++ii) {
        q[ii] = q[ii] * q_div;
        k[ii] = k[ii] * k_div;
      }
    }
#pragma unroll
    for (int ii = 0; ii < ROWS; ++ii) {
      q[ii] = q[ii] * scale;
    }
    float h_scale = expf(g);
#pragma unroll
    for (int ii = 0; ii < ROWS; ++ii) {
#pragma unroll
      for (int j = 0; j < BV; ++j) {
        h[ii * BV + j] = h[ii * BV + j] * h_scale;
      }
    }
    float vp[BV];
#pragma unroll
    for (int j = 0; j < BV; ++j) {
      vp[j] = 0.0f;
#pragma unroll
      for (int ii = 0; ii < ROWS; ++ii)
        vp[j] = fmaf(h[ii * BV + j], k[ii], vp[j]);
    }
#pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
#pragma unroll
      for (int j = 0; j < BV; ++j)
        vp[j] += SHFL_XOR(vp[j], offset, 32);
    }
    v_local -= vp[tid];
    v_local *= beta;

    float vb[BV];
#pragma unroll
    for (int j = 0; j < BV; ++j)
      vb[j] = __shfl_sync(uint32_t(-1), v_local, j, 32);
#pragma unroll
    for (int j = 0; j < BV; ++j) {
#pragma unroll
      for (int ii = 0; ii < ROWS; ++ii)
        h[ii * BV + j] += k[ii] * vb[j];
    }

    float op[BV];
#pragma unroll
    for (int j = 0; j < BV; ++j) {
      op[j] = 0.0f;
#pragma unroll
      for (int ii = 0; ii < ROWS; ++ii)
        op[j] = fmaf(h[ii * BV + j], q[ii], op[j]);
    }
#pragma unroll
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
#pragma unroll
      for (int j = 0; j < BV; ++j)
        op[j] += SHFL_XOR(op[j], offset, 32);
    }
#pragma unroll
    for (int j = 0; j < BV; ++j)
      o_bf16[j] = __float2bfloat16(op[j]);
  }


// clang-format off
__global__ void __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel(float * A_log, bf16 * a, bf16 * dt_bias, bf16 * q, bf16 * k, bf16 * v, bf16 * b, bf16 * o, float * initial_state_source, int * initial_state_indices, int * cu_seqlens, float scale, float softplus_beta, float softplus_threshold, bool IS_KDA, bool IS_VARLEN, bool USE_QK_L2NORM_IN_KERNEL, bool USE_INITAL_STATE, unsigned B, unsigned H, unsigned HV, unsigned K, unsigned N, unsigned T, unsigned V) {
  auto __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__ = nullptr;
  { // parallel-by: gdn_bf16_regonly.co:150.12
  alignas(16) unsigned char anon_1[576];
  auto __choreo_vtid_x = threadIdx.x;
  int i_h = blockIdx.z / (HV / H);
  int bos = blockIdx.y * T;
  int eos = blockIdx.y * T + T;
  int LEN = T;
  // if-else: gdn_bf16_regonly.co:156.5
  if (false) {
    bos = *((int*)cu_seqlens + blockIdx.y);
    eos = *((int*)cu_seqlens + (blockIdx.y + 1));
    LEN = (eos - bos);
  } // end if-else: gdn_bf16_regonly.co:156.5
  float a_log_l = *((float*)A_log + blockIdx.z);
  float dt_bias_l = static_cast<float>(*((bf16*)dt_bias + blockIdx.z));
  float* hidden_l = (float*)(anon_1 + 0);
  for (int i = 0; i < 128; ++i) hidden_l[i] = 0;
  int idx = *((int*)initial_state_indices + blockIdx.y);
  // if-else: gdn_bf16_regonly.co:165.5
  if ((idx >= 0)) {
    // with-in: gdn_bf16_regonly.co:166.7
    {
      int __iv_r__elem__0 = 0;
      // foreach: gdn_bf16_regonly.co:167.9
      for (__iv_r__elem__0 = 0; __iv_r__elem__0 < 4; ++__iv_r__elem__0) {
        future __choreo_anon_fut__0("", 168, 11, hidden_l);
        auto __shape1_initial_state_source = cute::make_shape(cute::Int<1>{}, cute::Int<32>{});
        auto __stride1_initial_state_source = cute::make_stride(cute::Int<32>{}, cute::Int<1>{});
        auto __layout1_initial_state_source = cute::make_layout(__shape1_initial_state_source, __stride1_initial_state_source);
        auto __tensor1_initial_state_source = cute::make_tensor(cute::make_gmem_ptr<float>((float*)initial_state_source + (V / ((V + 31) / 32) * blockIdx.x + (K * V * blockIdx.z + (HV * (K * V) * idx + V * (__iv_r__elem__0 * 32 + __choreo_vtid_x))))), __layout1_initial_state_source);
        auto __shape2_hidden_l = cute::make_shape(cute::Int<1>{}, cute::Int<32>{});
        auto __stride2_hidden_l = cute::make_stride(cute::Int<32>{}, cute::Int<1>{});
        auto __layout2_hidden_l = cute::make_layout(__shape2_hidden_l, __stride2_hidden_l);
        auto __tensor2_hidden_l = cute::make_tensor(((float*)hidden_l + (__iv_r__elem__0 * 32)), __layout2_hidden_l);
        opt_copy(__tensor1_initial_state_source, __tensor2_hidden_l);
        __syncthreads();
      } // r__elem__0
      __iv_r__elem__0 = 0;
    }
  } // end if-else: gdn_bf16_regonly.co:165.5
  {
    unsigned tid = __choreo_vtid_x;
    unsigned v_chunk = V / ((V + 31) / 32);

    // Shared memory for coalesced load + transpose of q/k
    // 128 bf16 = 256 bytes, reused for q then k
    extern __shared__ char smem_raw[];
    bf16* smem_buf = (bf16*)smem_raw;

    // with-in: gdn_bf16_regonly.co:175.5
    int __iv_i_l__elem__0 = 0;
    // foreach: gdn_bf16_regonly.co:176.7
    for (__iv_i_l__elem__0 = 0; __iv_i_l__elem__0 < LEN; ++__iv_i_l__elem__0) {
      float a_val = static_cast<float>(*((bf16*)a + (HV * T * blockIdx.y) + (HV * __iv_i_l__elem__0) + blockIdx.z));
      float b_val = static_cast<float>(*((bf16*)b + (HV * T * blockIdx.y) + (HV * __iv_i_l__elem__0) + blockIdx.z));
      bf16* o_l = (bf16*)(anon_1 + 512);

      // Coalesced load q[128] → smem → pass smem pointer to computation
      unsigned ts_q = (unsigned)(bos + __iv_i_l__elem__0);
      bf16* q_src = (bf16*)q + K * i_h + (unsigned)(H * K) * ts_q;
      *(uint2*)&smem_buf[tid * 4] = *(uint2*)&q_src[tid * 4];
      __syncthreads();

      // q is now in smem_buf[0..127], computation reads smem_buf[tid + ii*32] (stride-32 from smem = fast)
      // Save q smem to registers so we can reuse smem for k
      bf16 q_reg[4];
      #pragma unroll
      for (int ii = 0; ii < 4; ++ii)
        q_reg[ii] = smem_buf[tid + ii * 32];

      // Coalesced load k[128] → smem
      bf16* k_src = (bf16*)k + K * i_h + (unsigned)(H * K) * ts_q;
      __syncthreads();
      *(uint2*)&smem_buf[tid * 4] = *(uint2*)&k_src[tid * 4];
      __syncthreads();

      // Save k smem to registers
      bf16 k_reg[4];
      #pragma unroll
      for (int ii = 0; ii < 4; ++ii)
        k_reg[ii] = smem_buf[tid + ii * 32];

      // Write transposed q/k back to smem in the layout computation expects: smem[tid + ii*32] already correct
      // Actually, we can pack q_reg and k_reg back to smem in the transposed order
      // so computation can read them as bf16* with stride-1 per row
      // But computation expects q_bf16[tid + ii * 32] — which is exactly what smem already has after transpose!
      // So just write the transposed q/k values back
      __syncthreads();
      #pragma unroll
      for (int ii = 0; ii < 4; ++ii) {
        smem_buf[tid + ii * 32] = q_reg[ii];          // q in [0..127]
        smem_buf[128 + tid + ii * 32] = k_reg[ii];    // k in [128..255]
      }
      __syncthreads();

      computation<128, 32, true, false>(a_val, b_val, dt_bias_l, a_log_l, softplus_beta, softplus_threshold, scale,
          smem_buf,          // q transposed in smem
          smem_buf + 128,    // k transposed in smem
          (bf16*)(V / ((V + 31) / 32) * blockIdx.x + (V * blockIdx.z + HV * V * (bos + __iv_i_l__elem__0)) + v),
          (bf16*)o_l, (float*)hidden_l);

      // Use opt_copy for output (same as regonly.cu) to preserve register promotion
      future __choreo_anon_fut__1("", 189, 9);
      auto __shape3_o_l = cute::make_shape(cute::Int<32>{});
      auto __stride3_o_l = cute::make_stride(cute::Int<1>{});
      auto __layout3_o_l = cute::make_layout(__shape3_o_l, __stride3_o_l);
      auto __tensor3_o_l = cute::make_tensor(((bf16*)o_l + 0), __layout3_o_l);
      auto __shape4_o = cute::make_shape(cute::Int<32>{});
      auto __stride4_o = cute::make_stride(cute::Int<1>{});
      auto __layout4_o = cute::make_layout(__shape4_o, __stride4_o);
      auto __tensor4_o = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)o + (V / ((V + 31) / 32) * blockIdx.x + (V * blockIdx.z + HV * V * (bos + __iv_i_l__elem__0)))), __layout4_o);
      opt_copy(__tensor3_o_l, __tensor4_o);
      __syncthreads();
    } // i_l__elem__0
    __iv_i_l__elem__0 = 0;
  }
  // if-else: gdn_bf16_regonly.co:195.5
  if (true) {
    idx = *((int*)initial_state_indices + blockIdx.y);
    // if-else: gdn_bf16_regonly.co:197.7
    if ((idx >= 0)) {
      // with-in: gdn_bf16_regonly.co:198.9
      {
        int __iv_r__elem__0 = 0;
        // foreach: gdn_bf16_regonly.co:199.11
        for (__iv_r__elem__0 = 0; __iv_r__elem__0 < 4; ++__iv_r__elem__0) {
          future __choreo_anon_fut__2("", 200, 13, initial_state_source);
          auto __shape5_hidden_l = cute::make_shape(cute::Int<1>{}, cute::Int<32>{});
          auto __stride5_hidden_l = cute::make_stride(cute::Int<32>{}, cute::Int<1>{});
          auto __layout5_hidden_l = cute::make_layout(__shape5_hidden_l, __stride5_hidden_l);
          auto __tensor5_hidden_l = cute::make_tensor(((float*)hidden_l + (__iv_r__elem__0 * 32)), __layout5_hidden_l);
          auto __shape6_initial_state_source = cute::make_shape(cute::Int<1>{}, cute::Int<32>{});
          auto __stride6_initial_state_source = cute::make_stride(cute::Int<32>{}, cute::Int<1>{});
          auto __layout6_initial_state_source = cute::make_layout(__shape6_initial_state_source, __stride6_initial_state_source);
          auto __tensor6_initial_state_source = cute::make_tensor(cute::make_gmem_ptr<float>((float*)initial_state_source + (V / ((V + 31) / 32) * blockIdx.x + (K * V * blockIdx.z + (HV * (K * V) * idx + V * (__iv_r__elem__0 * 32 + __choreo_vtid_x))))), __layout6_initial_state_source);
          opt_copy(__tensor5_hidden_l, __tensor6_initial_state_source);
          __syncthreads();
        } // r__elem__0
        __iv_r__elem__0 = 0;
      }
    } // end if-else: gdn_bf16_regonly.co:197.7
  } // end if-else: gdn_bf16_regonly.co:195.5
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

  choreo::runtime_check(((static_cast<long long>(V) + 31LL) / 32LL > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, gdn_bf16_regonly.co:150.13");
  choreo::runtime_check((static_cast<long long>(N) > 0LL), "The 2nd bound item of parallelby is invalid: should be greater than 0, gdn_bf16_regonly.co:150.18");
  choreo::runtime_check((static_cast<long long>(HV) > 0LL), "The 3rd bound item of parallelby is invalid: should be greater than 0, gdn_bf16_regonly.co:150.23");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 1st index `i_hv` of element access 'A_log' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_regonly.co:161.29");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 1st index `i_hv` of element access 'dt_bias' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_regonly.co:162.33");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'initial_state_indices' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_regonly.co:164.36");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'a' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_regonly.co:177.27");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 3rd index `i_hv` of element access 'a' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_regonly.co:177.37");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'b' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_regonly.co:178.27");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 3rd index `i_hv` of element access 'b' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_regonly.co:178.37");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'initial_state_indices' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_regonly.co:196.38");
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
  unsigned __smem_size = 256 * sizeof(bf16);
  __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel<<<__fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_gdims0, __fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_bdims0, __smem_size>>>(A_log__device, a__device, dt_bias__device, q__device, k__device, v__device, b__device, o__device, initial_state_source__device, initial_state_indices__device, cu_seqlens__device, scale, softplus_beta, softplus_threshold, IS_KDA, IS_VARLEN, USE_QK_L2NORM_IN_KERNEL, USE_INITAL_STATE, B, H, HV, K, N, T, V);
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

