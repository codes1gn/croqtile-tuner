
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

// gdn_bf16_2buffer.co - Double-buffered pipeline choreo kernel.
//
// Uses local (register) hidden state with r#t interleaving pattern.
// Overlaps DMA load[i+1] with compute[i] using 2 sets of DMA buffers + rotate.
//
// Known .cu fix required after choreo compilation:
//   Stride fix for initial_state_source DMA (span_as bug):
//     Find: cute::make_stride(cute::Int<32>{}, cute::Int<1>{})  (in initial_state lines)
//     Replace cute::Int<32>{} with V
//     Typically 2 occurrences (load + store).
//
// Compile:
//   ../choreo/choreo -t cute -arch=sm_90a -es gdn_bf16_2buffer.co -o gdn_bf16_2buffer.cu
//
// Test:
//   python3 test_choreo_kernel.py gdn_bf16_2buffer.cu --fix-stride -T 4,128

#ifndef ENABLE_DEBUG
#define ENABLE_DEBUG 1
#endif



  template <typename T>
  __device__ __forceinline__ T SHFL_XOR(T var, int lane_mask, int width) {
    return __shfl_xor_sync(uint32_t(-1), var, lane_mask, width);
  }
  template <int BK, int BV, bool USE_QK_L2NORM_IN_KERNEL, bool IS_KDA>
  __device__ void computation(
      const choreo::bf16 a_bf16, const choreo::bf16 b_bf16,
      const choreo::bf16 dt_bias_bf16, const choreo::f32 a_log_f32,
      const choreo::f32 softplus_beta, const choreo::f32 softplus_threshold,
      const choreo::f32 scale, choreo::bf16 *__restrict__ q_bf16,
      choreo::bf16 *__restrict__ k_bf16, choreo::bf16 *__restrict__ v_bf16,
      choreo::bf16 *__restrict__ o_bf16, choreo::f32 *__restrict__ h) {
    int tid = threadIdx.x;
    constexpr int ROWS = BK / 32;

    const float a = __bfloat162float(a_bf16);
    const float b = __bfloat162float(b_bf16);
    const float dt_bias = __bfloat162float(dt_bias_bf16);
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

#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float v_sum = 0.0;
#pragma unroll
      for (int ii = 0; ii < ROWS; ++ii) {
        v_sum = fmaf(h[ii * BV + j], k[ii], v_sum);
      }
#pragma unroll
      for (int offset = 32 / 2; offset > 0; offset /= 2) {
        v_sum += SHFL_XOR(v_sum, offset, 32);
      }
      if (j == tid) v_local -= v_sum;
    }

    v_local *= beta;

#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float vj = __shfl_sync(uint32_t(-1), v_local, j, 32);
#pragma unroll
      for (int ii = 0; ii < ROWS; ++ii) {
        h[ii * BV + j] += k[ii] * vj;
      }
    }

#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float o_sum = 0.0;
#pragma unroll
      for (int ii = 0; ii < ROWS; ++ii) {
        o_sum = fmaf(h[ii * BV + j], q[ii], o_sum);
      }
#pragma unroll
      for (int offset = 32 / 2; offset > 0; offset /= 2) {
        o_sum += SHFL_XOR(o_sum, offset, 32);
      }
      o_bf16[j] = __float2bfloat16(o_sum);
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
__global__ void __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel(float * A_log, bf16 * a, bf16 * dt_bias, bf16 * q, bf16 * k, bf16 * v, bf16 * b, bf16 * o, float * initial_state_source, int * initial_state_indices, int * cu_seqlens, float scale, float softplus_beta, float softplus_threshold, bool IS_KDA, bool IS_VARLEN, bool USE_QK_L2NORM_IN_KERNEL, bool USE_INITAL_STATE, unsigned B, unsigned H, unsigned HV, unsigned K, unsigned N, unsigned T, unsigned V, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_a_s0__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_a_s1__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_b_s0__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_b_s1__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_k_s0__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_k_s1__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_q_s0__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_q_s1__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_v_s0__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_v_s1__buf__, unsigned __co__shared_spm_size) {
  extern __shared__ char __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__runtime_shared_buffer__raw[];
  auto __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<128 * 8>(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__runtime_shared_buffer__raw));
  auto __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__ = reinterpret_cast<choreo::future_ring<6>*>(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__runtime_shared_buffer__ + __co__shared_spm_size);
  if (threadIdx.x <= 1 && threadIdx.y == 0 && threadIdx.z == 0)    __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__[threadIdx.x].init();
  __syncthreads();  // must sync
  { // parallel-by: gdn_bf16_2buffer.co:190.12
  alignas(16) unsigned char anon_4[576];
  auto anon_3 = (unsigned char*)__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__runtime_shared_buffer__;
  auto __choreo_vtid_x = threadIdx.x;
  int i_h = blockIdx.z / (HV / H);
  int bos = blockIdx.y * T;
  int eos = blockIdx.y * T + T;
  int LEN = T;
  // if-else: gdn_bf16_2buffer.co:196.5
  if (false) {
    bos = *((int*)cu_seqlens + blockIdx.y);
    eos = *((int*)cu_seqlens + (blockIdx.y + 1));
    LEN = (eos - bos);
  } // end if-else: gdn_bf16_2buffer.co:196.5
  float a_log_l = *((float*)A_log + blockIdx.z);
  bf16 dt_bias_l = *((bf16*)dt_bias + blockIdx.z);
  float* hidden_l = (float*)(anon_4 + 0);
  for (int i = 0; i < 128; ++i) hidden_l[i] = 0;
  int idx = *((int*)initial_state_indices + blockIdx.y);
  // if-else: gdn_bf16_2buffer.co:205.5
  if ((idx >= 0)) {
    // with-in: gdn_bf16_2buffer.co:206.7
    {
      int __iv_r__elem__0 = 0;
      // foreach: gdn_bf16_2buffer.co:207.9
      for (__iv_r__elem__0 = 0; __iv_r__elem__0 < 4; ++__iv_r__elem__0) {
        future __choreo_anon_fut__0("", 208, 11, hidden_l);
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
  } // end if-else: gdn_bf16_2buffer.co:205.5
  // with-in: gdn_bf16_2buffer.co:215.5
  {
    int __iv_i_l__elem__0 = 0;
    bf16* a_s0__buf__ = (bf16*)(anon_3 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_a_s0__buf__);
    AsyncCopyAtom choreo_copy_atom_d_1{};
    future a_s0("a_s0", 216, 7, a_s0__buf__);
    a_s0.set_atom(&choreo_copy_atom_d_1);
    a_s0.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
    a_s0.id = 2;
    bf16* b_s0__buf__ = (bf16*)(anon_3 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_b_s0__buf__);
    AsyncCopyAtom choreo_copy_atom_d_2{};
    future b_s0("b_s0", 217, 7, b_s0__buf__);
    b_s0.set_atom(&choreo_copy_atom_d_2);
    b_s0.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
    b_s0.id = 3;
    bf16* q_s0__buf__ = (bf16*)(anon_3 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_q_s0__buf__);
    AsyncCopyAtom choreo_copy_atom_d_3{};
    future q_s0("q_s0", 218, 7, q_s0__buf__);
    q_s0.set_atom(&choreo_copy_atom_d_3);
    q_s0.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
    q_s0.id = 4;
    bf16* k_s0__buf__ = (bf16*)(anon_3 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_k_s0__buf__);
    AsyncCopyAtom choreo_copy_atom_d_4{};
    future k_s0("k_s0", 219, 7, k_s0__buf__);
    k_s0.set_atom(&choreo_copy_atom_d_4);
    k_s0.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
    k_s0.id = 5;
    bf16* v_s0__buf__ = (bf16*)(anon_3 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_v_s0__buf__);
    AsyncCopyAtom choreo_copy_atom_d_5{};
    future v_s0("v_s0", 220, 7, v_s0__buf__);
    v_s0.set_atom(&choreo_copy_atom_d_5);
    v_s0.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
    v_s0.id = 6;
    bf16* a_s1__buf__ = (bf16*)(anon_3 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_a_s1__buf__);
    AsyncCopyAtom choreo_copy_atom_d_6{};
    future a_s1("a_s1", 221, 7, a_s1__buf__);
    a_s1.set_atom(&choreo_copy_atom_d_6);
    a_s1.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
    a_s1.id = 7;
    bf16* b_s1__buf__ = (bf16*)(anon_3 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_b_s1__buf__);
    AsyncCopyAtom choreo_copy_atom_d_7{};
    future b_s1("b_s1", 222, 7, b_s1__buf__);
    b_s1.set_atom(&choreo_copy_atom_d_7);
    b_s1.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
    b_s1.id = 8;
    bf16* q_s1__buf__ = (bf16*)(anon_3 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_q_s1__buf__);
    AsyncCopyAtom choreo_copy_atom_d_8{};
    future q_s1("q_s1", 223, 7, q_s1__buf__);
    q_s1.set_atom(&choreo_copy_atom_d_8);
    q_s1.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
    q_s1.id = 9;
    bf16* k_s1__buf__ = (bf16*)(anon_3 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_k_s1__buf__);
    AsyncCopyAtom choreo_copy_atom_d_9{};
    future k_s1("k_s1", 224, 7, k_s1__buf__);
    k_s1.set_atom(&choreo_copy_atom_d_9);
    k_s1.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
    k_s1.id = 10;
    bf16* v_s1__buf__ = (bf16*)(anon_3 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_v_s1__buf__);
    AsyncCopyAtom choreo_copy_atom_d_10{};
    future v_s1("v_s1", 225, 7, v_s1__buf__);
    v_s1.set_atom(&choreo_copy_atom_d_10);
    v_s1.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
    v_s1.id = 11;
    bf16* o_l = (bf16*)(anon_4 + 512);
    auto __shape3_a = cute::make_shape(cute::Int<1>{}, cute::Int<1>{});
    auto __stride3_a = cute::make_stride(HV, cute::Int<1>{});
    auto __layout3_a = cute::make_layout(__shape3_a, __stride3_a);
    auto __tensor3_a = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)a + (blockIdx.z + HV * (bos + __iv_i_l__elem__0))), __layout3_a);
    auto __shape4_a_s1 = cute::make_shape(cute::Int<1>{}, cute::Int<1>{});
    auto __stride4_a_s1 = cute::make_stride(cute::Int<1>{}, cute::Int<1>{});
    auto __layout4_a_s1 = cute::make_layout(__shape4_a_s1, __stride4_a_s1);
    auto __tensor4_a_s1 = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)a_s1.data() + 0), __layout4_a_s1);
    cute::copy(*(AsyncCopyAtom*)a_s1.get_atom(), __tensor3_a, __tensor4_a_s1);
    cute::cp_async_fence();
    a_s1.trigger();
    auto __shape5_b = cute::make_shape(cute::Int<1>{}, cute::Int<1>{});
    auto __stride5_b = cute::make_stride(HV, cute::Int<1>{});
    auto __layout5_b = cute::make_layout(__shape5_b, __stride5_b);
    auto __tensor5_b = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)b + (blockIdx.z + HV * (bos + __iv_i_l__elem__0))), __layout5_b);
    auto __shape6_b_s1 = cute::make_shape(cute::Int<1>{}, cute::Int<1>{});
    auto __stride6_b_s1 = cute::make_stride(cute::Int<1>{}, cute::Int<1>{});
    auto __layout6_b_s1 = cute::make_layout(__shape6_b_s1, __stride6_b_s1);
    auto __tensor6_b_s1 = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)b_s1.data() + 0), __layout6_b_s1);
    cute::copy(*(AsyncCopyAtom*)b_s1.get_atom(), __tensor5_b, __tensor6_b_s1);
    cute::cp_async_fence();
    b_s1.trigger();
    auto __shape7_q = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K);
    auto __stride7_q = cute::make_stride((H * K), K, cute::Int<1>{});
    auto __layout7_q = cute::make_layout(__shape7_q, __stride7_q);
    auto __tensor7_q = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)q + (K * (blockIdx.z / (HV / H)) + H * K * (bos + __iv_i_l__elem__0))), __layout7_q);
    auto __shape8_q_s1 = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K);
    auto __stride8_q_s1 = cute::make_stride(K, K, cute::Int<1>{});
    auto __layout8_q_s1 = cute::make_layout(__shape8_q_s1, __stride8_q_s1);
    auto __tensor8_q_s1 = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)q_s1.data() + 0), __layout8_q_s1);
    cute::copy(*(AsyncCopyAtom*)q_s1.get_atom(), __tensor7_q, __tensor8_q_s1);
    cute::cp_async_fence();
    q_s1.trigger();
    auto __shape9_k = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K);
    auto __stride9_k = cute::make_stride((H * K), K, cute::Int<1>{});
    auto __layout9_k = cute::make_layout(__shape9_k, __stride9_k);
    auto __tensor9_k = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)k + (K * (blockIdx.z / (HV / H)) + H * K * (bos + __iv_i_l__elem__0))), __layout9_k);
    auto __shape10_k_s1 = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K);
    auto __stride10_k_s1 = cute::make_stride(K, K, cute::Int<1>{});
    auto __layout10_k_s1 = cute::make_layout(__shape10_k_s1, __stride10_k_s1);
    auto __tensor10_k_s1 = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)k_s1.data() + 0), __layout10_k_s1);
    cute::copy(*(AsyncCopyAtom*)k_s1.get_atom(), __tensor9_k, __tensor10_k_s1);
    cute::cp_async_fence();
    k_s1.trigger();
    auto __shape11_v = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, (V / ((V + cute::Int<31>{}) / cute::Int<32>{})));
    auto __stride11_v = cute::make_stride((HV * V), V, cute::Int<1>{});
    auto __layout11_v = cute::make_layout(__shape11_v, __stride11_v);
    auto __tensor11_v = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)v + (V / ((V + 31) / 32) * blockIdx.x + (V * blockIdx.z + HV * V * (bos + __iv_i_l__elem__0)))), __layout11_v);
    auto __shape12_v_s1 = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, (V / ((V + cute::Int<31>{}) / cute::Int<32>{})));
    auto __stride12_v_s1 = cute::make_stride((V / ((V + cute::Int<31>{}) / cute::Int<32>{})), (V / ((V + cute::Int<31>{}) / cute::Int<32>{})), cute::Int<1>{});
    auto __layout12_v_s1 = cute::make_layout(__shape12_v_s1, __stride12_v_s1);
    auto __tensor12_v_s1 = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)v_s1.data() + 0), __layout12_v_s1);
    cute::copy(*(AsyncCopyAtom*)v_s1.get_atom(), __tensor11_v, __tensor12_v_s1);
    cute::cp_async_fence();
    v_s1.trigger();
    auto anon_1 = 1;
    auto anon_2 = 0;
    // foreach: gdn_bf16_2buffer.co:242.7
    for (__iv_i_l__elem__0 = (1); __iv_i_l__elem__0 < LEN + 0; ++__iv_i_l__elem__0) {
      auto __shape13_a = cute::make_shape(cute::Int<1>{}, cute::Int<1>{});
      auto __stride13_a = cute::make_stride(HV, cute::Int<1>{});
      auto __layout13_a = cute::make_layout(__shape13_a, __stride13_a);
      auto __tensor13_a = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)a + (blockIdx.z + HV * (bos + __iv_i_l__elem__0))), __layout13_a);
      auto __shape14_a_s0 = cute::make_shape(cute::Int<1>{}, cute::Int<1>{});
      auto __stride14_a_s0 = cute::make_stride(cute::Int<1>{}, cute::Int<1>{});
      auto __layout14_a_s0 = cute::make_layout(__shape14_a_s0, __stride14_a_s0);
      auto __tensor14_a_s0 = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)a_s0.data() + 0), __layout14_a_s0);
      cute::copy(*(AsyncCopyAtom*)a_s0.get_atom(), __tensor13_a, __tensor14_a_s0);
      cute::cp_async_fence();
      a_s0.trigger();
      auto __shape15_b = cute::make_shape(cute::Int<1>{}, cute::Int<1>{});
      auto __stride15_b = cute::make_stride(HV, cute::Int<1>{});
      auto __layout15_b = cute::make_layout(__shape15_b, __stride15_b);
      auto __tensor15_b = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)b + (blockIdx.z + HV * (bos + __iv_i_l__elem__0))), __layout15_b);
      auto __shape16_b_s0 = cute::make_shape(cute::Int<1>{}, cute::Int<1>{});
      auto __stride16_b_s0 = cute::make_stride(cute::Int<1>{}, cute::Int<1>{});
      auto __layout16_b_s0 = cute::make_layout(__shape16_b_s0, __stride16_b_s0);
      auto __tensor16_b_s0 = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)b_s0.data() + 0), __layout16_b_s0);
      cute::copy(*(AsyncCopyAtom*)b_s0.get_atom(), __tensor15_b, __tensor16_b_s0);
      cute::cp_async_fence();
      b_s0.trigger();
      auto __shape17_q = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K);
      auto __stride17_q = cute::make_stride((H * K), K, cute::Int<1>{});
      auto __layout17_q = cute::make_layout(__shape17_q, __stride17_q);
      auto __tensor17_q = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)q + (K * (blockIdx.z / (HV / H)) + H * K * (bos + __iv_i_l__elem__0))), __layout17_q);
      auto __shape18_q_s0 = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K);
      auto __stride18_q_s0 = cute::make_stride(K, K, cute::Int<1>{});
      auto __layout18_q_s0 = cute::make_layout(__shape18_q_s0, __stride18_q_s0);
      auto __tensor18_q_s0 = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)q_s0.data() + 0), __layout18_q_s0);
      cute::copy(*(AsyncCopyAtom*)q_s0.get_atom(), __tensor17_q, __tensor18_q_s0);
      cute::cp_async_fence();
      q_s0.trigger();
      auto __shape19_k = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K);
      auto __stride19_k = cute::make_stride((H * K), K, cute::Int<1>{});
      auto __layout19_k = cute::make_layout(__shape19_k, __stride19_k);
      auto __tensor19_k = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)k + (K * (blockIdx.z / (HV / H)) + H * K * (bos + __iv_i_l__elem__0))), __layout19_k);
      auto __shape20_k_s0 = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K);
      auto __stride20_k_s0 = cute::make_stride(K, K, cute::Int<1>{});
      auto __layout20_k_s0 = cute::make_layout(__shape20_k_s0, __stride20_k_s0);
      auto __tensor20_k_s0 = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)k_s0.data() + 0), __layout20_k_s0);
      cute::copy(*(AsyncCopyAtom*)k_s0.get_atom(), __tensor19_k, __tensor20_k_s0);
      cute::cp_async_fence();
      k_s0.trigger();
      auto __shape21_v = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, (V / ((V + cute::Int<31>{}) / cute::Int<32>{})));
      auto __stride21_v = cute::make_stride((HV * V), V, cute::Int<1>{});
      auto __layout21_v = cute::make_layout(__shape21_v, __stride21_v);
      auto __tensor21_v = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)v + (V / ((V + 31) / 32) * blockIdx.x + (V * blockIdx.z + HV * V * (bos + __iv_i_l__elem__0)))), __layout21_v);
      auto __shape22_v_s0 = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, (V / ((V + cute::Int<31>{}) / cute::Int<32>{})));
      auto __stride22_v_s0 = cute::make_stride((V / ((V + cute::Int<31>{}) / cute::Int<32>{})), (V / ((V + cute::Int<31>{}) / cute::Int<32>{})), cute::Int<1>{});
      auto __layout22_v_s0 = cute::make_layout(__shape22_v_s0, __stride22_v_s0);
      auto __tensor22_v_s0 = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)v_s0.data() + 0), __layout22_v_s0);
      cute::copy(*(AsyncCopyAtom*)v_s0.get_atom(), __tensor21_v, __tensor22_v_s0);
      cute::cp_async_fence();
      v_s0.trigger();
      a_s1.wait();
      b_s1.wait();
      q_s1.wait();
      k_s1.wait();
      v_s1.wait();
      computation<128, 32, true, false>(*((bf16*)a_s1.data()), *((bf16*)b_s1.data()), dt_bias_l, a_log_l, softplus_beta, softplus_threshold, scale, (bf16*)q_s1.data(), (bf16*)k_s1.data(), (bf16*)v_s1.data(), (bf16*)o_l, (float*)hidden_l);
      future __choreo_anon_fut__11("", 260, 9);
      auto __shape23_o_l = cute::make_shape(cute::Int<32>{});
      auto __stride23_o_l = cute::make_stride(cute::Int<1>{});
      auto __layout23_o_l = cute::make_layout(__shape23_o_l, __stride23_o_l);
      auto __tensor23_o_l = cute::make_tensor(((bf16*)o_l + 0), __layout23_o_l);
      auto __shape24_o = cute::make_shape(cute::Int<32>{});
      auto __stride24_o = cute::make_stride(cute::Int<1>{});
      auto __layout24_o = cute::make_layout(__shape24_o, __stride24_o);
      auto __tensor24_o = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)o + (V / ((V + 31) / 32) * blockIdx.x + (V * blockIdx.z + HV * V * (bos + __iv_i_l__elem__0 - 1)))), __layout24_o);
      opt_copy(__tensor23_o_l, __tensor24_o);
      __syncthreads();
      choreo::rotate(a_s0, a_s1);
      choreo::rotate(b_s0, b_s1);
      choreo::rotate(q_s0, q_s1);
      choreo::rotate(k_s0, k_s1);
      choreo::rotate(v_s0, v_s1);
    } // i_l__elem__0
    __iv_i_l__elem__0 = 0;
    a_s1.wait();
    b_s1.wait();
    q_s1.wait();
    k_s1.wait();
    v_s1.wait();
    computation<128, 32, true, false>(*((bf16*)a_s1.data()), *((bf16*)b_s1.data()), dt_bias_l, a_log_l, softplus_beta, softplus_threshold, scale, (bf16*)q_s1.data(), (bf16*)k_s1.data(), (bf16*)v_s1.data(), (bf16*)o_l, (float*)hidden_l);
    future __choreo_anon_fut__12("", 278, 7);
    auto __shape25_o_l = cute::make_shape(cute::Int<32>{});
    auto __stride25_o_l = cute::make_stride(cute::Int<1>{});
    auto __layout25_o_l = cute::make_layout(__shape25_o_l, __stride25_o_l);
    auto __tensor25_o_l = cute::make_tensor(((bf16*)o_l + 0), __layout25_o_l);
    auto __shape26_o = cute::make_shape(cute::Int<32>{});
    auto __stride26_o = cute::make_stride(cute::Int<1>{});
    auto __layout26_o = cute::make_layout(__shape26_o, __stride26_o);
    auto __tensor26_o = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)o + (V / ((V + 31) / 32) * blockIdx.x + (V * blockIdx.z + HV * V * (eos - 1)))), __layout26_o);
    opt_copy(__tensor25_o_l, __tensor26_o);
    __syncthreads();
  }
  // if-else: gdn_bf16_2buffer.co:283.5
  if (true) {
    idx = *((int*)initial_state_indices + blockIdx.y);
    // if-else: gdn_bf16_2buffer.co:285.7
    if ((idx >= 0)) {
      // with-in: gdn_bf16_2buffer.co:286.9
      {
        int __iv_r__elem__0 = 0;
        // foreach: gdn_bf16_2buffer.co:287.11
        for (__iv_r__elem__0 = 0; __iv_r__elem__0 < 4; ++__iv_r__elem__0) {
          future __choreo_anon_fut__13("", 288, 13, initial_state_source);
          auto __shape27_hidden_l = cute::make_shape(cute::Int<1>{}, cute::Int<32>{});
          auto __stride27_hidden_l = cute::make_stride(cute::Int<32>{}, cute::Int<1>{});
          auto __layout27_hidden_l = cute::make_layout(__shape27_hidden_l, __stride27_hidden_l);
          auto __tensor27_hidden_l = cute::make_tensor(((float*)hidden_l + (__iv_r__elem__0 * 32)), __layout27_hidden_l);
          auto __shape28_initial_state_source = cute::make_shape(cute::Int<1>{}, cute::Int<32>{});
          auto __stride28_initial_state_source = cute::make_stride(cute::Int<32>{}, cute::Int<1>{});
          auto __layout28_initial_state_source = cute::make_layout(__shape28_initial_state_source, __stride28_initial_state_source);
          auto __tensor28_initial_state_source = cute::make_tensor(cute::make_gmem_ptr<float>((float*)initial_state_source + (V / ((V + 31) / 32) * blockIdx.x + (K * V * blockIdx.z + (HV * (K * V) * idx + V * (__iv_r__elem__0 * 32 + __choreo_vtid_x))))), __layout28_initial_state_source);
          opt_copy(__tensor27_hidden_l, __tensor28_initial_state_source);
          __syncthreads();
        } // r__elem__0
        __iv_r__elem__0 = 0;
      }
    } // end if-else: gdn_bf16_2buffer.co:285.7
  } // end if-else: gdn_bf16_2buffer.co:283.5
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

  choreo::runtime_check(((static_cast<long long>(V) + 31LL) / 32LL > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, gdn_bf16_2buffer.co:190.13");
  choreo::runtime_check((static_cast<long long>(N) > 0LL), "The 2nd bound item of parallelby is invalid: should be greater than 0, gdn_bf16_2buffer.co:190.18");
  choreo::runtime_check((static_cast<long long>(HV) > 0LL), "The 3rd bound item of parallelby is invalid: should be greater than 0, gdn_bf16_2buffer.co:190.23");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 1st index `i_hv` of element access 'A_log' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_2buffer.co:201.29");
  choreo::runtime_check((static_cast<long long>(HV) - 1 < static_cast<long long>(HV)), "The 1st index `i_hv` of element access 'dt_bias' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::HV, gdn_bf16_2buffer.co:202.34");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'initial_state_indices' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_2buffer.co:204.36");
  choreo::runtime_check((static_cast<long long>(N) - 1 < static_cast<long long>(B)), "The 1st index `i_n` of element access 'initial_state_indices' should be less than ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::B, gdn_bf16_2buffer.co:284.38");
  choreo::runtime_check((static_cast<long long>(K) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::K < 16777216., gdn_bf16_2buffer.co:233.29");
  choreo::runtime_check((static_cast<long long>(K) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::K < 16777216., gdn_bf16_2buffer.co:234.73");
  choreo::runtime_check((static_cast<long long>(K) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::K < 16777216., gdn_bf16_2buffer.co:235.29");
  choreo::runtime_check((static_cast<long long>(K) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::K < 16777216., gdn_bf16_2buffer.co:236.73");
  choreo::runtime_check((static_cast<long long>(V) / ((static_cast<long long>(V) + 31LL) / 32LL) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim (::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::V / ((::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::V + 31) / 32)) < 16777216., gdn_bf16_2buffer.co:237.29");
  choreo::runtime_check((static_cast<long long>(V) / ((static_cast<long long>(V) + 31LL) / 32LL) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim (::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::V / ((::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::V + 31) / 32)) < 16777216., gdn_bf16_2buffer.co:239.54");
  choreo::runtime_check((static_cast<long long>(K) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::K < 16777216., gdn_bf16_2buffer.co:247.31");
  choreo::runtime_check((static_cast<long long>(K) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::K < 16777216., gdn_bf16_2buffer.co:248.75");
  choreo::runtime_check((static_cast<long long>(K) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::K < 16777216., gdn_bf16_2buffer.co:249.31");
  choreo::runtime_check((static_cast<long long>(K) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim ::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::K < 16777216., gdn_bf16_2buffer.co:250.75");
  choreo::runtime_check((static_cast<long long>(V) / ((static_cast<long long>(V) + 31LL) / 32LL) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim (::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::V / ((::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::V + 31) / 32)) < 16777216., gdn_bf16_2buffer.co:251.31");
  choreo::runtime_check((static_cast<long long>(V) / ((static_cast<long long>(V) + 31LL) / 32LL) < 16777216LL), "On SM_90A, must satisfy: the 3rd dim (::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::V / ((::fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel::V + 31) / 32)) < 16777216., gdn_bf16_2buffer.co:253.56");
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
  // JIT memory reuse begin
  HeapSimulator::Chunks __co__shared_chunks;
  __co__shared_chunks.push_back({2, {{35,79}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_a_s0__buf__"});
  __co__shared_chunks.push_back({2, {{45,79}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_a_s1__buf__"});
  __co__shared_chunks.push_back({2, {{37,79}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_b_s0__buf__"});
  __co__shared_chunks.push_back({2, {{47,79}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_b_s1__buf__"});
  __co__shared_chunks.push_back({static_cast<size_t>((K * 2)), {{41,79}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_k_s0__buf__"});
  __co__shared_chunks.push_back({static_cast<size_t>((K * 2)), {{51,79}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_k_s1__buf__"});
  __co__shared_chunks.push_back({static_cast<size_t>((K * 2)), {{39,79}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_q_s0__buf__"});
  __co__shared_chunks.push_back({static_cast<size_t>((K * 2)), {{49,79}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_q_s1__buf__"});
  __co__shared_chunks.push_back({static_cast<size_t>(((V / ((V + 31) / 32)) * 2)), {{43,79}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_v_s0__buf__"});
  __co__shared_chunks.push_back({static_cast<size_t>(((V / ((V + 31) / 32)) * 2)), {{53,79}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_1_v_s1__buf__"});
  HeapSimulator __co_shared_heap_simulator;
  HeapSimulator::Result __co__shared_result = __co_shared_heap_simulator.Allocate(__co__shared_chunks, 512);
  unsigned __co__shared_spm_size = __co__shared_result.heap_size;
  choreo::runtime_check(__co__shared_spm_size <= (size_t)233472, "In the memory reuse of dynamic shapes, the size of the initial shared spm should not exceed the memory usage limit 233472 bytes.");
  unsigned long __co__shared_chunk_offsets[10];
  size_t __co__shared_chunks_idx = 0;
  for (const auto& [buffer_id, offset] : __co__shared_result.chunk_offsets)
    __co__shared_chunk_offsets[__co__shared_chunks_idx++] = offset;
  // JIT memory reuse end
  dim3 __fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_gdims0(((V + 31) / 32), N, HV);
  dim3 __fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_bdims0(32, 1, 1);
  cudaFuncSetAttribute(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (__co__shared_spm_size + 8) + (128 - 1));
  __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel<<<__fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_gdims0, __fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_bdims0, (__co__shared_spm_size + 8) + (128 - 1)>>>(A_log__device, a__device, dt_bias__device, q__device, k__device, v__device, b__device, o__device, initial_state_source__device, initial_state_indices__device, cu_seqlens__device, scale, softplus_beta, softplus_threshold, IS_KDA, IS_VARLEN, USE_QK_L2NORM_IN_KERNEL, USE_INITAL_STATE, B, H, HV, K, N, T, V, __co__shared_chunk_offsets[0], __co__shared_chunk_offsets[1], __co__shared_chunk_offsets[2], __co__shared_chunk_offsets[3], __co__shared_chunk_offsets[4], __co__shared_chunk_offsets[5], __co__shared_chunk_offsets[6], __co__shared_chunk_offsets[7], __co__shared_chunk_offsets[8], __co__shared_chunk_offsets[9], __co__shared_spm_size);
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

