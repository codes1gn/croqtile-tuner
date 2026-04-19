
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
      const choreo::bf16 a_bf16, const choreo::bf16 b_bf16,
      const choreo::bf16 dt_bias_bf16, const choreo::f32 a_log_f32,
      const choreo::f32 softplus_beta, const choreo::f32 softplus_threshold,
      const choreo::f32 scale, choreo::bf16 *__restrict__ q_bf16,
      choreo::bf16 *__restrict__ k_bf16, choreo::bf16 *__restrict__ v_bf16,
      choreo::bf16 *__restrict__ o_bf16, choreo::f32 *__restrict__ h) {
    int tid = threadIdx.x;

    const float a = __bfloat162float(a_bf16);
    const float b = __bfloat162float(b_bf16);
    const float dt_bias = __bfloat162float(dt_bias_bf16);
    const float a_log = a_log_f32;
    float q[BK];
    float k[BK];

#pragma unroll
    for (int i = tid; i < BK; i += 32) {
      q[i] = __bfloat162float(q_bf16[i]);
      k[i] = __bfloat162float(k_bf16[i]);
    }
    // Each thread holds exactly one v element (BV == warp size == 32)
    float v_local = __bfloat162float(v_bf16[tid]);

    // Compute g = -exp(A_log) * softplus(a + dt_bias)
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
      for (int i = tid; i < BK; i += 32) {
        q_sum = fmaf(q[i], q[i], q_sum);
        k_sum = fmaf(k[i], k[i], k_sum);
      }
#pragma unroll
      for (int offset = 32 / 2; offset > 0; offset /= 2) {
        q_sum += SHFL_XOR(q_sum, offset, 32);
        k_sum += SHFL_XOR(k_sum, offset, 32);
      }
      float q_div = rsqrtf(q_sum + 1e-6);
      float k_div = rsqrtf(k_sum + 1e-6);
#pragma unroll
      for (int i = tid; i < BK; i += 32) {
        q[i] = q[i] * q_div;
        k[i] = k[i] * k_div;
      }
    }
#pragma unroll
    for (int i = tid; i < BK; i += 32) {
      q[i] = q[i] * scale;
    }

    // Apply gating to hidden state: h *= exp(g)
    float h_scale = expf(g);
#pragma unroll
    for (int i = tid; i < BK; i += 32) {
#pragma unroll
      for (int j = 0; j < BV; ++j) {
        h[i * BV + j] = h[i * BV + j] * h_scale;
      }
    }

    // Delta rule: v[j] -= sum_k(h[k,j] * k[k])
    // Only thread j updates its own v_local for column j
#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float v_sum = 0.0;
#pragma unroll
      for (int i = tid; i < BK; i += 32) {
        v_sum = fmaf(h[i * BV + j], k[i], v_sum);
      }
#pragma unroll
      for (int offset = 32 / 2; offset > 0; offset /= 2) {
        v_sum += SHFL_XOR(v_sum, offset, 32);
      }
      if (j == tid) v_local -= v_sum;
    }

    // Apply beta gating
    v_local *= beta;

    // Update hidden state: h[i,j] += k[i] * v[j]
    // Broadcast v[j] from thread j to all threads via shuffle
#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float vj = __shfl_sync(uint32_t(-1), v_local, j, 32);
#pragma unroll
      for (int i = tid; i < BK; i += 32) {
        h[i * BV + j] += k[i] * vj;
      }
    }

    // Compute output: o[j] = sum_k(h[k,j] * q[k])
#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float o_sum = 0.0;
#pragma unroll
      for (int i = tid; i < BK; i += 32) {
        o_sum = fmaf(h[i * BV + j], q[i], o_sum);
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
__global__ void __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel(float * A_log, bf16 * a, bf16 * dt_bias, bf16 * q, bf16 * k, bf16 * v, bf16 * b, bf16 * o, float * initial_state_source, int * initial_state_indices, int * cu_seqlens, float scale, float softplus_beta, float softplus_threshold, bool IS_KDA, bool IS_VARLEN, bool USE_QK_L2NORM_IN_KERNEL, bool USE_INITAL_STATE, unsigned B, unsigned H, unsigned HV, unsigned K, unsigned N, unsigned T, unsigned V, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_hidden_s, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_a_s__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_b_s__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_k_s__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_q_s__buf__, unsigned long mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_v_s__buf__, unsigned __co__shared_spm_size) {
  extern __shared__ char __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__runtime_shared_buffer__raw[];
  auto __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<128 * 8>(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__runtime_shared_buffer__raw));
  auto __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__ = reinterpret_cast<choreo::future_ring<6>*>(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__runtime_shared_buffer__ + __co__shared_spm_size);
  if (threadIdx.x <= 1 && threadIdx.y == 0 && threadIdx.z == 0)    __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__[threadIdx.x].init();
  __syncthreads();  // must sync
  { // parallel-by: gdn_bf16_simple.co:180.12
  alignas(16) unsigned char anon_2[64];
  auto anon_1 = (unsigned char*)__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__runtime_shared_buffer__;
  auto __choreo_vtid_x = threadIdx.x;
  int i_h = blockIdx.z / (HV / H);
  int bos = blockIdx.y * T;
  int eos = blockIdx.y * T + T;
  int LEN = T;
  // if-else: gdn_bf16_simple.co:186.5
  if (false) {
    bos = *((int*)cu_seqlens + blockIdx.y);
    eos = *((int*)cu_seqlens + (blockIdx.y + 1));
    LEN = (eos - bos);
  } // end if-else: gdn_bf16_simple.co:186.5
  float a_log_l = *((float*)A_log + blockIdx.z);
  bf16 dt_bias_l = *((bf16*)dt_bias + blockIdx.z);
  float* hidden_s = (float*)(anon_1 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_hidden_s);
  if (__CHOREO_BLOCK_SINGLE__)  {
    for (int i = 0; i < 4096; ++i) hidden_s[i] = 0;
  } // single instance
  __syncthreads();
  int idx = *((int*)initial_state_indices + blockIdx.y);
  // if-else: gdn_bf16_simple.co:195.5
  if ((idx >= 0)) {
    future __choreo_anon_fut__0("", 196, 7, hidden_s);
    auto __shape1_initial_state_source = cute::make_shape(cute::Int<128>{}, cute::Int<32>{});
    auto __stride1_initial_state_source = cute::make_stride(cute::Int<32>{}, cute::Int<1>{});
    auto __layout1_initial_state_source = cute::make_layout(__shape1_initial_state_source, __stride1_initial_state_source);
    auto __tensor1_initial_state_source = cute::make_tensor(cute::make_gmem_ptr<float>((float*)initial_state_source + (V / ((V + 31) / 32) * blockIdx.x + (K * V * blockIdx.z + HV * (K * V) * idx))), __layout1_initial_state_source);
    auto __shape2_hidden_s = cute::make_shape(cute::Int<128>{}, cute::Int<32>{});
    auto __stride2_hidden_s = cute::make_stride(cute::Int<32>{}, cute::Int<1>{});
    auto __layout2_hidden_s = cute::make_layout(__shape2_hidden_s, __stride2_hidden_s);
    auto __tensor2_hidden_s = cute::make_tensor(cute::make_smem_ptr<float>((float*)hidden_s + 0), __layout2_hidden_s);
    {
      auto tiled_copy = cute::make_tiled_copy(
        cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, f32>{},
        cute::make_layout(cute::make_shape(cute::Int<4>{}, cute::Int<8>{}), cute::make_stride(cute::Int<8>{}, cute::Int<1>{})),
        cute::make_layout(cute::make_shape(cute::Int<32>{}, cute::Int<4>{}))
      );
      auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
      auto src_thr = thr_copy.partition_S(__tensor1_initial_state_source);
      auto dst_thr = thr_copy.partition_D(__tensor2_hidden_s);
      cute::copy(tiled_copy, src_thr, dst_thr);
    }
    __syncthreads();
  } // end if-else: gdn_bf16_simple.co:195.5
  // with-in: gdn_bf16_simple.co:201.5
  {
    int __iv_i_l__elem__0 = 0;
    // foreach: gdn_bf16_simple.co:202.7
    for (__iv_i_l__elem__0 = 0; __iv_i_l__elem__0 < LEN; ++__iv_i_l__elem__0) {
      bf16* a_s__buf__ = (bf16*)(anon_1 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_a_s__buf__);
      AsyncCopyAtom choreo_copy_atom_d_1{};
      future a_s("a_s", 203, 15, a_s__buf__);
      a_s.set_atom(&choreo_copy_atom_d_1);
      a_s.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
      a_s.id = 2;
      auto __shape3_a = cute::make_shape(cute::Int<1>{}, cute::Int<1>{});
      auto __stride3_a = cute::make_stride(HV, cute::Int<1>{});
      auto __layout3_a = cute::make_layout(__shape3_a, __stride3_a);
      auto __tensor3_a = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)a + (blockIdx.z + HV * (bos + __iv_i_l__elem__0))), __layout3_a);
      auto __shape4_a_s__buf__ = cute::make_shape(cute::Int<1>{}, cute::Int<1>{});
      auto __stride4_a_s__buf__ = cute::make_stride(cute::Int<1>{}, cute::Int<1>{});
      auto __layout4_a_s__buf__ = cute::make_layout(__shape4_a_s__buf__, __stride4_a_s__buf__);
      auto __tensor4_a_s__buf__ = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)a_s__buf__ + 0), __layout4_a_s__buf__);
      cute::copy(*(AsyncCopyAtom*)a_s.get_atom(), __tensor3_a, __tensor4_a_s__buf__);
      cute::cp_async_fence();
      a_s.trigger();
      bf16* b_s__buf__ = (bf16*)(anon_1 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_b_s__buf__);
      AsyncCopyAtom choreo_copy_atom_d_2{};
      future b_s("b_s", 205, 15, b_s__buf__);
      b_s.set_atom(&choreo_copy_atom_d_2);
      b_s.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
      b_s.id = 3;
      auto __shape5_b = cute::make_shape(cute::Int<1>{}, cute::Int<1>{});
      auto __stride5_b = cute::make_stride(HV, cute::Int<1>{});
      auto __layout5_b = cute::make_layout(__shape5_b, __stride5_b);
      auto __tensor5_b = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)b + (blockIdx.z + HV * (bos + __iv_i_l__elem__0))), __layout5_b);
      auto __shape6_b_s__buf__ = cute::make_shape(cute::Int<1>{}, cute::Int<1>{});
      auto __stride6_b_s__buf__ = cute::make_stride(cute::Int<1>{}, cute::Int<1>{});
      auto __layout6_b_s__buf__ = cute::make_layout(__shape6_b_s__buf__, __stride6_b_s__buf__);
      auto __tensor6_b_s__buf__ = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)b_s__buf__ + 0), __layout6_b_s__buf__);
      cute::copy(*(AsyncCopyAtom*)b_s.get_atom(), __tensor5_b, __tensor6_b_s__buf__);
      cute::cp_async_fence();
      b_s.trigger();
      bf16* q_s__buf__ = (bf16*)(anon_1 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_q_s__buf__);
      AsyncCopyAtom choreo_copy_atom_d_3{};
      future q_s("q_s", 207, 15, q_s__buf__);
      q_s.set_atom(&choreo_copy_atom_d_3);
      q_s.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
      q_s.id = 4;
      auto __shape7_q = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K);
      auto __stride7_q = cute::make_stride((H * K), K, cute::Int<1>{});
      auto __layout7_q = cute::make_layout(__shape7_q, __stride7_q);
      auto __tensor7_q = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)q + (K * (blockIdx.z / (HV / H)) + H * K * (bos + __iv_i_l__elem__0))), __layout7_q);
      auto __shape8_q_s__buf__ = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K);
      auto __stride8_q_s__buf__ = cute::make_stride(K, K, cute::Int<1>{});
      auto __layout8_q_s__buf__ = cute::make_layout(__shape8_q_s__buf__, __stride8_q_s__buf__);
      auto __tensor8_q_s__buf__ = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)q_s__buf__ + 0), __layout8_q_s__buf__);
      cute::copy(*(AsyncCopyAtom*)q_s.get_atom(), __tensor7_q, __tensor8_q_s__buf__);
      cute::cp_async_fence();
      q_s.trigger();
      bf16* k_s__buf__ = (bf16*)(anon_1 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_k_s__buf__);
      AsyncCopyAtom choreo_copy_atom_d_4{};
      future k_s("k_s", 209, 15, k_s__buf__);
      k_s.set_atom(&choreo_copy_atom_d_4);
      k_s.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
      k_s.id = 5;
      auto __shape9_k = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K);
      auto __stride9_k = cute::make_stride((H * K), K, cute::Int<1>{});
      auto __layout9_k = cute::make_layout(__shape9_k, __stride9_k);
      auto __tensor9_k = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)k + (K * (blockIdx.z / (HV / H)) + H * K * (bos + __iv_i_l__elem__0))), __layout9_k);
      auto __shape10_k_s__buf__ = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K);
      auto __stride10_k_s__buf__ = cute::make_stride(K, K, cute::Int<1>{});
      auto __layout10_k_s__buf__ = cute::make_layout(__shape10_k_s__buf__, __stride10_k_s__buf__);
      auto __tensor10_k_s__buf__ = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)k_s__buf__ + 0), __layout10_k_s__buf__);
      cute::copy(*(AsyncCopyAtom*)k_s.get_atom(), __tensor9_k, __tensor10_k_s__buf__);
      cute::cp_async_fence();
      k_s.trigger();
      bf16* v_s__buf__ = (bf16*)(anon_1 + mr_offset_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_v_s__buf__);
      AsyncCopyAtom choreo_copy_atom_d_5{};
      future v_s("v_s", 211, 15, v_s__buf__);
      v_s.set_atom(&choreo_copy_atom_d_5);
      v_s.set_ring(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel__ring__);
      v_s.id = 6;
      auto __shape11_v = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, (V / ((V + cute::Int<31>{}) / cute::Int<32>{})));
      auto __stride11_v = cute::make_stride((HV * V), V, cute::Int<1>{});
      auto __layout11_v = cute::make_layout(__shape11_v, __stride11_v);
      auto __tensor11_v = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)v + (V / ((V + 31) / 32) * blockIdx.x + (V * blockIdx.z + HV * V * (bos + __iv_i_l__elem__0)))), __layout11_v);
      auto __shape12_v_s__buf__ = cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, (V / ((V + cute::Int<31>{}) / cute::Int<32>{})));
      auto __stride12_v_s__buf__ = cute::make_stride((V / ((V + cute::Int<31>{}) / cute::Int<32>{})), (V / ((V + cute::Int<31>{}) / cute::Int<32>{})), cute::Int<1>{});
      auto __layout12_v_s__buf__ = cute::make_layout(__shape12_v_s__buf__, __stride12_v_s__buf__);
      auto __tensor12_v_s__buf__ = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)v_s__buf__ + 0), __layout12_v_s__buf__);
      cute::copy(*(AsyncCopyAtom*)v_s.get_atom(), __tensor11_v, __tensor12_v_s__buf__);
      cute::cp_async_fence();
      v_s.trigger();
      a_s.wait();
      b_s.wait();
      q_s.wait();
      k_s.wait();
      v_s.wait();
      bf16* o_l = (bf16*)(anon_2 + 0);
      computation<128, 32, true, false>(*((bf16*)a_s.data()), *((bf16*)b_s.data()), dt_bias_l, a_log_l, softplus_beta, softplus_threshold, scale, (bf16*)q_s.data(), (bf16*)k_s.data(), (bf16*)v_s.data(), (bf16*)o_l, (float*)hidden_s);
      future __choreo_anon_fut__6("", 223, 9);
      auto __shape13_o_l = cute::make_shape(cute::Int<32>{});
      auto __stride13_o_l = cute::make_stride(cute::Int<1>{});
      auto __layout13_o_l = cute::make_layout(__shape13_o_l, __stride13_o_l);
      auto __tensor13_o_l = cute::make_tensor(((bf16*)o_l + 0), __layout13_o_l);
      auto __shape14_o = cute::make_shape(cute::Int<32>{});
      auto __stride14_o = cute::make_stride(cute::Int<1>{});
      auto __layout14_o = cute::make_layout(__shape14_o, __stride14_o);
      auto __tensor14_o = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)o + (V / ((V + 31) / 32) * blockIdx.x + (V * blockIdx.z + HV * V * (bos + __iv_i_l__elem__0)))), __layout14_o);
      opt_copy(__tensor13_o_l, __tensor14_o);
      __syncthreads();
    } // i_l__elem__0
    __iv_i_l__elem__0 = 0;
  }
  // if-else: gdn_bf16_simple.co:229.5
  if (true) {
    idx = *((int*)initial_state_indices + blockIdx.y);
    // if-else: gdn_bf16_simple.co:231.7
    if ((idx >= 0)) {
      future __choreo_anon_fut__7("", 232, 9);
      auto __shape15_hidden_s = cute::make_shape(cute::Int<128>{}, cute::Int<32>{});
      auto __stride15_hidden_s = cute::make_stride(cute::Int<32>{}, cute::Int<1>{});
      auto __layout15_hidden_s = cute::make_layout(__shape15_hidden_s, __stride15_hidden_s);
      auto __tensor15_hidden_s = cute::make_tensor(cute::make_smem_ptr<float>((float*)hidden_s + 0), __layout15_hidden_s);
      auto __shape16_initial_state_source = cute::make_shape(cute::Int<128>{}, cute::Int<32>{});
      auto __stride16_initial_state_source = cute::make_stride(cute::Int<32>{}, cute::Int<1>{});
      auto __layout16_initial_state_source = cute::make_layout(__shape16_initial_state_source, __stride16_initial_state_source);
      auto __tensor16_initial_state_source = cute::make_tensor(cute::make_gmem_ptr<float>((float*)initial_state_source + (V / ((V + 31) / 32) * blockIdx.x + (K * V * blockIdx.z + HV * (K * V) * idx))), __layout16_initial_state_source);
      {
        auto tiled_copy = cute::make_tiled_copy(
          cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, f32>{},
          cute::make_layout(cute::make_shape(cute::Int<4>{}, cute::Int<8>{}), cute::make_stride(cute::Int<8>{}, cute::Int<1>{})),
          cute::make_layout(cute::make_shape(cute::Int<32>{}, cute::Int<4>{}))
        );
        auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
        auto src_thr = thr_copy.partition_S(__tensor15_hidden_s);
        auto dst_thr = thr_copy.partition_D(__tensor16_initial_state_source);
        cute::copy(tiled_copy, src_thr, dst_thr);
      }
      __syncthreads();
    } // end if-else: gdn_bf16_simple.co:231.7
  } // end if-else: gdn_bf16_simple.co:229.5
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
  // JIT memory reuse begin
  HeapSimulator::Chunks __co__shared_chunks;
  __co__shared_chunks.push_back({16384, {{24,50}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_hidden_s"});
  __co__shared_chunks.push_back({2, {{31,43}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_a_s__buf__"});
  __co__shared_chunks.push_back({2, {{33,43}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_b_s__buf__"});
  __co__shared_chunks.push_back({static_cast<size_t>((K * 2)), {{37,43}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_k_s__buf__"});
  __co__shared_chunks.push_back({static_cast<size_t>((K * 2)), {{35,43}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_q_s__buf__"});
  __co__shared_chunks.push_back({static_cast<size_t>(((V / ((V + 31) / 32)) * 2)), {{39,43}}, "_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_paraby_0_paraby_1_paraby_2_paraby_3_within_0_foreach_0_v_s__buf__"});
  HeapSimulator __co_shared_heap_simulator;
  HeapSimulator::Result __co__shared_result = __co_shared_heap_simulator.Allocate(__co__shared_chunks, 512);
  unsigned __co__shared_spm_size = __co__shared_result.heap_size;
  choreo::runtime_check(__co__shared_spm_size <= (size_t)233472, "In the memory reuse of dynamic shapes, the size of the initial shared spm should not exceed the memory usage limit 233472 bytes.");
  unsigned long __co__shared_chunk_offsets[6];
  size_t __co__shared_chunks_idx = 0;
  for (const auto& [buffer_id, offset] : __co__shared_result.chunk_offsets)
    __co__shared_chunk_offsets[__co__shared_chunks_idx++] = offset;
  // JIT memory reuse end
  dim3 __fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_gdims0(((V + 31) / 32), N, HV);
  dim3 __fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_bdims0(32, 1, 1);
  cudaFuncSetAttribute(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (__co__shared_spm_size + 8) + (128 - 1));
  __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel<<<__fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_gdims0, __fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel_bdims0, (__co__shared_spm_size + 8) + (128 - 1)>>>(A_log__device, a__device, dt_bias__device, q__device, k__device, v__device, b__device, o__device, initial_state_source__device, initial_state_indices__device, cu_seqlens__device, scale, softplus_beta, softplus_threshold, IS_KDA, IS_VARLEN, USE_QK_L2NORM_IN_KERNEL, USE_INITAL_STATE, B, H, HV, K, N, T, V, __co__shared_chunk_offsets[0], __co__shared_chunk_offsets[1], __co__shared_chunk_offsets[2], __co__shared_chunk_offsets[3], __co__shared_chunk_offsets[4], __co__shared_chunk_offsets[5], __co__shared_spm_size);
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
  const int B = 2;
  const int T = 4;
  const int H = 8;
  const int K = 128;
  const int V = 128;
  const int HV = H * K / 64;
  const int N = B;

  const int BK = K;
  const int BV = std::min(32, V);

  const size_t o_count = B * T * HV * V;

  const choreo::f32 softplus_beta = 1.0f;
  const choreo::f32 softplus_threshold = 20.0f;
  const choreo::f32 scale = 1.0f / std::sqrt(static_cast<choreo::f32>(K));
  const bool USE_INITAL_STATE = true;
  const bool USE_QK_L2NORM_IN_KERNEL = true;
  const bool IS_VARLEN = false;
  const bool IS_KDA = false;

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

  auto A_log = choreo::make_spanview<1, choreo::f32>(
      reinterpret_cast<choreo::f32 *>(A_log_bits.data()),
      {static_cast<size_t>(HV)});
  auto a = choreo::make_spanview<3, choreo::bf16>(
      reinterpret_cast<choreo::bf16 *>(a_bits.data()),
      {static_cast<size_t>(B), static_cast<size_t>(T),
       static_cast<size_t>(HV)});
  auto dt_bias = choreo::make_spanview<1, choreo::bf16>(
      reinterpret_cast<choreo::bf16 *>(dt_bias_bits.data()),
      {static_cast<size_t>(HV)});
  auto q = choreo::make_spanview<4, choreo::bf16>(
      reinterpret_cast<choreo::bf16 *>(q_bits.data()),
      {static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(H),
       static_cast<size_t>(K)});
  auto k = choreo::make_spanview<4, choreo::bf16>(
      reinterpret_cast<choreo::bf16 *>(k_bits.data()),
      {static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(H),
       static_cast<size_t>(K)});
  auto v = choreo::make_spanview<4, choreo::bf16>(
      reinterpret_cast<choreo::bf16 *>(v_bits.data()),
      {static_cast<size_t>(B), static_cast<size_t>(T), static_cast<size_t>(HV),
       static_cast<size_t>(V)});
  auto b = choreo::make_spanview<3, choreo::bf16>(
      reinterpret_cast<choreo::bf16 *>(b_bits.data()),
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

  auto o = choreo::make_spandata<choreo::bf16>(B, T, HV, V);
  fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel(
      A_log, a, dt_bias, q, k, v, b, o.view(), initial_state_source.view(),
      indices, cu_seqlens, scale, softplus_beta, softplus_threshold);

  auto o_ptr = reinterpret_cast<uint16_t *>(o.data());
  size_t o_exact_mismatch = 0;
  size_t o_ulp1_mismatch = 0;
  for (size_t i = 0; i < o_count; ++i) {
    if (o_ptr[i] != o_expected[i]) {
      int diff = std::abs(static_cast<int>(o_ptr[i]) - static_cast<int>(o_expected[i]));
      if (diff <= 1) {
        o_ulp1_mismatch++;
      } else {
        o_exact_mismatch++;
        if (o_exact_mismatch <= 10) {
          std::cout << "o[" << i << "]: 0x" << std::hex << o_ptr[i] << " vs 0x"
                    << o_expected[i] << std::dec << " (diff=" << diff << ")" << std::endl;
        }
      }
    }
  }

  std::cout << "o exact mismatches: " << o_exact_mismatch << " / " << o_count << std::endl;
  std::cout << "o 1-ULP mismatches: " << o_ulp1_mismatch << " / " << o_count << std::endl;

  if (o_exact_mismatch == 0) {
    std::cout << "TEST PASSED" << std::endl;
  } else {
    std::cout << "TEST FAILED" << std::endl;
  }

  return 0;
}

