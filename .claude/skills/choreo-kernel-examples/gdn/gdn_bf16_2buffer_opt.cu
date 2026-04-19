
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "choreo.h"
#include <cooperative_groups.h>
using namespace choreo;

#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ 90

static inline void __choreo_check_cuda_environment__() {
  static bool already_checked = false;
  if (already_checked) return;
  already_checked = true;
  auto decode_cuda_version = [](int v, int& major, int& minor, int& patch) {
    major = v / 1000; minor = (v % 1000) / 10; patch = v % 10;
  };
  int runtime_ver = 0;
  cudaError_t err = cudaRuntimeGetVersion(&runtime_ver);
  if (err != cudaSuccess) { std::fprintf(stderr, "[choreo] CUDA runtime not available: %s\n", cudaGetErrorString(err)); std::exit(EXIT_FAILURE); }
  int driver_ver = 0;
  err = cudaDriverGetVersion(&driver_ver);
  if (err != cudaSuccess) { std::fprintf(stderr, "[choreo] CUDA driver not available: %s\n", cudaGetErrorString(err)); std::exit(EXIT_FAILURE); }
  int rMaj, rMin, rPat, dMaj, dMin, dPat, reqMaj, reqMin, reqPat;
  decode_cuda_version(runtime_ver, rMaj, rMin, rPat);
  decode_cuda_version(driver_ver, dMaj, dMin, dPat);
  decode_cuda_version(CUDART_VERSION, reqMaj, reqMin, reqPat);
  if (runtime_ver < CUDART_VERSION) { std::fprintf(stderr, "[choreo] CUDA runtime too old\n"); std::exit(EXIT_FAILURE); }
  int device_count = 0;
  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) { std::fprintf(stderr, "[choreo] No CUDA devices\n"); std::exit(EXIT_FAILURE); }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  int sm = prop.major * 10 + prop.minor;
  if (sm < __CHOREO_REQUIRED_GPU_DEVICE_SM__) { std::fprintf(stderr, "[choreo] SM too low\n"); std::exit(EXIT_FAILURE); }
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

// clang-format off

#define ISSUE_QKV_COPIES(q_dst, k_dst, v_dst, timestep) do { \
  { \
    auto src = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)q + (K * i_h + (unsigned)(H * K) * (timestep))), \
      cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K), cute::make_stride((H * K), K, cute::Int<1>{}))); \
    auto dst = cute::make_tensor(cute::make_smem_ptr<bf16>(q_dst), \
      cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K), cute::make_stride(K, K, cute::Int<1>{}))); \
    cute::copy(copy_atom, src, dst); \
  } \
  { \
    auto src = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)k + (K * i_h + (unsigned)(H * K) * (timestep))), \
      cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K), cute::make_stride((H * K), K, cute::Int<1>{}))); \
    auto dst = cute::make_tensor(cute::make_smem_ptr<bf16>(k_dst), \
      cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, K), cute::make_stride(K, K, cute::Int<1>{}))); \
    cute::copy(copy_atom, src, dst); \
  } \
  { \
    auto src = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)v + (v_chunk * blockIdx.x + V * blockIdx.z + (unsigned)(HV * V) * (timestep))), \
      cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, v_chunk), cute::make_stride((HV * V), V, cute::Int<1>{}))); \
    auto dst = cute::make_tensor(cute::make_smem_ptr<bf16>(v_dst), \
      cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<1>{}, v_chunk), cute::make_stride(v_chunk, v_chunk, cute::Int<1>{}))); \
    cute::copy(copy_atom, src, dst); \
  } \
  cute::cp_async_fence(); \
} while(0)

__global__ void __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel(
    float * A_log, bf16 * a, bf16 * dt_bias, bf16 * q, bf16 * k, bf16 * v,
    bf16 * b, bf16 * o, float * initial_state_source,
    int * initial_state_indices, int * cu_seqlens,
    float scale, float softplus_beta, float softplus_threshold,
    bool IS_KDA, bool IS_VARLEN, bool USE_QK_L2NORM_IN_KERNEL, bool USE_INITAL_STATE,
    unsigned B, unsigned H, unsigned HV, unsigned K, unsigned N, unsigned T, unsigned V,
    unsigned long mr_offset_a0, unsigned long mr_offset_a1,
    unsigned long mr_offset_b0, unsigned long mr_offset_b1,
    unsigned long mr_offset_k0, unsigned long mr_offset_k1,
    unsigned long mr_offset_q0, unsigned long mr_offset_q1,
    unsigned long mr_offset_v0, unsigned long mr_offset_v1,
    unsigned __co__shared_spm_size)
{
  extern __shared__ char smem_raw[];
  auto smem = reinterpret_cast<char*>(aligned_up_ptr<128 * 8>(smem_raw));

  alignas(16) unsigned char local_storage[576];
  float* hidden_l = (float*)(local_storage);
  bf16* o_l = (bf16*)(local_storage + 512);

  unsigned tid = threadIdx.x;
  int i_h = blockIdx.z / (HV / H);
  int bos = blockIdx.y * T;
  int LEN = T;
  unsigned v_chunk = V / ((V + 31) / 32);

  bf16* q_s[2] = { (bf16*)(smem + mr_offset_q0), (bf16*)(smem + mr_offset_q1) };
  bf16* k_s[2] = { (bf16*)(smem + mr_offset_k0), (bf16*)(smem + mr_offset_k1) };
  bf16* v_s[2] = { (bf16*)(smem + mr_offset_v0), (bf16*)(smem + mr_offset_v1) };

  float a_log_l = A_log[blockIdx.z];
  bf16 dt_bias_l = dt_bias[blockIdx.z];

  #pragma unroll
  for (int i = 0; i < 128; i++) hidden_l[i] = 0.0f;

  int idx = initial_state_indices[blockIdx.y];
  if (idx >= 0) {
    #pragma unroll
    for (int r = 0; r < 4; r++) {
      auto src_t = cute::make_tensor(
          cute::make_gmem_ptr<float>((float*)initial_state_source
              + (v_chunk * blockIdx.x + K * V * blockIdx.z
                 + HV * K * V * idx + V * (r * 32 + tid))),
          cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<32>{}),
                           cute::make_stride(V, cute::Int<1>{})));
      auto dst_t = cute::make_tensor(
          (float*)hidden_l + r * 32,
          cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<32>{}),
                           cute::make_stride(cute::Int<32>{}, cute::Int<1>{})));
      cute::copy(src_t, dst_t);
    }
  }

  AsyncCopyAtom copy_atom{};
  int cur = 0;

  ISSUE_QKV_COPIES(q_s[cur], k_s[cur], v_s[cur], (unsigned)(bos));

  for (int i_l = 0; i_l < LEN; i_l++) {
    int nxt = cur ^ 1;
    if (i_l + 1 < LEN) {
      ISSUE_QKV_COPIES(q_s[nxt], k_s[nxt], v_s[nxt], (unsigned)(bos + i_l + 1));
      cute::cp_async_wait<1>();
    } else {
      cute::cp_async_wait<0>();
    }

    bf16 a_val = a[blockIdx.z + HV * (unsigned)(bos + i_l)];
    bf16 b_val = b[blockIdx.z + HV * (unsigned)(bos + i_l)];

    computation<128, 32, true, false>(
        a_val, b_val, dt_bias_l, a_log_l,
        softplus_beta, softplus_threshold, scale,
        q_s[cur], k_s[cur], v_s[cur],
        o_l, hidden_l);

    unsigned o_off = v_chunk * blockIdx.x + V * blockIdx.z + HV * V * (unsigned)(bos + i_l);
    #pragma unroll
    for (unsigned j = 0; j < 32; j++) {
      o[o_off + j] = o_l[j];
    }

    cur = nxt;
  }

  if (idx >= 0) {
    #pragma unroll
    for (int r = 0; r < 4; r++) {
      auto src_t = cute::make_tensor(
          (float*)hidden_l + r * 32,
          cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<32>{}),
                           cute::make_stride(cute::Int<32>{}, cute::Int<1>{})));
      auto dst_t = cute::make_tensor(
          cute::make_gmem_ptr<float>((float*)initial_state_source
              + (v_chunk * blockIdx.x + K * V * blockIdx.z
                 + HV * K * V * idx + V * (r * 32 + tid))),
          cute::make_layout(cute::make_shape(cute::Int<1>{}, cute::Int<32>{}),
                           cute::make_stride(V, cute::Int<1>{})));
      cute::copy(src_t, dst_t);
    }
  }
}
#undef ISSUE_QKV_COPIES

void fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel(const choreo::spanned_view<choreo::f32, 1> & A_log, const choreo::spanned_view<choreo::bf16, 3> & a, const choreo::spanned_view<choreo::bf16, 1> & dt_bias, const choreo::spanned_view<choreo::bf16, 4> & q, const choreo::spanned_view<choreo::bf16, 4> & k, const choreo::spanned_view<choreo::bf16, 4> & v, const choreo::spanned_view<choreo::bf16, 3> & b, const choreo::spanned_view<choreo::bf16, 4> & o, const choreo::spanned_view<choreo::f32, 4> & initial_state_source, const choreo::spanned_view<choreo::s32, 1> & initial_state_indices, const choreo::spanned_view<choreo::s32, 1> & cu_seqlens, float scale, float softplus_beta, float softplus_threshold) {
  __choreo_check_cuda_environment__();
  auto &B = a.shape()[0];
  auto &H = q.shape()[2];
  auto &HV = A_log.shape()[0];
  auto &K = q.shape()[3];
  auto &N = cu_seqlens.shape()[0];
  auto &T = a.shape()[1];
  auto &V = v.shape()[3];
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
  __co__shared_chunks.push_back({2, {{35,79}}, "_a_s0"});
  __co__shared_chunks.push_back({2, {{45,79}}, "_a_s1"});
  __co__shared_chunks.push_back({2, {{37,79}}, "_b_s0"});
  __co__shared_chunks.push_back({2, {{47,79}}, "_b_s1"});
  __co__shared_chunks.push_back({static_cast<size_t>((K * 2)), {{41,79}}, "_k_s0"});
  __co__shared_chunks.push_back({static_cast<size_t>((K * 2)), {{51,79}}, "_k_s1"});
  __co__shared_chunks.push_back({static_cast<size_t>((K * 2)), {{39,79}}, "_q_s0"});
  __co__shared_chunks.push_back({static_cast<size_t>((K * 2)), {{49,79}}, "_q_s1"});
  __co__shared_chunks.push_back({static_cast<size_t>(((V / ((V + 31) / 32)) * 2)), {{43,79}}, "_v_s0"});
  __co__shared_chunks.push_back({static_cast<size_t>(((V / ((V + 31) / 32)) * 2)), {{53,79}}, "_v_s1"});
  HeapSimulator __co_shared_heap_simulator;
  HeapSimulator::Result __co__shared_result = __co_shared_heap_simulator.Allocate(__co__shared_chunks, 512);
  unsigned __co__shared_spm_size = __co__shared_result.heap_size;
  unsigned long __co__shared_chunk_offsets[10];
  size_t __co__shared_chunks_idx = 0;
  for (const auto& [buffer_id, offset] : __co__shared_result.chunk_offsets)
    __co__shared_chunk_offsets[__co__shared_chunks_idx++] = offset;
  // JIT memory reuse end
  dim3 gdims(((V + 31) / 32), N, HV);
  dim3 bdims(32, 1, 1);
  unsigned smem_total = (__co__shared_spm_size + 8) + (128 - 1);
  cudaFuncSetAttribute(__choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_total);
  __choreo_device_fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel<<<gdims, bdims, smem_total>>>(A_log__device, a__device, dt_bias__device, q__device, k__device, v__device, b__device, o__device, initial_state_source__device, initial_state_indices__device, cu_seqlens__device, scale, softplus_beta, softplus_threshold, IS_KDA, IS_VARLEN, USE_QK_L2NORM_IN_KERNEL, USE_INITAL_STATE, B, H, HV, K, N, T, V, __co__shared_chunk_offsets[0], __co__shared_chunk_offsets[1], __co__shared_chunk_offsets[2], __co__shared_chunk_offsets[3], __co__shared_chunk_offsets[4], __co__shared_chunk_offsets[5], __co__shared_chunk_offsets[6], __co__shared_chunk_offsets[7], __co__shared_chunk_offsets[8], __co__shared_chunk_offsets[9], __co__shared_spm_size);
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
