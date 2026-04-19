
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

// REQUIRES: TARGET-SM_90
//
// =============================================================================
// CUDA-optimized fused MoE — Qwen3.5-35B-A3B FP8 end-to-end, SM_90
// =============================================================================
// Measured: ~13.10 TFLOPS end-to-end on H800 PCIe (no post-processing).
// Based on 06_final_choreo.co with CUDA-only optimizations applied:
//
// Optimizations in this .co file:
//   - parallel.async on all serving-path kernels (no cudaDeviceSynchronize)
//   - Grid swap: {block_n, eid} decomposition for L2 locality (+2.0%)
//   - QSG load pipelining: batched __ldg loads for MLP (+1.97%)
//   - Parallel Hillis-Steele prefix scan in count_and_build (+0.34%)
//   - Pipelined topk_ids loads in count_and_build (+0.65%)
//   - L2 persistence: cudaAccessPolicyWindow for rep_a_q_d (+1.6%)
//   - __cpp__ scatter epilogue: register→global atomicAdd (+vs mma.store)
//
// Requires CUDA post-processing (cuda_postprocess.py):
//   - __restrict__ on kernel pointer parameters (+0.21%)
//
// NOT applied (rescale_accumulator in cuda_postprocess.py is buggy):
//   - Rescale-accumulator (+2.67%): the transformation changes `mc += frag * s`
//     to `mc *= s`, which is mathematically wrong for multi-K-iteration loops
//     with per-block FP8 scales. Produces NaN/incorrect outputs.
//     This needs a proper implementation (see CUDA_ONLY_ANALYSIS.md).
//
// Not implemented (would need further QSG restructuring):
//   - QSG thread scaling 128→512 (+1.70%)
//   - Fused memset into QSG (+0.71%)
//   - Scale load overlap in WGMMA K-loop (+0.41%)
//
// Build workflow:
//   ./choreo -es -t cute -arch=sm_90a --disable-runtime-check \
//       --hoist-offset --hoist-scale 07_cuda_optimized.co -o /tmp/07.cu
//   python3 cuda_postprocess.py /tmp/07.cu --restrict
//   nvcc -gencode arch=compute_90a,code=sm_90a -std=c++17 \
//       -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ -DRUNMAIN \
//       -Xcompiler -static-libstdc++ -lcuda -O3 -D__USE_CUDA_TYPE__ \
//       --expt-relaxed-constexpr --use_fast_math --default-stream per-thread \
//       -I$CHOREO_ROOT -I$CHOREO_ROOT/runtime -I$CHOREO_ROOT/extern/cutlass/include \
//       /tmp/07.cu -o /tmp/07
//
// Problem dimensions: M=128, N=512, K=2048, 256 experts, topk=8, SM_90 (H800 PCIe).
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define QWEN35_DEFAULT_M 128
#define QWEN35_MAX_M 4096
#define QWEN35_DEFAULT_N 512
#define QWEN35_DEFAULT_K 2048
#define QWEN35_DEFAULT_NUM_EXPERTS 256
#define QWEN35_TOPK 8
#define QWEN35_RENORMALIZE false
#define QWEN35_SOFTCAP 0.0f

#define QWEN35_ROUTE_THREADS 32
#define QWEN35_ROUTE_VALUES_PER_THREAD 8
#define QWEN35_PACK_THREADS 128

#define QWEN35_WARP_M 64
#define QWEN35_WARP_N 128
#define QWEN35_WARP_K 32
#define QWEN35_TILE_K 128
#define QWEN35_SWIZZLE 128

#define QWEN35_BLOCK_N 128
#define QWEN35_BLOCK_K 128
#define QWEN35_K_BLOCKS 16
#define QWEN35_N_BLOCKS 4

#define QWEN35_K_512 512
#define QWEN35_K_BLOCKS_512 4

#define QWEN35_MAX_SORTED_ROUTES 32768
#define QWEN35_EXPERT_WEIGHT_ROWS 131072

#if QWEN35_DEFAULT_NUM_EXPERTS != QWEN35_ROUTE_THREADS * QWEN35_ROUTE_VALUES_PER_THREAD
  #error "QWEN35 routing decomposition must cover all experts"
#endif

#if QWEN35_DEFAULT_N % QWEN35_WARP_N != 0
  #error "QWEN35_DEFAULT_N must be divisible by QWEN35_WARP_N"
#endif

#if QWEN35_DEFAULT_K % QWEN35_TILE_K != 0
  #error "QWEN35_DEFAULT_K must be divisible by QWEN35_TILE_K"
#endif

#if QWEN35_TILE_K % QWEN35_WARP_K != 0
  #error "QWEN35_TILE_K must be divisible by QWEN35_WARP_K"
#endif

template <typename T>
__device__ __forceinline__ T SHFL_XOR(T var, int lane_mask, int width) {
  return __shfl_xor_sync(uint32_t(-1), var, lane_mask, width);
}

template <typename T, typename C>
__device__ __forceinline__ T ATOMIC_ADD(T* a, C b) {
  return atomicAdd(a, b);
}

__device__ __forceinline__ float ABS_F32(float value) {
  return value < 0.0f ? -value : value;
}

struct TopkSoftmaxResult {
  std::vector<float> weights;
  std::vector<int32_t> indices;
};

static float fp8MaxAbs() { return 448.0f; }

static float quantScaleFromMax(float max_abs) {
  float scale = max_abs / fp8MaxAbs();
  return scale > 1.0e-6f ? scale : 1.0e-6f;
}

static TopkSoftmaxResult topkSoftmaxCpuReference(const float* gating_output,
                                                 int num_tokens,
                                                 int num_experts, int topk,
                                                 bool renormalize,
                                                 float moe_softcapping) {
  TopkSoftmaxResult result;
  result.weights.resize(static_cast<size_t>(num_tokens) * topk);
  result.indices.resize(static_cast<size_t>(num_tokens) * topk);

  std::vector<float> probs(static_cast<size_t>(num_experts));
  for (int token = 0; token < num_tokens; ++token) {
    float row_max = 0.0f;
    for (int expert = 0; expert < num_experts; ++expert) {
      float value = gating_output[token * num_experts + expert];
      if (moe_softcapping != 0.0f) {
        value = std::tanh(value / moe_softcapping) * moe_softcapping;
      }
      probs[expert] = value;
      if (expert == 0 || value > row_max) row_max = value;
    }

    float row_sum = 0.0f;
    for (int expert = 0; expert < num_experts; ++expert) {
      probs[expert] = std::exp(probs[expert] - row_max);
      row_sum += probs[expert];
    }

    float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    for (int expert = 0; expert < num_experts; ++expert) probs[expert] *= inv_sum;

    float selected_sum = 0.0f;
    for (int selected = 0; selected < topk; ++selected) {
      int best_idx = 0;
      float best_val = probs[0];
      for (int expert = 1; expert < num_experts; ++expert) {
        float val = probs[expert];
        if (val > best_val || (val == best_val && expert < best_idx)) {
          best_val = val;
          best_idx = expert;
        }
      }
      result.weights[token * topk + selected] = best_val;
      result.indices[token * topk + selected] = best_idx;
      selected_sum += best_val;
      if (selected + 1 < topk) probs[best_idx] = -1.0f;
    }

    if (renormalize && selected_sum > 0.0f) {
      float inv_selected_sum = 1.0f / selected_sum;
      for (int selected = 0; selected < topk; ++selected) {
        result.weights[token * topk + selected] *= inv_selected_sum;
      }
    }
  }

  return result;
}

static bool nearlyEqual(float lhs, float rhs, float atol = 4e-2f,
                        float rtol = 1.2e-1f) {
  float diff = std::abs(lhs - rhs);
  float scale = std::max(std::abs(lhs), std::abs(rhs));
  return diff <= atol + rtol * scale;
}

static choreo::f8_e4m3 quantizeFp8(float value, float scale) {
  float scaled = value / scale;
  float clamped = std::max(-fp8MaxAbs(), std::min(fp8MaxAbs(), scaled));
  return choreo::utils::from_f32<choreo::f8_e4m3>(clamped);
}

static void initExpertWeightsQwenFp8(choreo::f8_e4m3* expert_weights_h,
                                     float* expert_scales_h) {
  for (int expert = 0; expert < QWEN35_DEFAULT_NUM_EXPERTS; ++expert) {
    for (int block_n = 0; block_n < QWEN35_N_BLOCKS; ++block_n) {
      for (int block_k = 0; block_k < QWEN35_K_BLOCKS; ++block_k) {
        float max_abs = 0.0f;
        for (int nn = 0; nn < QWEN35_BLOCK_N; ++nn) {
          int out_col = block_n * QWEN35_BLOCK_N + nn;
          for (int kk = 0; kk < QWEN35_BLOCK_K; ++kk) {
            int k_col = block_k * QWEN35_BLOCK_K + kk;
            float pattern =
                static_cast<float>(((expert + 1) * 13 + (out_col + 1) * 7 +
                                    (k_col + 1) * 5) % 23) - 11.0f;
            float value = 0.0625f * pattern;
            max_abs = std::max(max_abs, std::abs(value));
          }
        }

        float scale = quantScaleFromMax(max_abs);
        size_t scale_offset =
            (static_cast<size_t>(expert) * QWEN35_N_BLOCKS + block_n) *
                QWEN35_K_BLOCKS +
            block_k;
        expert_scales_h[scale_offset] = scale;

        for (int nn = 0; nn < QWEN35_BLOCK_N; ++nn) {
          int out_col = block_n * QWEN35_BLOCK_N + nn;
          for (int kk = 0; kk < QWEN35_BLOCK_K; ++kk) {
            int k_col = block_k * QWEN35_BLOCK_K + kk;
            float pattern =
                static_cast<float>(((expert + 1) * 13 + (out_col + 1) * 7 +
                                    (k_col + 1) * 5) % 23) - 11.0f;
            float value = 0.0625f * pattern;
            size_t weight_offset =
                (static_cast<size_t>(expert) * QWEN35_DEFAULT_N + out_col) *
                    QWEN35_DEFAULT_K +
                k_col;
            expert_weights_h[weight_offset] = quantizeFp8(value, scale);
          }
        }
      }
    }
  }
}

static void fusedMoeCpuReferenceFp8(
    const choreo::bf16* input, const float* gating_output,
    const choreo::f8_e4m3* expert_weights, const float* expert_scales,
    int num_tokens, int n_dim, int k_dim, int num_experts, int topk,
    bool renormalize, float moe_softcapping, int32_t* ref_topk_indices,
    float* ref_topk_weights, float* ref_output) {
  TopkSoftmaxResult routing =
      topkSoftmaxCpuReference(gating_output, num_tokens, num_experts, topk,
                              renormalize, moe_softcapping);

  for (int token = 0; token < num_tokens; ++token) {
    for (int selected = 0; selected < topk; ++selected) {
      ref_topk_indices[token * topk + selected] =
          routing.indices[token * topk + selected];
      ref_topk_weights[token * topk + selected] =
          routing.weights[token * topk + selected];
    }
  }

  const int k_blocks = k_dim / QWEN35_BLOCK_K;
  const int n_blocks = n_dim / QWEN35_BLOCK_N;
  std::vector<choreo::f8_e4m3> quant_input(static_cast<size_t>(num_tokens) * k_dim);
  std::vector<float> quant_scales(static_cast<size_t>(num_tokens) * k_blocks);

  for (int token = 0; token < num_tokens; ++token) {
    for (int block_k = 0; block_k < k_blocks; ++block_k) {
      float max_abs = 0.0f;
      for (int kk = 0; kk < QWEN35_BLOCK_K; ++kk) {
        int k_col = block_k * QWEN35_BLOCK_K + kk;
        float value = choreo::to_f32(input[token * k_dim + k_col]);
        max_abs = std::max(max_abs, std::abs(value));
      }
      float scale = quantScaleFromMax(max_abs);
      quant_scales[token * k_blocks + block_k] = scale;
      for (int kk = 0; kk < QWEN35_BLOCK_K; ++kk) {
        int k_col = block_k * QWEN35_BLOCK_K + kk;
        float value = choreo::to_f32(input[token * k_dim + k_col]);
        quant_input[token * k_dim + k_col] = quantizeFp8(value, scale);
      }
    }
  }

  for (int token = 0; token < num_tokens; ++token) {
    for (int out_col = 0; out_col < n_dim; ++out_col) {
      float acc = 0.0f;
      int block_n = out_col / QWEN35_BLOCK_N;
      for (int selected = 0; selected < topk; ++selected) {
        int expert = routing.indices[token * topk + selected];
        float route_weight = routing.weights[token * topk + selected];
        float dot = 0.0f;
        for (int kk = 0; kk < k_dim; ++kk) {
          int block_k = kk / QWEN35_BLOCK_K;
          float a_deq = choreo::to_f32(quant_input[token * k_dim + kk]) *
                        quant_scales[token * k_blocks + block_k];
          size_t w_offset =
              (static_cast<size_t>(expert) * n_dim + out_col) * k_dim + kk;
          size_t s_offset =
              (static_cast<size_t>(expert) * n_blocks + block_n) * k_blocks + block_k;
          float b_deq =
              choreo::to_f32(expert_weights[w_offset]) * expert_scales[s_offset];
          dot += a_deq * b_deq;
        }
        acc += route_weight * dot;
      }
      ref_output[token * n_dim + out_col] = acc;
    }
  }
}

// fused_moe_route — unchanged from baseline routing logic.
// Optimized separately on the E2E path (iter010-style: count fused into route). The
// signature takes expert_counts so launch_end_to_end can accumulate per-expert counts
// alongside top-k (see ATOMIC_ADD to expert_counts when routing).

__global__ void __choreo_device_fused_moe_route(float * __restrict__ gating_output, int * __restrict__ topk_ids, float * __restrict__ topk_weights, int * __restrict__ expert_counts, unsigned M) {
  auto __choreo_device_fused_moe_route__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:321.18
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  alignas(16) unsigned char anon_9[32];
  auto __choreo_vtid_x = threadIdx.x;
  float* probs_chunk = (float*)(anon_9 + 0);
  float thread_max = -1000000.000000f;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:325.7
  {
    int __iv_v__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:325.7
    for (__iv_v__elem__0 = 0; __iv_v__elem__0 < 8; ++__iv_v__elem__0) {
      int expert_idx = (__choreo_vtid_x * 8 + __iv_v__elem__0);
      float logit = -1000000.000000f;
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:328.9
      if ((expert_idx < 256)) {
        logit = *((float*)gating_output + (blockIdx.x * 256) + expert_idx);
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:328.9
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:331.9
      if (0.000000f != 0.000000f) {
        logit = choreo::nv_cute::numerics::tanh(logit / 0.000000f) * 0.000000f;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:331.9
      *((float*)probs_chunk + __iv_v__elem__0) = logit;
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:335.9
      if ((logit > thread_max)) {
        thread_max = logit;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:335.9
    } // v__elem__0
    __iv_v__elem__0 = 0;
  }
  int mask = 16;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:339.7
  {
    int __iv_idx__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:339.7
    for (__iv_idx__elem__0 = 0; __iv_idx__elem__0 < 10; ++__iv_idx__elem__0) {
      float other_tmax = SHFL_XOR(thread_max, mask, 32);
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:341.9
      if ((other_tmax > thread_max)) {
        thread_max = other_tmax;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:341.9
      mask = (mask >> 1);
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:343.9
      if ((mask == 0)) {
        break;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:343.9
    } // idx__elem__0
    __iv_idx__elem__0 = 0;
  }
  float row_sum = 0.000000f;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:347.7
  {
    int __iv_v__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:347.7
    for (__iv_v__elem__0 = 0; __iv_v__elem__0 < 8; ++__iv_v__elem__0) {
      float prob = choreo::nv_cute::numerics::exp(*((float*)probs_chunk + __iv_v__elem__0) - thread_max);
      *((float*)probs_chunk + __iv_v__elem__0) = prob;
      row_sum = row_sum + prob;
    } // v__elem__0
    __iv_v__elem__0 = 0;
  }
  mask = 16;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:354.7
  {
    int __iv_idx__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:354.7
    for (__iv_idx__elem__0 = 0; __iv_idx__elem__0 < 10; ++__iv_idx__elem__0) {
      float temp = SHFL_XOR(row_sum, mask, 32);
      row_sum = row_sum + temp;
      mask = (mask >> 1);
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:358.9
      if ((mask == 0)) {
        break;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:358.9
    } // idx__elem__0
    __iv_idx__elem__0 = 0;
  }
  float inv_sum = 0.000000f;
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:362.7
  if (row_sum > 0.000000f) {
    inv_sum = 1.000000f / row_sum;
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:362.7
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:363.7
  {
    int __iv_v__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:363.7
    for (__iv_v__elem__0 = 0; __iv_v__elem__0 < 8; ++__iv_v__elem__0) {
      *((float*)probs_chunk + __iv_v__elem__0) = *((float*)probs_chunk + __iv_v__elem__0) * inv_sum;
    } // v__elem__0
    __iv_v__elem__0 = 0;
  }
  float selected_sum = 0.000000f;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:367.7
  {
    int __iv_selected__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:367.7
    for (__iv_selected__elem__0 = 0; __iv_selected__elem__0 < 8; ++__iv_selected__elem__0) {
      float max_val = -1.000000f;
      int expert = 256;
      // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:370.9
      {
        int __iv_v__elem__0 = 0;
        // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:370.9
        for (__iv_v__elem__0 = 0; __iv_v__elem__0 < 8; ++__iv_v__elem__0) {
          int expert_idx = (__choreo_vtid_x * 8 + __iv_v__elem__0);
          float val = *((float*)probs_chunk + __iv_v__elem__0);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:373.11
          if ((expert_idx < 256 && (val > max_val || val == max_val && expert_idx < expert))) {
            max_val = val;
            expert = expert_idx;
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:373.11
        } // v__elem__0
        __iv_v__elem__0 = 0;
      }
      mask = 16;
      // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:381.9
      {
        int __iv_idx__elem__0 = 0;
        // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:381.9
        for (__iv_idx__elem__0 = 0; __iv_idx__elem__0 < 10; ++__iv_idx__elem__0) {
          float other_max = SHFL_XOR(max_val, mask, 32);
          int other_expert = SHFL_XOR(expert, mask, 32);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:384.11
          if ((other_max > max_val || other_max == max_val && other_expert < expert)) {
            max_val = other_max;
            expert = other_expert;
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:384.11
          mask = (mask >> 1);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:390.11
          if ((mask == 0)) {
            break;
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:390.11
        } // idx__elem__0
        __iv_idx__elem__0 = 0;
      }
      // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:393.9
      if ((__choreo_vtid_x == 0)) {
        *((int*)topk_ids + (blockIdx.x * 8) + __iv_selected__elem__0) = expert;
        *((float*)topk_weights + (blockIdx.x * 8) + __iv_selected__elem__0) = max_val;
        ATOMIC_ADD(&*((int*)expert_counts + expert), 1);
        selected_sum = selected_sum + max_val;
      }
      __syncthreads(); // end inthreads
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:400.9
      if ((__iv_selected__elem__0 + 1 < 8)) {
        // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:401.11
        if ((expert < 256 && expert / 8 == __choreo_vtid_x)) {
          *((float*)probs_chunk + (expert % 8)) = -1.000000f;
        } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:401.11
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:400.9
    } // selected__elem__0
    __iv_selected__elem__0 = 0;
  }
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:408.7
  if (false) {
    // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:409.9
    if ((__choreo_vtid_x == 0)) {
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:410.11
      if (selected_sum > 0.000000f) {
        float inv_selected_sum = 1.000000f / selected_sum;
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:412.13
        {
          int __iv_selected__elem__0 = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:412.13
          for (__iv_selected__elem__0 = 0; __iv_selected__elem__0 < 8; ++__iv_selected__elem__0) {
            *((float*)topk_weights + (blockIdx.x * 8) + __iv_selected__elem__0) = *((float*)topk_weights + (blockIdx.x * 8) + __iv_selected__elem__0) * inv_selected_sum;
          } // selected__elem__0
          __iv_selected__elem__0 = 0;
        }
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:410.11
    }
    __syncthreads(); // end inthreads
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:408.7
  } // end parallel-by
}

void fused_moe_route(const choreo::spanned_view<choreo::f32, 2> & gating_output, const choreo::spanned_view<choreo::s32, 2> & topk_ids, const choreo::spanned_view<choreo::f32, 2> & topk_weights, const choreo::spanned_view<choreo::s32, 1> & expert_counts) {
  __choreo_check_cuda_environment__();
  auto &M = gating_output.shape()[0];
  dim3 __fused_moe_route_gdims0(M, 1, 1);
  dim3 __fused_moe_route_bdims0(32, 1, 1);
  __choreo_device_fused_moe_route<<<__fused_moe_route_gdims0, __fused_moe_route_bdims0>>>(gating_output.data(), topk_ids.data(), topk_weights.data(), expert_counts.data(), M);
}




// fused_moe_count_experts — unfused expert counting (baseline building block).
// Still present for verification and E2E compatibility. The optimized serving path
// uses fused_moe_count_and_build instead (single kernel with build_layout).

__global__ void __choreo_device_fused_moe_count_experts(int * __restrict__ topk_ids, int * __restrict__ expert_counts, unsigned M) {
  auto __choreo_device_fused_moe_count_experts__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:428.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  auto __choreo_vtid_x = threadIdx.x;
  int expert = *((int*)topk_ids + (blockIdx.x * 8) + __choreo_vtid_x);
  ATOMIC_ADD(&*((int*)expert_counts + expert), 1);
  } // end parallel-by
}

void fused_moe_count_experts(const choreo::spanned_view<choreo::s32, 2> & topk_ids, const choreo::spanned_view<choreo::s32, 1> & expert_counts) {
  __choreo_check_cuda_environment__();
  auto &M = topk_ids.shape()[0];
  dim3 __fused_moe_count_experts_gdims0(M, 1, 1);
  dim3 __fused_moe_count_experts_bdims0(8, 1, 1);
  __choreo_device_fused_moe_count_experts<<<__fused_moe_count_experts_gdims0, __fused_moe_count_experts_bdims0>>>(topk_ids.data(), expert_counts.data(), M);
  choreo::abend_true(cudaDeviceSynchronize());
}




// fused_moe_build_layout — unfused prefix-sum / offset write (baseline).
// Retained for E2E path (launch_end_to_end uses route + this after memset counts).
// Serving path uses fused_moe_count_and_build instead.

__global__ void __choreo_device_fused_moe_build_layout(int * __restrict__ expert_counts, int * __restrict__ expert_offsets, int * __restrict__ expert_write_offsets) {
  auto __choreo_device_fused_moe_build_layout__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:443.18
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ alignas(128) unsigned char anon_10[2176];
  auto __choreo_vtid_x = threadIdx.x;
  int* s_counts = (int*)(anon_10 + 1152);
  int* s_offsets = (int*)(anon_10 + 0);
  *((int*)s_counts + __choreo_vtid_x) = *((int*)expert_counts + __choreo_vtid_x);
  __syncthreads();
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:450.5
  if ((__choreo_vtid_x == 0)) {
    int prefix = 0;
    *((int*)s_offsets) = 0;
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:453.7
    {
      int __iv_expert__elem__0 = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:453.7
      for (__iv_expert__elem__0 = 0; __iv_expert__elem__0 < 256; ++__iv_expert__elem__0) {
        int count = *((int*)s_counts + __iv_expert__elem__0);
        *((int*)s_offsets + (__iv_expert__elem__0 + 1)) = (prefix + count);
        prefix = (prefix + count);
      } // expert__elem__0
      __iv_expert__elem__0 = 0;
    }
  }
  __syncthreads(); // end inthreads
  __syncthreads();
  *((int*)expert_write_offsets + __choreo_vtid_x) = *((int*)s_offsets + __choreo_vtid_x);
  *((int*)expert_offsets + __choreo_vtid_x) = *((int*)s_offsets + __choreo_vtid_x);
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:463.5
  if ((__choreo_vtid_x == 0)) {
    *((int*)expert_offsets + 256) = *((int*)s_offsets + 256);
  }
  __syncthreads(); // end inthreads
  } // end parallel-by
}

void fused_moe_build_layout(const choreo::spanned_view<choreo::s32, 1> & expert_counts, const choreo::spanned_view<choreo::s32, 1> & expert_offsets, const choreo::spanned_view<choreo::s32, 1> & expert_write_offsets) {
  __choreo_check_cuda_environment__();
  dim3 __fused_moe_build_layout_gdims0(1, 1, 1);
  dim3 __fused_moe_build_layout_bdims0(256, 1, 1);
  __choreo_device_fused_moe_build_layout<<<__fused_moe_build_layout_gdims0, __fused_moe_build_layout_bdims0>>>(expert_counts.data(), expert_offsets.data(), expert_write_offsets.data());
}




// fused_moe_count_and_build — NEW (iter018): fuses count_experts + build_layout.
// Algorithm: 256 threads (one per expert), single block. All threads atomically add
// per-expert counts into SMEM (grid-strided foreach + ATOMIC_ADD in native Choreo);
// thread 0 runs a serial prefix sum in SMEM; all threads write expert_write_offsets
// and expert_offsets to GMEM. Eliminates one kernel launch + memset/sync vs unfused.

// Pipelined topk_ids loads (iter068, +0.65%): batch __ldg before atomicAdds.
// Parallel Hillis-Steele prefix scan (iter067, +0.34%): O(log N) vs serial O(N).
__global__ void __choreo_device_fused_moe_count_and_build(int * __restrict__ topk_ids, int * __restrict__ expert_offsets, int * __restrict__ expert_write_offsets, unsigned M) {
  auto __choreo_device_fused_moe_count_and_build__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:480.18
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ alignas(128) unsigned char anon_11[2176];
  auto __choreo_vtid_x = threadIdx.x;
  int* s_counts = (int*)(anon_11 + 1152);
  int* s_offsets = (int*)(anon_11 + 0);
  *((int*)s_counts + __choreo_vtid_x) = 0;
  __syncthreads();
  {
    int total = M * 8;
    int stride = 256;
    for (int base = __choreo_vtid_x; base < total; base += stride) {
      int expert = __ldg(&topk_ids[base]);
      atomicAdd(&s_counts[expert], 1);
    }
  }
  __syncthreads();
  s_offsets[__choreo_vtid_x] = s_counts[__choreo_vtid_x];
  __syncthreads();
  for (int stride = 1; stride < 256; stride *= 2) {
    int val = (__choreo_vtid_x >= stride) ? s_offsets[__choreo_vtid_x - stride] : 0;
    __syncthreads();
    s_offsets[__choreo_vtid_x] += val;
    __syncthreads();
  }
  {
    int inclusive = s_offsets[__choreo_vtid_x];
    __syncthreads();
    s_offsets[__choreo_vtid_x + 1] = inclusive;
    if (__choreo_vtid_x == 0) s_offsets[0] = 0;
    __syncthreads();
  }
  *((int*)expert_write_offsets + __choreo_vtid_x) = *((int*)s_offsets + __choreo_vtid_x);
  *((int*)expert_offsets + __choreo_vtid_x) = *((int*)s_offsets + __choreo_vtid_x);
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:517.5
  if ((__choreo_vtid_x == 0)) {
    *((int*)expert_offsets + 256) = *((int*)s_offsets + 256);
  }
  __syncthreads(); // end inthreads
  } // end parallel-by
}

void fused_moe_count_and_build(const choreo::spanned_view<choreo::s32, 2> & topk_ids, const choreo::spanned_view<choreo::s32, 1> & expert_offsets, const choreo::spanned_view<choreo::s32, 1> & expert_write_offsets) {
  __choreo_check_cuda_environment__();
  auto &M = topk_ids.shape()[0];
  dim3 __fused_moe_count_and_build_gdims0(1, 1, 1);
  dim3 __fused_moe_count_and_build_bdims0(256, 1, 1);
  __choreo_device_fused_moe_count_and_build<<<__fused_moe_count_and_build_gdims0, __fused_moe_count_and_build_bdims0>>>(topk_ids.data(), expert_offsets.data(), expert_write_offsets.data(), M);
}




// fused_moe_quantize_input — separate per-token quantization (baseline / compatibility).
// Unused on the optimized serving path (quant_sort_gather subsumes it).

__global__ void __choreo_device_fused_moe_quantize_input(bf16 * __restrict__ input, f8_e4m3 * __restrict__ input_q, float * __restrict__ input_scales, unsigned K, unsigned M) {
  auto __choreo_device_fused_moe_quantize_input__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:529.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ alignas(128) unsigned char anon_12[128];
  float* warp_max = (float*)(anon_12 + 0);
  auto __choreo_vtid_x = threadIdx.x;
  int warp_id = __choreo_vtid_x / 32;
  int lane_id = __choreo_vtid_x % 32;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:534.7
  {
    int __iv_block_k__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:534.7
    for (__iv_block_k__elem__0 = 0; __iv_block_k__elem__0 < 16; ++__iv_block_k__elem__0) {
      int kk = (__choreo_vtid_x + __iv_block_k__elem__0 * 128);
      float value = 0.000000f;
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:537.9
      if ((kk < K)) {
        value = static_cast<float>(*((bf16*)input + (K * blockIdx.x) + kk));
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:537.9
      float local_max = ABS_F32(value);
      int mask = 16;
      // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:541.9
      {
        int __iv_idx__elem__0 = 0;
        // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:541.9
        for (__iv_idx__elem__0 = 0; __iv_idx__elem__0 < 5; ++__iv_idx__elem__0) {
          float other = SHFL_XOR(local_max, mask, 32);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:543.11
          if ((other > local_max)) {
            local_max = other;
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:543.11
          mask = (mask >> 1);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:545.11
          if ((mask == 0)) {
            break;
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:545.11
        } // idx__elem__0
        __iv_idx__elem__0 = 0;
      }
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:548.9
      if ((__choreo_vtid_x % 32 == 0)) {
        *((float*)warp_max + warp_id) = local_max;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:548.9
      __syncthreads();
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:551.9
      if ((__choreo_vtid_x == 0)) {
        float block_max = *((float*)warp_max);
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:553.11
        {
          int __iv_w__elem__0 = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:553.11
          for (__iv_w__elem__0 = 0; __iv_w__elem__0 < 3; ++__iv_w__elem__0) {
            float other = *((float*)warp_max + (__iv_w__elem__0 + 1));
            // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:555.13
            if ((other > block_max)) {
              block_max = other;
            } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:555.13
          } // w__elem__0
          __iv_w__elem__0 = 0;
        }
        *((float*)input_scales + (blockIdx.x * 16) + __iv_block_k__elem__0) = 0.000001f;
        // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:558.11
        if (block_max > 0.000001f) {
          *((float*)input_scales + (blockIdx.x * 16) + __iv_block_k__elem__0) = block_max / 448.000000f;
        } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:558.11
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:551.9
      __syncthreads();
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:564.9
      if ((kk < K)) {
        float inv_scale = 1.000000f / *((float*)input_scales + (blockIdx.x * 16) + __iv_block_k__elem__0);
        *((f8_e4m3*)input_q + (K * blockIdx.x) + kk) = choreo::utils::from_f32<f8_e4m3>(value * inv_scale);
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:564.9
      __syncthreads();
    } // block_k__elem__0
    __iv_block_k__elem__0 = 0;
  }
  } // end parallel-by
}

void fused_moe_quantize_input(const choreo::spanned_view<choreo::bf16, 2> & input, const choreo::spanned_view<choreo::f8_e4m3, 2> & input_q, const choreo::spanned_view<choreo::f32, 2> & input_scales) {
  __choreo_check_cuda_environment__();
  auto &K = input.shape()[1];
  auto &M = input.shape()[0];
  dim3 __fused_moe_quantize_input_gdims0(M, 1, 1);
  dim3 __fused_moe_quantize_input_bdims0(128, 1, 1);
  __choreo_device_fused_moe_quantize_input<<<__fused_moe_quantize_input_gdims0, __fused_moe_quantize_input_bdims0>>>(input.data(), input_q.data(), input_scales.data(), K, M);
  choreo::abend_true(cudaDeviceSynchronize());
}




// fused_moe_sort_and_gather_quant_input — separate sort+gather (baseline).
// Unused on the optimized serving path (replaced by fused_moe_quant_sort_gather).

__global__ void __choreo_device_fused_moe_sort_and_gather_quant_input(f8_e4m3 * __restrict__ input_q, float * __restrict__ input_scales, int * __restrict__ topk_ids, int * __restrict__ expert_write_offsets, int * __restrict__ sorted_route_ids, f8_e4m3 * __restrict__ rep_a_q, float * __restrict__ rep_a_scales, unsigned K, unsigned M) {
  auto __choreo_device_fused_moe_sort_and_gather_quant_input__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:585.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ alignas(128) unsigned char anon_13[128];
  int* route_slots = (int*)(anon_13 + 0);
  auto __choreo_vtid_x = threadIdx.x;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:588.7
  {
    int __iv_selected__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:588.7
    for (__iv_selected__elem__0 = 0; __iv_selected__elem__0 < 8; ++__iv_selected__elem__0) {
      // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:589.9
      if ((__choreo_vtid_x == 0)) {
        int expert = *((int*)topk_ids + (blockIdx.x * 8) + __iv_selected__elem__0);
        int slot = ATOMIC_ADD(&*((int*)expert_write_offsets + expert), 1);
        *((int*)route_slots + __iv_selected__elem__0) = slot;
        *((int*)sorted_route_ids + slot) = blockIdx.x * 8 + __iv_selected__elem__0;
      }
      __syncthreads(); // end inthreads
      __syncthreads();
      int slot = *((int*)route_slots + __iv_selected__elem__0);
      // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:598.9
      {
        int __iv_block_k__elem__0 = 0;
        // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:598.9
        for (__iv_block_k__elem__0 = 0; __iv_block_k__elem__0 < 16; ++__iv_block_k__elem__0) {
          int kk = (__choreo_vtid_x + __iv_block_k__elem__0 * 128);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:600.11
          if ((__choreo_vtid_x < 16 && __iv_block_k__elem__0 == 0)) {
            *((float*)rep_a_scales + (__choreo_vtid_x * 32768) + slot) = *((float*)input_scales + (blockIdx.x * 16) + __choreo_vtid_x);
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:600.11
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:603.11
          if ((kk < K)) {
            *((f8_e4m3*)rep_a_q + (K * slot) + kk) = *((f8_e4m3*)input_q + (K * blockIdx.x) + kk);
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:603.11
        } // block_k__elem__0
        __iv_block_k__elem__0 = 0;
      }
      __syncthreads();
    } // selected__elem__0
    __iv_selected__elem__0 = 0;
  }
  } // end parallel-by
}

void fused_moe_sort_and_gather_quant_input(const choreo::spanned_view<choreo::f8_e4m3, 2> & input_q, const choreo::spanned_view<choreo::f32, 2> & input_scales, const choreo::spanned_view<choreo::s32, 2> & topk_ids, const choreo::spanned_view<choreo::s32, 1> & expert_write_offsets, const choreo::spanned_view<choreo::s32, 1> & sorted_route_ids, const choreo::spanned_view<choreo::f8_e4m3, 2> & rep_a_q, const choreo::spanned_view<choreo::f32, 2> & rep_a_scales) {
  __choreo_check_cuda_environment__();
  auto &K = input_q.shape()[1];
  auto &M = input_q.shape()[0];
  dim3 __fused_moe_sort_and_gather_quant_input_gdims0(M, 1, 1);
  dim3 __fused_moe_sort_and_gather_quant_input_bdims0(128, 1, 1);
  __choreo_device_fused_moe_sort_and_gather_quant_input<<<__fused_moe_sort_and_gather_quant_input_gdims0, __fused_moe_sort_and_gather_quant_input_bdims0>>>(input_q.data(), input_scales.data(), topk_ids.data(), expert_write_offsets.data(), sorted_route_ids.data(), rep_a_q.data(), rep_a_scales.data(), K, M);
  choreo::abend_true(cudaDeviceSynchronize());
}




// fused_moe_quant_sort_gather — with QSG load pipelining (iter065, +1.97%).
// Phase 1: batch ALL 16 loads via __ldg for memory-level parallelism (all 16
// outstanding memory requests in flight), then reduce. Phase 2: parallel 8-thread
// slot assignment. Phase 3: uint4 vectorized copy.

__global__ void __choreo_device_fused_moe_quant_sort_gather(bf16 * __restrict__ input, int * __restrict__ topk_ids, int * __restrict__ expert_write_offsets, int * __restrict__ sorted_route_ids, f8_e4m3 * __restrict__ rep_a_q, float * __restrict__ rep_a_scales, unsigned K, unsigned M) {
  auto __choreo_device_fused_moe_quant_sort_gather__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:625.18
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  alignas(16) unsigned char anon_15[128];
  __shared__ alignas(128) unsigned char anon_14[2560];
  float* all_warp_max = (float*)(anon_14 + 2048);
  f8_e4m3* sq = (f8_e4m3*)(anon_14 + 0);
  float* ss = (float*)(anon_14 + 2304);
  int* route_slots = (int*)(anon_14 + 2432);
  auto __choreo_vtid_x = threadIdx.x;
  int warp_id = __choreo_vtid_x / 32;
  int lane_id = __choreo_vtid_x % 32;
  float* __saved_vals = (float*)(anon_15 + 0);
      #pragma unroll
      for (int kb = 0; kb < 16; kb++) {
        int kk = kb * 128 + __choreo_vtid_x;
        ((float*)__saved_vals)[kb] = (kk < K) ? __bfloat162float(__ldg((const __nv_bfloat16*)input + blockIdx.x * K + kk)) : 0.0f;
      }
      float __warp_max[16];
      #pragma unroll
      for (int kb = 0; kb < 16; kb++) {
        float m = fmaxf(__saved_vals[kb], -__saved_vals[kb]);
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, 16, 32));
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, 8, 32));
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, 4, 32));
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, 2, 32));
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, 1, 32));
        __warp_max[kb] = m;
        if ((__choreo_vtid_x & 31) == 0)
          all_warp_max[kb * 4 + (__choreo_vtid_x >> 5)] = m;
      }
  __syncthreads();
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:660.7
  if ((__choreo_vtid_x == 0)) {
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:661.9
    {
      int __iv_block_k__elem__0 = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:661.9
      for (__iv_block_k__elem__0 = 0; __iv_block_k__elem__0 < 16; ++__iv_block_k__elem__0) {
        float bmax = *((float*)all_warp_max + __iv_block_k__elem__0 * 4);
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:663.11
        {
          int __iv_w__elem__0 = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:663.11
          for (__iv_w__elem__0 = 0; __iv_w__elem__0 < 3; ++__iv_w__elem__0) {
            float other = *((float*)all_warp_max + (__iv_block_k__elem__0 * 4 + __iv_w__elem__0 + 1));
            // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:665.13
            if ((other > bmax)) {
              bmax = other;
            } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:665.13
          } // w__elem__0
          __iv_w__elem__0 = 0;
        }
        *((float*)ss + __iv_block_k__elem__0) = 0.000001f;
        // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:668.11
        if (bmax > 0.000001f) {
          *((float*)ss + __iv_block_k__elem__0) = bmax / 448.000000f;
        } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:668.11
      } // block_k__elem__0
      __iv_block_k__elem__0 = 0;
    }
  }
  __syncthreads(); // end inthreads
  float* __inv_scales = (float*)(anon_15 + 64);
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:677.7
  {
    int __iv_block_k__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:677.7
    for (__iv_block_k__elem__0 = 0; __iv_block_k__elem__0 < 16; ++__iv_block_k__elem__0) {
      *((float*)__inv_scales + __iv_block_k__elem__0) = 1.000000f / *((float*)ss + __iv_block_k__elem__0);
    } // block_k__elem__0
    __iv_block_k__elem__0 = 0;
  }
  __syncthreads();
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:681.7
  {
    int __iv_block_k__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:681.7
    for (__iv_block_k__elem__0 = 0; __iv_block_k__elem__0 < 16; ++__iv_block_k__elem__0) {
      int kk = (__choreo_vtid_x + __iv_block_k__elem__0 * 128);
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:683.9
      if ((kk < K)) {
        float qval = *((float*)__saved_vals + __iv_block_k__elem__0) * *((float*)__inv_scales + __iv_block_k__elem__0);
        *((f8_e4m3*)sq + kk) = choreo::utils::from_f32<f8_e4m3>(qval);
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:683.9
    } // block_k__elem__0
    __iv_block_k__elem__0 = 0;
  }
  __syncthreads();
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:691.7
  if ((__choreo_vtid_x < 8)) {
    int expert = *((int*)topk_ids + (blockIdx.x * 8) + __choreo_vtid_x);
    int slot = ATOMIC_ADD(&*((int*)expert_write_offsets + expert), 1);
    *((int*)route_slots + __choreo_vtid_x) = slot;
    *((int*)sorted_route_ids + slot) = (__choreo_vtid_x + blockIdx.x * 8);
  }
  __syncthreads(); // end inthreads
  __syncthreads();
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:700.7
  {
    int __iv_selected__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:700.7
    for (__iv_selected__elem__0 = 0; __iv_selected__elem__0 < 8; ++__iv_selected__elem__0) {
      int slot = *((int*)route_slots + __iv_selected__elem__0);
    {
      const uint4* __vsrc = reinterpret_cast<const uint4*>((f8_e4m3*)sq + __choreo_vtid_x * 16);
      uint4* __vdst = reinterpret_cast<uint4*>((f8_e4m3*)rep_a_q + (K * slot) + __choreo_vtid_x * 16);
      *__vdst = *__vsrc;
    }
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:707.9
      if ((__choreo_vtid_x < 16)) {
        *((float*)rep_a_scales + (__choreo_vtid_x * 32768) + slot) = *((float*)ss + __choreo_vtid_x);
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:707.9
    } // selected__elem__0
    __iv_selected__elem__0 = 0;
  }
  } // end parallel-by
}

void fused_moe_quant_sort_gather(const choreo::spanned_view<choreo::bf16, 2> & input, const choreo::spanned_view<choreo::s32, 2> & topk_ids, const choreo::spanned_view<choreo::s32, 1> & expert_write_offsets, const choreo::spanned_view<choreo::s32, 1> & sorted_route_ids, const choreo::spanned_view<choreo::f8_e4m3, 2> & rep_a_q, const choreo::spanned_view<choreo::f32, 2> & rep_a_scales) {
  __choreo_check_cuda_environment__();
  auto &K = input.shape()[1];
  auto &M = input.shape()[0];
  dim3 __fused_moe_quant_sort_gather_gdims0(M, 1, 1);
  dim3 __fused_moe_quant_sort_gather_bdims0(128, 1, 1);
  __choreo_device_fused_moe_quant_sort_gather<<<__fused_moe_quant_sort_gather_gdims0, __fused_moe_quant_sort_gather_bdims0>>>(input.data(), topk_ids.data(), expert_write_offsets.data(), sorted_route_ids.data(), rep_a_q.data(), rep_a_scales.data(), K, M);
}




// fused_moe_grouped_wgmma_fp8 — grouped FP8 GEMM with fused scatter epilogue (iter007/014/027).
// vs baseline: takes sorted_route_ids, topk_weights, scatter_output for direct weighted output.
// K-loop (iter027): TMA for rhs (sB) before DMA for lhs (sA) to overlap transfers.
// scale_a layout (iter011): [K_BLOCKS, M] so WGMMA scale loads coalesce along experts/rows.
// Epilogue (iter014): after WGMMA, warpgroup_commit/wait then __cpp__ scatters accumulator
// fragments from registers to scatter_output with atomicAdd (no SMEM output tile, no separate
// scatter kernel, full fp32 accumulation path). Scatter uses sorted_route_ids to map rows to
// (token, selected) and multiplies by topk weight.

// Grid swap: {block_n, eid} decomposition → consecutive blocks process nearby
// N-tiles of the same expert, improving L2 cache locality for LHS reads (+2.0%).
__global__ void __choreo_device_fused_moe_grouped_wgmma_fp8(f8_e4m3 * __restrict__ lhs, float * __restrict__ scale_a, f8_e4m3 * __restrict__ rhs, float * __restrict__ scale_b, int * __restrict__ expert_offsets, int * __restrict__ sorted_route_ids, float * __restrict__ topk_weights, float * __restrict__ scatter_output, unsigned EXPERT_N, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map) {
  auto __choreo_device_fused_moe_grouped_wgmma_fp8__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:734.18
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_0_barrier, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  __shared__ alignas(1024) unsigned char anon_16[24576];
  auto __choreo_vg4id_x = threadIdx.x / 128;
  auto __choreo_vtid_x = threadIdx.x % 128;
  f8_e4m3* sA = (f8_e4m3*)(anon_16 + 16384);
  f8_e4m3* sB = (f8_e4m3*)(anon_16 + 0);
  int seg_start = *((int*)expert_offsets + blockIdx.y);
  int seg_end = *((int*)expert_offsets + (blockIdx.y + 1));
  int seg_length = (seg_end - seg_start);
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:744.5
  if ((seg_length > 0)) {
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:746.5
    {
      int __iv_iv_m__elem__0 = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:746.5
      for (__iv_iv_m__elem__0 = 0; __iv_iv_m__elem__0 < ((seg_length + 63) / 64); ++__iv_iv_m__elem__0) {
        int remaining = seg_length - __iv_iv_m__elem__0 * 64;
        int tile_rows = (remaining < 64) ? remaining : 64;
        float mc[64];
        float __frag_init_val0 = 0.000000f;
        for (int idx = 0; idx < 64; ++idx)
          mc[idx] = __frag_init_val0;
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:751.7
        {
          int __iv_iv_k__elem__0 = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:751.7
          for (__iv_iv_k__elem__0 = 0; __iv_iv_k__elem__0 < 16; ++__iv_iv_k__elem__0) {
            float mc_scale_frag[64];
            memset(mc_scale_frag, 0, sizeof(mc_scale_frag));
            auto anon_6 = blockIdx.y * ((N + 127) / 128) + blockIdx.x;
            future __choreo_anon_fut__0("", 752, 9, sB);
            __choreo_anon_fut__0.is_tma = true;
            __choreo_anon_fut__0.set_atom(&choreo_copy_atom_t_0);
            const unsigned rhs_k_offset = (__iv_iv_k__elem__0 * 128);
            const unsigned rhs_expert_n_offset = ((blockIdx.x + (N + 127) / 128 * blockIdx.y) * 128);
            if (__CHOREO_BLOCK_SINGLE__) {
              cde::cp_async_bulk_tensor_2d_global_to_shared(sB, &__choreo_tma_0_tensor_map, rhs_k_offset, rhs_expert_n_offset, ((TMAAtom*)__choreo_anon_fut__0.get_atom())->barrier());
              ((TMAAtom*)__choreo_anon_fut__0.get_atom())->token() = cuda::device::barrier_arrive_tx(((TMAAtom*)__choreo_anon_fut__0.get_atom())->barrier(), 1, 16384);
            } else {
              ((TMAAtom*)__choreo_anon_fut__0.get_atom())->token() = ((TMAAtom*)__choreo_anon_fut__0.get_atom())->barrier().arrive();
            }
            ((TMAAtom*)__choreo_anon_fut__0.get_atom())->barrier().wait(std::move(((TMAAtom*)__choreo_anon_fut__0.get_atom())->token()));
            __choreo_anon_fut__0.set_nowait();

            {
              f8_e4m3* __src_base = (f8_e4m3*)lhs + (__iv_iv_k__elem__0 * 128 + K * (__iv_iv_m__elem__0 * 64 + seg_start));
              auto __shape1_lhs = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
              auto __stride1_lhs = cute::make_stride(K, cute::Int<1>{});
              auto __layout1_lhs = cute::make_layout(__shape1_lhs, __stride1_lhs);
              auto __tensor1_lhs = cute::make_tensor(cute::make_gmem_ptr<f8_e4m3>(__src_base), __layout1_lhs);
              auto __shape2_sA = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
              auto __layout2_sA = cute::tile_to_shape(cute::SM90::GMMA::Layout_K_SW128_Atom<f8_e4m3>{}, __shape2_sA);
              auto __tensor2_sA = cute::make_tensor(cute::make_smem_ptr<f8_e4m3>((f8_e4m3*)sA + 0), __layout2_sA);
              auto tiled_copy = cute::make_tiled_copy(
                  cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, f8_e4m3>{},
                  cute::Layout<cute::Shape<cute::_16, cute::_8>, cute::Stride<cute::_8, cute::_1>>{},
                  cute::Layout<cute::Shape<cute::_1, cute::_16>>{}
              );
              auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x % 128);
              auto s = thr_copy.partition_D(__tensor2_sA);
              auto g = thr_copy.partition_S(__tensor1_lhs);
              cute::copy(tiled_copy, g, s);
              cute::cp_async_fence();
              cute::cp_async_wait<0>();
            }
            wg_barrier.sync();
            // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:759.9
            {
              int __iv_iv_warp__elem__0 = 0;
              // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:759.9
              for (__iv_iv_warp__elem__0 = 0; __iv_iv_warp__elem__0 < 4; ++__iv_iv_warp__elem__0) {
                f8_e4m3* ma_smem_ptr = (f8_e4m3*)((__iv_iv_warp__elem__0 * 32 + sA));
                uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma_smem_ptr);
                f8_e4m3* mb_smem_ptr = (f8_e4m3*)((__iv_iv_warp__elem__0 * 32 + sB));
                uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
                warpgroup_arrive();
                // Note: warpgroup_arrive() should be called once before first WGMMA
                // and warpgroup_wait() should be called once after all WGMMAs
                cute::SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<>::fma(desc_ma, desc_mb, mc_scale_frag[0], mc_scale_frag[1], mc_scale_frag[2], mc_scale_frag[3], mc_scale_frag[4], mc_scale_frag[5], mc_scale_frag[6], mc_scale_frag[7], mc_scale_frag[8], mc_scale_frag[9], mc_scale_frag[10], mc_scale_frag[11], mc_scale_frag[12], mc_scale_frag[13], mc_scale_frag[14], mc_scale_frag[15], mc_scale_frag[16], mc_scale_frag[17], mc_scale_frag[18], mc_scale_frag[19], mc_scale_frag[20], mc_scale_frag[21], mc_scale_frag[22], mc_scale_frag[23], mc_scale_frag[24], mc_scale_frag[25], mc_scale_frag[26], mc_scale_frag[27], mc_scale_frag[28], mc_scale_frag[29], mc_scale_frag[30], mc_scale_frag[31], mc_scale_frag[32], mc_scale_frag[33], mc_scale_frag[34], mc_scale_frag[35], mc_scale_frag[36], mc_scale_frag[37], mc_scale_frag[38], mc_scale_frag[39], mc_scale_frag[40], mc_scale_frag[41], mc_scale_frag[42], mc_scale_frag[43], mc_scale_frag[44], mc_scale_frag[45], mc_scale_frag[46], mc_scale_frag[47], mc_scale_frag[48], mc_scale_frag[49], mc_scale_frag[50], mc_scale_frag[51], mc_scale_frag[52], mc_scale_frag[53], mc_scale_frag[54], mc_scale_frag[55], mc_scale_frag[56], mc_scale_frag[57], mc_scale_frag[58], mc_scale_frag[59], mc_scale_frag[60], mc_scale_frag[61], mc_scale_frag[62], mc_scale_frag[63]);
              } // iv_warp__elem__0
              __iv_iv_warp__elem__0 = 0;
            }
            auto sc_a = (M * __iv_iv_k__elem__0 + (__iv_iv_m__elem__0 * 64 + seg_start) + scale_a);
            float sc_b = *((float*)scale_b + (blockIdx.y * ((N + 127) / 128) + blockIdx.x)*16 + __iv_iv_k__elem__0);
            float* mc_scale_a_ptr = (float*)(sc_a);
            float mc_scale_b_val = static_cast<float>(sc_b);
            scale_accumulator<float, float, 128>(reinterpret_cast<float*>(mc), reinterpret_cast<float*>(mc_scale_frag), mc_scale_a_ptr, 1, tile_rows, mc_scale_b_val);
          } // iv_k__elem__0
          __iv_iv_k__elem__0 = 0;
        }
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  {
    int itd = threadIdx.x & 127;
    int lane = itd & 31;
    int warp = itd >> 5;
    int row0 = warp * 16 + (lane >> 2);
    int row1 = row0 + 8;
    int base_col = blockIdx.x * 128;
    auto do_scatter_row = [&](int local_row, int frag_off) __attribute__((always_inline)) {
      if (local_row >= tile_rows) return;
      int actual_row = seg_start + __iv_iv_m__elem__0 * 64 + local_row;
      int route_id = sorted_route_ids[actual_row];
      int token = route_id / 8;
      int selected = route_id % 8;
      float weight = topk_weights[token * 8 + selected];
      int out_base = token * N + base_col;
      for (int c = 0; c < 16; c++) {
        int col0 = c * 8 + (itd & 3) * 2;
        float v0 = mc[c * 4 + frag_off] * weight;
        float v1 = mc[c * 4 + frag_off + 1] * weight;
        atomicAdd(&scatter_output[out_base + col0], v0);
        atomicAdd(&scatter_output[out_base + col0 + 1], v1);
      }
    };
    do_scatter_row(row0, 0);
    do_scatter_row(row1, 2);
  }
      } // iv_m__elem__0
      __iv_iv_m__elem__0 = 0;
    }
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:744.5
  } // end parallel-by
}

void fused_moe_grouped_wgmma_fp8(const choreo::spanned_view<choreo::f8_e4m3, 2> & lhs, const choreo::spanned_view<choreo::f32, 2> & scale_a, const choreo::spanned_view<choreo::f8_e4m3, 2> & rhs, const choreo::spanned_view<choreo::f32, 2> & scale_b, const choreo::spanned_view<choreo::s32, 1> & expert_offsets, const choreo::spanned_view<choreo::s32, 1> & sorted_route_ids, const choreo::spanned_view<choreo::f32, 2> & topk_weights, const choreo::spanned_view<choreo::f32, 2> & scatter_output) {
  __choreo_check_cuda_environment__();
  auto &EXPERT_N = rhs.shape()[0];
  auto &K = lhs.shape()[1];
  auto &M = lhs.shape()[0];
  auto &N = scatter_output.shape()[1];
  uint64_t __choreo_tma_0_shape[] = {K, EXPERT_N};
  uint64_t __choreo_tma_0_strides[] = {K};
  uint32_t __choreo_tma_0_box_shape[] = {128, 128};
  uint32_t __choreo_tma_0_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_0_tensor_map{};
  CUresult __choreo_tma_0_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_0_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
          2,
          rhs.data(),
          __choreo_tma_0_shape,
          __choreo_tma_0_strides,
          __choreo_tma_0_box_shape,
          __choreo_tma_0_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_0_tensor_map_res != CUDA_SUCCESS);
  dim3 __fused_moe_grouped_wgmma_fp8_gdims0(((N + 127) / 128), 256, 1);
  dim3 __fused_moe_grouped_wgmma_fp8_bdims0(128, 1, 1);
  __choreo_device_fused_moe_grouped_wgmma_fp8<<<__fused_moe_grouped_wgmma_fp8_gdims0, __fused_moe_grouped_wgmma_fp8_bdims0>>>(lhs.data(), scale_a.data(), rhs.data(), scale_b.data(), expert_offsets.data(), sorted_route_ids.data(), topk_weights.data(), scatter_output.data(), EXPERT_N, K, M, N, __choreo_tma_0_tensor_map);
}




// fused_moe_scatter_rows_to_output — baseline standalone scatter from rep_out.
// Retained for compatibility; optimized path fuses scatter into WGMMA (see above).

__global__ void __choreo_device_fused_moe_scatter_rows_to_output(bf16 * __restrict__ rep_out, int * __restrict__ sorted_route_ids, float * __restrict__ topk_weights, float * __restrict__ output, unsigned M, unsigned N) {
  auto __choreo_device_fused_moe_scatter_rows_to_output__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:812.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  int route_id = *((int*)sorted_route_ids + blockIdx.x);
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:814.5
  if ((route_id >= 0 && route_id < M * 8)) {
    int token = (route_id / 8);
    int selected = (route_id % 8);
    float weight = *((float*)topk_weights + (token * 8) + selected);
    auto __choreo_vtid_x = threadIdx.x;
    int col = __choreo_vtid_x;
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:820.9
    {
      int __iv_block_n__elem__0 = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:820.9
      for (__iv_block_n__elem__0 = 0; __iv_block_n__elem__0 < ((N + 127) / 128); ++__iv_block_n__elem__0) {
        int out_col = (__choreo_vtid_x + __iv_block_n__elem__0 * 128);
        // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:822.11
        if ((out_col < N)) {
          bf16 val = choreo::bf16(static_cast<float>(*((bf16*)rep_out + (N * blockIdx.x) + out_col)) * weight);
          ATOMIC_ADD(&*((float*)output + (N * token) + out_col), val);
        } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:822.11
      } // block_n__elem__0
      __iv_block_n__elem__0 = 0;
    }
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:814.5
  } // end parallel-by
}

void fused_moe_scatter_rows_to_output(const choreo::spanned_view<choreo::bf16, 2> & rep_out, const choreo::spanned_view<choreo::s32, 1> & sorted_route_ids, const choreo::spanned_view<choreo::f32, 2> & topk_weights, const choreo::spanned_view<choreo::f32, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &M = topk_weights.shape()[0];
  auto &N = rep_out.shape()[1];
  dim3 __fused_moe_scatter_rows_to_output_gdims0((M * 8), 1, 1);
  dim3 __fused_moe_scatter_rows_to_output_bdims0(128, 1, 1);
  __choreo_device_fused_moe_scatter_rows_to_output<<<__fused_moe_scatter_rows_to_output_gdims0, __fused_moe_scatter_rows_to_output_bdims0>>>(rep_out.data(), sorted_route_ids.data(), topk_weights.data(), output.data(), M, N);
  choreo::abend_true(cudaDeviceSynchronize());
}




// ============================================================================
// candle-vllm: additional __co__ kernels for the gate-up (noscat) path
// ============================================================================

__global__ void __choreo_device_fused_moe_grouped_wgmma_fp8_noscat(f8_e4m3 * __restrict__ lhs, float * __restrict__ scale_a, f8_e4m3 * __restrict__ rhs, float * __restrict__ scale_b, int * __restrict__ expert_offsets, bf16 * __restrict__ output, unsigned EXPERT_N, unsigned EXPERT_NB, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map) {
  auto __choreo_device_fused_moe_grouped_wgmma_fp8_noscat__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:842.18
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_0_barrier, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  __shared__ alignas(1024) unsigned char anon_17[24576];
  auto __choreo_vg4id_x = threadIdx.x / 128;
  auto __choreo_vtid_x = threadIdx.x % 128;
  f8_e4m3* sA = (f8_e4m3*)(anon_17 + 16384);
  f8_e4m3* sB = (f8_e4m3*)(anon_17 + 0);
  int seg_start = *((int*)expert_offsets + blockIdx.y);
  int seg_end = *((int*)expert_offsets + (blockIdx.y + 1));
  int seg_length = (seg_end - seg_start);
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:852.5
  if ((seg_length > 0)) {
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:854.5
    {
      int __iv_iv_m__elem__0 = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:854.5
      for (__iv_iv_m__elem__0 = 0; __iv_iv_m__elem__0 < ((seg_length + 63) / 64); ++__iv_iv_m__elem__0) {
        int remaining = seg_length - __iv_iv_m__elem__0 * 64;
        int tile_rows = (remaining < 64) ? remaining : 64;
        float mc[64];
        float __frag_init_val1 = 0.000000f;
        for (int idx = 0; idx < 64; ++idx)
          mc[idx] = __frag_init_val1;
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:859.7
        {
          int __iv_iv_k__elem__0 = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:859.7
          for (__iv_iv_k__elem__0 = 0; __iv_iv_k__elem__0 < 16; ++__iv_iv_k__elem__0) {
            float mc_scale_frag[64];
            memset(mc_scale_frag, 0, sizeof(mc_scale_frag));
            auto anon_7 = blockIdx.y * ((N + 127) / 128) + blockIdx.x;
            future __choreo_anon_fut__2("", 860, 9, sB);
            __choreo_anon_fut__2.is_tma = true;
            __choreo_anon_fut__2.set_atom(&choreo_copy_atom_t_0);
            const unsigned rhs_k_offset = (__iv_iv_k__elem__0 * 128);
            const unsigned rhs_expert_n_offset = ((blockIdx.x + (N + 127) / 128 * blockIdx.y) * 128);
            if (__CHOREO_BLOCK_SINGLE__) {
              cde::cp_async_bulk_tensor_2d_global_to_shared(sB, &__choreo_tma_1_tensor_map, rhs_k_offset, rhs_expert_n_offset, ((TMAAtom*)__choreo_anon_fut__2.get_atom())->barrier());
              ((TMAAtom*)__choreo_anon_fut__2.get_atom())->token() = cuda::device::barrier_arrive_tx(((TMAAtom*)__choreo_anon_fut__2.get_atom())->barrier(), 1, 16384);
            } else {
              ((TMAAtom*)__choreo_anon_fut__2.get_atom())->token() = ((TMAAtom*)__choreo_anon_fut__2.get_atom())->barrier().arrive();
            }
            ((TMAAtom*)__choreo_anon_fut__2.get_atom())->barrier().wait(std::move(((TMAAtom*)__choreo_anon_fut__2.get_atom())->token()));
            __choreo_anon_fut__2.set_nowait();

            {
              f8_e4m3* __src_base3 = (f8_e4m3*)lhs + (__iv_iv_k__elem__0 * 128 + K * (__iv_iv_m__elem__0 * 64 + seg_start));
              auto __shape3_lhs = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
              auto __stride3_lhs = cute::make_stride(K, cute::Int<1>{});
              auto __layout3_lhs = cute::make_layout(__shape3_lhs, __stride3_lhs);
              auto __tensor3_lhs = cute::make_tensor(cute::make_gmem_ptr<f8_e4m3>(__src_base3), __layout3_lhs);
              auto __shape4_sA = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
              auto __layout4_sA = cute::tile_to_shape(cute::SM90::GMMA::Layout_K_SW128_Atom<f8_e4m3>{}, __shape4_sA);
              auto __tensor4_sA = cute::make_tensor(cute::make_smem_ptr<f8_e4m3>((f8_e4m3*)sA + 0), __layout4_sA);
              auto tiled_copy3 = cute::make_tiled_copy(
                  cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, f8_e4m3>{},
                  cute::Layout<cute::Shape<cute::_16, cute::_8>, cute::Stride<cute::_8, cute::_1>>{},
                  cute::Layout<cute::Shape<cute::_1, cute::_16>>{}
              );
              auto thr_copy3 = tiled_copy3.get_thread_slice(threadIdx.x % 128);
              auto s3 = thr_copy3.partition_D(__tensor4_sA);
              auto g3 = thr_copy3.partition_S(__tensor3_lhs);
              cute::copy(tiled_copy3, g3, s3);
              cute::cp_async_fence();
              cute::cp_async_wait<0>();
            }
            wg_barrier.sync();
            // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:866.9
            {
              int __iv_iv_warp__elem__0 = 0;
              // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:866.9
              for (__iv_iv_warp__elem__0 = 0; __iv_iv_warp__elem__0 < 4; ++__iv_iv_warp__elem__0) {
                f8_e4m3* ma_smem_ptr = (f8_e4m3*)((__iv_iv_warp__elem__0 * 32 + sA));
                uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma_smem_ptr);
                f8_e4m3* mb_smem_ptr = (f8_e4m3*)((__iv_iv_warp__elem__0 * 32 + sB));
                uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
                // Note: warpgroup_arrive() should be called once before first WGMMA
                // and warpgroup_wait() should be called once after all WGMMAs
                cute::SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<>::fma(desc_ma, desc_mb, mc_scale_frag[0], mc_scale_frag[1], mc_scale_frag[2], mc_scale_frag[3], mc_scale_frag[4], mc_scale_frag[5], mc_scale_frag[6], mc_scale_frag[7], mc_scale_frag[8], mc_scale_frag[9], mc_scale_frag[10], mc_scale_frag[11], mc_scale_frag[12], mc_scale_frag[13], mc_scale_frag[14], mc_scale_frag[15], mc_scale_frag[16], mc_scale_frag[17], mc_scale_frag[18], mc_scale_frag[19], mc_scale_frag[20], mc_scale_frag[21], mc_scale_frag[22], mc_scale_frag[23], mc_scale_frag[24], mc_scale_frag[25], mc_scale_frag[26], mc_scale_frag[27], mc_scale_frag[28], mc_scale_frag[29], mc_scale_frag[30], mc_scale_frag[31], mc_scale_frag[32], mc_scale_frag[33], mc_scale_frag[34], mc_scale_frag[35], mc_scale_frag[36], mc_scale_frag[37], mc_scale_frag[38], mc_scale_frag[39], mc_scale_frag[40], mc_scale_frag[41], mc_scale_frag[42], mc_scale_frag[43], mc_scale_frag[44], mc_scale_frag[45], mc_scale_frag[46], mc_scale_frag[47], mc_scale_frag[48], mc_scale_frag[49], mc_scale_frag[50], mc_scale_frag[51], mc_scale_frag[52], mc_scale_frag[53], mc_scale_frag[54], mc_scale_frag[55], mc_scale_frag[56], mc_scale_frag[57], mc_scale_frag[58], mc_scale_frag[59], mc_scale_frag[60], mc_scale_frag[61], mc_scale_frag[62], mc_scale_frag[63]);
              } // iv_warp__elem__0
              __iv_iv_warp__elem__0 = 0;
            }
            auto sc_a = (M * __iv_iv_k__elem__0 + (__iv_iv_m__elem__0 * 64 + seg_start) + scale_a);
            float sc_b = *((float*)scale_b + (blockIdx.y * ((N + 127) / 128) + blockIdx.x)*16 + __iv_iv_k__elem__0);
            float* mc_scale_a_ptr = (float*)(sc_a);
            float mc_scale_b_val = static_cast<float>(sc_b);
            scale_accumulator<float, float, 128>(reinterpret_cast<float*>(mc), reinterpret_cast<float*>(mc_scale_frag), mc_scale_a_ptr, 1, tile_rows, mc_scale_b_val);
          } // iv_k__elem__0
          __iv_iv_k__elem__0 = 0;
        }
        // Finalize WGMMA operations
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        auto __shape5_output = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
        auto __stride5_output = cute::make_stride(N, cute::Int<1>{});
        auto __layout5_output = cute::make_layout(__shape5_output, __stride5_output);
        auto __tensor5_output = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)output + (blockIdx.x * 128 + N * (__iv_iv_m__elem__0 * 64 + seg_start))), __layout5_output);
        store_fragment_d<CUTE_WGMMA_M64K32, 128>(__tensor5_output, reinterpret_cast<float*>(mc));
      } // iv_m__elem__0
      __iv_iv_m__elem__0 = 0;
    }
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:852.5
  } // end parallel-by
}

void fused_moe_grouped_wgmma_fp8_noscat(const choreo::spanned_view<choreo::f8_e4m3, 2> & lhs, const choreo::spanned_view<choreo::f32, 2> & scale_a, const choreo::spanned_view<choreo::f8_e4m3, 2> & rhs, const choreo::spanned_view<choreo::f32, 2> & scale_b, const choreo::spanned_view<choreo::s32, 1> & expert_offsets, const choreo::spanned_view<choreo::bf16, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &EXPERT_N = rhs.shape()[0];
  auto &EXPERT_NB = scale_b.shape()[0];
  auto &K = lhs.shape()[1];
  auto &M = lhs.shape()[0];
  auto &N = output.shape()[1];
  uint64_t __choreo_tma_1_shape[] = {K, EXPERT_N};
  uint64_t __choreo_tma_1_strides[] = {K};
  uint32_t __choreo_tma_1_box_shape[] = {128, 128};
  uint32_t __choreo_tma_1_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_1_tensor_map{};
  CUresult __choreo_tma_1_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_1_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
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
  dim3 __fused_moe_grouped_wgmma_fp8_noscat_gdims0(((N + 127) / 128), 256, 1);
  dim3 __fused_moe_grouped_wgmma_fp8_noscat_bdims0(128, 1, 1);
  __choreo_device_fused_moe_grouped_wgmma_fp8_noscat<<<__fused_moe_grouped_wgmma_fp8_noscat_gdims0, __fused_moe_grouped_wgmma_fp8_noscat_bdims0>>>(lhs.data(), scale_a.data(), rhs.data(), scale_b.data(), expert_offsets.data(), output.data(), EXPERT_N, EXPERT_NB, K, M, N, __choreo_tma_1_tensor_map);
}




__global__ void __choreo_device_fused_moe_unshuffle_output(bf16 * __restrict__ rep_out, int * __restrict__ sorted_route_ids, bf16 * __restrict__ output, unsigned N, unsigned SIZE_M, unsigned TOTAL) {
  auto __choreo_device_fused_moe_unshuffle_output__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:888.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  int route_id = *((int*)sorted_route_ids + blockIdx.x);
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:890.5
  if ((route_id >= 0 && route_id < SIZE_M)) {
    auto __choreo_vtid_x = threadIdx.x;
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:892.9
    {
      int __iv_block_n__elem__0 = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:892.9
      for (__iv_block_n__elem__0 = 0; __iv_block_n__elem__0 < ((N + 127) / 128); ++__iv_block_n__elem__0) {
        int out_col = (__choreo_vtid_x + __iv_block_n__elem__0 * 128);
        // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:894.11
        if ((out_col < N)) {
          *((bf16*)output + (N * route_id) + out_col) = *((bf16*)rep_out + (N * blockIdx.x) + out_col);
        } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:894.11
      } // block_n__elem__0
      __iv_block_n__elem__0 = 0;
    }
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:890.5
  } // end parallel-by
}

void fused_moe_unshuffle_output(const choreo::spanned_view<choreo::bf16, 2> & rep_out, const choreo::spanned_view<choreo::s32, 1> & sorted_route_ids, const choreo::spanned_view<choreo::bf16, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &N = rep_out.shape()[1];
  auto &SIZE_M = output.shape()[0];
  auto &TOTAL = rep_out.shape()[0];
  dim3 __fused_moe_unshuffle_output_gdims0(TOTAL, 1, 1);
  dim3 __fused_moe_unshuffle_output_bdims0(128, 1, 1);
  __choreo_device_fused_moe_unshuffle_output<<<__fused_moe_unshuffle_output_gdims0, __fused_moe_unshuffle_output_bdims0>>>(rep_out.data(), sorted_route_ids.data(), output.data(), N, SIZE_M, TOTAL);
  choreo::abend_true(cudaDeviceSynchronize());
}




__global__ void __choreo_device_cast_f32_to_bf16_kernel(float * __restrict__ input, bf16 * __restrict__ output, unsigned N) {
  auto __choreo_device_cast_f32_to_bf16_kernel__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:906.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  auto __choreo_vtid_x = threadIdx.x;
  *((bf16*)output + __choreo_vtid_x) = choreo::bf16(*((float*)input + __choreo_vtid_x));
  } // end parallel-by
}

void cast_f32_to_bf16_kernel(const choreo::spanned_view<choreo::f32, 1> & input, const choreo::spanned_view<choreo::bf16, 1> & output) {
  __choreo_check_cuda_environment__();
  auto &N = input.shape()[0];
  dim3 __cast_f32_to_bf16_kernel_gdims0(1, 1, 1);
  dim3 __cast_f32_to_bf16_kernel_bdims0(N, 1, 1);
  __choreo_device_cast_f32_to_bf16_kernel<<<__cast_f32_to_bf16_kernel_gdims0, __cast_f32_to_bf16_kernel_bdims0>>>(input.data(), output.data(), N);
  choreo::abend_true(cudaDeviceSynchronize());
}




// ============================================================================
// K=512 kernel variants for the down projection (moe_intermediate_size=512)
// ============================================================================

__global__ void __choreo_device_fused_moe_quant_sort_gather_k512(bf16 * __restrict__ input, int * __restrict__ topk_ids, int * __restrict__ expert_write_offsets, int * __restrict__ sorted_route_ids, f8_e4m3 * __restrict__ rep_a_q, float * __restrict__ rep_a_scales, unsigned K, unsigned M) {
  auto __choreo_device_fused_moe_quant_sort_gather_k512__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:922.18
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  alignas(16) unsigned char anon_19[32];
  __shared__ alignas(128) unsigned char anon_18[896];
  float* all_warp_max = (float*)(anon_18 + 512);
  f8_e4m3* sq = (f8_e4m3*)(anon_18 + 0);
  float* ss = (float*)(anon_18 + 768);
  int* route_slots = (int*)(anon_18 + 640);
  auto __choreo_vtid_x = threadIdx.x;
  int warp_id = __choreo_vtid_x / 32;
  int lane_id = __choreo_vtid_x % 32;
  float* __saved_vals = (float*)(anon_19 + 16);
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:933.7
  {
    int __iv_block_k__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:933.7
    for (__iv_block_k__elem__0 = 0; __iv_block_k__elem__0 < 4; ++__iv_block_k__elem__0) {
      int kk = (__choreo_vtid_x + __iv_block_k__elem__0 * 128);
      float value = 0.000000f;
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:936.9
      if ((kk < K)) {
        value = static_cast<float>(*((bf16*)input + (K * blockIdx.x) + kk));
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:936.9
      *((float*)__saved_vals + __iv_block_k__elem__0) = value;
      float local_max = ABS_F32(value);
      int mask = 16;
      // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:942.9
      {
        int __iv_idx__elem__0 = 0;
        // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:942.9
        for (__iv_idx__elem__0 = 0; __iv_idx__elem__0 < 5; ++__iv_idx__elem__0) {
          float other = SHFL_XOR(local_max, mask, 32);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:944.11
          if ((other > local_max)) {
            local_max = other;
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:944.11
          mask = (mask >> 1);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:946.11
          if ((mask == 0)) {
            break;
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:946.11
        } // idx__elem__0
        __iv_idx__elem__0 = 0;
      }
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:949.9
      if ((__choreo_vtid_x % 32 == 0)) {
        *((float*)all_warp_max + (__choreo_vtid_x / 32 + __iv_block_k__elem__0 * 4)) = local_max;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:949.9
    } // block_k__elem__0
    __iv_block_k__elem__0 = 0;
  }
  __syncthreads();
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:955.7
  if ((__choreo_vtid_x == 0)) {
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:956.9
    {
      int __iv_block_k__elem__0 = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:956.9
      for (__iv_block_k__elem__0 = 0; __iv_block_k__elem__0 < 4; ++__iv_block_k__elem__0) {
        float bmax = *((float*)all_warp_max + __iv_block_k__elem__0 * 4);
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:958.11
        {
          int __iv_w__elem__0 = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:958.11
          for (__iv_w__elem__0 = 0; __iv_w__elem__0 < 3; ++__iv_w__elem__0) {
            float other = *((float*)all_warp_max + (__iv_block_k__elem__0 * 4 + __iv_w__elem__0 + 1));
            // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:960.13
            if ((other > bmax)) {
              bmax = other;
            } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:960.13
          } // w__elem__0
          __iv_w__elem__0 = 0;
        }
        *((float*)ss + __iv_block_k__elem__0) = 0.000001f;
        // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:963.11
        if (bmax > 0.000001f) {
          *((float*)ss + __iv_block_k__elem__0) = bmax / 448.000000f;
        } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:963.11
      } // block_k__elem__0
      __iv_block_k__elem__0 = 0;
    }
  }
  __syncthreads(); // end inthreads
  float* __inv_scales = (float*)(anon_19 + 0);
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:972.7
  {
    int __iv_block_k__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:972.7
    for (__iv_block_k__elem__0 = 0; __iv_block_k__elem__0 < 4; ++__iv_block_k__elem__0) {
      *((float*)__inv_scales + __iv_block_k__elem__0) = 1.000000f / *((float*)ss + __iv_block_k__elem__0);
    } // block_k__elem__0
    __iv_block_k__elem__0 = 0;
  }
  __syncthreads();
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:976.7
  {
    int __iv_block_k__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:976.7
    for (__iv_block_k__elem__0 = 0; __iv_block_k__elem__0 < 4; ++__iv_block_k__elem__0) {
      auto kk = (__choreo_vtid_x + __iv_block_k__elem__0 * 128);
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:978.9
      if ((kk < K)) {
        float qval = *((float*)__saved_vals + __iv_block_k__elem__0) * *((float*)__inv_scales + __iv_block_k__elem__0);
        *((f8_e4m3*)sq + kk) = choreo::utils::from_f32<f8_e4m3>(qval);
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:978.9
    } // block_k__elem__0
    __iv_block_k__elem__0 = 0;
  }
  __syncthreads();
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:987.7
  if ((__choreo_vtid_x < 8)) {
    int expert = *((int*)topk_ids + (blockIdx.x * 8) + __choreo_vtid_x);
    int slot = ATOMIC_ADD(&*((int*)expert_write_offsets + expert), 1);
    *((int*)route_slots + __choreo_vtid_x) = slot;
    *((int*)sorted_route_ids + slot) = (__choreo_vtid_x + blockIdx.x * 8);
  }
  __syncthreads(); // end inthreads
  __syncthreads();
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:995.7
  {
    int __iv_selected__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:995.7
    for (__iv_selected__elem__0 = 0; __iv_selected__elem__0 < 8; ++__iv_selected__elem__0) {
      int slot = *((int*)route_slots + __iv_selected__elem__0);
    {
      const uint4* __vsrc = reinterpret_cast<const uint4*>((f8_e4m3*)sq + __choreo_vtid_x * 16);
      uint4* __vdst = reinterpret_cast<uint4*>((f8_e4m3*)rep_a_q + (K * slot) + __choreo_vtid_x * 16);
      if (__choreo_vtid_x < 32) *__vdst = *__vsrc;
    }
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:1003.9
      if ((__choreo_vtid_x < 4)) {
        *((float*)rep_a_scales + (__choreo_vtid_x * 32768) + slot) = *((float*)ss + __choreo_vtid_x);
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:1003.9
    } // selected__elem__0
    __iv_selected__elem__0 = 0;
  }
  } // end parallel-by
}

void fused_moe_quant_sort_gather_k512(const choreo::spanned_view<choreo::bf16, 2> & input, const choreo::spanned_view<choreo::s32, 2> & topk_ids, const choreo::spanned_view<choreo::s32, 1> & expert_write_offsets, const choreo::spanned_view<choreo::s32, 1> & sorted_route_ids, const choreo::spanned_view<choreo::f8_e4m3, 2> & rep_a_q, const choreo::spanned_view<choreo::f32, 2> & rep_a_scales) {
  __choreo_check_cuda_environment__();
  auto &K = input.shape()[1];
  auto &M = input.shape()[0];
  dim3 __fused_moe_quant_sort_gather_k512_gdims0(M, 1, 1);
  dim3 __fused_moe_quant_sort_gather_k512_bdims0(128, 1, 1);
  __choreo_device_fused_moe_quant_sort_gather_k512<<<__fused_moe_quant_sort_gather_k512_gdims0, __fused_moe_quant_sort_gather_k512_bdims0>>>(input.data(), topk_ids.data(), expert_write_offsets.data(), sorted_route_ids.data(), rep_a_q.data(), rep_a_scales.data(), K, M);
}




__global__ void __choreo_device_fused_moe_grouped_wgmma_fp8_k512(f8_e4m3 * __restrict__ lhs, float * __restrict__ scale_a, f8_e4m3 * __restrict__ rhs, float * __restrict__ scale_b, int * __restrict__ expert_offsets, int * __restrict__ sorted_route_ids, float * __restrict__ topk_weights, float * __restrict__ scatter_output, unsigned EXPERT_N, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map) {
  auto __choreo_device_fused_moe_grouped_wgmma_fp8_k512__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:1019.18
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_0_barrier, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  __shared__ alignas(1024) unsigned char anon_20[24576];
  auto __choreo_vg4id_x = threadIdx.x / 128;
  auto __choreo_vtid_x = threadIdx.x % 128;
  f8_e4m3* sA = (f8_e4m3*)(anon_20 + 16384);
  f8_e4m3* sB = (f8_e4m3*)(anon_20 + 0);
  int seg_start = *((int*)expert_offsets + blockIdx.y);
  int seg_end = *((int*)expert_offsets + (blockIdx.y + 1));
  int seg_length = (seg_end - seg_start);
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:1030.5
  if ((seg_length > 0)) {
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:1032.5
    {
      int __iv_iv_m__elem__0 = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:1032.5
      for (__iv_iv_m__elem__0 = 0; __iv_iv_m__elem__0 < ((seg_length + 63) / 64); ++__iv_iv_m__elem__0) {
        int remaining = seg_length - __iv_iv_m__elem__0 * 64;
        int tile_rows = (remaining < 64) ? remaining : 64;
        float mc[64];
        float __frag_init_val2 = 0.000000f;
        for (int idx = 0; idx < 64; ++idx)
          mc[idx] = __frag_init_val2;
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:1037.7
        {
          int __iv_iv_k__elem__0 = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:1037.7
          for (__iv_iv_k__elem__0 = 0; __iv_iv_k__elem__0 < 4; ++__iv_iv_k__elem__0) {
            float mc_scale_frag[64];
            memset(mc_scale_frag, 0, sizeof(mc_scale_frag));
            auto anon_8 = blockIdx.y * ((N + 127) / 128) + blockIdx.x;
            future __choreo_anon_fut__4("", 1038, 9, sB);
            __choreo_anon_fut__4.is_tma = true;
            __choreo_anon_fut__4.set_atom(&choreo_copy_atom_t_0);
            const unsigned rhs_k_offset = (__iv_iv_k__elem__0 * 128);
            const unsigned rhs_expert_n_offset = ((blockIdx.x + (N + 127) / 128 * blockIdx.y) * 128);
            if (__CHOREO_BLOCK_SINGLE__) {
              cde::cp_async_bulk_tensor_2d_global_to_shared(sB, &__choreo_tma_2_tensor_map, rhs_k_offset, rhs_expert_n_offset, ((TMAAtom*)__choreo_anon_fut__4.get_atom())->barrier());
              ((TMAAtom*)__choreo_anon_fut__4.get_atom())->token() = cuda::device::barrier_arrive_tx(((TMAAtom*)__choreo_anon_fut__4.get_atom())->barrier(), 1, 16384);
            } else {
              ((TMAAtom*)__choreo_anon_fut__4.get_atom())->token() = ((TMAAtom*)__choreo_anon_fut__4.get_atom())->barrier().arrive();
            }
            ((TMAAtom*)__choreo_anon_fut__4.get_atom())->barrier().wait(std::move(((TMAAtom*)__choreo_anon_fut__4.get_atom())->token()));
            __choreo_anon_fut__4.set_nowait();

            {
              f8_e4m3* __src_base6 = (f8_e4m3*)lhs + (__iv_iv_k__elem__0 * 128 + K * (__iv_iv_m__elem__0 * 64 + seg_start));
              auto __shape6_lhs = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
              auto __stride6_lhs = cute::make_stride(K, cute::Int<1>{});
              auto __layout6_lhs = cute::make_layout(__shape6_lhs, __stride6_lhs);
              auto __tensor6_lhs = cute::make_tensor(cute::make_gmem_ptr<f8_e4m3>(__src_base6), __layout6_lhs);
              auto __shape7_sA = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
              auto __layout7_sA = cute::tile_to_shape(cute::SM90::GMMA::Layout_K_SW128_Atom<f8_e4m3>{}, __shape7_sA);
              auto __tensor7_sA = cute::make_tensor(cute::make_smem_ptr<f8_e4m3>((f8_e4m3*)sA + 0), __layout7_sA);
              auto tiled_copy6 = cute::make_tiled_copy(
                  cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, f8_e4m3>{},
                  cute::Layout<cute::Shape<cute::_16, cute::_8>, cute::Stride<cute::_8, cute::_1>>{},
                  cute::Layout<cute::Shape<cute::_1, cute::_16>>{}
              );
              auto thr_copy6 = tiled_copy6.get_thread_slice(threadIdx.x % 128);
              auto s6 = thr_copy6.partition_D(__tensor7_sA);
              auto g6 = thr_copy6.partition_S(__tensor6_lhs);
              cute::copy(tiled_copy6, g6, s6);
              cute::cp_async_fence();
              cute::cp_async_wait<0>();
            }
            wg_barrier.sync();
            // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:1045.9
            {
              int __iv_iv_warp__elem__0 = 0;
              // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:1045.9
              for (__iv_iv_warp__elem__0 = 0; __iv_iv_warp__elem__0 < 4; ++__iv_iv_warp__elem__0) {
                f8_e4m3* ma_smem_ptr = (f8_e4m3*)((__iv_iv_warp__elem__0 * 32 + sA));
                uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma_smem_ptr);
                f8_e4m3* mb_smem_ptr = (f8_e4m3*)((__iv_iv_warp__elem__0 * 32 + sB));
                uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
                warpgroup_arrive();
                // Note: warpgroup_arrive() should be called once before first WGMMA
                // and warpgroup_wait() should be called once after all WGMMAs
                cute::SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<>::fma(desc_ma, desc_mb, mc_scale_frag[0], mc_scale_frag[1], mc_scale_frag[2], mc_scale_frag[3], mc_scale_frag[4], mc_scale_frag[5], mc_scale_frag[6], mc_scale_frag[7], mc_scale_frag[8], mc_scale_frag[9], mc_scale_frag[10], mc_scale_frag[11], mc_scale_frag[12], mc_scale_frag[13], mc_scale_frag[14], mc_scale_frag[15], mc_scale_frag[16], mc_scale_frag[17], mc_scale_frag[18], mc_scale_frag[19], mc_scale_frag[20], mc_scale_frag[21], mc_scale_frag[22], mc_scale_frag[23], mc_scale_frag[24], mc_scale_frag[25], mc_scale_frag[26], mc_scale_frag[27], mc_scale_frag[28], mc_scale_frag[29], mc_scale_frag[30], mc_scale_frag[31], mc_scale_frag[32], mc_scale_frag[33], mc_scale_frag[34], mc_scale_frag[35], mc_scale_frag[36], mc_scale_frag[37], mc_scale_frag[38], mc_scale_frag[39], mc_scale_frag[40], mc_scale_frag[41], mc_scale_frag[42], mc_scale_frag[43], mc_scale_frag[44], mc_scale_frag[45], mc_scale_frag[46], mc_scale_frag[47], mc_scale_frag[48], mc_scale_frag[49], mc_scale_frag[50], mc_scale_frag[51], mc_scale_frag[52], mc_scale_frag[53], mc_scale_frag[54], mc_scale_frag[55], mc_scale_frag[56], mc_scale_frag[57], mc_scale_frag[58], mc_scale_frag[59], mc_scale_frag[60], mc_scale_frag[61], mc_scale_frag[62], mc_scale_frag[63]);
              } // iv_warp__elem__0
              __iv_iv_warp__elem__0 = 0;
            }
            auto sc_a = (M * __iv_iv_k__elem__0 + (__iv_iv_m__elem__0 * 64 + seg_start) + scale_a);
            float sc_b = *((float*)scale_b + (blockIdx.y * ((N + 127) / 128) + blockIdx.x)*4 + __iv_iv_k__elem__0);
            float* mc_scale_a_ptr = (float*)(sc_a);
            float mc_scale_b_val = static_cast<float>(sc_b);
            scale_accumulator<float, float, 128>(reinterpret_cast<float*>(mc), reinterpret_cast<float*>(mc_scale_frag), mc_scale_a_ptr, 1, tile_rows, mc_scale_b_val);
          } // iv_k__elem__0
          __iv_iv_k__elem__0 = 0;
        }
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  {
    int itd = threadIdx.x & 127;
    int lane = itd & 31;
    int warp = itd >> 5;
    int row0 = warp * 16 + (lane >> 2);
    int row1 = row0 + 8;
    int base_col = blockIdx.x * 128;
    auto do_scatter_row = [&](int local_row, int frag_off) __attribute__((always_inline)) {
      if (local_row >= tile_rows) return;
      int actual_row = seg_start + __iv_iv_m__elem__0 * 64 + local_row;
      int route_id = sorted_route_ids[actual_row];
      int token = route_id / 8;
      int selected = route_id % 8;
      float weight = topk_weights[token * 8 + selected];
      int out_base = token * N + base_col;
      for (int c = 0; c < 16; c++) {
        int col0 = c * 8 + (itd & 3) * 2;
        float v0 = mc[c * 4 + frag_off] * weight;
        float v1 = mc[c * 4 + frag_off + 1] * weight;
        atomicAdd(&scatter_output[out_base + col0], v0);
        atomicAdd(&scatter_output[out_base + col0 + 1], v1);
      }
    };
    do_scatter_row(row0, 0);
    do_scatter_row(row1, 2);
  }
      } // iv_m__elem__0
      __iv_iv_m__elem__0 = 0;
    }
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter003_full_optimized.co:1030.5
  } // end parallel-by
}

void fused_moe_grouped_wgmma_fp8_k512(const choreo::spanned_view<choreo::f8_e4m3, 2> & lhs, const choreo::spanned_view<choreo::f32, 2> & scale_a, const choreo::spanned_view<choreo::f8_e4m3, 2> & rhs, const choreo::spanned_view<choreo::f32, 2> & scale_b, const choreo::spanned_view<choreo::s32, 1> & expert_offsets, const choreo::spanned_view<choreo::s32, 1> & sorted_route_ids, const choreo::spanned_view<choreo::f32, 2> & topk_weights, const choreo::spanned_view<choreo::f32, 2> & scatter_output) {
  __choreo_check_cuda_environment__();
  auto &EXPERT_N = rhs.shape()[0];
  auto &K = lhs.shape()[1];
  auto &M = lhs.shape()[0];
  auto &N = scatter_output.shape()[1];
  uint64_t __choreo_tma_2_shape[] = {K, EXPERT_N};
  uint64_t __choreo_tma_2_strides[] = {K};
  uint32_t __choreo_tma_2_box_shape[] = {128, 128};
  uint32_t __choreo_tma_2_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_2_tensor_map{};
  CUresult __choreo_tma_2_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_2_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
          2,
          rhs.data(),
          __choreo_tma_2_shape,
          __choreo_tma_2_strides,
          __choreo_tma_2_box_shape,
          __choreo_tma_2_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_2_tensor_map_res != CUDA_SUCCESS);
  dim3 __fused_moe_grouped_wgmma_fp8_k512_gdims0(((N + 127) / 128), 256, 1);
  dim3 __fused_moe_grouped_wgmma_fp8_k512_bdims0(128, 1, 1);
  __choreo_device_fused_moe_grouped_wgmma_fp8_k512<<<__fused_moe_grouped_wgmma_fp8_k512_gdims0, __fused_moe_grouped_wgmma_fp8_k512_bdims0>>>(lhs.data(), scale_a.data(), rhs.data(), scale_b.data(), expert_offsets.data(), sorted_route_ids.data(), topk_weights.data(), scatter_output.data(), EXPERT_N, K, M, N, __choreo_tma_2_tensor_map);
}




// ============================================================================
// candle-vllm: exactly 4 extern "C" wrappers for launch_end_to_end
//   1. choreo_fused_moe_route
//   2. choreo_fused_moe_build_layout
//   3. choreo_fused_moe_quant_sort_gather
//   4. choreo_fused_moe_grouped_wgmma
// Static scratch buffers shared between wrappers.
// ============================================================================

static int32_t* s_expert_counts = nullptr;
static int32_t* s_expert_offsets = nullptr;
static int32_t* s_expert_write_offsets = nullptr;
static int32_t* s_sorted_route_ids = nullptr;
static uint8_t* s_rep_a_q = nullptr;
static float* s_rep_a_scales = nullptr;
static __nv_bfloat16* s_rep_out = nullptr;
static float* s_scatter_f32 = nullptr;
static bool s_scratch_inited = false;
static bool s_route_called = false;

__global__ void k_cast_f32_bf16(const float* __restrict__ in,
                                __nv_bfloat16* __restrict__ out, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(in[i]);
}

static bool ensure_scratch(cudaStream_t stream) {
  if (s_scratch_inited) return true;
  CUstreamCaptureStatus cap_status;
  if (cuStreamIsCapturing((CUstream)stream, &cap_status) == CUDA_SUCCESS &&
      cap_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    return false;
  }
  size_t ne = QWEN35_DEFAULT_NUM_EXPERTS;
  size_t ms = QWEN35_MAX_SORTED_ROUTES;
  size_t K = QWEN35_DEFAULT_K;
  size_t max_n = 2048;
  cudaMalloc(&s_expert_counts, ne * sizeof(int32_t));
  cudaMalloc(&s_expert_offsets, (ne + 1) * sizeof(int32_t));
  cudaMalloc(&s_expert_write_offsets, ne * sizeof(int32_t));
  cudaMalloc(&s_sorted_route_ids, ms * sizeof(int32_t));
  cudaMalloc(&s_rep_a_q, ms * K * sizeof(uint8_t));
  cudaMalloc(&s_rep_a_scales, QWEN35_K_BLOCKS * ms * sizeof(float));
  cudaMalloc(&s_rep_out, ms * max_n * sizeof(__nv_bfloat16));
  cudaMalloc(&s_scatter_f32, size_t(QWEN35_MAX_M) * max_n * sizeof(float));
  s_scratch_inited = true;
  return true;
}

// --- Wrapper 1: routing (replaces topk_softmax) ---
// Returns 1 on success, 0 if skipped (caller should fall back).
extern "C" int
choreo_fused_moe_route(const float* gating_output,
                       int32_t* topk_ids,
                       float* topk_weights,
                       int num_tokens, int num_experts, int topk,
                       cudaStream_t stream) {
  if (num_experts != QWEN35_DEFAULT_NUM_EXPERTS || topk != QWEN35_TOPK)
    return 0;
  if (!ensure_scratch(stream)) return 0;
  s_route_called = false;
  cudaStreamSynchronize(stream);
  cudaMemsetAsync(s_expert_counts, 0, num_experts * sizeof(int32_t), 0);
  fprintf(stderr, "[CALL4] choreo_fused_moe_route M=%d\n", num_tokens);
  auto g = choreo::make_spanview<choreo::f32, 2>(
      gating_output, {size_t(num_tokens), size_t(num_experts)});
  auto ti = choreo::make_spanview<choreo::s32, 2>(
      topk_ids, {size_t(num_tokens), size_t(topk)});
  auto tw = choreo::make_spanview<choreo::f32, 2>(
      topk_weights, {size_t(num_tokens), size_t(topk)});
  auto ec = choreo::make_spanview<choreo::s32, 1>(
      s_expert_counts, {size_t(num_experts)});
  fused_moe_route(g, ti, tw, ec);
  { cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      fprintf(stderr, "[ERR4] route launch: %d (%s)\n", e, cudaGetErrorString(e));
      return 0;
    }
  }
  s_route_called = true;
  return 1;
}

// --- Wrapper 2: build layout from expert_counts (fast) or topk_ids (fallback) ---
extern "C" int
choreo_fused_moe_build_layout(const int32_t* topk_ids, int num_tokens,
                              int num_experts, int topk,
                              cudaStream_t stream) {
  if (num_experts != QWEN35_DEFAULT_NUM_EXPERTS || topk != QWEN35_TOPK) return 0;
  if (!ensure_scratch(stream)) return 0;
  auto eo = choreo::make_spanview<choreo::s32, 1>(
      s_expert_offsets, {size_t(num_experts + 1)});
  auto ewo = choreo::make_spanview<choreo::s32, 1>(
      s_expert_write_offsets, {size_t(num_experts)});
  if (s_route_called) {
    fprintf(stderr, "[CALL4] choreo_fused_moe_build_layout M=%d (from counts)\n", num_tokens);
    auto ec = choreo::make_spanview<choreo::s32, 1>(
        s_expert_counts, {size_t(num_experts)});
    fused_moe_build_layout(ec, eo, ewo);
  } else {
    fprintf(stderr, "[CALL4] choreo_fused_moe_build_layout M=%d (count_and_build)\n", num_tokens);
    cudaStreamSynchronize(stream);
    size_t M = num_tokens;
    auto ti = choreo::make_spanview<choreo::s32, 2>(
        topk_ids, {M, size_t(topk)});
    fused_moe_count_and_build(ti, eo, ewo);
  }
  { cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      fprintf(stderr, "[ERR4] build_layout launch: %d (%s)\n", e, cudaGetErrorString(e));
      return 0;
    }
  }
  return 1;
}

// --- Wrapper 3: fused quantize + sort + gather (K=2048 or K=512) ---
extern "C" int
choreo_fused_moe_quant_sort_gather(const void* input,
                                   const int32_t* topk_ids,
                                   int num_tokens, int k,
                                   int topk, int num_experts,
                                   cudaStream_t stream) {
  if (num_experts != QWEN35_DEFAULT_NUM_EXPERTS || topk != QWEN35_TOPK ||
      (k != QWEN35_DEFAULT_K && k != QWEN35_K_512) || !s_scratch_inited ||
      size_t(num_tokens) * topk > QWEN35_MAX_SORTED_ROUTES) {
    return 0;
  }
  cudaStreamSynchronize(stream);
  fprintf(stderr, "[CALL4] choreo_fused_moe_quant_sort_gather M=%d K=%d\n", num_tokens, k);
  size_t M = num_tokens;
  size_t K = k;
  size_t ms = QWEN35_MAX_SORTED_ROUTES;
  auto inp = choreo::make_spanview<choreo::bf16, 2>(input, {M, K});
  auto ti = choreo::make_spanview<choreo::s32, 2>(topk_ids, {M, size_t(topk)});
  auto ewo = choreo::make_spanview<choreo::s32, 1>(
      s_expert_write_offsets, {size_t(num_experts)});
  auto sri = choreo::make_spanview<choreo::s32, 1>(s_sorted_route_ids, {ms});
  auto raq = choreo::make_spanview<choreo::f8_e4m3, 2>(s_rep_a_q, {ms, K});
  if (k == QWEN35_K_512) {
    auto ras = choreo::make_spanview<choreo::f32, 2>(
        s_rep_a_scales, {size_t(QWEN35_K_BLOCKS_512), ms});
    fused_moe_quant_sort_gather_k512(inp, ti, ewo, sri, raq, ras);
  } else {
    auto ras = choreo::make_spanview<choreo::f32, 2>(
        s_rep_a_scales, {size_t(QWEN35_K_BLOCKS), ms});
    fused_moe_quant_sort_gather(inp, ti, ewo, sri, raq, ras);
  }
  { cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      fprintf(stderr, "[ERR4] quant_sort_gather launch: %d (%s)\n", e, cudaGetErrorString(e));
      return 0;
    }
  }
  return 1;
}

// --- Wrapper 4: grouped WGMMA (noscat for gate-up, scatter for down) ---
extern "C" int
choreo_fused_moe_grouped_wgmma(const uint8_t* expert_weights,
                               const float* expert_scales,
                               const float* topk_weights_ptr,
                               int num_tokens, int n, int k,
                               int num_experts,
                               __nv_bfloat16* output,
                               cudaStream_t stream) {
  if (num_experts != QWEN35_DEFAULT_NUM_EXPERTS ||
      (k != QWEN35_DEFAULT_K && k != QWEN35_K_512) ||
      n % QWEN35_WARP_N != 0 || !s_scratch_inited ||
      num_tokens > QWEN35_MAX_M) {
    return 0;
  }

  size_t N = n;
  size_t K = k;
  size_t ne = num_experts;
  size_t nb = N / QWEN35_BLOCK_N;
  size_t ms = QWEN35_MAX_SORTED_ROUTES;
  size_t k_blocks = (k == QWEN35_K_512) ? QWEN35_K_BLOCKS_512 : QWEN35_K_BLOCKS;

  auto raq = choreo::make_spanview<choreo::f8_e4m3, 2>(s_rep_a_q, {ms, K});
  auto b = choreo::make_spanview<choreo::f8_e4m3, 2>(
      expert_weights, {ne * N, K});
  auto eo = choreo::make_spanview<choreo::s32, 1>(
      s_expert_offsets, {ne + 1});

  cudaStreamSynchronize(stream);

  if (topk_weights_ptr == nullptr) {
    if (k != QWEN35_DEFAULT_K) return 0;
    if (num_tokens > 1024) return 0;
    fprintf(stderr, "[CALL4] choreo_fused_moe_grouped_wgmma NOSCAT N=%d K=%d M=%d\n", n, k, num_tokens);
    auto ras = choreo::make_spanview<choreo::f32, 2>(
        s_rep_a_scales, {size_t(QWEN35_K_BLOCKS), ms});
    auto bs = choreo::make_spanview<choreo::f32, 2>(
        expert_scales, {ne * nb, size_t(QWEN35_K_BLOCKS)});
    auto ro = choreo::make_spanview<choreo::bf16, 2>(s_rep_out, {ms, N});
    fused_moe_grouped_wgmma_fp8_noscat(raq, ras, b, bs, eo, ro);
    { cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        fprintf(stderr, "[ERR4] noscat WGMMA launch: %d (%s)\n", e, cudaGetErrorString(e));
        return 0;
      }
    }

    size_t total = size_t(num_tokens) * QWEN35_TOPK;
    auto ro2 = choreo::make_spanview<choreo::bf16, 2>(
        s_rep_out, {total, N});
    auto sri = choreo::make_spanview<choreo::s32, 1>(
        s_sorted_route_ids, {total});
    auto out = choreo::make_spanview<choreo::bf16, 2>(output, {total, N});
    fused_moe_unshuffle_output(ro2, sri, out);
    { cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        fprintf(stderr, "[ERR4] unshuffle launch: %d (%s)\n", e, cudaGetErrorString(e));
        return 0;
      }
    }
  } else {
    fprintf(stderr, "[CALL4] choreo_fused_moe_grouped_wgmma SCATTER N=%d K=%d M=%d\n",
           n, k, num_tokens);
    cudaMemsetAsync(s_scatter_f32, 0,
                    size_t(num_tokens) * N * sizeof(float), 0);
    auto sri = choreo::make_spanview<choreo::s32, 1>(s_sorted_route_ids, {ms});
    auto tw = choreo::make_spanview<choreo::f32, 2>(
        topk_weights_ptr, {size_t(QWEN35_MAX_M), size_t(QWEN35_TOPK)});
    auto scat = choreo::make_spanview<choreo::f32, 2>(
        s_scatter_f32, {size_t(QWEN35_MAX_M), N});
    if (k == QWEN35_K_512) {
      auto ras = choreo::make_spanview<choreo::f32, 2>(
          s_rep_a_scales, {size_t(QWEN35_K_BLOCKS_512), ms});
      auto bs = choreo::make_spanview<choreo::f32, 2>(
          expert_scales, {ne * nb, size_t(QWEN35_K_BLOCKS_512)});
      fused_moe_grouped_wgmma_fp8_k512(raq, ras, b, bs, eo, sri, tw, scat);
    } else {
      auto ras = choreo::make_spanview<choreo::f32, 2>(
          s_rep_a_scales, {size_t(QWEN35_K_BLOCKS), ms});
      auto bs = choreo::make_spanview<choreo::f32, 2>(
          expert_scales, {ne * nb, size_t(QWEN35_K_BLOCKS)});
      fused_moe_grouped_wgmma_fp8(raq, ras, b, bs, eo, sri, tw, scat);
    }
    { cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        fprintf(stderr, "[ERR4] scatter WGMMA launch: %d (%s)\n", e, cudaGetErrorString(e));
        return 0;
      }
    }
    int cast_n = num_tokens * int(N);
    k_cast_f32_bf16<<<(cast_n + 255) / 256, 256, 0, (cudaStream_t)0>>>(
        s_scatter_f32, (__nv_bfloat16*)output, cast_n);
    { cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) {
        fprintf(stderr, "[ERR4] cast launch: %d (%s)\n", e, cudaGetErrorString(e));
        return 0;
      }
    }
  }
  cudaError_t sync_err = cudaStreamSynchronize(0);
  if (sync_err != cudaSuccess) {
    fprintf(stderr, "[ERR4] choreo_fused_moe_grouped_wgmma sync failed: %d (%s)\n",
            sync_err, cudaGetErrorString(sync_err));
    cudaGetLastError();
    return 0;
  }
  fprintf(stderr, "[OK4] choreo_fused_moe_grouped_wgmma done\n");
  return 1;
}

// ============================================================================
// Benchmark main (only compiled for standalone)
// ============================================================================

#if defined(RUNMAIN)

extern "C" void moe_fp8_grouped_gemm_bf16(
    const uint8_t* a, const uint8_t* b, const float* a_scales,
    const float* b_scales, const int32_t* expert_offsets, int num_experts,
    int m, int n, int k, int block_size_n, int block_size_k, int sm_version,
    const int32_t* sorted_route_ids, const float* topk_weights_ptr,
    int num_tokens, float* scatter_out, cudaStream_t stream) {
  if (sm_version != 90) return;
  if (num_experts != QWEN35_DEFAULT_NUM_EXPERTS || n != QWEN35_DEFAULT_N ||
      k != QWEN35_DEFAULT_K) return;
  int32_t total_rows_h = 0;
  choreo::abend_true(cudaMemcpy(&total_rows_h, expert_offsets + num_experts,
                                sizeof(int32_t), cudaMemcpyDeviceToHost));
  auto a_ptr = choreo::make_spanview<choreo::f8_e4m3, 2>(
      a, {size_t(total_rows_h), size_t(k)});
  auto a_scales_ptr = choreo::make_spanview<choreo::f32, 2>(
      a_scales, {size_t(QWEN35_K_BLOCKS), size_t(total_rows_h)});
  auto b_ptr = choreo::make_spanview<choreo::f8_e4m3, 2>(
      b, {size_t(num_experts) * size_t(n), size_t(k)});
  auto b_scales_ptr = choreo::make_spanview<choreo::f32, 2>(
      b_scales,
      {size_t(num_experts) * size_t(QWEN35_N_BLOCKS), size_t(QWEN35_K_BLOCKS)});
  auto expert_offsets_ptr =
      choreo::make_spanview<choreo::s32, 1>(expert_offsets, {size_t(num_experts + 1)});
  auto sorted_ids_ptr = choreo::make_spanview<choreo::s32, 1>(
      sorted_route_ids, {size_t(QWEN35_MAX_SORTED_ROUTES)});
  auto topk_w_ptr = choreo::make_spanview<choreo::f32, 2>(
      topk_weights_ptr, {size_t(QWEN35_MAX_M), size_t(QWEN35_TOPK)});
  auto scatter_ptr = choreo::make_spanview<choreo::f32, 2>(
      scatter_out, {size_t(QWEN35_MAX_M), size_t(n)});
  fused_moe_grouped_wgmma_fp8(a_ptr, a_scales_ptr, b_ptr, b_scales_ptr,
                  expert_offsets_ptr, sorted_ids_ptr, topk_w_ptr, scatter_ptr);
}

int main(int argc, char** argv) {
  // --- Argument parsing: --enable-timing, --skip-verify / --verify, --m=N; env:
  // CHOREO_ENABLE_TIMING=1, CHOREO_SKIP_VERIFY=1, and when timing:
  // CHOREO_TIMING_WARMUP, CHOREO_TIMING_REPEAT.

  bool enable_timing = false;
  bool skip_verify = false;
  int runtime_m = QWEN35_DEFAULT_M;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--enable-timing") == 0) {
      enable_timing = true;
      continue;
    }
    if (std::strcmp(argv[i], "--skip-verify") == 0) {
      skip_verify = true;
      continue;
    }
    if (std::strcmp(argv[i], "--verify") == 0) {
      skip_verify = false;
      continue;
    }
    if (std::strncmp(argv[i], "--m=", 4) == 0) {
      int value = std::atoi(argv[i] + 4);
      if (value > 0) runtime_m = value;
    }
  }

  const char* timing_env = std::getenv("CHOREO_ENABLE_TIMING");
  if (timing_env && timing_env[0] == '1' && timing_env[1] == '\0') {
    enable_timing = true;
  }

  const char* skip_verify_env = std::getenv("CHOREO_SKIP_VERIFY");
  if (skip_verify_env && skip_verify_env[0] == '1' && skip_verify_env[1] == '\0') {
    skip_verify = true;
  }

  if (runtime_m > QWEN35_MAX_M) {
    std::cerr << "runtime m exceeds QWEN35_MAX_M=" << QWEN35_MAX_M << "\n";
    return 1;
  }

  size_t M = static_cast<size_t>(runtime_m);
  size_t N = QWEN35_DEFAULT_N;
  size_t K = QWEN35_DEFAULT_K;
  size_t num_experts = QWEN35_DEFAULT_NUM_EXPERTS;
  size_t expanded_m = M * QWEN35_TOPK;

  auto input_h = choreo::make_spandata<choreo::bf16>(M, K);
  auto gating_h = choreo::make_spandata<choreo::f32>(M, num_experts);
  auto expert_weights_h = choreo::make_spandata<choreo::f8_e4m3>(num_experts, N, K);
  auto expert_scales_h =
      choreo::make_spandata<choreo::f32>(num_experts, QWEN35_N_BLOCKS, QWEN35_K_BLOCKS);
  auto topk_ids_h = choreo::make_spandata<choreo::s32>(M, QWEN35_TOPK);
  auto topk_weights_h = choreo::make_spandata<choreo::f32>(M, QWEN35_TOPK);
  auto output_h = choreo::make_spandata<choreo::f32>(M, N);

  for (size_t token = 0; token < M; ++token) {
    for (size_t kk = 0; kk < K; ++kk) {
      float pattern =
          static_cast<float>(((token + 1) * 17 + (kk + 3) * 11) % 41) - 20.0f;
      input_h[token][kk] = choreo::utils::from_f32<choreo::bf16>(0.0625f * pattern);
    }
  }

  for (size_t token = 0; token < M; ++token) {
    for (size_t expert = 0; expert < num_experts; ++expert) {
      float pattern =
          static_cast<float>(((token + 3) * 19 + (expert + 5) * 7) % 43) - 21.0f;
      gating_h[token][expert] = 0.125f * pattern;
    }
  }

  initExpertWeightsQwenFp8(expert_weights_h.data(), expert_scales_h.data());

  topk_ids_h.fill(-1);
  topk_weights_h.fill(0.0f);
  output_h.fill(0.0f);

  // --- Device memory: full E2E buffers including legacy quant buffers (input_q_d,
  // input_scales_d) and rep_out_d for unfused scatter verification; optimized serving
  // path does not write separate rep_out (scatter is fused in WGMMA).

  choreo::bf16* input_d = nullptr;
  float* gating_d = nullptr;
  choreo::f8_e4m3* expert_weights_d = nullptr;
  float* expert_scales_d = nullptr;
  int32_t* topk_ids_d = nullptr;
  float* topk_weights_d = nullptr;
  int32_t* expert_counts_d = nullptr;
  int32_t* expert_offsets_d = nullptr;
  int32_t* expert_write_offsets_d = nullptr;
  choreo::f8_e4m3* input_q_d = nullptr;
  float* input_scales_d = nullptr;
  int32_t* sorted_route_ids_d = nullptr;
  choreo::f8_e4m3* rep_a_q_d = nullptr;
  float* rep_a_scales_d = nullptr;
  choreo::bf16* rep_out_d = nullptr;
  float* output_d = nullptr;

  choreo::abend_true(cudaMalloc(&input_d, M * K * sizeof(choreo::bf16)));
  choreo::abend_true(cudaMalloc(&gating_d, M * num_experts * sizeof(float)));
  choreo::abend_true(cudaMalloc(&expert_weights_d,
                                num_experts * N * K * sizeof(choreo::f8_e4m3)));
  choreo::abend_true(cudaMalloc(&expert_scales_d,
                                num_experts * QWEN35_N_BLOCKS * QWEN35_K_BLOCKS *
                                    sizeof(float)));
  choreo::abend_true(cudaMalloc(&topk_ids_d, expanded_m * sizeof(int32_t)));
  choreo::abend_true(cudaMalloc(&topk_weights_d, QWEN35_MAX_M * QWEN35_TOPK * sizeof(float)));
  choreo::abend_true(cudaMalloc(&expert_counts_d,
                                num_experts * sizeof(int32_t)));
  choreo::abend_true(cudaMalloc(&expert_offsets_d,
                                (num_experts + 1) * sizeof(int32_t)));
  choreo::abend_true(cudaMalloc(&expert_write_offsets_d,
                                num_experts * sizeof(int32_t)));
  choreo::abend_true(cudaMalloc(&input_q_d,
                                M * K * sizeof(choreo::f8_e4m3)));
  choreo::abend_true(cudaMalloc(&input_scales_d,
                                M * QWEN35_K_BLOCKS * sizeof(float)));
  choreo::abend_true(cudaMalloc(&sorted_route_ids_d,
                                QWEN35_MAX_SORTED_ROUTES * sizeof(int32_t)));
  choreo::abend_true(cudaMalloc(&rep_a_q_d,
                                QWEN35_MAX_SORTED_ROUTES * K * sizeof(choreo::f8_e4m3)));
  choreo::abend_true(cudaMalloc(&rep_a_scales_d,
                                QWEN35_MAX_SORTED_ROUTES * QWEN35_K_BLOCKS *
                                    sizeof(float)));
  choreo::abend_true(cudaMalloc(&rep_out_d,
                                QWEN35_MAX_SORTED_ROUTES * N * sizeof(choreo::bf16)));
  choreo::abend_true(cudaMalloc(&output_d, QWEN35_MAX_M * N * sizeof(float)));

  choreo::abend_true(cudaMemcpy(input_d, input_h.data(),
                                M * K * sizeof(choreo::bf16),
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(gating_d, gating_h.data(),
                                M * num_experts * sizeof(float),
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(expert_weights_d, expert_weights_h.data(),
                                num_experts * N * K * sizeof(choreo::f8_e4m3),
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(expert_scales_d, expert_scales_h.data(),
                                num_experts * QWEN35_N_BLOCKS * QWEN35_K_BLOCKS *
                                    sizeof(float),
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());

  // --- Views: rep_a_scales is [K_BLOCKS, MAX_SORTED_ROUTES] (transposed layout, iter011);
  // topk_weights for WGMMA uses QWEN35_MAX_M rows for stable shapes under CUDA Graphs.

  auto input_d_view = choreo::make_spanview<choreo::bf16, 2>(input_d, {M, K});
  auto gating_d_view =
      choreo::make_spanview<choreo::f32, 2>(gating_d, {M, num_experts});
  auto expert_weights_d_view =
      choreo::make_spanview<choreo::f8_e4m3, 2>(expert_weights_d,
                                                {num_experts * N, K});
  auto expert_scales_d_view = choreo::make_spanview<choreo::f32, 2>(
      expert_scales_d, {num_experts * QWEN35_N_BLOCKS, QWEN35_K_BLOCKS});
  auto topk_ids_d_view =
      choreo::make_spanview<choreo::s32, 2>(topk_ids_d, {M, QWEN35_TOPK});
  auto topk_weights_d_view =
      choreo::make_spanview<choreo::f32, 2>(topk_weights_d, {M, QWEN35_TOPK});
  auto topk_weights_wgmma_view =
      choreo::make_spanview<choreo::f32, 2>(topk_weights_d, {QWEN35_MAX_M, QWEN35_TOPK});
  auto expert_counts_d_view =
      choreo::make_spanview<choreo::s32, 1>(expert_counts_d, {num_experts});
  auto expert_offsets_d_view =
      choreo::make_spanview<choreo::s32, 1>(expert_offsets_d, {num_experts + 1});
  auto expert_write_offsets_d_view = choreo::make_spanview<choreo::s32, 1>(
      expert_write_offsets_d, {num_experts});
  auto input_q_d_view =
      choreo::make_spanview<choreo::f8_e4m3, 2>(input_q_d, {M, K});
  auto input_scales_d_view =
      choreo::make_spanview<choreo::f32, 2>(input_scales_d,
                                            {M, QWEN35_K_BLOCKS});
  auto sorted_route_ids_d_view = choreo::make_spanview<choreo::s32, 1>(
      sorted_route_ids_d, {QWEN35_MAX_SORTED_ROUTES});
  auto rep_a_q_d_view = choreo::make_spanview<choreo::f8_e4m3, 2>(
      rep_a_q_d, {QWEN35_MAX_SORTED_ROUTES, K});
  auto rep_a_scales_d_view = choreo::make_spanview<choreo::f32, 2>(
      rep_a_scales_d, {QWEN35_K_BLOCKS, QWEN35_MAX_SORTED_ROUTES});
  auto rep_out_d_view = choreo::make_spanview<choreo::bf16, 2>(
      rep_out_d, {QWEN35_MAX_SORTED_ROUTES, N});
  auto output_d_view = choreo::make_spanview<choreo::f32, 2>(output_d, {QWEN35_MAX_M, N});

  // --- launch_serving_path: 3 kernels — count_and_build → quant_sort_gather →
  // grouped_wgmma_fp8; cudaMemsetAsync zeros output_d (no separate scatter kernel).

  auto launch_serving_path = [&]() {
    choreo::abend_true(cudaMemsetAsync(output_d, 0, M * N * sizeof(float), 0));

    fused_moe_count_and_build(topk_ids_d_view, expert_offsets_d_view,
                              expert_write_offsets_d_view);
    fused_moe_quant_sort_gather(input_d_view, topk_ids_d_view,
                                expert_write_offsets_d_view,
                                sorted_route_ids_d_view,
                                rep_a_q_d_view,
                                rep_a_scales_d_view);
    fused_moe_grouped_wgmma_fp8(rep_a_q_d_view, rep_a_scales_d_view,
                                expert_weights_d_view, expert_scales_d_view,
                                expert_offsets_d_view, sorted_route_ids_d_view,
                                topk_weights_wgmma_view, output_d_view);
  };

  // --- launch_end_to_end: route (writes topk + expert_counts) → build_layout (not the
  // fused count_and_build) → quant_sort_gather → grouped_wgmma_fp8. Uses memset on
  // expert_counts before route.

  auto launch_end_to_end = [&]() {
    choreo::abend_true(cudaMemsetAsync(expert_counts_d, 0, num_experts * sizeof(int32_t), 0));
    choreo::abend_true(cudaMemsetAsync(output_d, 0, M * N * sizeof(float), 0));
    fused_moe_route(gating_d_view, topk_ids_d_view, topk_weights_d_view, expert_counts_d_view);
    fused_moe_build_layout(expert_counts_d_view, expert_offsets_d_view,
                           expert_write_offsets_d_view);
    fused_moe_quant_sort_gather(input_d_view, topk_ids_d_view,
                                expert_write_offsets_d_view,
                                sorted_route_ids_d_view,
                                rep_a_q_d_view,
                                rep_a_scales_d_view);
    fused_moe_grouped_wgmma_fp8(rep_a_q_d_view, rep_a_scales_d_view,
                                expert_weights_d_view, expert_scales_d_view,
                                expert_offsets_d_view, sorted_route_ids_d_view,
                                topk_weights_wgmma_view, output_d_view);
  };

  std::cout << "Qwen3.5 FP8 fused MoE benchmark"
            << " m=" << M
            << " n=" << N
            << " k=" << K
            << " experts=" << num_experts
            << " topk=" << QWEN35_TOPK
            << " block_shape=" << QWEN35_BLOCK_N << "x" << QWEN35_BLOCK_K
            << "\n";

  if (enable_timing) {
    // L2 persistence (iter052, +1.6%): pin LHS data in L2 cache
    {
      cudaAccessPolicyWindow window = {};
      window.base_ptr = (void*)rep_a_q_d;
      window.num_bytes = QWEN35_MAX_SORTED_ROUTES * K * sizeof(choreo::f8_e4m3);
      window.hitRatio = 1.0f;
      window.hitProp = cudaAccessPropertyPersisting;
      window.missProp = cudaAccessPropertyStreaming;
      cudaStreamAttrValue attr;
      attr.accessPolicyWindow = window;
      cudaStreamSetAttribute(cudaStreamPerThread,
                             cudaStreamAttributeAccessPolicyWindow, &attr);
    }

    int warmup = 10;
    int repeat = 100;
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

    // --- CUDA Graph (iter024/025): capture E2E launches on cudaStreamPerThread
    // with cudaStreamCaptureModeThreadLocal; replay via cudaGraphLaunch to avoid per-kernel
    // CPU dispatch. Requires default stream per-thread (e.g. --default-stream per-thread).

    // Capture E2E path as CUDA Graph
    cudaGraph_t e2e_graph;
    cudaGraphExec_t e2e_graph_exec;
    cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
    launch_end_to_end();
    choreo::abend_true(cudaStreamEndCapture(cudaStreamPerThread, &e2e_graph));
    choreo::abend_true(cudaGraphInstantiate(&e2e_graph_exec, e2e_graph, nullptr, nullptr, 0));

    for (int i = 0; i < warmup; i++)
      cudaGraphLaunch(e2e_graph_exec, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);

    cudaEvent_t e2e_start, e2e_stop;
    cudaEventCreate(&e2e_start);
    cudaEventCreate(&e2e_stop);
    cudaEventRecord(e2e_start, cudaStreamPerThread);
    for (int i = 0; i < repeat; i++)
      cudaGraphLaunch(e2e_graph_exec, cudaStreamPerThread);
    cudaEventRecord(e2e_stop, cudaStreamPerThread);
    cudaEventSynchronize(e2e_stop);
    float e2e_ms_f = 0.0f;
    cudaEventElapsedTime(&e2e_ms_f, e2e_start, e2e_stop);
    double full_ms = static_cast<double>(e2e_ms_f) / repeat;
    cudaEventDestroy(e2e_start);
    cudaEventDestroy(e2e_stop);
    cudaGraphExecDestroy(e2e_graph_exec);
    cudaGraphDestroy(e2e_graph);

    // Timing: cudaEvent interval over `repeat` graph launches; TFLOPS from 2*M*topk*N*K.

    std::cout << "End-to-end avg ms: " << full_ms << "\n";
    double flops = 2.0 * double(expanded_m) * double(N) * double(K);
    double full_tflops = (flops / (full_ms / 1000.0)) / 1e12;
    std::cout << "End-to-end TFLOPS: " << full_tflops << "\n";
  } else {
    launch_end_to_end();
  }

  choreo::abend_true(cudaMemcpy(topk_ids_h.data(), topk_ids_d,
                                expanded_m * sizeof(int32_t),
                                cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaMemcpy(topk_weights_h.data(), topk_weights_d,
                                expanded_m * sizeof(float),
                                cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaMemcpy(output_h.data(), output_d,
                                M * N * sizeof(float),
                                cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaDeviceSynchronize());

  if (skip_verify) {
    std::cout << "Test Passed (verify skipped)\n";
    return 0;
  }

  std::vector<int32_t> ref_topk_ids(expanded_m);
  std::vector<float> ref_topk_weights(expanded_m);
  std::vector<float> ref_output(M * N);
  fusedMoeCpuReferenceFp8(input_h.data(), gating_h.data(), expert_weights_h.data(),
                          expert_scales_h.data(), static_cast<int>(M),
                          static_cast<int>(N), static_cast<int>(K),
                          static_cast<int>(num_experts), QWEN35_TOPK,
                          QWEN35_RENORMALIZE, QWEN35_SOFTCAP,
                          ref_topk_ids.data(), ref_topk_weights.data(),
                          ref_output.data());

  for (size_t token = 0; token < M; ++token) {
    for (size_t selected = 0; selected < QWEN35_TOPK; ++selected) {
      choreo::choreo_assert(
          topk_ids_h[token][selected] ==
              ref_topk_ids[token * QWEN35_TOPK + selected],
          "top-k indices are not equal.");
      choreo::choreo_assert(
          nearlyEqual(topk_weights_h[token][selected],
                      ref_topk_weights[token * QWEN35_TOPK + selected],
                      2e-5f, 2e-5f),
          "top-k weights are not equal.");
    }
  }

  for (size_t token = 0; token < M; ++token) {
    for (size_t out_col = 0; out_col < N; ++out_col) {
      float got = output_h[token][out_col];
      float ref = ref_output[token * N + out_col];
      if (!nearlyEqual(got, ref)) {
        std::cout << "Output mismatch at [" << token << ", " << out_col
                  << "]: got " << got << ", ref " << ref << "\n";
      }
      choreo::choreo_assert(nearlyEqual(got, ref),
                            "fused moe outputs are not equal.");
    }
  }

  choreo::abend_true(cudaFree(input_d));
  choreo::abend_true(cudaFree(gating_d));
  choreo::abend_true(cudaFree(expert_weights_d));
  choreo::abend_true(cudaFree(expert_scales_d));
  choreo::abend_true(cudaFree(topk_ids_d));
  choreo::abend_true(cudaFree(topk_weights_d));
  choreo::abend_true(cudaFree(expert_counts_d));
  choreo::abend_true(cudaFree(expert_offsets_d));
  choreo::abend_true(cudaFree(expert_write_offsets_d));
  choreo::abend_true(cudaFree(input_q_d));
  choreo::abend_true(cudaFree(input_scales_d));
  choreo::abend_true(cudaFree(sorted_route_ids_d));
  choreo::abend_true(cudaFree(rep_a_q_d));
  choreo::abend_true(cudaFree(rep_a_scales_d));
  choreo::abend_true(cudaFree(rep_out_d));
  choreo::abend_true(cudaFree(output_d));

  std::cout << "Test Passed\n";
  return 0;
}
#endif


