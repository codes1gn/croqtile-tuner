
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
// 00_baseline — Fused MoE Baseline: 7 Separate Kernels
// =============================================================================
// Tuning stage: BASELINE (iter000)
// End-to-end TFLOPS: ~8.09 (H800 PCIe)
// Compilation: ./choreo -es -t cute -arch=sm_90a 00_baseline.co -o /tmp/out.cu
//
// This is the unoptimized starting point with 7 separate kernel launches:
//   1. fused_moe_route           — softmax routing + top-8 selection
//   2. fused_moe_count_experts   — atomicAdd per-expert token counts
//   3. fused_moe_build_layout    — SERIAL prefix sum on single thread (26.6μs!)
//   4. fused_moe_quantize_input  — per-block BF16→FP8 quantization
//   5. fused_moe_sort_and_gather — assign sorted slots + copy quantized data
//   6. fused_moe_grouped_wgmma   — grouped FP8 GEMM via WGMMA (57% of runtime)
//   7. fused_moe_scatter_rows    — read GEMM output from GMEM, atomicAdd scatter
//
// Each kernel launch incurs overhead: parameter packing, TMA descriptor creation,
// and cudaDeviceSynchronize between launches. Intermediate buffers (rep_out,
// input_q, etc.) are materialized in global memory between kernels.
//
// Problem: M=128, N=512, K=2048, 256 experts, topk=8, FP8 E4M3, SM_90 (H100).
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define QWEN35_DEFAULT_M 128
#define QWEN35_MAX_M 1024
#define QWEN35_DEFAULT_N 512
#define QWEN35_DEFAULT_K 2048
#define QWEN35_DEFAULT_NUM_EXPERTS 256
#define QWEN35_TOPK 8
#define QWEN35_RENORMALIZE true
#define QWEN35_SOFTCAP 2.5f

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

#define QWEN35_MAX_SORTED_ROUTES 8192
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

__global__ void __choreo_device_fused_moe_route(float * gating_output, int * topk_ids, float * topk_weights, unsigned M) {
  auto __choreo_device_fused_moe_route__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:291.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  alignas(16) unsigned char anon_3[32];
  auto __choreo_vtid_x = threadIdx.x;
  float* probs_chunk = (float*)(anon_3 + 0);
  float thread_max = -1000000.000000f;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:295.7
  {
    int __iv_v__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:295.7
    for (__iv_v__elem__0 = 0; __iv_v__elem__0 < 8; ++__iv_v__elem__0) {
      int expert_idx = (__choreo_vtid_x * 8 + __iv_v__elem__0);
      float logit = -1000000.000000f;
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:298.9
      if ((expert_idx < 256)) {
        logit = *((float*)gating_output + (blockIdx.x * 256) + expert_idx);
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:298.9
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:301.9
      if (2.500000f != 0.000000f) {
        logit = choreo::nv_cute::numerics::tanh(logit / 2.500000f) * 2.500000f;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:301.9
      *((float*)probs_chunk + __iv_v__elem__0) = logit;
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:305.9
      if ((logit > thread_max)) {
        thread_max = logit;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:305.9
    } // v__elem__0
    __iv_v__elem__0 = 0;
  }
  int mask = 16;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:309.7
  {
    int __iv_idx__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:309.7
    for (__iv_idx__elem__0 = 0; __iv_idx__elem__0 < 10; ++__iv_idx__elem__0) {
      float other_tmax = SHFL_XOR(thread_max, mask, 32);
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:311.9
      if ((other_tmax > thread_max)) {
        thread_max = other_tmax;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:311.9
      mask = (mask >> 1);
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:313.9
      if ((mask == 0)) {
        break;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:313.9
    } // idx__elem__0
    __iv_idx__elem__0 = 0;
  }
  float row_sum = 0.000000f;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:317.7
  {
    int __iv_v__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:317.7
    for (__iv_v__elem__0 = 0; __iv_v__elem__0 < 8; ++__iv_v__elem__0) {
      float prob = choreo::nv_cute::numerics::exp(*((float*)probs_chunk + __iv_v__elem__0) - thread_max);
      *((float*)probs_chunk + __iv_v__elem__0) = prob;
      row_sum = row_sum + prob;
    } // v__elem__0
    __iv_v__elem__0 = 0;
  }
  mask = 16;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:324.7
  {
    int __iv_idx__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:324.7
    for (__iv_idx__elem__0 = 0; __iv_idx__elem__0 < 10; ++__iv_idx__elem__0) {
      float temp = SHFL_XOR(row_sum, mask, 32);
      row_sum = row_sum + temp;
      mask = (mask >> 1);
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:328.9
      if ((mask == 0)) {
        break;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:328.9
    } // idx__elem__0
    __iv_idx__elem__0 = 0;
  }
  float inv_sum = 0.000000f;
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:332.7
  if (row_sum > 0.000000f) {
    inv_sum = 1.000000f / row_sum;
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:332.7
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:333.7
  {
    int __iv_v__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:333.7
    for (__iv_v__elem__0 = 0; __iv_v__elem__0 < 8; ++__iv_v__elem__0) {
      *((float*)probs_chunk + __iv_v__elem__0) = *((float*)probs_chunk + __iv_v__elem__0) * inv_sum;
    } // v__elem__0
    __iv_v__elem__0 = 0;
  }
  float selected_sum = 0.000000f;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:337.7
  {
    int __iv_selected__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:337.7
    for (__iv_selected__elem__0 = 0; __iv_selected__elem__0 < 8; ++__iv_selected__elem__0) {
      float max_val = -1.000000f;
      int expert = 256;
      // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:340.9
      {
        int __iv_v__elem__0 = 0;
        // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:340.9
        for (__iv_v__elem__0 = 0; __iv_v__elem__0 < 8; ++__iv_v__elem__0) {
          int expert_idx = (__choreo_vtid_x * 8 + __iv_v__elem__0);
          float val = *((float*)probs_chunk + __iv_v__elem__0);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:343.11
          if ((expert_idx < 256 && (val > max_val || val == max_val && expert_idx < expert))) {
            max_val = val;
            expert = expert_idx;
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:343.11
        } // v__elem__0
        __iv_v__elem__0 = 0;
      }
      mask = 16;
      // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:351.9
      {
        int __iv_idx__elem__0 = 0;
        // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:351.9
        for (__iv_idx__elem__0 = 0; __iv_idx__elem__0 < 10; ++__iv_idx__elem__0) {
          float other_max = SHFL_XOR(max_val, mask, 32);
          int other_expert = SHFL_XOR(expert, mask, 32);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:354.11
          if ((other_max > max_val || other_max == max_val && other_expert < expert)) {
            max_val = other_max;
            expert = other_expert;
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:354.11
          mask = (mask >> 1);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:360.11
          if ((mask == 0)) {
            break;
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:360.11
        } // idx__elem__0
        __iv_idx__elem__0 = 0;
      }
      // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:363.9
      if ((__choreo_vtid_x == 0)) {
        *((int*)topk_ids + (blockIdx.x * 8) + __iv_selected__elem__0) = expert;
        *((float*)topk_weights + (blockIdx.x * 8) + __iv_selected__elem__0) = max_val;
        selected_sum = selected_sum + max_val;
      }
      __syncthreads(); // end inthreads
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:370.9
      if ((__iv_selected__elem__0 + 1 < 8)) {
        // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:371.11
        if ((expert < 256 && expert / 8 == __choreo_vtid_x)) {
          *((float*)probs_chunk + (expert % 8)) = -1.000000f;
        } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:371.11
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:370.9
    } // selected__elem__0
    __iv_selected__elem__0 = 0;
  }
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:378.7
  if (true) {
    // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:379.9
    if ((__choreo_vtid_x == 0)) {
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:380.11
      if (selected_sum > 0.000000f) {
        float inv_selected_sum = 1.000000f / selected_sum;
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:382.13
        {
          int __iv_selected__elem__0 = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:382.13
          for (__iv_selected__elem__0 = 0; __iv_selected__elem__0 < 8; ++__iv_selected__elem__0) {
            *((float*)topk_weights + (blockIdx.x * 8) + __iv_selected__elem__0) = *((float*)topk_weights + (blockIdx.x * 8) + __iv_selected__elem__0) * inv_selected_sum;
          } // selected__elem__0
          __iv_selected__elem__0 = 0;
        }
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:380.11
    }
    __syncthreads(); // end inthreads
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:378.7
  } // end parallel-by
}

void fused_moe_route(const choreo::spanned_view<choreo::f32, 2> & gating_output, const choreo::spanned_view<choreo::s32, 2> & topk_ids, const choreo::spanned_view<choreo::f32, 2> & topk_weights) {
  __choreo_check_cuda_environment__();
  auto &M = gating_output.shape()[0];
  choreo::runtime_check(gating_output.shape()[1] == 256, "shape inconsistent on the 1st parameter ('gating_output', dim: 1): expect: 256, but got " + std::to_string(gating_output.shape()[1]) + ".");
  choreo::runtime_check(topk_ids.shape()[1] == 8, "shape inconsistent on the 2nd parameter ('topk_ids', dim: 1): expect: 8, but got " + std::to_string(topk_ids.shape()[1]) + ".");
  choreo::runtime_check(topk_weights.shape()[1] == 8, "shape inconsistent on the 3rd parameter ('topk_weights', dim: 1): expect: 8, but got " + std::to_string(topk_weights.shape()[1]) + ".");
  choreo::runtime_check(gating_output.shape()[0] == topk_ids.shape()[0], "The shapes of the 1st parameter (dim: 0) and the 2nd parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(topk_ids.shape()[0] == topk_weights.shape()[0], "The shapes of the 2nd parameter (dim: 0) and the 3rd parameter (dim: 0) are inconsistent.");

  choreo::runtime_check((static_cast<long long>(M) > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:291.12");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'gating_output' should be less than ::fused_moe_route::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:299.36");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'topk_ids' should be less than ::fused_moe_route::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:364.23");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'topk_weights' should be less than ::fused_moe_route::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:365.27");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'topk_weights' should be less than ::fused_moe_route::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:383.31");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'topk_weights' should be less than ::fused_moe_route::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:384.35");
  dim3 __fused_moe_route_gdims0(M, 1, 1);
  dim3 __fused_moe_route_bdims0(32, 1, 1);
  __choreo_device_fused_moe_route<<<__fused_moe_route_gdims0, __fused_moe_route_bdims0>>>(gating_output.data(), topk_ids.data(), topk_weights.data(), M);
  choreo::abend_true(cudaDeviceSynchronize());
}




__global__ void __choreo_device_fused_moe_count_experts(int * topk_ids, int * expert_counts, unsigned M) {
  auto __choreo_device_fused_moe_count_experts__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:394.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  auto __choreo_vtid_x = threadIdx.x;
  int expert = *((int*)topk_ids + (blockIdx.x * 8) + __choreo_vtid_x);
  ATOMIC_ADD(&*((int*)expert_counts + expert), 1);
  } // end parallel-by
}

void fused_moe_count_experts(const choreo::spanned_view<choreo::s32, 2> & topk_ids, const choreo::spanned_view<choreo::s32, 1> & expert_counts) {
  __choreo_check_cuda_environment__();
  auto &M = topk_ids.shape()[0];
  choreo::runtime_check(topk_ids.shape()[1] == 8, "shape inconsistent on the 1st parameter ('topk_ids', dim: 1): expect: 8, but got " + std::to_string(topk_ids.shape()[1]) + ".");
  choreo::runtime_check(expert_counts.shape()[0] == 256, "shape inconsistent on the 2nd parameter ('expert_counts', dim: 0): expect: 256, but got " + std::to_string(expert_counts.shape()[0]) + ".");

  choreo::runtime_check((static_cast<long long>(M) > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:394.12");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'topk_ids' should be less than ::fused_moe_count_experts::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:396.28");
  dim3 __fused_moe_count_experts_gdims0(M, 1, 1);
  dim3 __fused_moe_count_experts_bdims0(8, 1, 1);
  __choreo_device_fused_moe_count_experts<<<__fused_moe_count_experts_gdims0, __fused_moe_count_experts_bdims0>>>(topk_ids.data(), expert_counts.data(), M);
  choreo::abend_true(cudaDeviceSynchronize());
}




__global__ void __choreo_device_fused_moe_build_layout(int * expert_counts, int * expert_offsets, int * expert_write_offsets) {
  auto __choreo_device_fused_moe_build_layout__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:405.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  auto __choreo_vtid_x = threadIdx.x;
  // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:406.5
  if ((__choreo_vtid_x == 0)) {
    int prefix = 0;
    *((int*)expert_offsets) = 0;
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:409.7
    {
      int __iv_expert__elem__0 = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:409.7
      for (__iv_expert__elem__0 = 0; __iv_expert__elem__0 < 256; ++__iv_expert__elem__0) {
        int count = *((int*)expert_counts + __iv_expert__elem__0);
        *((int*)expert_offsets + (__iv_expert__elem__0 + 1)) = (prefix + count);
        *((int*)expert_write_offsets + __iv_expert__elem__0) = prefix;
        prefix = (prefix + count);
      } // expert__elem__0
      __iv_expert__elem__0 = 0;
    }
  }
  __syncthreads(); // end inthreads
  } // end parallel-by
}

void fused_moe_build_layout(const choreo::spanned_view<choreo::s32, 1> & expert_counts, const choreo::spanned_view<choreo::s32, 1> & expert_offsets, const choreo::spanned_view<choreo::s32, 1> & expert_write_offsets) {
  __choreo_check_cuda_environment__();
  choreo::runtime_check(expert_counts.shape()[0] == 256, "shape inconsistent on the 1st parameter ('expert_counts', dim: 0): expect: 256, but got " + std::to_string(expert_counts.shape()[0]) + ".");
  choreo::runtime_check(expert_offsets.shape()[0] == 257, "shape inconsistent on the 2nd parameter ('expert_offsets', dim: 0): expect: 257, but got " + std::to_string(expert_offsets.shape()[0]) + ".");
  choreo::runtime_check(expert_write_offsets.shape()[0] == 256, "shape inconsistent on the 3rd parameter ('expert_write_offsets', dim: 0): expect: 256, but got " + std::to_string(expert_write_offsets.shape()[0]) + ".");

  dim3 __fused_moe_build_layout_gdims0(1, 1, 1);
  dim3 __fused_moe_build_layout_bdims0(256, 1, 1);
  __choreo_device_fused_moe_build_layout<<<__fused_moe_build_layout_gdims0, __fused_moe_build_layout_bdims0>>>(expert_counts.data(), expert_offsets.data(), expert_write_offsets.data());
  choreo::abend_true(cudaDeviceSynchronize());
}




__global__ void __choreo_device_fused_moe_quantize_input(bf16 * input, f8_e4m3 * input_q, float * input_scales, unsigned K, unsigned M) {
  auto __choreo_device_fused_moe_quantize_input__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:422.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ alignas(128) unsigned char anon_4[128];
  float* warp_max = (float*)(anon_4 + 0);
  auto __choreo_vtid_x = threadIdx.x;
  int warp_id = __choreo_vtid_x / 32;
  int lane_id = __choreo_vtid_x % 32;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:427.7
  {
    int __iv_block_k__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:427.7
    for (__iv_block_k__elem__0 = 0; __iv_block_k__elem__0 < 16; ++__iv_block_k__elem__0) {
      int kk = (__choreo_vtid_x + __iv_block_k__elem__0 * 128);
      float value = 0.000000f;
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:430.9
      if ((kk < K)) {
        value = static_cast<float>(*((bf16*)input + (K * blockIdx.x) + kk));
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:430.9
      float local_max = ABS_F32(value);
      int mask = 16;
      // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:434.9
      {
        int __iv_idx__elem__0 = 0;
        // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:434.9
        for (__iv_idx__elem__0 = 0; __iv_idx__elem__0 < 5; ++__iv_idx__elem__0) {
          float other = SHFL_XOR(local_max, mask, 32);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:436.11
          if ((other > local_max)) {
            local_max = other;
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:436.11
          mask = (mask >> 1);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:438.11
          if ((mask == 0)) {
            break;
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:438.11
        } // idx__elem__0
        __iv_idx__elem__0 = 0;
      }
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:441.9
      if ((__choreo_vtid_x % 32 == 0)) {
        *((float*)warp_max + warp_id) = local_max;
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:441.9
      __syncthreads();
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:444.9
      if ((__choreo_vtid_x == 0)) {
        float block_max = *((float*)warp_max);
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:446.11
        {
          int __iv_w__elem__0 = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:446.11
          for (__iv_w__elem__0 = 0; __iv_w__elem__0 < 3; ++__iv_w__elem__0) {
            float other = *((float*)warp_max + (__iv_w__elem__0 + 1));
            // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:448.13
            if ((other > block_max)) {
              block_max = other;
            } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:448.13
          } // w__elem__0
          __iv_w__elem__0 = 0;
        }
        *((float*)input_scales + (blockIdx.x * 16) + __iv_block_k__elem__0) = 0.000001f;
        // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:451.11
        if (block_max > 0.000001f) {
          *((float*)input_scales + (blockIdx.x * 16) + __iv_block_k__elem__0) = block_max / 448.000000f;
        } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:451.11
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:444.9
      __syncthreads();
      // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:457.9
      if ((kk < K)) {
        float inv_scale = 1.000000f / *((float*)input_scales + (blockIdx.x * 16) + __iv_block_k__elem__0);
        *((f8_e4m3*)input_q + (K * blockIdx.x) + kk) = choreo::utils::from_f32<f8_e4m3>(value * inv_scale);
      } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:457.9
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
  choreo::runtime_check(input_scales.shape()[1] == 16, "shape inconsistent on the 3rd parameter ('input_scales', dim: 1): expect: 16, but got " + std::to_string(input_scales.shape()[1]) + ".");
  choreo::runtime_check(input.shape()[1] == input_q.shape()[1], "The shapes of the 1st parameter (dim: 1) and the 2nd parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(input.shape()[0] == input_q.shape()[0], "The shapes of the 1st parameter (dim: 0) and the 2nd parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(input_q.shape()[0] == input_scales.shape()[0], "The shapes of the 2nd parameter (dim: 0) and the 3rd parameter (dim: 0) are inconsistent.");

  choreo::runtime_check((static_cast<long long>(M) > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:422.12");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'input' should be less than ::fused_moe_quantize_input::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:430.38");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'input_scales' should be less than ::fused_moe_quantize_input::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:450.27");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'input_scales' should be less than ::fused_moe_quantize_input::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:452.29");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'input_scales' should be less than ::fused_moe_quantize_input::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:458.51");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'input_q' should be less than ::fused_moe_quantize_input::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:459.22");
  dim3 __fused_moe_quantize_input_gdims0(M, 1, 1);
  dim3 __fused_moe_quantize_input_bdims0(128, 1, 1);
  __choreo_device_fused_moe_quantize_input<<<__fused_moe_quantize_input_gdims0, __fused_moe_quantize_input_bdims0>>>(input.data(), input_q.data(), input_scales.data(), K, M);
  choreo::abend_true(cudaDeviceSynchronize());
}




__global__ void __choreo_device_fused_moe_sort_and_gather_quant_input(f8_e4m3 * input_q, float * input_scales, int * topk_ids, int * expert_write_offsets, int * sorted_route_ids, f8_e4m3 * rep_a_q, float * rep_a_scales, unsigned K, unsigned M) {
  auto __choreo_device_fused_moe_sort_and_gather_quant_input__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:475.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ alignas(128) unsigned char anon_5[128];
  int* route_slots = (int*)(anon_5 + 0);
  auto __choreo_vtid_x = threadIdx.x;
  // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:478.7
  {
    int __iv_selected__elem__0 = 0;
    // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:478.7
    for (__iv_selected__elem__0 = 0; __iv_selected__elem__0 < 8; ++__iv_selected__elem__0) {
      // inthreads: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:479.9
      if ((__choreo_vtid_x == 0)) {
        int expert = *((int*)topk_ids + (blockIdx.x * 8) + __iv_selected__elem__0);
        int slot = ATOMIC_ADD(&*((int*)expert_write_offsets + expert), 1);
        *((int*)route_slots + __iv_selected__elem__0) = slot;
        *((int*)sorted_route_ids + slot) = blockIdx.x * 8 + __iv_selected__elem__0;
      }
      __syncthreads(); // end inthreads
      __syncthreads();
      int slot = *((int*)route_slots + __iv_selected__elem__0);
      // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:488.9
      {
        int __iv_block_k__elem__0 = 0;
        // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:488.9
        for (__iv_block_k__elem__0 = 0; __iv_block_k__elem__0 < 16; ++__iv_block_k__elem__0) {
          int kk = (__choreo_vtid_x + __iv_block_k__elem__0 * 128);
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:490.11
          if ((__choreo_vtid_x < 16 && __iv_block_k__elem__0 == 0)) {
            *((float*)rep_a_scales + (slot * 16) + __choreo_vtid_x) = *((float*)input_scales + (blockIdx.x * 16) + __choreo_vtid_x);
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:490.11
          // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:493.11
          if ((kk < K)) {
            *((f8_e4m3*)rep_a_q + (K * slot) + kk) = *((f8_e4m3*)input_q + (K * blockIdx.x) + kk);
          } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:493.11
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
  choreo::runtime_check(input_scales.shape()[1] == 16, "shape inconsistent on the 2nd parameter ('input_scales', dim: 1): expect: 16, but got " + std::to_string(input_scales.shape()[1]) + ".");
  choreo::runtime_check(topk_ids.shape()[1] == 8, "shape inconsistent on the 3rd parameter ('topk_ids', dim: 1): expect: 8, but got " + std::to_string(topk_ids.shape()[1]) + ".");
  choreo::runtime_check(expert_write_offsets.shape()[0] == 256, "shape inconsistent on the 4th parameter ('expert_write_offsets', dim: 0): expect: 256, but got " + std::to_string(expert_write_offsets.shape()[0]) + ".");
  choreo::runtime_check(sorted_route_ids.shape()[0] == 8192, "shape inconsistent on the 5th parameter ('sorted_route_ids', dim: 0): expect: 8192, but got " + std::to_string(sorted_route_ids.shape()[0]) + ".");
  choreo::runtime_check(rep_a_q.shape()[0] == 8192, "shape inconsistent on the 6th parameter ('rep_a_q', dim: 0): expect: 8192, but got " + std::to_string(rep_a_q.shape()[0]) + ".");
  choreo::runtime_check(rep_a_scales.shape()[0] == 8192, "shape inconsistent on the 7th parameter ('rep_a_scales', dim: 0): expect: 8192, but got " + std::to_string(rep_a_scales.shape()[0]) + ".");
  choreo::runtime_check(rep_a_scales.shape()[1] == 16, "shape inconsistent on the 7th parameter ('rep_a_scales', dim: 1): expect: 16, but got " + std::to_string(rep_a_scales.shape()[1]) + ".");
  choreo::runtime_check(input_q.shape()[1] == rep_a_q.shape()[1], "The shapes of the 1st parameter (dim: 1) and the 6th parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(input_q.shape()[0] == input_scales.shape()[0], "The shapes of the 1st parameter (dim: 0) and the 2nd parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(input_scales.shape()[0] == topk_ids.shape()[0], "The shapes of the 2nd parameter (dim: 0) and the 3rd parameter (dim: 0) are inconsistent.");

  choreo::runtime_check((static_cast<long long>(M) > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:475.12");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'topk_ids' should be less than ::fused_moe_sort_and_gather_quant_input::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:480.32");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'input_scales' should be less than ::fused_moe_sort_and_gather_quant_input::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:491.59");
  choreo::runtime_check((static_cast<long long>(M) - 1 < static_cast<long long>(M)), "The 1st index `token` of element access 'input_q' should be less than ::fused_moe_sort_and_gather_quant_input::M, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:494.47");
  dim3 __fused_moe_sort_and_gather_quant_input_gdims0(M, 1, 1);
  dim3 __fused_moe_sort_and_gather_quant_input_bdims0(128, 1, 1);
  __choreo_device_fused_moe_sort_and_gather_quant_input<<<__fused_moe_sort_and_gather_quant_input_gdims0, __fused_moe_sort_and_gather_quant_input_bdims0>>>(input_q.data(), input_scales.data(), topk_ids.data(), expert_write_offsets.data(), sorted_route_ids.data(), rep_a_q.data(), rep_a_scales.data(), K, M);
  choreo::abend_true(cudaDeviceSynchronize());
}




__global__ void __choreo_device_fused_moe_grouped_wgmma_fp8(f8_e4m3 * lhs, float * scale_a, f8_e4m3 * rhs, float * scale_b, int * expert_offsets, bf16 * output, unsigned EXPERT_N, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map) {
  auto __choreo_device_fused_moe_grouped_wgmma_fp8__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:509.18
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_0_barrier, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  __shared__ alignas(1024) unsigned char anon_6[24576];
  auto __choreo_vg4id_x = threadIdx.x / 128;
  auto __choreo_vtid_x = threadIdx.x % 128;
  f8_e4m3* sA = (f8_e4m3*)(anon_6 + 16384);
  f8_e4m3* sB = (f8_e4m3*)(anon_6 + 0);
  int seg_start = *((int*)expert_offsets + blockIdx.x);
  int seg_end = *((int*)expert_offsets + (blockIdx.x + 1));
  int seg_length = (seg_end - seg_start);
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:519.5
  if ((seg_length > 0)) {
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:520.5
    {
      int __iv_iv_m__elem__0 = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:520.5
      for (__iv_iv_m__elem__0 = 0; __iv_iv_m__elem__0 < ((seg_length + 63) / 64); ++__iv_iv_m__elem__0) {
        int remaining = seg_length - __iv_iv_m__elem__0 * 64;
        int ROWS_M = (remaining < 64) ? remaining : 64;
        float mc[64];
        float __frag_init_val0 = 0.000000f;
        for (int idx = 0; idx < 64; ++idx)
          mc[idx] = __frag_init_val0;
        // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:525.7
        {
          int __iv_iv_k__elem__0 = 0;
          // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:525.7
          for (__iv_iv_k__elem__0 = 0; __iv_iv_k__elem__0 < 16; ++__iv_iv_k__elem__0) {
            float mc_scale_frag[64];
            memset(mc_scale_frag, 0, sizeof(mc_scale_frag));
            future __choreo_anon_fut__0("", 526, 9, sA);
            auto __shape1_lhs = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
            auto __stride1_lhs = cute::make_stride(K, cute::Int<1>{});
            auto __layout1_lhs = cute::make_layout(__shape1_lhs, __stride1_lhs);
            auto __tensor1_lhs = cute::make_tensor(cute::make_gmem_ptr<f8_e4m3>((f8_e4m3*)lhs + (__iv_iv_k__elem__0 * 128 + K * (__iv_iv_m__elem__0 * 64 + seg_start))), __layout1_lhs);
            auto __shape2_sA = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
            auto __layout2_sA = cute::tile_to_shape(cute::SM90::GMMA::Layout_K_SW128_Atom<f8_e4m3>{}, __shape2_sA);
            auto __tensor2_sA = cute::make_tensor(cute::make_smem_ptr<f8_e4m3>((f8_e4m3*)sA + 0), __layout2_sA);
            opt_copy(__tensor1_lhs, __tensor2_sA);
            __syncthreads();
            auto anon_2 = blockIdx.x * ((N + 127) / 128) + blockIdx.y;
            future __choreo_anon_fut__1("", 530, 9, sB);
            __choreo_anon_fut__1.is_tma = true;
            __choreo_anon_fut__1.set_atom(&choreo_copy_atom_t_0);
            if (__CHOREO_BLOCK_SINGLE__) {
              cde::cp_async_bulk_tensor_2d_global_to_shared(sB, &__choreo_tma_0_tensor_map, (__iv_iv_k__elem__0 * 128), ((blockIdx.y + (N + 127) / 128 * blockIdx.x) * 128), ((TMAAtom*)__choreo_anon_fut__1.get_atom())->barrier());
              ((TMAAtom*)__choreo_anon_fut__1.get_atom())->token() = cuda::device::barrier_arrive_tx(((TMAAtom*)__choreo_anon_fut__1.get_atom())->barrier(), 1, 16384);
            } else {
              ((TMAAtom*)__choreo_anon_fut__1.get_atom())->token() = ((TMAAtom*)__choreo_anon_fut__1.get_atom())->barrier().arrive();
            }
            ((TMAAtom*)__choreo_anon_fut__1.get_atom())->barrier().wait(std::move(((TMAAtom*)__choreo_anon_fut__1.get_atom())->token()));
            __choreo_anon_fut__1.set_nowait();

            // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:533.9
            {
              int __iv_iv_warp__elem__0 = 0;
              // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:533.9
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
            auto sc_a = (__iv_iv_k__elem__0 + (__iv_iv_m__elem__0 * 64 + seg_start) * 16 + scale_a);
            float sc_b = *((float*)scale_b + (blockIdx.x * ((N + 127) / 128) + blockIdx.y)*16 + __iv_iv_k__elem__0);
            float* mc_scale_a_ptr = (float*)(sc_a);
            float mc_scale_b_val = static_cast<float>(sc_b);
            scale_accumulator<float, float, 128>(reinterpret_cast<float*>(mc), reinterpret_cast<float*>(mc_scale_frag), mc_scale_a_ptr, 16, ROWS_M, mc_scale_b_val);
          } // iv_k__elem__0
          __iv_iv_k__elem__0 = 0;
        }
        // Finalize WGMMA operations
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        auto __shape3_output = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
        auto __stride3_output = cute::make_stride(N, cute::Int<1>{});
        auto __layout3_output = cute::make_layout(__shape3_output, __stride3_output);
        auto __tensor3_output = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)output + (blockIdx.y * 128 + N * (__iv_iv_m__elem__0 * 64 + seg_start))), __layout3_output);
        store_fragment_d<CUTE_WGMMA_M64K32, 128>(__tensor3_output, reinterpret_cast<float*>(mc));
      } // iv_m__elem__0
      __iv_iv_m__elem__0 = 0;
    }
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:519.5
  choreo::choreo_assert((static_cast<long long>(blockIdx.y) + (static_cast<long long>(N) + 127LL) / 128LL * static_cast<long long>(blockIdx.x) >= 0LL), "The 1st index ` (eid # block_n) ` of element access 'scale_b' should be greater than or equal to 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:541.27");
  choreo::choreo_assert((static_cast<long long>(blockIdx.y) + (static_cast<long long>(N) + 127LL) / 128LL * static_cast<long long>(blockIdx.x) < 1024LL), "The 1st index ` (eid # block_n) ` of element access 'scale_b' should be less than 1024, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:541.27");
  } // end parallel-by
}

void fused_moe_grouped_wgmma_fp8(const choreo::spanned_view<choreo::f8_e4m3, 2> & lhs, const choreo::spanned_view<choreo::f32, 2> & scale_a, const choreo::spanned_view<choreo::f8_e4m3, 2> & rhs, const choreo::spanned_view<choreo::f32, 2> & scale_b, const choreo::spanned_view<choreo::s32, 1> & expert_offsets, const choreo::spanned_view<choreo::bf16, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &EXPERT_N = rhs.shape()[0];
  auto &K = lhs.shape()[1];
  auto &M = lhs.shape()[0];
  auto &N = output.shape()[1];
  choreo::runtime_check(scale_a.shape()[1] == 16, "shape inconsistent on the 2nd parameter ('scale_a', dim: 1): expect: 16, but got " + std::to_string(scale_a.shape()[1]) + ".");
  choreo::runtime_check(scale_b.shape()[0] == 1024, "shape inconsistent on the 4th parameter ('scale_b', dim: 0): expect: 1024, but got " + std::to_string(scale_b.shape()[0]) + ".");
  choreo::runtime_check(scale_b.shape()[1] == 16, "shape inconsistent on the 4th parameter ('scale_b', dim: 1): expect: 16, but got " + std::to_string(scale_b.shape()[1]) + ".");
  choreo::runtime_check(expert_offsets.shape()[0] == 257, "shape inconsistent on the 5th parameter ('expert_offsets', dim: 0): expect: 257, but got " + std::to_string(expert_offsets.shape()[0]) + ".");
  choreo::runtime_check(lhs.shape()[1] == rhs.shape()[1], "The shapes of the 1st parameter (dim: 1) and the 3rd parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(lhs.shape()[0] == scale_a.shape()[0], "The shapes of the 1st parameter (dim: 0) and the 2nd parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(scale_a.shape()[0] == output.shape()[0], "The shapes of the 2nd parameter (dim: 0) and the 6th parameter (dim: 0) are inconsistent.");

  choreo::runtime_check(((static_cast<long long>(N) + 127LL) / 128LL > 0LL), "The 2nd bound item of parallelby is invalid: should be greater than 0, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:509.24");
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
  dim3 __fused_moe_grouped_wgmma_fp8_gdims0(256, ((N + 127) / 128), 1);
  dim3 __fused_moe_grouped_wgmma_fp8_bdims0(128, 1, 1);
  __choreo_device_fused_moe_grouped_wgmma_fp8<<<__fused_moe_grouped_wgmma_fp8_gdims0, __fused_moe_grouped_wgmma_fp8_bdims0>>>(lhs.data(), scale_a.data(), rhs.data(), scale_b.data(), expert_offsets.data(), output.data(), EXPERT_N, K, M, N, __choreo_tma_0_tensor_map);
}




__global__ void __choreo_device_fused_moe_scatter_rows_to_output(bf16 * rep_out, int * sorted_route_ids, float * topk_weights, float * output, unsigned M, unsigned N) {
  auto __choreo_device_fused_moe_scatter_rows_to_output__ring__ = nullptr;
  { // parallel-by: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:558.12
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  int route_id = *((int*)sorted_route_ids + blockIdx.x);
  // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:560.5
  if ((route_id >= 0 && route_id < M * 8)) {
    int token = (route_id / 8);
    int selected = (route_id % 8);
    float weight = *((float*)topk_weights + (token * 8) + selected);
    auto __choreo_vtid_x = threadIdx.x;
    int col = __choreo_vtid_x;
    // with-in: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:566.9
    {
      int __iv_block_n__elem__0 = 0;
      // foreach: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:566.9
      for (__iv_block_n__elem__0 = 0; __iv_block_n__elem__0 < ((N + 127) / 128); ++__iv_block_n__elem__0) {
        int out_col = (__choreo_vtid_x + __iv_block_n__elem__0 * 128);
        // if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:568.11
        if ((out_col < N)) {
          bf16 val = choreo::bf16(static_cast<float>(*((bf16*)rep_out + (N * blockIdx.x) + out_col)) * weight);
          ATOMIC_ADD(&*((float*)output + (N * token) + out_col), val);
        } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:568.11
      } // block_n__elem__0
      __iv_block_n__elem__0 = 0;
    }
  } // end if-else: tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:560.5
  } // end parallel-by
}

void fused_moe_scatter_rows_to_output(const choreo::spanned_view<choreo::bf16, 2> & rep_out, const choreo::spanned_view<choreo::s32, 1> & sorted_route_ids, const choreo::spanned_view<choreo::f32, 2> & topk_weights, const choreo::spanned_view<choreo::f32, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &M = topk_weights.shape()[0];
  auto &N = rep_out.shape()[1];
  choreo::runtime_check(rep_out.shape()[0] == 8192, "shape inconsistent on the 1st parameter ('rep_out', dim: 0): expect: 8192, but got " + std::to_string(rep_out.shape()[0]) + ".");
  choreo::runtime_check(sorted_route_ids.shape()[0] == 8192, "shape inconsistent on the 2nd parameter ('sorted_route_ids', dim: 0): expect: 8192, but got " + std::to_string(sorted_route_ids.shape()[0]) + ".");
  choreo::runtime_check(topk_weights.shape()[1] == 8, "shape inconsistent on the 3rd parameter ('topk_weights', dim: 1): expect: 8, but got " + std::to_string(topk_weights.shape()[1]) + ".");
  choreo::runtime_check(topk_weights.shape()[0] == output.shape()[0], "The shapes of the 3rd parameter (dim: 0) and the 4th parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(rep_out.shape()[1] == output.shape()[1], "The shapes of the 1st parameter (dim: 1) and the 4th parameter (dim: 1) are inconsistent.");

  choreo::runtime_check(((static_cast<long long>(N) + 127LL) / 128LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/M128_N512_K2048/qwen35_moe_fp8/iter001_baseline.co:566.28");
  dim3 __fused_moe_scatter_rows_to_output_gdims0(8192, 1, 1);
  dim3 __fused_moe_scatter_rows_to_output_bdims0(128, 1, 1);
  __choreo_device_fused_moe_scatter_rows_to_output<<<__fused_moe_scatter_rows_to_output_gdims0, __fused_moe_scatter_rows_to_output_bdims0>>>(rep_out.data(), sorted_route_ids.data(), topk_weights.data(), output.data(), M, N);
  choreo::abend_true(cudaDeviceSynchronize());
}




extern "C" void moe_fp8_grouped_gemm_bf16(
    const uint8_t* a, const uint8_t* b, const float* a_scales,
    const float* b_scales, const int32_t* expert_offsets, int num_experts,
    int m, int n, int k, int block_size_n, int block_size_k, int sm_version,
    __nv_bfloat16* out, cudaStream_t stream) {
  if (sm_version != 90) {
    std::printf("moe_fp8_grouped_gemm_bf16 unsupported sm_version %d\n",
                sm_version);
    return;
  }
  if (num_experts != QWEN35_DEFAULT_NUM_EXPERTS || n != QWEN35_DEFAULT_N ||
      k != QWEN35_DEFAULT_K || block_size_n != QWEN35_BLOCK_N ||
      block_size_k != QWEN35_BLOCK_K) {
    std::printf("moe_fp8_grouped_gemm_bf16 unsupported shape e=%d n=%d k=%d bn=%d bk=%d\n",
                num_experts, n, k, block_size_n, block_size_k);
    return;
  }

  int32_t total_rows_h = 0;
  choreo::abend_true(cudaMemcpy(&total_rows_h, expert_offsets + num_experts,
                                sizeof(int32_t), cudaMemcpyDeviceToHost));
  if (total_rows_h < 0 || total_rows_h > QWEN35_MAX_SORTED_ROUTES) {
    std::printf("moe_fp8_grouped_gemm_bf16 invalid routed rows %d\n", total_rows_h);
    return;
  }
  if (m != total_rows_h) {
    std::printf("moe_fp8_grouped_gemm_bf16 mismatched m=%d total_rows=%d\n",
                m, total_rows_h);
    return;
  }

  auto a_ptr = choreo::make_spanview<choreo::f8_e4m3, 2>(
      a, {size_t(total_rows_h), size_t(k)});
  auto a_scales_ptr = choreo::make_spanview<choreo::f32, 2>(
      a_scales, {size_t(total_rows_h), size_t(QWEN35_K_BLOCKS)});
  auto b_ptr = choreo::make_spanview<choreo::f8_e4m3, 2>(
      b, {size_t(num_experts) * size_t(n), size_t(k)});
  auto b_scales_ptr = choreo::make_spanview<choreo::f32, 2>(
      b_scales,
      {size_t(num_experts) * size_t(QWEN35_N_BLOCKS), size_t(QWEN35_K_BLOCKS)});
  auto expert_offsets_ptr =
      choreo::make_spanview<choreo::s32, 1>(expert_offsets, {size_t(num_experts + 1)});
  auto out_ptr = choreo::make_spanview<choreo::bf16, 2>(
      out, {size_t(total_rows_h), size_t(n)});
  fused_moe_grouped_wgmma_fp8(a_ptr, a_scales_ptr, b_ptr, b_scales_ptr,
                  expert_offsets_ptr, out_ptr);
}

int main(int argc, char** argv) {
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
  choreo::abend_true(cudaMalloc(&topk_weights_d, expanded_m * sizeof(float)));
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
  choreo::abend_true(cudaMalloc(&output_d, M * N * sizeof(float)));

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
      rep_a_scales_d, {QWEN35_MAX_SORTED_ROUTES, QWEN35_K_BLOCKS});
  auto rep_out_d_view = choreo::make_spanview<choreo::bf16, 2>(
      rep_out_d, {QWEN35_MAX_SORTED_ROUTES, N});
  auto output_d_view = choreo::make_spanview<choreo::f32, 2>(output_d, {M, N});

  auto launch_serving_path = [&]() {
    choreo::abend_true(cudaMemset(expert_counts_d, 0, num_experts * sizeof(int32_t)));
    choreo::abend_true(cudaMemset(sorted_route_ids_d, 0xff,
                                  QWEN35_MAX_SORTED_ROUTES * sizeof(int32_t)));
    choreo::abend_true(cudaMemset(output_d, 0, M * N * sizeof(float)));

    fused_moe_count_experts(topk_ids_d_view, expert_counts_d_view);
    fused_moe_build_layout(expert_counts_d_view, expert_offsets_d_view,
                           expert_write_offsets_d_view);
    fused_moe_quantize_input(input_d_view, input_q_d_view, input_scales_d_view);
    fused_moe_sort_and_gather_quant_input(input_q_d_view, input_scales_d_view,
                                          topk_ids_d_view,
                                          expert_write_offsets_d_view,
                                          sorted_route_ids_d_view,
                                          rep_a_q_d_view,
                                          rep_a_scales_d_view);
    fused_moe_grouped_wgmma_fp8(rep_a_q_d_view, rep_a_scales_d_view,
                                expert_weights_d_view, expert_scales_d_view,
                                expert_offsets_d_view, rep_out_d_view);
    fused_moe_scatter_rows_to_output(rep_out_d_view, sorted_route_ids_d_view,
                                     topk_weights_d_view, output_d_view);
    choreo::abend_true(cudaDeviceSynchronize());
  };

  auto launch_end_to_end = [&]() {
    fused_moe_route(gating_d_view, topk_ids_d_view, topk_weights_d_view);
    launch_serving_path();
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

    auto full_ms = choreo::timing([&]() { launch_end_to_end(); }, topt);
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


