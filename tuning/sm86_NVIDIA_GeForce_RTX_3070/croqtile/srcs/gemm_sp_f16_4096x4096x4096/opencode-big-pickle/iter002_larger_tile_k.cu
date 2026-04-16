
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
// include the choreo header;
#include "choreo.h"
#include <cooperative_groups.h>
using namespace choreo;

#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ 86

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

// Sparse GEMM f16 for Ampere (sm_80/sm_86) using MMA with sparse support
// Adapted from gemm_sp_sm80.co for 4096x4096x4096
// Round 2: Increase TILE_K from 64 to 128 to improve L2 cache utilization

#include <cstring>
#include <cstdlib>
#include <random>

#define RTX3070_PEAK_F16_TFLOPS 19.5

#ifndef SPMM_DEFAULT_M
#define SPMM_DEFAULT_M 4096
#endif

#ifndef SPMM_DEFAULT_N
#define SPMM_DEFAULT_N 4096
#endif

#ifndef SPMM_DEFAULT_K
#define SPMM_DEFAULT_K 4096
#endif

// Tile sizes optimized for RTX 3070 (sm_86) - Ampere sparse MMA
// Round 2 change: TILE_K 64 -> 128 (reduces K-iterations from 64 to 32)
#define TILE_M 64
#define TILE_N 64
#define TILE_K 128
#define WARP_M 16
#define WARP_N 8
#define WARP_K 16
#define META_K 16  // WARP_K for sparse MMA (Ampere constraint)

template <typename T>
using SparsePolicy = choreo::utils::SparsePolicy<T>;

template <typename T>
void init_random_b(choreo::spanned_data<T, 2>& rhs, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t j = 0; j < rhs.shape()[0]; ++j)
    for (size_t k = 0; k < rhs.shape()[1]; ++k) {
      float value = dist(gen);
      if (std::fabs(value) < 0.1f) value = (value < 0.0f ? -0.25f : 0.25f);
      rhs[j][k] = choreo::utils::from_f32<T>(value);
    }
}

__global__ void __choreo_device_spmm_kernel(f16 * lhs_packed, unsigned int * lhs_meta, f16 * rhs, f16 * output) {
  auto __choreo_device_spmm_kernel__ring__ = nullptr;
  { // parallel-by: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_f16_4096x4096x4096/opencode-big-pickle/iter002_larger_tile_k.co:53.12
  alignas(16) unsigned char anon_5[2048];
  __shared__ alignas(16) unsigned char anon_4[24576];
  auto __choreo_vgid = threadIdx.x / 32;
  auto __choreo_vgid_x = __choreo_vgid / 8;
  auto __choreo_vgid_y = __choreo_vgid % 8;
  auto anon_2 = blockIdx.x * 4 + __choreo_vgid_x;
  auto anon_3 = blockIdx.y * 8 + __choreo_vgid_y;
  float mc[4];
  float __frag_init_val0 = 0.000000f;
  for (int idx = 0; idx < 4; ++idx)
    mc[idx] = __frag_init_val0;
  // with-in: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_f16_4096x4096x4096/opencode-big-pickle/iter002_larger_tile_k.co:61.7
  {
    int __iv_iv_k = 0;
    // foreach: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_f16_4096x4096x4096/opencode-big-pickle/iter002_larger_tile_k.co:61.7
    for (__iv_iv_k = 0; __iv_iv_k < 32; ++__iv_iv_k) {
      f16* lhs_load_s__buf__ = (f16*)(anon_4 + 16384);
      future lhs_load_s("lhs_load_s", 63, 22, lhs_load_s__buf__);
      auto __shape1_lhs_packed = cute::make_shape(cute::Int<64>{}, cute::Int<64>{});
      auto __stride1_lhs_packed = cute::make_stride(cute::Int<2048>{}, cute::Int<1>{});
      auto __layout1_lhs_packed = cute::make_layout(__shape1_lhs_packed, __stride1_lhs_packed);
      auto __tensor1_lhs_packed = cute::make_tensor(cute::make_gmem_ptr<f16>((f16*)lhs_packed + (blockIdx.x * 131072 + __iv_iv_k * 64)), __layout1_lhs_packed);
      auto __shape2_lhs_load_s__buf__ = cute::make_shape(cute::Int<64>{}, cute::Int<64>{});
      auto __stride2_lhs_load_s__buf__ = cute::make_stride(cute::Int<64>{}, cute::Int<1>{});
      auto __layout2_lhs_load_s__buf__ = cute::make_layout(__shape2_lhs_load_s__buf__, __stride2_lhs_load_s__buf__);
      auto __tensor2_lhs_load_s__buf__ = cute::make_tensor(cute::make_smem_ptr<f16>((f16*)lhs_load_s__buf__ + 0), __layout2_lhs_load_s__buf__);
        opt_copy(__tensor1_lhs_packed, __tensor2_lhs_load_s__buf__);
      __syncthreads();
      f16* rhs_load_s__buf__ = (f16*)(anon_4 + 0);
      future rhs_load_s("rhs_load_s", 64, 22, rhs_load_s__buf__);
      auto __shape3_rhs = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
      auto __stride3_rhs = cute::make_stride(cute::Int<4096>{}, cute::Int<1>{});
      auto __layout3_rhs = cute::make_layout(__shape3_rhs, __stride3_rhs);
      auto __tensor3_rhs = cute::make_tensor(cute::make_gmem_ptr<f16>((f16*)rhs + (blockIdx.y * 262144 + __iv_iv_k * 128)), __layout3_rhs);
      auto __shape4_rhs_load_s__buf__ = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
      auto __stride4_rhs_load_s__buf__ = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
      auto __layout4_rhs_load_s__buf__ = cute::make_layout(__shape4_rhs_load_s__buf__, __stride4_rhs_load_s__buf__);
      auto __tensor4_rhs_load_s__buf__ = cute::make_tensor(cute::make_smem_ptr<f16>((f16*)rhs_load_s__buf__ + 0), __layout4_rhs_load_s__buf__);
        opt_copy(__tensor3_rhs, __tensor4_rhs_load_s__buf__);
      __syncthreads();
      unsigned int* lhs_meta_s__buf__ = (unsigned int*)(anon_5 + 0);
      future lhs_meta_s("lhs_meta_s", 65, 22, lhs_meta_s__buf__);
      auto __shape5_lhs_meta = cute::make_shape(cute::Int<64>{}, cute::Int<8>{});
      auto __stride5_lhs_meta = cute::make_stride(cute::Int<256>{}, cute::Int<1>{});
      auto __layout5_lhs_meta = cute::make_layout(__shape5_lhs_meta, __stride5_lhs_meta);
      auto __tensor5_lhs_meta = cute::make_tensor(cute::make_gmem_ptr<unsigned int>((unsigned int*)lhs_meta + (blockIdx.x * 16384 + __iv_iv_k * 8)), __layout5_lhs_meta);
      auto __shape6_lhs_meta_s__buf__ = cute::make_shape(cute::Int<64>{}, cute::Int<8>{});
      auto __stride6_lhs_meta_s__buf__ = cute::make_stride(cute::Int<8>{}, cute::Int<1>{});
      auto __layout6_lhs_meta_s__buf__ = cute::make_layout(__shape6_lhs_meta_s__buf__, __stride6_lhs_meta_s__buf__);
      auto __tensor6_lhs_meta_s__buf__ = cute::make_tensor(((unsigned int*)lhs_meta_s__buf__ + 0), __layout6_lhs_meta_s__buf__);
        opt_copy(__tensor5_lhs_meta, __tensor6_lhs_meta_s__buf__);
      __syncthreads();
      // with-in: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_f16_4096x4096x4096/opencode-big-pickle/iter002_larger_tile_k.co:67.9
      {
        int __iv_iv_warp_k = 0;
        // foreach: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_f16_4096x4096x4096/opencode-big-pickle/iter002_larger_tile_k.co:67.9
        for (__iv_iv_warp_k = 0; __iv_iv_warp_k < 8; ++__iv_iv_warp_k) {
          auto __shape7_lhs_load_s = cute::make_shape(cute::Int<16>{}, cute::Int<8>{});
          auto __stride7_lhs_load_s = cute::make_stride(cute::Int<64>{}, cute::Int<1>{});
          auto __layout7_lhs_load_s = cute::make_layout(__shape7_lhs_load_s, __stride7_lhs_load_s);
          auto __tensor7_lhs_load_s = cute::make_tensor(cute::make_smem_ptr<f16>((f16*)lhs_load_s.data() + (__choreo_vgid_x * 1024 + __iv_iv_warp_k * 8)), __layout7_lhs_load_s);
          auto ma = load_fragment_a<CUTE_MMA_SPARSE_M16N8K16>(__tensor7_lhs_load_s);
          uint8_t* ma_mdata_ptr = (uint8_t*)lhs_load_s.mdata();
          auto __shape8_rhs_load_s = cute::make_shape(cute::Int<8>{}, cute::Int<16>{});
          auto __stride8_rhs_load_s = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
          auto __layout8_rhs_load_s = cute::make_layout(__shape8_rhs_load_s, __stride8_rhs_load_s);
          auto __tensor8_rhs_load_s = cute::make_tensor(cute::make_smem_ptr<f16>((f16*)rhs_load_s.data() + (__choreo_vgid_y * 1024 + __iv_iv_warp_k * 16)), __layout8_rhs_load_s);
          auto mb = load_fragment_b<CUTE_MMA_SPARSE_M16N8K16>(__tensor8_rhs_load_s);
          auto __shape9_lhs_meta_s = cute::make_shape(cute::Int<16>{}, cute::Int<1>{});
          auto __stride9_lhs_meta_s = cute::make_stride(cute::Int<8>{}, cute::Int<1>{});
          auto __layout9_lhs_meta_s = cute::make_layout(__shape9_lhs_meta_s, __stride9_lhs_meta_s);
          auto __tensor9_lhs_meta_s = cute::make_tensor(((unsigned int*)lhs_meta_s.data() + (__choreo_vgid_x * 128 + __iv_iv_warp_k)), __layout9_lhs_meta_s);
          auto me = load_fragment_e<CUTE_MMA_SPARSE_M16N8K16>(__tensor9_lhs_meta_s);
          cute::SM80_SPARSE_16x8x16_F32F16F16F32_TN::fma(mc[0], mc[1], mc[2], mc[3], ma[0], ma[1], mb[0], mb[1], mc[0], mc[1], mc[2], mc[3], me, 0);
        } // iv_warp_k
        __iv_iv_warp_k = 0;
      }
    } // iv_k
    __iv_iv_k = 0;
  }
  auto __shape10_output = cute::make_shape(cute::Int<16>{}, cute::Int<8>{});
  auto __stride10_output = cute::make_stride(cute::Int<4096>{}, cute::Int<1>{});
  auto __layout10_output = cute::make_layout(__shape10_output, __stride10_output);
  auto __tensor10_output = cute::make_tensor(cute::make_gmem_ptr<f16>((f16*)output + ((blockIdx.x * 4 + __choreo_vgid_x) * 65536 + (blockIdx.y * 8 + __choreo_vgid_y) * 8)), __layout10_output);
  store_fragment_d<CUTE_MMA_SPARSE_M16N8K16>(__tensor10_output, reinterpret_cast<float*> (mc));
  } // end parallel-by
}

void spmm_kernel(const choreo::spanned_view<choreo::f16, 2> & lhs_packed, const choreo::spanned_view<choreo::u32, 2> & lhs_meta, const choreo::spanned_view<choreo::f16, 2> & rhs, const choreo::spanned_view<choreo::f16, 2> & output) {
  __choreo_check_cuda_environment__();
  choreo::runtime_check(lhs_packed.shape()[0] == 4096, "shape inconsistent on the 1st parameter ('lhs_packed', dim: 0): expect: 4096, but got " + std::to_string(lhs_packed.shape()[0]) + ".");
  choreo::runtime_check(lhs_packed.shape()[1] == 2048, "shape inconsistent on the 1st parameter ('lhs_packed', dim: 1): expect: 2048, but got " + std::to_string(lhs_packed.shape()[1]) + ".");
  choreo::runtime_check(lhs_meta.shape()[0] == 4096, "shape inconsistent on the 2nd parameter ('lhs_meta', dim: 0): expect: 4096, but got " + std::to_string(lhs_meta.shape()[0]) + ".");
  choreo::runtime_check(lhs_meta.shape()[1] == 256, "shape inconsistent on the 2nd parameter ('lhs_meta', dim: 1): expect: 256, but got " + std::to_string(lhs_meta.shape()[1]) + ".");
  choreo::runtime_check(rhs.shape()[0] == 4096, "shape inconsistent on the 3rd parameter ('rhs', dim: 0): expect: 4096, but got " + std::to_string(rhs.shape()[0]) + ".");
  choreo::runtime_check(rhs.shape()[1] == 4096, "shape inconsistent on the 3rd parameter ('rhs', dim: 1): expect: 4096, but got " + std::to_string(rhs.shape()[1]) + ".");
  choreo::runtime_check(output.shape()[0] == 4096, "shape inconsistent on the 4th parameter ('output', dim: 0): expect: 4096, but got " + std::to_string(output.shape()[0]) + ".");
  choreo::runtime_check(output.shape()[1] == 4096, "shape inconsistent on the 4th parameter ('output', dim: 1): expect: 4096, but got " + std::to_string(output.shape()[1]) + ".");

  dim3 __spmm_kernel_gdims0(64, 64, 1);
  dim3 __spmm_kernel_bdims0(1024, 1, 1);
  __choreo_device_spmm_kernel<<<__spmm_kernel_gdims0, __spmm_kernel_bdims0>>>(lhs_packed.data(), lhs_meta.data(), rhs.data(), output.data());
  choreo::abend_true(cudaDeviceSynchronize());
}




int main(int argc, char** argv) {
  bool enable_timing = true;
  bool skip_verify = false;
  
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
  }

  const char* timing_env = std::getenv("CHOREO_DISABLE_TIMING");
  if (timing_env && timing_env[0] == '1' && timing_env[1] == '\0') {
    enable_timing = false;
  }

  const char* skip_verify_env = std::getenv("CHOREO_SKIP_VERIFY");
  if (skip_verify_env && skip_verify_env[0] == '1' && skip_verify_env[1] == '\0') {
    skip_verify = true;
  }

  size_t M = SPMM_DEFAULT_M;
  size_t N = SPMM_DEFAULT_N;
  size_t K = SPMM_DEFAULT_K;

  std::mt19937 gen(42);
  auto lhs_dense = choreo::make_spandata<choreo::f16>(M, K);
  auto lhs_packed = choreo::make_spandata<choreo::f16>(M, K / 2);
  auto lhs_meta = choreo::make_spandata<choreo::u32>(M, K / META_K);
  auto rhs = choreo::make_spandata<choreo::f16>(N, K);
  auto res = choreo::make_spandata<choreo::f16>(M, N);
  
  SparsePolicy<choreo::f16>::init_structured_sparse_A(lhs_dense, gen);
  init_random_b<choreo::f16>(rhs, gen);
  SparsePolicy<choreo::f16>::encode(lhs_dense, lhs_packed, lhs_meta);

  half *lhs_packed_d = nullptr, *rhs_d = nullptr, *res_d = nullptr;
  u32 *lhs_meta_d = nullptr;
  
  choreo::abend_true(cudaMalloc(&lhs_packed_d, M * (K / 2) * sizeof(half)));
  choreo::abend_true(cudaMalloc(&lhs_meta_d, M * (K / META_K) * sizeof(u32)));
  choreo::abend_true(cudaMalloc(&rhs_d, N * K * sizeof(half)));
  choreo::abend_true(cudaMalloc(&res_d, M * N * sizeof(half)));
  
  choreo::abend_true(cudaMemcpy(lhs_packed_d, lhs_packed.data(), M * (K / 2) * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(lhs_meta_d, lhs_meta.data(), M * (K / META_K) * sizeof(u32), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(rhs_d, rhs.data(), N * K * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(res_d, res.data(), M * N * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());

  auto lhs_packed_dv = choreo::make_spanview<choreo::f16, 2>(lhs_packed_d, {M, K / 2});
  auto lhs_meta_dv = choreo::make_spanview<choreo::u32, 2>(lhs_meta_d, {M, K / META_K});
  auto rhs_dv = choreo::make_spanview<choreo::f16, 2>(rhs_d, {N, K});
  auto res_dv = choreo::make_spanview<choreo::f16, 2>(res_d, {M, N});

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

    auto avg_ms = choreo::timing([&]() { 
      spmm_kernel(lhs_packed_dv, lhs_meta_dv, rhs_dv, res_dv); 
      cudaDeviceSynchronize(); 
    }, topt);
    
    double flops = 2.0 * double(M) * double(N) * double(K);
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    std::cout << "Timing avg ms: " << avg_ms << "\n";
    std::cout << "TFLOPS: " << tflops << "\n";
    
    double eff = (tflops / RTX3070_PEAK_F16_TFLOPS) * 100.0;
    std::cout << "HW efficiency: " << eff << "%\n";
  }

  if (skip_verify) {
    std::cout << "Test Passed (verify skipped)\n" << std::endl;
    return 0;
  }

  spmm_kernel(lhs_packed_dv, lhs_meta_dv, rhs_dv, res_dv);
  choreo::abend_true(cudaDeviceSynchronize());
  choreo::abend_true(cudaMemcpy(res.data(), res_d, M * N * sizeof(half), cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaDeviceSynchronize());

  // Verify a subset
  float tolerance = 0.5f;
  size_t verify_m = (M < 128) ? M : 128;
  size_t verify_n = (N < 256) ? N : 256;
  int errors = 0;
  
  for (size_t i = 0; i < verify_m; ++i) {
    for (size_t j = 0; j < verify_n; ++j) {
      float ref = 0.0f;
      for (size_t k = 0; k < K; ++k)
        ref += choreo::to_f32(lhs_dense[i][k]) * choreo::to_f32(rhs[j][k]);
      float got = choreo::to_f32(res[i][j]);
      float diff = std::abs(got - ref);
      if (diff > tolerance) {
        if (errors < 8)
          std::cout << "[" << i << ", " << j << "] ref=" << ref
                    << " got=" << got << " diff=" << diff << std::endl;
        ++errors;
      }
    }
  }
  std::cout << "f16_sp: " << errors << " errors\n";
  
  if (errors == 0) {
    std::cout << "Test Passed\n";
  } else {
    std::cout << "Test FAILED\n";
    return 1;
  }
}


