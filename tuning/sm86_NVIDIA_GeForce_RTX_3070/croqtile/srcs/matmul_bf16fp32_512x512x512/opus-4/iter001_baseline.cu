
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

#include <cstring>
#include <cstdlib>

// SM86 (RTX 3070) BF16 Tensor Core peak throughput.
#define SM86_PEAK_bf16_TFLOPS 163

// SM86 mma.sync tile: m16n16k16
#define MATMUL_MMA_M 16
#define MATMUL_MMA_N 16
#define MATMUL_MMA_K 16

// Block tile defaults - minimal single warp
#define MATMUL_TILE_M 16
#define MATMUL_TILE_N 16
#define MATMUL_TILE_K 16

#if MATMUL_TILE_M % MATMUL_MMA_M != 0
#error "MATMUL_TILE_M must be divisible by MATMUL_MMA_M"
#endif

#if MATMUL_TILE_N % MATMUL_MMA_N != 0
#error "MATMUL_TILE_N must be divisible by MATMUL_MMA_N"
#endif

#if MATMUL_TILE_K % MATMUL_MMA_K != 0
#error "MATMUL_TILE_K must be divisible by MATMUL_MMA_K"
#endif

// SM86 register pressure: opt_copy uses 255 regs/thread on SM86
#if (MATMUL_TILE_M / MATMUL_MMA_M) * (MATMUL_TILE_N / MATMUL_MMA_N) > 8
#error "Too many warps per block"
#endif

#define MATMUL_DEFAULT_M 512
#define MATMUL_DEFAULT_N 512
#define MATMUL_DEFAULT_K 512

__global__ void __choreo_device_matmul(bf16 * lhs, bf16 * rhs, float * output, unsigned K, unsigned M, unsigned N) {
  extern __shared__ char __choreo_device_matmul__runtime_shared_buffer__raw[];
  auto __choreo_device_matmul__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<16 * 8>(__choreo_device_matmul__runtime_shared_buffer__raw));
  auto __choreo_device_matmul__ring__ = reinterpret_cast<choreo::future_ring<6>*>(__choreo_device_matmul__runtime_shared_buffer__ + 0);
  if (threadIdx.x <= 1 && threadIdx.y == 0 && threadIdx.z == 0)    __choreo_device_matmul__ring__[threadIdx.x].init();
  __syncthreads();  // must sync
  { // parallel-by: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/matmul_bf16fp32_512x512x512/opus-4/iter001_minimal.co:42.12
  __shared__ alignas(16) unsigned char anon_2[2048];
  bf16* lhs_s = (bf16*)(anon_2 + 1536);
  bf16* rhs_s = (bf16*)(anon_2 + 1024);
  float* output_s = (float*)(anon_2 + 0);
  auto __choreo_vgid = threadIdx.x / 32;
  auto __choreo_vgid_x = __choreo_vgid / 1;
  auto __choreo_vgid_y = __choreo_vgid % 1;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> mc;
  nvcuda::wmma::fill_fragment(mc, (float)0.000000);
  // with-in: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/matmul_bf16fp32_512x512x512/opus-4/iter001_minimal.co:48.7
  {
    int __iv_iv_k = 0;
    // foreach: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/matmul_bf16fp32_512x512x512/opus-4/iter001_minimal.co:48.7
    for (__iv_iv_k = 0; __iv_iv_k < ((K + 15) / 16); ++__iv_iv_k) {
      AsyncCopyAtom choreo_copy_atom0{};
      future lhs_load_s("lhs_load_s", 49, 22, lhs_s);
      lhs_load_s.set_atom(&choreo_copy_atom0);
      lhs_load_s.set_ring(__choreo_device_matmul__ring__);
      lhs_load_s.id = 1;
      auto __shape1_lhs = cute::make_shape(cute::Int<16>{}, cute::Int<16>{});
      auto __stride1_lhs = cute::make_stride(K, cute::Int<1>{});
      auto __layout1_lhs = cute::make_layout(__shape1_lhs, __stride1_lhs);
      auto __tensor1_lhs = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)lhs + (K * blockIdx.x * 16 + __iv_iv_k * 16)), __layout1_lhs);
      auto __shape2_lhs_s = cute::make_shape(cute::Int<16>{}, cute::Int<16>{});
      auto __stride2_lhs_s = cute::make_stride(cute::Int<16>{}, cute::Int<1>{});
      auto __layout2_lhs_s = cute::make_layout(__shape2_lhs_s, __stride2_lhs_s);
      auto __tensor2_lhs_s = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)lhs_s + 0), __layout2_lhs_s);
        cute::copy(*(AsyncCopyAtom*)lhs_load_s.get_atom(), __tensor1_lhs, __tensor2_lhs_s);
        cute::cp_async_fence();
        lhs_load_s.trigger();
      AsyncCopyAtom choreo_copy_atom1{};
      future rhs_load_s("rhs_load_s", 50, 22, rhs_s);
      rhs_load_s.set_atom(&choreo_copy_atom1);
      rhs_load_s.set_ring(__choreo_device_matmul__ring__);
      rhs_load_s.id = 2;
      auto __shape3_rhs = cute::make_shape(cute::Int<16>{}, cute::Int<16>{});
      auto __stride3_rhs = cute::make_stride(K, cute::Int<1>{});
      auto __layout3_rhs = cute::make_layout(__shape3_rhs, __stride3_rhs);
      auto __tensor3_rhs = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)rhs + (K * blockIdx.y * 16 + __iv_iv_k * 16)), __layout3_rhs);
      auto __shape4_rhs_s = cute::make_shape(cute::Int<16>{}, cute::Int<16>{});
      auto __stride4_rhs_s = cute::make_stride(cute::Int<16>{}, cute::Int<1>{});
      auto __layout4_rhs_s = cute::make_layout(__shape4_rhs_s, __stride4_rhs_s);
      auto __tensor4_rhs_s = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)rhs_s + 0), __layout4_rhs_s);
        cute::copy(*(AsyncCopyAtom*)rhs_load_s.get_atom(), __tensor3_rhs, __tensor4_rhs_s);
        cute::cp_async_fence();
        rhs_load_s.trigger();
      lhs_load_s.wait();
      rhs_load_s.wait();
      // with-in: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/matmul_bf16fp32_512x512x512/opus-4/iter001_minimal.co:53.9
      {
        int __iv_iv_warp_k = 0;
        // foreach: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/matmul_bf16fp32_512x512x512/opus-4/iter001_minimal.co:53.9
        for (__iv_iv_warp_k = 0; __iv_iv_warp_k < 1; ++__iv_iv_warp_k) {
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, bf16, nvcuda::wmma::row_major> ma;
          nvcuda::wmma::load_matrix_sync(ma, ((bf16*)lhs_load_s.data() + (__choreo_vgid_x * 256 + __iv_iv_warp_k * 16)), 16);
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, bf16, nvcuda::wmma::col_major> mb;
          nvcuda::wmma::load_matrix_sync(mb, ((bf16*)rhs_load_s.data() + (__choreo_vgid_y * 256 + __iv_iv_warp_k * 16)), 16);
          nvcuda::wmma::mma_sync(mc, ma, mb, mc);
        } // iv_warp_k
        __iv_iv_warp_k = 0;
      }
    } // iv_k
    __iv_iv_k = 0;
  }
  nvcuda::wmma::store_matrix_sync((__choreo_vgid_x * 256 + __choreo_vgid_y * 16 + output_s), mc, 16, nvcuda::wmma::mem_row_major);
  future __choreo_anon_fut__2("", 63, 5);
  auto __shape5_output_s = cute::make_shape(cute::Int<16>{}, cute::Int<16>{});
  auto __stride5_output_s = cute::make_stride(cute::Int<16>{}, cute::Int<1>{});
  auto __layout5_output_s = cute::make_layout(__shape5_output_s, __stride5_output_s);
  auto __tensor5_output_s = cute::make_tensor(cute::make_smem_ptr<float>((float*)output_s + 0), __layout5_output_s);
  auto __shape6_output = cute::make_shape(cute::Int<16>{}, cute::Int<16>{});
  auto __stride6_output = cute::make_stride(N, cute::Int<1>{});
  auto __layout6_output = cute::make_layout(__shape6_output, __stride6_output);
  auto __tensor6_output = cute::make_tensor(cute::make_gmem_ptr<float>((float*)output + (N * blockIdx.x * 16 + blockIdx.y * 16)), __layout6_output);
    opt_copy(__tensor5_output_s, __tensor6_output);
  __syncthreads();
  } // end parallel-by
}

void matmul(const choreo::spanned_view<choreo::bf16, 2> & lhs, const choreo::spanned_view<choreo::bf16, 2> & rhs, const choreo::spanned_view<choreo::f32, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &K = lhs.shape()[1];
  auto &M = lhs.shape()[0];
  auto &N = rhs.shape()[0];
  choreo::runtime_check(lhs.shape()[1] == rhs.shape()[1], "The shapes of the 1st parameter (dim: 1) and the 2nd parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(lhs.shape()[0] == output.shape()[0], "The shapes of the 1st parameter (dim: 0) and the 3rd parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(rhs.shape()[0] == output.shape()[1], "The shapes of the 2nd parameter (dim: 0) and the 3rd parameter (dim: 1) are inconsistent.");

  choreo::runtime_check(((static_cast<long long>(M) + 15) / 16 > 0), "The 1st bound item of parallelby is invalid: should be greater than 0, tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/matmul_bf16fp32_512x512x512/opus-4/iter001_minimal.co:42.13");
  choreo::runtime_check(((static_cast<long long>(N) + 15) / 16 > 0), "The 2nd bound item of parallelby is invalid: should be greater than 0, tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/matmul_bf16fp32_512x512x512/opus-4/iter001_minimal.co:42.22");
  choreo::runtime_check(((static_cast<long long>(K) + 15) / 16 != 0), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/matmul_bf16fp32_512x512x512/opus-4/iter001_minimal.co:48.25");
  dim3 __matmul_gdims0(((M + 15) / 16), ((N + 15) / 16), 1);
  dim3 __matmul_bdims0(32, 1, 1);
  __choreo_device_matmul<<<__matmul_gdims0, __matmul_bdims0, 8 + (16 - 1)>>>(lhs.data(), rhs.data(), output.data(), K, M, N);
  choreo::abend_true(cudaDeviceSynchronize());
}




int main(int argc, char** argv) {
  bool enable_timing = true;
  bool skip_verify = false;
  double user_flops = -1.0;
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
    if (std::strncmp(argv[i], "--flops=", 8) == 0) {
      user_flops = std::atof(argv[i] + 8);
      continue;
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

  size_t M = MATMUL_DEFAULT_M;
  size_t N = MATMUL_DEFAULT_N;
  size_t K = MATMUL_DEFAULT_K;

  auto lhs_h = choreo::make_spandata<choreo::bf16>(M, K);
  auto rhs_h = choreo::make_spandata<choreo::bf16>(N, K);
  auto res_h = choreo::make_spandata<choreo::f32>(M, N);
  lhs_h.fill_random(0, 2);
  rhs_h.fill_random(0, 2);
  res_h.fill(0.0f);

  __nv_bfloat16 *a_d = nullptr;
  __nv_bfloat16 *b_d = nullptr;
  float *c_d = nullptr;
  choreo::abend_true(cudaMalloc(&a_d, M * K * sizeof(__nv_bfloat16)));
  choreo::abend_true(cudaMalloc(&b_d, N * K * sizeof(__nv_bfloat16)));
  choreo::abend_true(cudaMalloc(&c_d, M * N * sizeof(float)));

  // Copy bf16 data
  choreo::abend_true(cudaMemcpy(a_d, lhs_h.data(), M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(b_d, rhs_h.data(), N * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(c_d, res_h.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());

  auto lhs_d = choreo::make_spanview<choreo::bf16, 2>(a_d, {M, K});
  auto rhs_d = choreo::make_spanview<choreo::bf16, 2>(b_d, {N, K});
  auto res_d = choreo::make_spanview<choreo::f32, 2>(c_d, {M, N});
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
    auto avg_ms = choreo::timing([&]() { matmul(lhs_d, rhs_d, res_d); cudaDeviceSynchronize(); }, topt);
    std::cout << "Timing avg ms: " << avg_ms << "\n";

    double flops = (user_flops > 0.0) ? user_flops : (2.0 * double(M) * double(N) * double(K));
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    std::cout << "TFLOPS: " << tflops << "\n";

    double eff = (tflops / SM86_PEAK_bf16_TFLOPS) * 100.0;
    std::cout << "HW efficiency: " << eff << "%\n";
  } else {
    matmul(lhs_d, rhs_d, res_d);
  }

  choreo::abend_true(cudaMemcpy(res_h.data(), c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaDeviceSynchronize());

  if (skip_verify) {
    std::cout << "Test Passed (verify skipped)\n" << std::endl;
    return 0;
  }

  auto lhs_view = lhs_h.view();
  auto rhs_view = rhs_h.view();
  auto res_view = res_h.view();

  float tolerance = 1.0f;
  auto rel_error = [](float ref, float got) {
    float abs_ref = std::abs(ref);
    float denom = abs_ref > 1e-6f ? abs_ref : 1.0f;
    return std::abs(ref - got) / denom;
  };
  // verification
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float ref = 0.0f;
      for (size_t k = 0; k < lhs_view.shape()[1]; ++k)
        ref += float(lhs_view[i][k]) * float(rhs_view[j][k]);
      float got = res_view[i][j];
      auto delta = rel_error(ref, got);
      if (delta >= tolerance) {
        std::cout << "[" << i << ", " << j << "] " << ref << " <-> " << got << ", delta: " << delta * 100 << "%\n";
      }
      choreo::choreo_assert((delta < tolerance), "values are not equal.");
    }
  }

  std::cout << "Test Passed\n" << std::endl;
}


