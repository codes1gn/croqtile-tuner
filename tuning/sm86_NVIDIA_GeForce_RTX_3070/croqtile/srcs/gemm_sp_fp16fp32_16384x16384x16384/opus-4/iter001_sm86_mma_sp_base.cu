
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

// Sparse GEMM f16*f16->f32, SM86, mma.sp.sync.aligned m16n8k16
// Dynamic dimensions, 64x32x64 tiling, 16 warps per CTA (512 threads)

#define M 16384
#define N 16384
#define K 16384
#define META_K 16

#define TILE_M 64
#define TILE_N 32
#define TILE_K 64
#define WARP_M 16
#define WARP_N 8
#define WARP_K 16

__global__ void __choreo_device_gemm_sp_f16(f16 * lhs_packed, unsigned int * lhs_meta, f16 * rhs, float * output) {
  auto __choreo_device_gemm_sp_f16__ring__ = nullptr;
  { // parallel-by: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_fp16fp32_16384x16384x16384/opus-4/iter001_sm86_mma_sp_base.co:18.12
  alignas(16) unsigned char anon_5[1024];
  __shared__ alignas(16) unsigned char anon_4[8192];
  auto __choreo_vgid = threadIdx.x / 32;
  auto __choreo_vgid_x = __choreo_vgid / 4;
  auto __choreo_vgid_y = __choreo_vgid % 4;
  auto anon_2 = blockIdx.x * 4 + __choreo_vgid_x;
  auto anon_3 = blockIdx.y * 4 + __choreo_vgid_y;
  float mc[4];
  float __frag_init_val0 = 0.000000f;
  for (int idx = 0; idx < 4; ++idx)
    mc[idx] = __frag_init_val0;
  // with-in: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_fp16fp32_16384x16384x16384/opus-4/iter001_sm86_mma_sp_base.co:21.7
  {
    int __iv_iv_k = 0;
    // foreach: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_fp16fp32_16384x16384x16384/opus-4/iter001_sm86_mma_sp_base.co:21.7
    for (__iv_iv_k = 0; __iv_iv_k < 256; ++__iv_iv_k) {
      f16* lhs_load_s__buf__ = (f16*)(anon_4 + 0);
      future lhs_load_s("lhs_load_s", 22, 22, lhs_load_s__buf__);
      auto __shape1_lhs_packed = cute::make_shape(cute::Int<64>{}, cute::Int<32>{});
      auto __stride1_lhs_packed = cute::make_stride(cute::Int<8192>{}, cute::Int<1>{});
      auto __layout1_lhs_packed = cute::make_layout(__shape1_lhs_packed, __stride1_lhs_packed);
      auto __tensor1_lhs_packed = cute::make_tensor(cute::make_gmem_ptr<f16>((f16*)lhs_packed + (blockIdx.x * 524288 + __iv_iv_k * 32)), __layout1_lhs_packed);
      auto __shape2_lhs_load_s__buf__ = cute::make_shape(cute::Int<64>{}, cute::Int<32>{});
      auto __stride2_lhs_load_s__buf__ = cute::make_stride(cute::Int<32>{}, cute::Int<1>{});
      auto __layout2_lhs_load_s__buf__ = cute::make_layout(__shape2_lhs_load_s__buf__, __stride2_lhs_load_s__buf__);
      auto __tensor2_lhs_load_s__buf__ = cute::make_tensor(cute::make_smem_ptr<f16>((f16*)lhs_load_s__buf__ + 0), __layout2_lhs_load_s__buf__);
        opt_copy(__tensor1_lhs_packed, __tensor2_lhs_load_s__buf__);
      __syncthreads();
      f16* rhs_load_s__buf__ = (f16*)(anon_4 + 4096);
      future rhs_load_s("rhs_load_s", 23, 22, rhs_load_s__buf__);
      auto __shape3_rhs = cute::make_shape(cute::Int<32>{}, cute::Int<64>{});
      auto __stride3_rhs = cute::make_stride(cute::Int<16384>{}, cute::Int<1>{});
      auto __layout3_rhs = cute::make_layout(__shape3_rhs, __stride3_rhs);
      auto __tensor3_rhs = cute::make_tensor(cute::make_gmem_ptr<f16>((f16*)rhs + (blockIdx.y * 524288 + __iv_iv_k * 64)), __layout3_rhs);
      auto __shape4_rhs_load_s__buf__ = cute::make_shape(cute::Int<32>{}, cute::Int<64>{});
      auto __stride4_rhs_load_s__buf__ = cute::make_stride(cute::Int<64>{}, cute::Int<1>{});
      auto __layout4_rhs_load_s__buf__ = cute::make_layout(__shape4_rhs_load_s__buf__, __stride4_rhs_load_s__buf__);
      auto __tensor4_rhs_load_s__buf__ = cute::make_tensor(cute::make_smem_ptr<f16>((f16*)rhs_load_s__buf__ + 0), __layout4_rhs_load_s__buf__);
        opt_copy(__tensor3_rhs, __tensor4_rhs_load_s__buf__);
      __syncthreads();
      unsigned int* lhs_load_s_mdata__buf__ = (unsigned int*)(anon_5 + 0);
      future lhs_load_s_mdata("lhs_load_s_mdata", 24, 28, lhs_load_s_mdata__buf__);
      auto __shape5_lhs_meta = cute::make_shape(cute::Int<64>{}, cute::Int<4>{});
      auto __stride5_lhs_meta = cute::make_stride(cute::Int<1024>{}, cute::Int<1>{});
      auto __layout5_lhs_meta = cute::make_layout(__shape5_lhs_meta, __stride5_lhs_meta);
      auto __tensor5_lhs_meta = cute::make_tensor(cute::make_gmem_ptr<unsigned int>((unsigned int*)lhs_meta + (blockIdx.x * 65536 + __iv_iv_k * 4)), __layout5_lhs_meta);
      auto __shape6_lhs_load_s_mdata__buf__ = cute::make_shape(cute::Int<64>{}, cute::Int<4>{});
      auto __stride6_lhs_load_s_mdata__buf__ = cute::make_stride(cute::Int<4>{}, cute::Int<1>{});
      auto __layout6_lhs_load_s_mdata__buf__ = cute::make_layout(__shape6_lhs_load_s_mdata__buf__, __stride6_lhs_load_s_mdata__buf__);
      auto __tensor6_lhs_load_s_mdata__buf__ = cute::make_tensor(((unsigned int*)lhs_load_s_mdata__buf__ + 0), __layout6_lhs_load_s_mdata__buf__);
        opt_copy(__tensor5_lhs_meta, __tensor6_lhs_load_s_mdata__buf__);
      __syncthreads();
      // with-in: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_fp16fp32_16384x16384x16384/opus-4/iter001_sm86_mma_sp_base.co:25.9
      {
        int __iv_iv_warp_k__elem__0 = 0;
        // foreach: tuning/sm86_NVIDIA_GeForce_RTX_3070/croqtile/srcs/gemm_sp_fp16fp32_16384x16384x16384/opus-4/iter001_sm86_mma_sp_base.co:25.9
        for (__iv_iv_warp_k__elem__0 = 0; __iv_iv_warp_k__elem__0 < 4; ++__iv_iv_warp_k__elem__0) {
          auto __shape7_lhs_load_s = cute::make_shape(cute::Int<16>{}, cute::Int<8>{});
          auto __stride7_lhs_load_s = cute::make_stride(cute::Int<32>{}, cute::Int<1>{});
          auto __layout7_lhs_load_s = cute::make_layout(__shape7_lhs_load_s, __stride7_lhs_load_s);
          auto __tensor7_lhs_load_s = cute::make_tensor(cute::make_smem_ptr<f16>((f16*)lhs_load_s.data() + (__choreo_vgid_x * 512 + __iv_iv_warp_k__elem__0 * 8)), __layout7_lhs_load_s);
          auto ma = load_fragment_a<CUTE_MMA_SPARSE_M16N8K16>(__tensor7_lhs_load_s);
          uint8_t* ma_mdata_ptr = (uint8_t*)lhs_load_s.mdata();
          auto __shape8_rhs_load_s = cute::make_shape(cute::Int<8>{}, cute::Int<16>{});
          auto __stride8_rhs_load_s = cute::make_stride(cute::Int<64>{}, cute::Int<1>{});
          auto __layout8_rhs_load_s = cute::make_layout(__shape8_rhs_load_s, __stride8_rhs_load_s);
          auto __tensor8_rhs_load_s = cute::make_tensor(cute::make_smem_ptr<f16>((f16*)rhs_load_s.data() + (__choreo_vgid_y * 512 + __iv_iv_warp_k__elem__0 * 16)), __layout8_rhs_load_s);
          auto mb = load_fragment_b<CUTE_MMA_SPARSE_M16N8K16>(__tensor8_rhs_load_s);
          auto __shape9_lhs_load_s_mdata = cute::make_shape(cute::Int<16>{}, cute::Int<1>{});
          auto __stride9_lhs_load_s_mdata = cute::make_stride(cute::Int<4>{}, cute::Int<1>{});
          auto __layout9_lhs_load_s_mdata = cute::make_layout(__shape9_lhs_load_s_mdata, __stride9_lhs_load_s_mdata);
          auto __tensor9_lhs_load_s_mdata = cute::make_tensor(((unsigned int*)lhs_load_s_mdata.data() + (__choreo_vgid_x * 64 + __iv_iv_warp_k__elem__0)), __layout9_lhs_load_s_mdata);
          auto me = load_fragment_e<CUTE_MMA_SPARSE_M16N8K16>(__tensor9_lhs_load_s_mdata);
          cute::SM80_SPARSE_16x8x16_F32F16F16F32_TN::fma(mc[0], mc[1], mc[2], mc[3], ma[0], ma[1], mb[0], mb[1], mc[0], mc[1], mc[2], mc[3], me, 0);
        } // iv_warp_k__elem__0
        __iv_iv_warp_k__elem__0 = 0;
      }
    } // iv_k
    __iv_iv_k = 0;
  }
  auto __shape10_output = cute::make_shape(cute::Int<16>{}, cute::Int<8>{});
  auto __stride10_output = cute::make_stride(cute::Int<16384>{}, cute::Int<1>{});
  auto __layout10_output = cute::make_layout(__shape10_output, __stride10_output);
  auto __tensor10_output = cute::make_tensor(cute::make_gmem_ptr<float>((float*)output + ((blockIdx.x * 4 + __choreo_vgid_x) * 262144 + (blockIdx.y * 4 + __choreo_vgid_y) * 8)), __layout10_output);
  store_fragment_d<CUTE_MMA_SPARSE_M16N8K16>(__tensor10_output, reinterpret_cast<float*> (mc));
  } // end parallel-by
}

choreo::spanned_data<choreo::f32, 2> gemm_sp_f16(const choreo::spanned_view<choreo::f16, 2> & lhs_packed, const choreo::spanned_view<choreo::u32, 2> & lhs_meta, const choreo::spanned_view<choreo::f16, 2> & rhs) {
  __choreo_check_cuda_environment__();
  choreo::runtime_check(lhs_packed.shape()[0] == 16384, "shape inconsistent on the 1st parameter ('lhs_packed', dim: 0): expect: 16384, but got " + std::to_string(lhs_packed.shape()[0]) + ".");
  choreo::runtime_check(lhs_packed.shape()[1] == 8192, "shape inconsistent on the 1st parameter ('lhs_packed', dim: 1): expect: 8192, but got " + std::to_string(lhs_packed.shape()[1]) + ".");
  choreo::runtime_check(lhs_meta.shape()[0] == 16384, "shape inconsistent on the 2nd parameter ('lhs_meta', dim: 0): expect: 16384, but got " + std::to_string(lhs_meta.shape()[0]) + ".");
  choreo::runtime_check(lhs_meta.shape()[1] == 1024, "shape inconsistent on the 2nd parameter ('lhs_meta', dim: 1): expect: 1024, but got " + std::to_string(lhs_meta.shape()[1]) + ".");
  choreo::runtime_check(rhs.shape()[0] == 16384, "shape inconsistent on the 3rd parameter ('rhs', dim: 0): expect: 16384, but got " + std::to_string(rhs.shape()[0]) + ".");
  choreo::runtime_check(rhs.shape()[1] == 16384, "shape inconsistent on the 3rd parameter ('rhs', dim: 1): expect: 16384, but got " + std::to_string(rhs.shape()[1]) + ".");

  f16 * lhs_packed__device = nullptr;
  choreo::abend_true(cudaMalloc(&lhs_packed__device, 268435456ULL));
  choreo::abend_true(cudaMemcpy(lhs_packed__device, lhs_packed.data(), 268435456ULL, cudaMemcpyHostToDevice));
  unsigned int * lhs_meta__device = nullptr;
  choreo::abend_true(cudaMalloc(&lhs_meta__device, 67108864ULL));
  choreo::abend_true(cudaMemcpy(lhs_meta__device, lhs_meta.data(), 67108864ULL, cudaMemcpyHostToDevice));
  f16 * rhs__device = nullptr;
  choreo::abend_true(cudaMalloc(&rhs__device, 536870912ULL));
  choreo::abend_true(cudaMemcpy(rhs__device, rhs.data(), 536870912ULL, cudaMemcpyHostToDevice));
  auto output = choreo::make_spandata<choreo::f32, 2>({16384, 16384});
  std::fill(output.data(), output.data()+output.element_count(), 0.000000f);
  float * output__device = nullptr;
  choreo::abend_true(cudaMalloc(&output__device, 1073741824ULL));
  choreo::abend_true(cudaMemcpy(output__device, output.data(), 1073741824ULL, cudaMemcpyHostToDevice));
  dim3 __gemm_sp_f16_gdims0(256, 512, 1);
  dim3 __gemm_sp_f16_bdims0(512, 1, 1);
  __choreo_device_gemm_sp_f16<<<__gemm_sp_f16_gdims0, __gemm_sp_f16_bdims0>>>(lhs_packed__device, lhs_meta__device, rhs__device, output__device);
  choreo::abend_true(cudaDeviceSynchronize());
  choreo::abend_true(cudaMemcpy(output.data(), output__device, 1073741824ULL, cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaFree(lhs_packed__device));
  choreo::abend_true(cudaFree(lhs_meta__device));
  choreo::abend_true(cudaFree(rhs__device));
  return output;
}




template <typename T>
using SparsePolicy = choreo::utils::SparsePolicy<T>;

int main(int argc, char** argv) {
  const int _M = M, _N = N, _K = K, _META_K = META_K;
  int warmup = 5, iters = 20;
  bool verify = true;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--no-verify") verify = false;
    if (arg == "--warmup" && i+1 < argc) warmup = std::atoi(argv[++i]);
    if (arg == "--iters" && i+1 < argc) iters = std::atoi(argv[++i]);
  }

  std::mt19937 gen(42);
  auto lhs_dense = choreo::make_spandata<choreo::f16>(_M, _K);
  auto rhs_host = choreo::make_spandata<choreo::f16>(_N, _K);
  SparsePolicy<choreo::f16>::init_structured_sparse_A(lhs_dense, gen);
  rhs_host.fill_random(-1.0f, 1.0f);
  auto lhs_packed = choreo::make_spandata<choreo::f16>(_M, _K / 2);
  auto lhs_meta = choreo::make_spandata<choreo::u32>(_M, _K / _META_K);
  SparsePolicy<choreo::f16>::encode(lhs_dense, lhs_packed, lhs_meta);

  if (verify) {
    auto res = gemm_sp_f16(lhs_packed.view(), lhs_meta.view(), rhs_host.view());
    int errors = 0;
    int checked = 0;
    for (int i = 0; i < std::min(_M, 64); i++) {
      for (int j = 0; j < std::min(_N, 64); j++) {
        float ref = 0.0f;
        for (int k = 0; k < _K; k++)
          ref += choreo::to_f32(lhs_dense[i][k]) * choreo::to_f32(rhs_host[j][k]);
        float got = res[i][j];
        float diff = std::abs(got - ref);
        float base_tol = 2.0f;
        float rel_tol = 0.02f;
        if (diff > base_tol + rel_tol * std::abs(ref)) errors++;
        checked++;
      }
    }
    std::cout << "Verification: " << errors << "/" << checked << " errors" << std::endl;
    if (errors > 0) {
      std::cout << "FAILED" << std::endl;
      return 1;
    }
    std::cout << "PASSED" << std::endl;
  }

  f16 *d_lhs_packed = nullptr, *d_rhs = nullptr;
  unsigned int *d_lhs_meta = nullptr;
  float *d_output = nullptr;
  choreo::abend_true(cudaMalloc(&d_lhs_packed, 268435456ULL));
  choreo::abend_true(cudaMalloc(&d_lhs_meta, 67108864ULL));
  choreo::abend_true(cudaMalloc(&d_rhs, 536870912ULL));
  choreo::abend_true(cudaMalloc(&d_output, 1073741824ULL));
  choreo::abend_true(cudaMemcpy(d_lhs_packed, lhs_packed.data(), 268435456ULL, cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(d_lhs_meta, lhs_meta.data(), 67108864ULL, cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(d_rhs, rhs_host.data(), 536870912ULL, cudaMemcpyHostToDevice));

  dim3 gdims(256, 512, 1);
  dim3 bdims(512, 1, 1);

  for (int i = 0; i < warmup; i++) {
    cudaMemset(d_output, 0, 1073741824ULL);
    __choreo_device_gemm_sp_f16<<<gdims, bdims>>>(d_lhs_packed, d_lhs_meta, d_rhs, d_output);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    cudaMemset(d_output, 0, 1073741824ULL);
    __choreo_device_gemm_sp_f16<<<gdims, bdims>>>(d_lhs_packed, d_lhs_meta, d_rhs, d_output);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_ms = ms / iters;
  double flops = 2.0 * (double)_M * (double)_N * (double)_K;
  double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
  std::cout << "avg_ms=" << avg_ms << " TFLOPS=" << tflops << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_lhs_packed);
  cudaFree(d_lhs_meta);
  cudaFree(d_rhs);
  cudaFree(d_output);
  return 0;
}


