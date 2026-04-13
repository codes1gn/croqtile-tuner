#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t status = (call);                                              \
    if (status != cudaSuccess) {                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl; \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                              \
  do {                                                                   \
    cublasStatus_t status = (call);                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                               \
      std::cerr << "cuBLAS error code: " << static_cast<int>(status)     \
                << std::endl;                                            \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

int main(int argc, char** argv) {
  int m = 512;
  int n = 16384;
  int k = 16384;
  int warmup = 3;
  int iters = 10;

  if (argc >= 4) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
  }
  if (argc >= 5) {
    warmup = std::atoi(argv[4]);
  }
  if (argc >= 6) {
    iters = std::atoi(argv[5]);
  }

  nv_bfloat16* dA = nullptr;
  nv_bfloat16* dB = nullptr;
  float* dC = nullptr;

  const size_t bytes_a = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(nv_bfloat16);
  const size_t bytes_b = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(nv_bfloat16);
  const size_t bytes_c = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);

  CUDA_CHECK(cudaMalloc(&dA, bytes_a));
  CUDA_CHECK(cudaMalloc(&dB, bytes_b));
  CUDA_CHECK(cudaMalloc(&dC, bytes_c));

  CUDA_CHECK(cudaMemset(dA, 0, bytes_a));
  CUDA_CHECK(cudaMemset(dB, 0, bytes_b));
  CUDA_CHECK(cudaMemset(dC, 0, bytes_c));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (int i = 0; i < warmup; ++i) {
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              &alpha,
                              dB,
                              CUDA_R_16BF,
                              n,
                              dA,
                              CUDA_R_16BF,
                              k,
                              &beta,
                              dC,
                              CUDA_R_32F,
                              n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              &alpha,
                              dB,
                              CUDA_R_16BF,
                              n,
                              dA,
                              CUDA_R_16BF,
                              k,
                              &beta,
                              dC,
                              CUDA_R_32F,
                              n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  const double time_ms = static_cast<double>(elapsed_ms) / static_cast<double>(iters);
  const double tflops =
      (2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k)) /
      (time_ms * 1e-3) / 1e12;

  std::cout << "time_ms=" << time_ms << ",tflops=" << tflops << std::endl;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}
