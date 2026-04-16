// iter000_baseline.cu - cuBLAS baseline for bf16 input -> fp32 output matmul
// Shape: M=16384, N=16384, K=16384
// Uses cuBLAS GemmEx with BF16 inputs and FP32 output/compute

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

constexpr int M = 16384;
constexpr int N = 16384;
constexpr int K = 16384;
constexpr int WARMUP = 10;
constexpr int ITERS = 50;
constexpr int SAMPLES = 5;

void init_bf16_random(__nv_bfloat16* data, size_t size, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; i++) {
        data[i] = __float2bfloat16(dist(gen));
    }
}

int main() {
    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;

    __nv_bfloat16 *h_A = new __nv_bfloat16[size_A];
    __nv_bfloat16 *h_B = new __nv_bfloat16[size_B];
    init_bf16_random(h_A, size_A, 42);
    init_bf16_random(h_B, size_B, 123);

    __nv_bfloat16 *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, size_C * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < WARMUP; i++) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            d_B, CUDA_R_16BF, N,
            d_A, CUDA_R_16BF, K,
            &beta, d_C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double best_tflops = 0.0;
    for (int s = 0; s < SAMPLES; s++) {
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++) {
            CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                d_B, CUDA_R_16BF, N,
                d_A, CUDA_R_16BF, K,
                &beta, d_C, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        double time_ms = ms / ITERS;
        double tflops = 2.0 * M * N * K / (time_ms * 1e-3) / 1e12;
        printf("sample %d: time=%.3f ms  TFLOPS: %.2f\n", s+1, time_ms, tflops);
        if (tflops > best_tflops) best_tflops = tflops;
    }

    printf("\nTFLOPS: %.2f   time_ms: %.3f\n", best_tflops,
           2.0 * M * N * K / (best_tflops * 1e12) * 1e3);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B;
    return 0;
}
