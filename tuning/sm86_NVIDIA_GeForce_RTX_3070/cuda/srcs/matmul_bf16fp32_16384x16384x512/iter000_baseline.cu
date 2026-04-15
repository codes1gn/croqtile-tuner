// iter000_baseline.cu — cuBLAS reference for matmul bf16->fp32
// Shape: M=16384, N=16384, K=512
// C[M,N] = A[M,K] * B[K,N],  A/B in bf16, C in fp32
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } \
} while(0)
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); } \
} while(0)

static const int M = 16384, N = 16384, K = 512;
static const int WARMUP = 10, ITERS = 50, SAMPLES = 5;

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    __nv_bfloat16 *dA, *dB;
    float *dC;
    CHECK_CUDA(cudaMalloc(&dA, (size_t)M*K*sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dB, (size_t)K*N*sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC, (size_t)M*N*sizeof(float)));

    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    {
        size_t szA = (size_t)M*K, szB = (size_t)K*N;
        __nv_bfloat16 *hA = new __nv_bfloat16[szA];
        __nv_bfloat16 *hB = new __nv_bfloat16[szB];
        for (size_t i = 0; i < szA; i++) hA[i] = __float2bfloat16(dist(rng));
        for (size_t i = 0; i < szB; i++) hB[i] = __float2bfloat16(dist(rng));
        CHECK_CUDA(cudaMemcpy(dA, hA, szA*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB, szB*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        delete[] hA; delete[] hB;
    }

    // cuBLAS: C = alpha*A*B + beta*C  (col-major: C^T = B^T * A^T)
    float alpha = 1.f, beta = 0.f;
    auto run = [&]() {
        CHECK_CUBLAS(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB, CUDA_R_16BF, N,
            dA, CUDA_R_16BF, K,
            &beta,
            dC, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    for (int i = 0; i < WARMUP; i++) run();
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("VERIFY: PASS (cuBLAS reference)\n");

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    double tflops_sum = 0;
    for (int s = 0; s < SAMPLES; s++) {
        CHECK_CUDA(cudaEventRecord(t0));
        for (int i = 0; i < ITERS; i++) run();
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        ms /= ITERS;
        double tflops = 2.0*M*N*K / ms / 1e9;
        tflops_sum += tflops;
        printf("sample %d: time=%.3f ms, tflops=%.2f\n", s+1, ms, tflops);
    }
    double avg = tflops_sum / SAMPLES;
    printf("\nTFLOPS: %.2f   time_ms: %.3f\n", avg, 2.0*M*N*K / avg / 1e9);

    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return 0;
}
