// baseline_cublas.cu - cuBLAS baseline for matmul bf16->fp32
// Used ONLY for baseline measurement, NOT for tuning iterations

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char** argv) {
    const int M = 512;
    const int N = 16384;
    const int K = 16384;
    
    int warmup = 10;
    int iters = 20;
    if (argc >= 2) warmup = atoi(argv[1]);
    if (argc >= 3) iters = atoi(argv[2]);
    
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    
    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B;
    float *d_C;
    
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Initialize with random values
    __nv_bfloat16* h_A = new __nv_bfloat16[M * K];
    __nv_bfloat16* h_B = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; i++) h_A[i] = __float2bfloat16((float)(rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2bfloat16((float)(rand() % 100) / 100.0f);
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        // cublasGemmEx: C = alpha * A * B + beta * C
        // Note: cuBLAS uses column-major, so we compute B^T * A^T = (A * B)^T
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,  // Note: reversed for row-major
            &alpha,
            d_B, CUDA_R_16BF, N,  // B is K x N in row-major = N x K in col-major
            d_A, CUDA_R_16BF, K,  // A is M x K in row-major = K x M in col-major
            &beta,
            d_C, CUDA_R_32F, N,   // C is M x N in row-major = N x M in col-major
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, CUDA_R_16BF, N,
            d_A, CUDA_R_16BF, K,
            &beta,
            d_C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;
    
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;
    
    printf("cuBLAS bf16->fp32: %.3f ms, %.2f TFLOPS\n", avg_ms, tflops);
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    
    return 0;
}
