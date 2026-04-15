// iter000_baseline.cu - cuBLAS baseline for bf16 input -> fp32 output matmul
// Shape: M=512, N=16384, K=16384
// This baseline uses cuBLAS GemmEx with BF16 inputs and FP32 output/compute

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
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

constexpr int M = 512;
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

void compute_reference_fp32(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += __bfloat162float(A[i * k + l]) * __bfloat162float(B[l * n + j]);
            }
            C[i * n + j] = sum;
        }
    }
}

bool verify_output(const float* ref, const float* out, int size, float atol = 1e-2f, float rtol = 1e-2f) {
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int errors = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(ref[i] - out[i]);
        float rel = (fabsf(ref[i]) > 1e-6f) ? diff / fabsf(ref[i]) : diff;
        max_abs_err = fmaxf(max_abs_err, diff);
        max_rel_err = fmaxf(max_rel_err, rel);
        if (diff > atol + rtol * fabsf(ref[i])) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at %d: ref=%f, out=%f, diff=%f\n", i, ref[i], out[i], diff);
            }
        }
    }
    printf("max_abs_err=%e max_rel_err=%e errors=%d/%d\n", max_abs_err, max_rel_err, errors, size);
    return errors == 0;
}

int main(int argc, char** argv) {
    bool verify_mode = (argc > 1 && std::string(argv[1]) == "--verify");
    
    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;

    __nv_bfloat16 *h_A, *h_B;
    float *h_C, *h_ref;
    __nv_bfloat16 *d_A, *d_B;
    float *d_C;

    h_A = new __nv_bfloat16[size_A];
    h_B = new __nv_bfloat16[size_B];
    h_C = new float[size_C];
    h_ref = new float[size_C];

    init_bf16_random(h_A, size_A, 42);
    init_bf16_random(h_B, size_B, 123);

    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, size_C * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS uses column-major, so we compute C = B^T * A^T to get row-major C = A * B
    // Actually, we'll use the standard approach: C = A * B in row-major = B^T * A^T in col-major
    // cublasGemmEx(handle, transB, transA, N, M, K, alpha, B, K, A, K, beta, C, N)
    
    // Warmup
    for (int i = 0; i < WARMUP; i++) {
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
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        // Compute CPU reference
        compute_reference_fp32(h_A, h_B, h_ref, M, N, K);
        
        // Run once and verify
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
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Copy back - note: cuBLAS gives us transposed result due to col-major
        float* h_C_transposed = new float[size_C];
        CHECK_CUDA(cudaMemcpy(h_C_transposed, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Transpose back to row-major
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                h_C[i * N + j] = h_C_transposed[j * M + i];
            }
        }
        delete[] h_C_transposed;
        
        bool passed = verify_output(h_ref, h_C, size_C);
        printf("verification: %s\n", passed ? "PASSED" : "FAILED");
        
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_ref;
        
        return passed ? 0 : 1;
    }

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double tflops_sum = 0.0;
    double tflops_min = 1e9;
    double tflops_max = 0.0;

    for (int s = 0; s < SAMPLES; s++) {
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++) {
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

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        double time_per_iter_ms = ms / ITERS;
        double flops = 2.0 * M * N * K;
        double tflops = flops / (time_per_iter_ms * 1e-3) / 1e12;

        printf("sample %d: time=%.3f ms, tflops=%.2f\n", s + 1, time_per_iter_ms, tflops);
        tflops_sum += tflops;
        tflops_min = fmin(tflops_min, tflops);
        tflops_max = fmax(tflops_max, tflops);
    }

    double tflops_avg = tflops_sum / SAMPLES;
    printf("\n=== BASELINE RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M, N, K);
    printf("dtype: bf16 input, fp32 output\n");
    printf("avg_tflops: %.2f\n", tflops_avg);
    printf("min_tflops: %.2f\n", tflops_min);
    printf("max_tflops: %.2f\n", tflops_max);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_ref;

    return 0;
}
