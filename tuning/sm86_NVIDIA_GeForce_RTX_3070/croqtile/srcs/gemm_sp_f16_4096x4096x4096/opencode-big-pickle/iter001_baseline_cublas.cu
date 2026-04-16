// Dense GEMM f16 baseline using cuBLAS
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define RTX3070_PEAK_F16_TFLOPS 19.5

int main(int argc, char** argv) {
    bool enable_timing = true;
    bool skip_verify = false;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--disable-timing") == 0) {
            enable_timing = false;
        } else if (strcmp(argv[i], "--skip-verify") == 0) {
            skip_verify = true;
        }
    }

    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate host memory
    __half *A_h, *B_h, *C_h, *D_h;
    size_t A_size = M * K * sizeof(__half);
    size_t B_size = N * K * sizeof(__half);
    size_t C_size = M * N * sizeof(__half);
    
    cudaMallocHost(&A_h, A_size);
    cudaMallocHost(&B_h, B_size);
    cudaMallocHost(&C_h, C_size);
    cudaMallocHost(&D_h, M * N * sizeof(__half));

    // Initialize with seed
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < M * K; ++i) {
        A_h[i] = __float2half(dist(gen));
    }
    for (int i = 0; i < N * K; ++i) {
        B_h[i] = __float2half(dist(gen));
    }
    for (int i = 0; i < M * N; ++i) {
        C_h[i] = __float2half(0.0f);
        D_h[i] = __float2half(0.0f);
    }

    // Allocate device memory
    __half *A_d, *B_d, *C_d, *D_d;
    cudaMalloc(&A_d, A_size);
    cudaMalloc(&B_d, B_size);
    cudaMalloc(&C_d, C_size);
    cudaMalloc(&D_d, M * N * sizeof(__half));

    cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);
    cudaMemset(D_d, 0, M * N * sizeof(__half));

    if (enable_timing) {
        int warmup = 10;
        int repeat = 50;
        
        for (int i = 0; i < warmup; ++i) {
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                         N, M, K,
                         &alpha, B_d, CUDA_R_16F, N,
                                 A_d, CUDA_R_16F, M,
                         &beta, D_d, CUDA_R_16F, N,
                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        }
        cudaDeviceSynchronize();
        
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        
        cudaEventRecord(start);
        for (int i = 0; i < repeat; ++i) {
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                         N, M, K,
                         &alpha, B_d, CUDA_R_16F, N,
                                 A_d, CUDA_R_16F, M,
                         &beta, D_d, CUDA_R_16F, N,
                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        
        float ms;
        cudaEventElapsedTime(&ms, start, end);
        ms /= repeat;
        
        double flops = 2.0 * double(M) * double(N) * double(K);
        double tflops = (flops / (ms / 1000.0)) / 1e12;
        
        std::cout << "Timing avg ms: " << ms << "\n";
        std::cout << "TFLOPS: " << tflops << "\n";
        std::cout << "HW efficiency: " << (tflops / RTX3070_PEAK_F16_TFLOPS) * 100.0 << "%\n";
    }

    if (skip_verify) {
        std::cout << "Test Passed (verify skipped)\n";
        cublasDestroy(handle);
        cudaFree(A_d); cudaFree(B_d); cudaFree(C_d); cudaFree(D_d);
        cudaFreeHost(A_h); cudaFreeHost(B_h); cudaFreeHost(C_h); cudaFreeHost(D_h);
        return 0;
    }

    // Run GEMM for verification
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                 N, M, K,
                 &alpha, B_d, CUDA_R_16F, N,
                         A_d, CUDA_R_16F, M,
                 &beta, D_d, CUDA_R_16F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    // Verify
    cudaMemcpy(D_h, D_d, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float tolerance = 0.5f;
    int errors = 0;
    size_t verify_m = 128;
    size_t verify_n = 256;
    
    for (size_t i = 0; i < verify_m && errors < 8; ++i) {
        for (size_t j = 0; j < verify_n && errors < 8; ++j) {
            float ref = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                ref += __half2float(A_h[i * K + k]) * __half2float(B_h[j * K + k]);
            }
            float got = __half2float(D_h[i * N + j]);
            float diff = std::abs(got - ref);
            if (diff > tolerance) {
                std::cout << "[" << i << ", " << j << "] ref=" << ref
                          << " got=" << got << " diff=" << diff << std::endl;
                ++errors;
            }
        }
    }
    
    std::cout << "f16_gemm: " << errors << " errors\n";
    if (errors == 0) {
        std::cout << "Test Passed\n";
    } else {
        std::cout << "Test FAILED\n";
    }

    cublasDestroy(handle);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d); cudaFree(D_d);
    cudaFreeHost(A_h); cudaFreeHost(B_h); cudaFreeHost(C_h); cudaFreeHost(D_h);
    return errors;
}
