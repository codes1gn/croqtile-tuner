#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;

constexpr int M_DIM = 16384;
constexpr int N_DIM = 16384;
constexpr int K_DIM = 16384;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = BM / WMMA_M;  // 4
constexpr int WARPS_N = BN / WMMA_N;  // 4
constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 16
constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;  // 512

__global__ void matmul_wmma_kernel(const half* __restrict__ A,
                                    const half* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;
    
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    
    __shared__ half As[BM][BK + 8];
    __shared__ half Bs[BK][BN + 8];
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    const int tid = threadIdx.x;
    const int loadA_count = (BM * BK + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int loadB_count = (BK * BN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int kBlock = 0; kBlock < K; kBlock += BK) {
        #pragma unroll
        for (int i = 0; i < loadA_count; i++) {
            int idx = tid + i * BLOCK_SIZE;
            if (idx < BM * BK) {
                int m = idx / BK;
                int k = idx % BK;
                int globalM = blockRow + m;
                int globalK = kBlock + k;
                if (globalM < M && globalK < K) {
                    As[m][k] = A[globalM * K + globalK];
                } else {
                    As[m][k] = __float2half(0.0f);
                }
            }
        }
        
        #pragma unroll
        for (int i = 0; i < loadB_count; i++) {
            int idx = tid + i * BLOCK_SIZE;
            if (idx < BK * BN) {
                int k = idx / BN;
                int n = idx % BN;
                int globalN = blockCol + n;
                int globalK = kBlock + k;
                if (globalN < N && globalK < K) {
                    Bs[k][n] = B[globalK * N + globalN];
                } else {
                    Bs[k][n] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        
        int fragRow = warpM * WMMA_M;
        int fragCol = warpN * WMMA_N;
        
        wmma::load_matrix_sync(a_frag, &As[fragRow][0], BK + 8);
        wmma::load_matrix_sync(b_frag, &Bs[0][fragCol], BN + 8);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        __syncthreads();
    }
    
    int outRow = blockRow + warpM * WMMA_M;
    int outCol = blockCol + warpN * WMMA_N;
    if (outRow < M && outCol < N) {
        wmma::store_matrix_sync(&C[outRow * N + outCol], c_frag, N, wmma::mem_row_major);
    }
}

void matmul_cuda(const half* A, const half* B, float* C, int M, int N, int K) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_wmma_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

void matmul_cublas_ref(const half* A, const half* B, float* C, int M, int N, int K, cublasHandle_t handle) {
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 B, CUDA_R_16F, N,
                 A, CUDA_R_16F, K,
                 &beta,
                 C, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

bool verify(float* C_test, float* C_ref, int size) {
    float max_err = 0.0f;
    float max_rel_err = 0.0f;
    for (int i = 0; i < size; i++) {
        float err = fabsf(C_test[i] - C_ref[i]);
        float rel_err = err / (fabsf(C_ref[i]) + 1e-6f);
        if (err > max_err) max_err = err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }
    float base_tol = 1.0f;
    float rel_tol = 0.01f;
    bool pass = (max_err < base_tol || max_rel_err < rel_tol);
    if (!pass) {
        printf("VERIFY: FAIL max_abs_err=%.6f max_rel_err=%.6f\n", max_err, max_rel_err);
    } else {
        printf("VERIFY: PASS max_abs_err=%.6f max_rel_err=%.6f\n", max_err, max_rel_err);
    }
    return pass;
}

int main(int argc, char** argv) {
    const int M = M_DIM;
    const int N = N_DIM;
    const int K = K_DIM;
    
    size_t size_A = (size_t)M * K * sizeof(half);
    size_t size_B = (size_t)K * N * sizeof(half);
    size_t size_C = (size_t)M * N * sizeof(float);
    
    half *h_A = (half*)malloc(size_A);
    half *h_B = (half*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);
    
    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++) h_A[i] = __float2half((rand() % 100 - 50) / 50.0f);
    for (size_t i = 0; i < (size_t)K * N; i++) h_B[i] = __float2half((rand() % 100 - 50) / 50.0f);
    
    half *d_A, *d_B;
    float *d_C, *d_C_ref;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_C_ref, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    matmul_cublas_ref(d_A, d_B, d_C_ref, M, N, K, handle);
    cudaDeviceSynchronize();
    
    matmul_cuda(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost);
    
    bool passed = verify(h_C, h_C_ref, M * N);
    if (!passed) {
        cublasDestroy(handle);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        return 1;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < 10; i++) {
        matmul_cuda(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 50; i++) {
        matmul_cuda(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / 50.0f;
    
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    
    printf("TFLOPS: %.3f   time_ms: %.4f\n", tflops, avg_ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    
    return 0;
}
