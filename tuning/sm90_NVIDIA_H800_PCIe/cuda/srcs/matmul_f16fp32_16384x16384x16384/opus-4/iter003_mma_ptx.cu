#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

constexpr int M_DIM = 16384;
constexpr int N_DIM = 16384;
constexpr int K_DIM = 16384;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int BLOCK_SIZE = 256;

__device__ __forceinline__ void mma_m16n8k16_f16f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

__global__ void matmul_mma_kernel(const half* __restrict__ A,
                                   const half* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    
    const int warpM = warpId / 2;
    const int warpN = warpId % 2;
    
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    
    __shared__ __align__(128) half As[BK][BM];
    __shared__ __align__(128) half Bs[BK][BN];
    
    float acc[4][8] = {{0.0f}};
    
    for (int kBlock = 0; kBlock < K; kBlock += BK) {
        for (int idx = threadIdx.x; idx < BM * BK; idx += BLOCK_SIZE) {
            int m = idx % BM;
            int k = idx / BM;
            int globalM = blockRow + m;
            int globalK = kBlock + k;
            As[k][m] = (globalM < M && globalK < K) ? A[globalM * K + globalK] : __float2half(0.0f);
        }
        
        for (int idx = threadIdx.x; idx < BK * BN; idx += BLOCK_SIZE) {
            int n = idx % BN;
            int k = idx / BN;
            int globalN = blockCol + n;
            int globalK = kBlock + k;
            Bs[k][n] = (globalN < N && globalK < K) ? B[globalK * N + globalN] : __float2half(0.0f);
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int kStep = 0; kStep < BK; kStep += 16) {
            uint32_t a_reg[4][4];
            uint32_t b_reg[8][2];
            
            #pragma unroll
            for (int tm = 0; tm < 4; tm++) {
                int row = warpM * 32 + tm * 8 + (laneId % 8);
                int col = kStep + (laneId / 8) * 8;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_reg[tm][0]), "=r"(a_reg[tm][1]), "=r"(a_reg[tm][2]), "=r"(a_reg[tm][3])
                    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&As[col][row])))
                );
            }
            
            #pragma unroll
            for (int tn = 0; tn < 8; tn++) {
                int row = kStep + (laneId % 16);
                int col = warpN * 64 + tn * 8 + (laneId / 16) * 4;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_reg[tn][0]), "=r"(b_reg[tn][1])
                    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[row][col])))
                );
            }
            
            #pragma unroll
            for (int tm = 0; tm < 4; tm++) {
                #pragma unroll
                for (int tn = 0; tn < 8; tn++) {
                    mma_m16n8k16_f16f32(
                        acc[tm][tn * 4 / 4], acc[tm][tn * 4 / 4 + 1], acc[tm][tn * 4 / 4 + 2], acc[tm][tn * 4 / 4 + 3],
                        a_reg[tm][0], a_reg[tm][1], a_reg[tm][2], a_reg[tm][3],
                        b_reg[tn][0], b_reg[tn][1],
                        acc[tm][tn * 4 / 4], acc[tm][tn * 4 / 4 + 1], acc[tm][tn * 4 / 4 + 2], acc[tm][tn * 4 / 4 + 3]
                    );
                }
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int tm = 0; tm < 4; tm++) {
        int outRow = blockRow + warpM * 32 + tm * 8 + (laneId / 4);
        #pragma unroll
        for (int tn = 0; tn < 8; tn++) {
            int outCol = blockCol + warpN * 64 + tn * 8 + (laneId % 4) * 2;
            
            if (outRow < M && outCol + 1 < N) {
                C[outRow * N + outCol] = acc[tm][tn];
                C[outRow * N + outCol + 1] = acc[tm][tn];
            }
        }
    }
}

void matmul_cuda(const half* A, const half* B, float* C, int M, int N, int K) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_mma_kernel<<<grid, block>>>(A, B, C, M, N, K);
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
    bool pass = (max_err < 1.0f || max_rel_err < 0.01f);
    printf("VERIFY: %s max_abs_err=%.6f max_rel_err=%.6f\n", pass ? "PASS" : "FAIL", max_err, max_rel_err);
    return pass;
}

int main() {
    const int M = M_DIM, N = N_DIM, K = K_DIM;
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
    
    half *d_A, *d_B; float *d_C, *d_C_ref;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_C_ref, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    matmul_cublas_ref(d_A, d_B, d_C_ref, M, N, K, handle);
    matmul_cuda(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost);
    
    if (!verify(h_C, h_C_ref, M * N)) {
        cublasDestroy(handle);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        return 1;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < 10; i++) matmul_cuda(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 50; i++) matmul_cuda(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double tflops = (2.0 * M * N * K / (ms / 50.0 / 1000.0)) / 1e12;
    
    printf("TFLOPS: %.3f   time_ms: %.4f\n", tflops, ms / 50.0);
    
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return 0;
}
