#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_pipeline.h>
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

// Reduce BM from 256 to 128 to reduce smem and register pressure
// Keep BN at 128, reduce BK to 16 for smaller tiles
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 8;  // Reduce from 16 to 8 warps
constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;  // 256 threads

// 8 warps covering 128x128 output tile
// 4 warps in M, 2 warps in N
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
// Each warp computes 2x4 WMMA tiles = 32x64 output
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 4;
constexpr int WARP_TILES_K = BK / WMMA_K;  // 1

__global__ void matmul_wmma_kernel(const half* __restrict__ A,
                                    const half* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    const int warpId = threadIdx.x / WARP_SIZE;
    const int tid = threadIdx.x;
    
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    
    // Smaller smem footprint: 128*24 + 16*136 = 3072 + 2176 = 5248 bytes
    __shared__ half As[BM][BK + 8];
    __shared__ half Bs[BK][BN + 8];
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_TILES_M][WARP_TILES_K];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WARP_TILES_N][WARP_TILES_K];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];
    
    #pragma unroll
    for (int m = 0; m < WARP_TILES_M; m++) {
        #pragma unroll
        for (int n = 0; n < WARP_TILES_N; n++) {
            wmma::fill_fragment(c_frag[m][n], 0.0f);
        }
    }
    
    // Warp arrangement: 4 warps in M direction, 2 in N
    const int warpRow = (warpId / WARPS_N) * (WARP_TILES_M * WMMA_M);  // 0, 32, 64, 96
    const int warpCol = (warpId % WARPS_N) * (WARP_TILES_N * WMMA_N);  // 0, 64
    
    // Load patterns for 256 threads loading BM=128 x BK=16 = 2048 halfs
    // Each thread loads 8 halfs (one float4)
    const int loadARow = tid / 2;   // 256/2 = 128 unique rows
    const int loadACol = (tid % 2) * 8;  // 0 or 8, covers 16 cols
    
    // Load patterns for 256 threads loading BK=16 x BN=128 = 2048 halfs
    const int loadBRow = tid / 16;  // 256/16 = 16 unique rows
    const int loadBCol = (tid % 16) * 8;  // covers 128 cols
    
    for (int kBlock = 0; kBlock < K; kBlock += BK) {
        // Load A tile: 128 x 16
        int row = loadARow;
        int globalM = blockRow + row;
        int globalK = kBlock + loadACol;
        
        if (globalM < M && globalK + 7 < K) {
            __pipeline_memcpy_async(
                reinterpret_cast<float4*>(&As[row][loadACol]),
                reinterpret_cast<const float4*>(&A[globalM * K + globalK]),
                sizeof(float4)
            );
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int gk = globalK + i;
                As[row][loadACol + i] = (globalM < M && gk < K) ? A[globalM * K + gk] : __float2half(0.0f);
            }
        }
        
        // Load B tile: 16 x 128 (each thread loads 1 float4)
        int bRow = loadBRow;
        int globalKB = kBlock + bRow;
        int globalN = blockCol + loadBCol;
        
        if (globalKB < K && globalN + 7 < N) {
            __pipeline_memcpy_async(
                reinterpret_cast<float4*>(&Bs[bRow][loadBCol]),
                reinterpret_cast<const float4*>(&B[globalKB * N + globalN]),
                sizeof(float4)
            );
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int gn = globalN + i;
                Bs[bRow][loadBCol + i] = (globalKB < K && gn < N) ? B[globalKB * N + gn] : __float2half(0.0f);
            }
        }
        
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        
        // Load A fragments
        #pragma unroll
        for (int tm = 0; tm < WARP_TILES_M; tm++) {
            #pragma unroll
            for (int tk = 0; tk < WARP_TILES_K; tk++) {
                wmma::load_matrix_sync(a_frag[tm][tk], &As[warpRow + tm * WMMA_M][tk * WMMA_K], BK + 8);
            }
        }
        
        // Load B fragments
        #pragma unroll
        for (int tn = 0; tn < WARP_TILES_N; tn++) {
            #pragma unroll
            for (int tk = 0; tk < WARP_TILES_K; tk++) {
                wmma::load_matrix_sync(b_frag[tn][tk], &Bs[tk * WMMA_K][warpCol + tn * WMMA_N], BN + 8);
            }
        }
        
        // Compute
        #pragma unroll
        for (int tk = 0; tk < WARP_TILES_K; tk++) {
            #pragma unroll
            for (int tm = 0; tm < WARP_TILES_M; tm++) {
                #pragma unroll
                for (int tn = 0; tn < WARP_TILES_N; tn++) {
                    wmma::mma_sync(c_frag[tm][tn], a_frag[tm][tk], b_frag[tn][tk], c_frag[tm][tn]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int tm = 0; tm < WARP_TILES_M; tm++) {
        #pragma unroll
        for (int tn = 0; tn < WARP_TILES_N; tn++) {
            int outRow = blockRow + warpRow + tm * WMMA_M;
            int outCol = blockCol + warpCol + tn * WMMA_N;
            if (outRow < M && outCol < N) {
                wmma::store_matrix_sync(&C[outRow * N + outCol], c_frag[tm][tn], N, wmma::mem_row_major);
            }
        }
    }
}

void matmul_cuda(const half* A, const half* B, float* C, int M, int N, int K) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_wmma_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

void matmul_cublas_ref(const half* A, const half* B, float* C, int M, int N, int K, cublasHandle_t handle) {
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

bool verify(float* C_test, float* C_ref, int size) {
    float max_err = 0.0f, max_rel_err = 0.0f;
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
    cudaMalloc(&d_A, size_A); cudaMalloc(&d_B, size_B); cudaMalloc(&d_C, size_C); cudaMalloc(&d_C_ref, size_C);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    cublasHandle_t handle; cublasCreate(&handle);
    matmul_cublas_ref(d_A, d_B, d_C_ref, M, N, K, handle);
    matmul_cuda(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost);
    
    if (!verify(h_C, h_C_ref, M * N)) {
        cublasDestroy(handle); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
        free(h_A); free(h_B); free(h_C); free(h_C_ref);
        return 1;
    }
    
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < 10; i++) matmul_cuda(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 50; i++) matmul_cuda(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    double tflops = (2.0 * M * N * K / (ms / 50.0 / 1000.0)) / 1e12;
    printf("TFLOPS: %.3f   time_ms: %.4f\n", tflops, ms / 50.0);
    
    cudaEventDestroy(start); cudaEventDestroy(stop); cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return 0;
}
