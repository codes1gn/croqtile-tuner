// iter001_draft.cu - First kernel draft for f16 matmul 512x16384x16384
// Pure CUDA implementation - NO library calls (cuBLAS, cuTLASS, etc.)
// This is the starting point for tuning iterations.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// Shape constants
constexpr int M = 512;
constexpr int N = 16384;
constexpr int K = 16384;

// Tile dimensions for shared memory blocking
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;

// Thread block configuration
constexpr int BLOCK_DIM_X = 16;
constexpr int BLOCK_DIM_Y = 16;

// Naive tiled matmul kernel with shared memory
// C[M,N] = A[M,K] @ B[K,N]
// All matrices in row-major: A[i,j] at A[i*K+j], B[i,j] at B[i*N+j], C[i,j] at C[i*N+j]
__global__ void matmul_f16_tiled(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    __shared__ half As[TILE_M][TILE_K];
    __shared__ half Bs[TILE_K][TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float acc = 0.0f;

    for (int t = 0; t < (k + TILE_K - 1) / TILE_K; ++t) {
        int a_col = t * TILE_K + tx;
        int b_row = t * TILE_K + ty;

        if (row < m && a_col < k) {
            As[ty][tx] = A[row * k + a_col];
        } else {
            As[ty][tx] = __float2half(0.0f);
        }

        if (b_row < k && col < n) {
            Bs[ty][tx] = B[b_row * n + col];
        } else {
            Bs[ty][tx] = __float2half(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_K; ++i) {
            acc += __half2float(As[ty][i]) * __half2float(Bs[i][tx]);
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = __float2half(acc);
    }
}

// Verification kernel using cuBLAS-equivalent computation (for reference only)
void init_matrix_random(half* mat, int rows, int cols, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < rows * cols; ++i) {
        float val = (float)(rand() % 1000) / 1000.0f - 0.5f;
        mat[i] = __float2half(val);
    }
}

float benchmark_kernel(
    const half* d_A, const half* d_B, half* d_C,
    int m, int n, int k,
    int warmup, int iters
) {
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);

    for (int i = 0; i < warmup; ++i) {
        matmul_f16_tiled<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        matmul_f16_tiled<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / iters;
}

int main(int argc, char** argv) {
    int warmup = 5;
    int iters = 10;

    if (argc > 1) warmup = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    size_t size_A = (size_t)M * K * sizeof(half);
    size_t size_B = (size_t)K * N * sizeof(half);
    size_t size_C = (size_t)M * N * sizeof(half);

    half *h_A = (half*)malloc(size_A);
    half *h_B = (half*)malloc(size_B);
    half *h_C = (half*)malloc(size_C);

    init_matrix_random(h_A, M, K, 42);
    init_matrix_random(h_B, K, N, 43);

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    float time_ms = benchmark_kernel(d_A, d_B, d_C, M, N, K, warmup, iters);

    double flops = 2.0 * M * N * K;
    double tflops = flops / (time_ms * 1e-3) / 1e12;

    printf("Shape: %dx%dx%d\n", M, N, K);
    printf("Time: %.3f ms\n", time_ms);
    printf("TFLOPS: %.2f\n", tflops);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
