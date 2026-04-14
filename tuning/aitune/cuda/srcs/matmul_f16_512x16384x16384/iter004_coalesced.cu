// iter004_coalesced.cu - Coalesced memory access with optimized tile sizes
// Focus on memory coalescing and avoiding bank conflicts

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

constexpr int M = 512;
constexpr int N = 16384;
constexpr int K = 16384;

// Optimal tile size for Ampere: 64x64x16
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;

// Thread block: 16x16 = 256 threads (8 warps)
constexpr int BLOCK_DIM_X = 16;
constexpr int BLOCK_DIM_Y = 16;

// Each thread computes a 4x4 sub-tile
constexpr int THREAD_TILE_M = 4;
constexpr int THREAD_TILE_N = 4;

__global__ void matmul_coalesced(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ half As[TILE_M][TILE_K + 2];
    __shared__ half Bs[TILE_K][TILE_N + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Global starting positions
    const int row_start = by * TILE_M + ty * THREAD_TILE_M;
    const int col_start = bx * TILE_N + tx * THREAD_TILE_N;

    // Register accumulators
    float acc[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    // Precompute thread position for loading
    const int tid = ty * BLOCK_DIM_X + tx;
    
    for (int tile = 0; tile < (k + TILE_K - 1) / TILE_K; ++tile) {
        // Load A: each thread loads 4 elements (coalesced along K dimension)
        // Total: 256 threads * 4 elements = 1024 = 64 * 16 ✓
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = (tid / TILE_K) * 4 + i;
            int col = tid % TILE_K;
            int g_row = by * TILE_M + row;
            int g_col = tile * TILE_K + col;
            As[row][col] = (g_row < m && g_col < k) ? A[g_row * k + g_col] : __float2half(0.0f);
        }

        // Load B: each thread loads 4 elements (coalesced along N dimension)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = tid / TILE_N;
            int col = (tid % TILE_N / 4) * 4 + i;
            if (tid < TILE_K * TILE_N / 4) {
                int actual_row = (tid * 4) / TILE_N;
                int actual_col = (tid * 4 + i) % TILE_N;
                int g_row = tile * TILE_K + actual_row;
                int g_col = bx * TILE_N + actual_col;
                Bs[actual_row][actual_col] = (g_row < k && g_col < n) ? B[g_row * n + g_col] : __float2half(0.0f);
            }
        }

        __syncthreads();

        // Compute: each thread computes 4x4 output
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a[THREAD_TILE_M];
            float b[THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                a[i] = __half2float(As[ty * THREAD_TILE_M + i][kk]);
            }

            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                b[j] = __half2float(Bs[kk][tx * THREAD_TILE_N + j]);
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        int row = row_start + i;
        if (row < m) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                int col = col_start + j;
                if (col < n) {
                    C[row * n + col] = __float2half(acc[i][j]);
                }
            }
        }
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
        matmul_coalesced<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        matmul_coalesced<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / iters;
}

void init_matrix_random(half* mat, int rows, int cols, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < rows * cols; ++i) {
        float val = (float)(rand() % 1000) / 1000.0f - 0.5f;
        mat[i] = __float2half(val);
    }
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

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
