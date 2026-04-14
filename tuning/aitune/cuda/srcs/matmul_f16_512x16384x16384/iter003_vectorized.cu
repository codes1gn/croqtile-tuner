// iter003_vectorized.cu - Vectorized matmul with half2 loads
// Idea: Use half2 for coalesced 4-byte loads, moderate tiles (64x64)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

constexpr int M = 512;
constexpr int N = 16384;
constexpr int K = 16384;

// Moderate tile size with vectorization
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;

// Thread configuration: 32 warps per block for better occupancy
constexpr int BLOCK_DIM_X = 32;
constexpr int BLOCK_DIM_Y = 8;

__global__ void matmul_vectorized(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    // Shared memory
    __shared__ half As[TILE_M][TILE_K + 1];
    __shared__ half Bs[TILE_K][TILE_N + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Each thread computes a 2x2 tile of output
    int row0 = by * TILE_M + ty * (TILE_M / BLOCK_DIM_Y);
    int col0 = bx * TILE_N + tx * 2;

    float acc[8][2] = {0};  // 8 rows, 2 cols per thread

    int tid = ty * BLOCK_DIM_X + tx;
    int total_threads = BLOCK_DIM_X * BLOCK_DIM_Y;

    for (int tile = 0; tile < (k + TILE_K - 1) / TILE_K; ++tile) {
        // Load A tile - each thread loads one element
        for (int offset = 0; offset < TILE_M * TILE_K; offset += total_threads) {
            int idx = tid + offset;
            if (idx < TILE_M * TILE_K) {
                int r = idx / TILE_K;
                int c = idx % TILE_K;
                int gr = by * TILE_M + r;
                int gc = tile * TILE_K + c;
                As[r][c] = (gr < m && gc < k) ? A[gr * k + gc] : __float2half(0.0f);
            }
        }

        // Load B tile
        for (int offset = 0; offset < TILE_K * TILE_N; offset += total_threads) {
            int idx = tid + offset;
            if (idx < TILE_K * TILE_N) {
                int r = idx / TILE_N;
                int c = idx % TILE_N;
                int gr = tile * TILE_K + r;
                int gc = bx * TILE_N + c;
                Bs[r][c] = (gr < k && gc < n) ? B[gr * n + gc] : __float2half(0.0f);
            }
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            half2 b_vec = *reinterpret_cast<const half2*>(&Bs[kk][tx * 2]);
            float b0 = __half2float(b_vec.x);
            float b1 = __half2float(b_vec.y);

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float a_val = __half2float(As[ty * 8 + i][kk]);
                acc[i][0] += a_val * b0;
                acc[i][1] += a_val * b1;
            }
        }

        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int row = row0 + i;
        if (row < m) {
            if (col0 < n) {
                C[row * n + col0] = __float2half(acc[i][0]);
            }
            if (col0 + 1 < n) {
                C[row * n + col0 + 1] = __float2half(acc[i][1]);
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
        matmul_vectorized<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        matmul_vectorized<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
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
