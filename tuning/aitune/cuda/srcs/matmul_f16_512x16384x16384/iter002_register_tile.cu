// iter002_register_tile.cu - Register-tiled matmul for improved arithmetic intensity
// Bottleneck addressed: memory_bandwidth (iter001 achieved only 24% of baseline)
// Idea: Larger tiles (128x128) with register blocking (8x8 per thread)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

constexpr int M = 512;
constexpr int N = 16384;
constexpr int K = 16384;

// Larger tile dimensions for better arithmetic intensity
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;

// Register tile: each thread computes TM x TN elements
constexpr int TM = 8;
constexpr int TN = 8;

// Thread block: (BM/TM) x (BN/TN) = 16 x 16 = 256 threads
constexpr int THREADS_X = BN / TN;  // 16
constexpr int THREADS_Y = BM / TM;  // 16

__global__ void matmul_register_tile(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ half As[BM][BK + 1];
    __shared__ half Bs[BK][BN + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Starting row/col for this thread's register tile
    int row_start = by * BM + ty * TM;
    int col_start = bx * BN + tx * TN;

    // Register accumulators
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Register fragments for A and B
    half a_frag[TM];
    half b_frag[TN];

    int num_tiles = (k + BK - 1) / BK;
    int tid = ty * THREADS_X + tx;
    int total_threads = THREADS_X * THREADS_Y;

    for (int tile = 0; tile < num_tiles; ++tile) {
        // Cooperative load of A tile into shared memory
        // Each thread loads multiple elements
        int a_loads = (BM * BK + total_threads - 1) / total_threads;
        for (int i = 0; i < a_loads; ++i) {
            int idx = tid + i * total_threads;
            if (idx < BM * BK) {
                int r = idx / BK;
                int c = idx % BK;
                int global_row = by * BM + r;
                int global_col = tile * BK + c;
                if (global_row < m && global_col < k) {
                    As[r][c] = A[global_row * k + global_col];
                } else {
                    As[r][c] = __float2half(0.0f);
                }
            }
        }

        // Cooperative load of B tile into shared memory
        int b_loads = (BK * BN + total_threads - 1) / total_threads;
        for (int i = 0; i < b_loads; ++i) {
            int idx = tid + i * total_threads;
            if (idx < BK * BN) {
                int r = idx / BN;
                int c = idx % BN;
                int global_row = tile * BK + r;
                int global_col = bx * BN + c;
                if (global_row < k && global_col < n) {
                    Bs[r][c] = B[global_row * n + global_col];
                } else {
                    Bs[r][c] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // Compute register tile
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // Load A fragment
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = As[ty * TM + i][kk];
            }
            // Load B fragment
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = Bs[kk][tx * TN + j];
            }
            // Accumulate
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += __half2float(a_frag[i]) * __half2float(b_frag[j]);
                }
            }
        }

        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = row_start + i;
        if (row < m) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
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
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);

    for (int i = 0; i < warmup; ++i) {
        matmul_register_tile<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        matmul_register_tile<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
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

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
