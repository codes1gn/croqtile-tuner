// iter007_wmma_multi.cu - WMMA with multiple fragments per warp
// Idea: Each warp computes multiple 16x16 output tiles for better ILP

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

constexpr int M = 512;
constexpr int N = 16384;
constexpr int K = 16384;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Each warp computes 2x2 = 4 WMMA tiles = 32x32 output
constexpr int WARP_TILE_M = 2;
constexpr int WARP_TILE_N = 2;

// Block tile: 64x128 with 8 warps (2x4 arrangement)
constexpr int TILE_M = 64;
constexpr int TILE_N = 128;
constexpr int TILE_K = 16;

constexpr int WARPS_X = 4;
constexpr int WARPS_Y = 2;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = WARPS_X * WARPS_Y * WARP_SIZE;

__global__ void matmul_wmma_multi(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    __shared__ half As[TILE_M][TILE_K + 8];
    __shared__ half Bs[TILE_K][TILE_N + 8];

    int warpId = threadIdx.x / WARP_SIZE;
    int warpX = warpId % WARPS_X;
    int warpY = warpId / WARPS_X;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Each warp's output position (32x32 per warp)
    int warp_row = by * TILE_M + warpY * WARP_TILE_M * WMMA_M;
    int warp_col = bx * TILE_N + warpX * WARP_TILE_N * WMMA_N;

    // 4 sets of WMMA fragments per warp
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WARP_TILE_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILE_M][WARP_TILE_N];

    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int tid = threadIdx.x;
    int num_tiles = (k + TILE_K - 1) / TILE_K;

    for (int tile = 0; tile < num_tiles; ++tile) {
        // Load A tile: 64 x 16 = 1024 elements, 256 threads -> 4 per thread
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid * 4 + i;
            if (idx < TILE_M * TILE_K) {
                int r = idx / TILE_K;
                int c = idx % TILE_K;
                int gr = by * TILE_M + r;
                int gc = tile * TILE_K + c;
                As[r][c] = (gr < m && gc < k) ? A[gr * k + gc] : __float2half(0.0f);
            }
        }

        // Load B tile: 16 x 128 = 2048 elements, 256 threads -> 8 per thread
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid * 8 + i;
            if (idx < TILE_K * TILE_N) {
                int r = idx / TILE_N;
                int c = idx % TILE_N;
                int gr = tile * TILE_K + r;
                int gc = bx * TILE_N + c;
                Bs[r][c] = (gr < k && gc < n) ? B[gr * n + gc] : __float2half(0.0f);
            }
        }

        __syncthreads();

        // Load A fragments (2 per warp)
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int a_row = warpY * WARP_TILE_M * WMMA_M + i * WMMA_M;
            if (warp_row + i * WMMA_M < m) {
                wmma::load_matrix_sync(a_frag[i], &As[a_row][0], TILE_K + 8);
            }
        }

        // Load B fragments (2 per warp)
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int b_col = warpX * WARP_TILE_N * WMMA_N + j * WMMA_N;
            if (warp_col + j * WMMA_N < n) {
                wmma::load_matrix_sync(b_frag[j], &Bs[0][b_col], TILE_N + 8);
            }
        }

        // Compute 4 WMMA operations
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            #pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                if (warp_row + i * WMMA_M < m && warp_col + j * WMMA_N < n) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int out_row = warp_row + i * WMMA_M;
            int out_col = warp_col + j * WMMA_N;
            if (out_row < m && out_col < n) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_f16;
                for (int e = 0; e < c_frag[i][j].num_elements; ++e) {
                    c_f16.x[e] = __float2half(c_frag[i][j].x[e]);
                }
                wmma::store_matrix_sync(C + out_row * n + out_col, c_f16, n, wmma::mem_row_major);
            }
        }
    }
}

float benchmark_kernel(
    const half* d_A, const half* d_B, half* d_C,
    int m, int n, int k,
    int warmup, int iters
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);

    for (int i = 0; i < warmup; ++i) {
        matmul_wmma_multi<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        matmul_wmma_multi<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
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
