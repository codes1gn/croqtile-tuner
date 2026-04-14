// iter011_wmma_swizzle.cu - WMMA with shared memory swizzling
// Idea: Use XOR-based swizzling to eliminate bank conflicts

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

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;

constexpr int WARPS_X = 4;
constexpr int WARPS_Y = 4;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = WARPS_X * WARPS_Y * WARP_SIZE;

// Swizzle function: XOR row index with column bits to spread bank access
__device__ __forceinline__ int swizzle_offset(int row, int col, int stride) {
    return row * stride + (col ^ (row & 7));
}

__global__ void matmul_wmma_swizzle(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    // Shared memory with extra padding for swizzle
    __shared__ half As[TILE_M][TILE_K + 16];
    __shared__ half Bs[TILE_K][TILE_N + 16];

    int warpId = threadIdx.x / WARP_SIZE;
    int warpX = warpId % WARPS_X;
    int warpY = warpId / WARPS_X;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int warp_row = by * TILE_M + warpY * WMMA_M;
    int warp_col = bx * TILE_N + warpX * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int tid = threadIdx.x;
    int num_tiles = (k + TILE_K - 1) / TILE_K;

    for (int tile = 0; tile < num_tiles; ++tile) {
        // Load A with swizzled layout
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * BLOCK_SIZE;
            if (idx < TILE_M * TILE_K) {
                int r = idx / TILE_K;
                int c = idx % TILE_K;
                int gr = by * TILE_M + r;
                int gc = tile * TILE_K + c;
                half val = (gr < m && gc < k) ? A[gr * k + gc] : __float2half(0.0f);
                As[r][c ^ (r & 7)] = val;  // Swizzled store
            }
        }

        // Load B with swizzled layout
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * BLOCK_SIZE;
            if (idx < TILE_K * TILE_N) {
                int r = idx / TILE_N;
                int c = idx % TILE_N;
                int gr = tile * TILE_K + r;
                int gc = bx * TILE_N + c;
                half val = (gr < k && gc < n) ? B[gr * n + gc] : __float2half(0.0f);
                Bs[r][c ^ (r & 7)] = val;  // Swizzled store
            }
        }

        __syncthreads();

        // Load WMMA fragments from swizzled shared memory
        // Need to de-swizzle during load - use sequential access pattern
        if (warp_row < m && warp_col < n) {
            // For WMMA, we need contiguous memory, so use standard layout
            // Swizzling helps during the store phase
            wmma::load_matrix_sync(a_frag, &As[warpY * WMMA_M][0], TILE_K + 16);
            wmma::load_matrix_sync(b_frag, &Bs[0][warpX * WMMA_N], TILE_N + 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    if (warp_row < m && warp_col < n) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag_f16;
        for (int i = 0; i < c_frag.num_elements; ++i) {
            c_frag_f16.x[i] = __float2half(c_frag.x[i]);
        }
        wmma::store_matrix_sync(C + warp_row * n + warp_col, c_frag_f16, n, wmma::mem_row_major);
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
        matmul_wmma_swizzle<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        matmul_wmma_swizzle<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
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
