// iter032_48x64.cu
// 48x64 block tile with 6 warps (192 threads)
// Different aspect ratio may improve memory access patterns

#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_M = 48;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;
constexpr int SMEM_PAD = 4;

constexpr int WARPS_M = 3;  // 3 warps in M
constexpr int WARPS_N = 2;  // 2 warps in N
constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = WARPS_M * WARPS_N * WARP_SIZE;  // 192

__global__ __launch_bounds__(192, 4)
void matmul_bf16_fp32_48x64(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpM = warpId / WARPS_N;  // 0, 1, 2
    const int warpN = warpId % WARPS_N;  // 0, 1
    
    const int tileM = by * BLOCK_M;
    const int tileN = bx * BLOCK_N;
    
    __shared__ __nv_bfloat16 sA[BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __nv_bfloat16 sB[BLOCK_N][BLOCK_K + SMEM_PAD];
    
    // Each warp handles 16x32 (1 A frag, 2 B frags)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2];
    
    wmma::fill_fragment(c_frag[0], 0.0f);
    wmma::fill_fragment(c_frag[1], 0.0f);
    
    const int tid = threadIdx.x;
    const int warpBaseM = warpM * WMMA_M;
    const int warpBaseN = warpN * 32;
    
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A: 192 threads, 48*16 = 768 elements, 4 per thread
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < BLOCK_M * BLOCK_K) {
                int row = idx / BLOCK_K;
                int col = idx % BLOCK_K;
                sA[row][col] = A[(tileM + row) * K + k + col];
            }
        }
        
        // Load B: 192 threads, 16*64 = 1024 elements, ~5.3 per thread
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < BLOCK_K * BLOCK_N) {
                int srcRow = idx / BLOCK_N;
                int srcCol = idx % BLOCK_N;
                sB[srcCol][srcRow] = B[(k + srcRow) * N + tileN + srcCol];
            }
        }
        
        __syncthreads();
        
        wmma::load_matrix_sync(a_frag, &sA[warpBaseM][0], BLOCK_K + SMEM_PAD);
        wmma::load_matrix_sync(b_frag[0], &sB[warpBaseN][0], BLOCK_K + SMEM_PAD);
        wmma::load_matrix_sync(b_frag[1], &sB[warpBaseN + WMMA_N][0], BLOCK_K + SMEM_PAD);
        
        wmma::mma_sync(c_frag[0], a_frag, b_frag[0], c_frag[0]);
        wmma::mma_sync(c_frag[1], a_frag, b_frag[1], c_frag[1]);
        
        __syncthreads();
    }
    
    const int outRowBase = tileM + warpBaseM;
    const int outColBase = tileN + warpBaseN;
    
    if (outRowBase < M) {
        wmma::store_matrix_sync(&C[outRowBase * N + outColBase], c_frag[0], N, wmma::mem_row_major);
        wmma::store_matrix_sync(&C[outRowBase * N + outColBase + WMMA_N], c_frag[1], N, wmma::mem_row_major);
    }
}

int main(int argc, char** argv) {
    int M = 512, N = 16384, K = 16384;
    int warmup = (argc > 1) ? atoi(argv[1]) : 5;
    int iters = (argc > 2) ? atoi(argv[2]) : 10;
    
    size_t sizeA = M * K * sizeof(__nv_bfloat16);
    size_t sizeB = K * N * sizeof(__nv_bfloat16);
    size_t sizeC = M * N * sizeof(float);
    
    __nv_bfloat16 *dA, *dB;
    float *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);
    
    __nv_bfloat16* hA = new __nv_bfloat16[M * K];
    __nv_bfloat16* hB = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; i++) hA[i] = __float2bfloat16((rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) hB[i] = __float2bfloat16((rand() % 100) / 100.0f);
    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);
    delete[] hA;
    delete[] hB;
    
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(NUM_THREADS);
    
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_48x64<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_48x64<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    ms /= iters;
    
    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("Time: %.3f ms, TFLOPS: %.2f\n", ms, tflops);
    
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
