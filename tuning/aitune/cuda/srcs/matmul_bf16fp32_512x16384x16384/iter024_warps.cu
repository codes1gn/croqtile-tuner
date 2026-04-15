// iter024_8warps.cu
// iter022 base + 8 warps (256 threads) instead of 4
// Each warp still handles 16x16, but 8 warps cover 64x32 (or 32x64)

#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;
constexpr int SMEM_PAD = 8;

// 8 warps: 4 in M, 2 in N direction
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = 256;

__global__ __launch_bounds__(256, 2)
void matmul_bf16_fp32_8warps(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int warpId = threadIdx.x / WARP_SIZE;
    int warpM = warpId / WARPS_N;  // 0-3
    int warpN = warpId % WARPS_N;  // 0-1
    
    int tileM = by * BLOCK_M;
    int tileN = bx * BLOCK_N;
    
    __shared__ __nv_bfloat16 sA[BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __nv_bfloat16 sB[BLOCK_N][BLOCK_K + SMEM_PAD];
    
    // Each warp handles 16x32 (1 in M, 2 in N)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2];
    
    wmma::fill_fragment(c_frag[0], 0.0f);
    wmma::fill_fragment(c_frag[1], 0.0f);
    
    int tid = threadIdx.x;
    
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A: 256 threads, 64*16 = 1024 elements, 4 per thread
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * NUM_THREADS;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            int gRow = tileM + row;
            int gCol = k + col;
            sA[row][col] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : __float2bfloat16(0.0f);
        }
        
        // Load B and transpose: 256 threads, 16*64 = 1024 elements, 4 per thread
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * NUM_THREADS;
            int srcRow = idx / BLOCK_N;
            int srcCol = idx % BLOCK_N;
            int gRow = k + srcRow;
            int gCol = tileN + srcCol;
            sB[srcCol][srcRow] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : __float2bfloat16(0.0f);
        }
        
        __syncthreads();
        
        // Each warp: row = warpM * 16, cols = warpN * 32 to warpN * 32 + 32
        int warpRowM = warpM * WMMA_M;
        int warpColN = warpN * 32;
        
        // Load A fragment
        wmma::load_matrix_sync(a_frag, &sA[warpRowM][0], BLOCK_K + SMEM_PAD);
        
        // Load 2 B fragments
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::load_matrix_sync(b_frag[j], &sB[warpColN + j * WMMA_N][0], BLOCK_K + SMEM_PAD);
        }
        
        // Compute
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::mma_sync(c_frag[j], a_frag, b_frag[j], c_frag[j]);
        }
        
        __syncthreads();
    }
    
    // Store results
    int warpRowM = warpM * WMMA_M;
    int warpColN = warpN * 32;
    
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        int outM = tileM + warpRowM;
        int outN = tileN + warpColN + j * WMMA_N;
        if (outM < M && outN < N) {
            wmma::store_matrix_sync(&C[outM * N + outN], c_frag[j], N, wmma::mem_row_major);
        }
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
    dim3 block(256);
    
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_8warps<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_8warps<<<grid, block>>>(dA, dB, dC, M, N, K);
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
