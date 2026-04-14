// iter002_no_bank_conflict.cu - Fix shared memory bank conflicts
// Problem from ncu: 5-way bank conflict on shared memory loads (79.56% wavefronts)
// Solution: Add padding to shared memory to avoid bank conflicts
// 
// BF16 is 2 bytes. Shared memory has 32 banks, each 4 bytes wide.
// Two bf16 values share the same bank. With padding, we shift access patterns.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;

// Padding to avoid bank conflicts
// Each row gets 8 extra bf16 values (16 bytes = 4 banks)
constexpr int SMEM_PAD = 8;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;

__global__ void matmul_bf16_fp32_no_bank_conflict(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;
    
    const int tileM = by * BLOCK_M;
    const int tileN = bx * BLOCK_N;
    
    // Padded shared memory to avoid bank conflicts
    __shared__ __nv_bfloat16 sA[BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __nv_bfloat16 sB[BLOCK_K][BLOCK_N + SMEM_PAD];
    
    // WMMA fragments - each warp handles 32x32 output (2x2 WMMA tiles)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    
    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }
    
    const int tid = threadIdx.x;
    
    // K-loop
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A tile with coalesced access
        // 128 threads, 64*16 = 1024 elements
        // Each thread loads 8 elements
        #pragma unroll
        for (int i = 0; i < (BLOCK_M * BLOCK_K) / (NUM_WARPS * WARP_SIZE); i++) {
            int idx = tid + i * NUM_WARPS * WARP_SIZE;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            int globalRow = tileM + row;
            int globalCol = k + col;
            if (globalRow < M && globalCol < K) {
                sA[row][col] = A[globalRow * K + globalCol];
            } else {
                sA[row][col] = __float2bfloat16(0.0f);
            }
        }
        
        // Load B tile with coalesced access
        #pragma unroll
        for (int i = 0; i < (BLOCK_K * BLOCK_N) / (NUM_WARPS * WARP_SIZE); i++) {
            int idx = tid + i * NUM_WARPS * WARP_SIZE;
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            int globalRow = k + row;
            int globalCol = tileN + col;
            if (globalRow < K && globalCol < N) {
                sB[row][col] = B[globalRow * N + globalCol];
            } else {
                sB[row][col] = __float2bfloat16(0.0f);
            }
        }
        
        __syncthreads();
        
        // Compute WMMA - each warp handles 32x32 (2x2 WMMA tiles)
        int warpBaseM = warpM * 32;  // 0 or 32
        int warpBaseN = warpN * 32;  // 0 or 32
        
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(a_frag[i], &sA[warpBaseM + i * WMMA_M][0], BLOCK_K + SMEM_PAD);
        }
        
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::load_matrix_sync(b_frag[j], &sB[0][warpBaseN + j * WMMA_N], BLOCK_N + SMEM_PAD);
        }
        
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    int warpBaseM = warpM * 32;
    int warpBaseN = warpN * 32;
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int outM = tileM + warpBaseM + i * WMMA_M;
            int outN = tileN + warpBaseN + j * WMMA_N;
            if (outM < M && outN < N) {
                wmma::store_matrix_sync(&C[outM * N + outN], c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

int main(int argc, char** argv) {
    const int M = 512;
    const int N = 16384;
    const int K = 16384;
    
    int warmup = 5;
    int iters = 10;
    if (argc >= 2) warmup = atoi(argv[1]);
    if (argc >= 3) iters = atoi(argv[2]);
    
    __nv_bfloat16 *d_A, *d_B;
    float *d_C;
    
    cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    __nv_bfloat16* h_A = new __nv_bfloat16[M * K];
    __nv_bfloat16* h_B = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; i++) h_A[i] = __float2bfloat16((float)(rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2bfloat16((float)(rand() % 100) / 100.0f);
    
    cudaMemcpy(d_A, h_A, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    dim3 block(NUM_WARPS * WARP_SIZE);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_no_bank_conflict<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_no_bank_conflict<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;
    
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;
    
    printf("Time: %.3f ms, TFLOPS: %.2f\n", avg_ms, tflops);
    
    delete[] h_A;
    delete[] h_B;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
