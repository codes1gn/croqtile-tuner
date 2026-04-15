// iter009_coalesced_load.cu - Fully coalesced memory loads
// Each warp loads a continuous row of memory for better coalescing
// Based on iter002 with improved load pattern

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

constexpr int SMEM_PAD = 8;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;
constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

__global__ void matmul_bf16_fp32_coalesced(
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
    const int laneId = threadIdx.x % WARP_SIZE;
    
    const int tileM = by * BLOCK_M;
    const int tileN = bx * BLOCK_N;
    
    __shared__ __nv_bfloat16 sA[BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __nv_bfloat16 sB[BLOCK_K][BLOCK_N + SMEM_PAD];
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }
    
    int warpBaseM = warpM * 32;
    int warpBaseN = warpN * 32;
    
    // K-loop
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A: 64x16 = 1024 elements
        // 4 warps, each warp loads 16 rows (256 elements per warp)
        // Each warp loads rows [warpId*16, warpId*16+16)
        // Each thread loads 8 elements (half a row)
        int aWarpRow = warpId * 16;
        #pragma unroll 8
        for (int r = 0; r < 16; r++) {
            int row = aWarpRow + r;
            // First half of threads load first half of row
            // Second half of threads load second half of row
            int col = (laneId < 16) ? laneId : (laneId - 16 + 8);
            int lanePart = (laneId < 16) ? 0 : 1;
            
            if (lanePart == 0 && col < 8) {
                int globalRow = tileM + row;
                int globalCol = k + col;
                sA[row][col] = (globalRow < M && globalCol < K) ?
                               A[globalRow * K + globalCol] : __float2bfloat16(0.0f);
            } else if (lanePart == 1) {
                int globalRow = tileM + row;
                int globalCol = k + col;
                sA[row][col] = (globalRow < M && globalCol < K) ?
                               A[globalRow * K + globalCol] : __float2bfloat16(0.0f);
            }
        }
        
        // Load B: 16x64 = 1024 elements
        // 4 warps, each warp loads 4 rows (256 elements per warp)
        // Each thread loads 8 elements
        int bWarpRow = warpId * 4;
        #pragma unroll 4
        for (int r = 0; r < 4; r++) {
            int row = bWarpRow + r;
            // Each thread loads 2 elements
            int col1 = laneId * 2;
            int col2 = laneId * 2 + 1;
            
            int globalRow = k + row;
            int globalCol1 = tileN + col1;
            int globalCol2 = tileN + col2;
            
            sB[row][col1] = (globalRow < K && globalCol1 < N) ?
                            B[globalRow * N + globalCol1] : __float2bfloat16(0.0f);
            sB[row][col2] = (globalRow < K && globalCol2 < N) ?
                            B[globalRow * N + globalCol2] : __float2bfloat16(0.0f);
        }
        
        __syncthreads();
        
        // Compute WMMA
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
    
    // Store
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
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_coalesced<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_coalesced<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
