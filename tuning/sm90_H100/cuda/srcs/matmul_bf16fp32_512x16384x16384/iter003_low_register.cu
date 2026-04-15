// iter003_low_register.cu - Reduce register pressure for better occupancy
// Problem from ncu: 126 registers/thread -> only 33% occupancy (16/48 warps)
// Solution: Use __launch_bounds__ and simplify to 1 WMMA tile per warp

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
constexpr int BLOCK_K = 32;  // Larger K tile for better arithmetic intensity

constexpr int SMEM_PAD = 8;

// 4x4 warps = 16 warps per block, each handles 16x16 output
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 4;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;
constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;  // 512 threads

// Limit registers to increase occupancy
// With 64 registers per thread: 65536/64 = 1024 threads/SM = 32 warps/SM
__launch_bounds__(BLOCK_SIZE, 2)  // 2 blocks per SM minimum
__global__ void matmul_bf16_fp32_low_register(
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
    
    // Padded shared memory
    __shared__ __nv_bfloat16 sA[BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __nv_bfloat16 sB[BLOCK_K][BLOCK_N + SMEM_PAD];
    
    // Single WMMA tile per warp (reduces register usage)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    const int tid = threadIdx.x;
    const int laneId = tid % WARP_SIZE;
    
    // K-loop with larger K-tile
    for (int k = 0; k < K; k += BLOCK_K) {
        // Cooperative load of A tile (64x32 = 2048 elements, 512 threads = 4 per thread)
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            int idx = tid * 4 + i;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            int globalRow = tileM + row;
            int globalCol = k + col;
            sA[row][col] = (globalRow < M && globalCol < K) ? 
                           A[globalRow * K + globalCol] : __float2bfloat16(0.0f);
        }
        
        // Cooperative load of B tile (32x64 = 2048 elements, 512 threads = 4 per thread)
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            int idx = tid * 4 + i;
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            int globalRow = k + row;
            int globalCol = tileN + col;
            sB[row][col] = (globalRow < K && globalCol < N) ? 
                           B[globalRow * N + globalCol] : __float2bfloat16(0.0f);
        }
        
        __syncthreads();
        
        // Compute WMMA - iterate over K in shared memory
        int warpBaseM = warpM * WMMA_M;
        int warpBaseN = warpN * WMMA_N;
        
        // Two WMMA_K iterations per BLOCK_K
        #pragma unroll 2
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::load_matrix_sync(a_frag, &sA[warpBaseM][kk], BLOCK_K + SMEM_PAD);
            wmma::load_matrix_sync(b_frag, &sB[kk][warpBaseN], BLOCK_N + SMEM_PAD);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        __syncthreads();
    }
    
    // Store result
    int outM = tileM + warpM * WMMA_M;
    int outN = tileN + warpN * WMMA_N;
    
    if (outM < M && outN < N) {
        wmma::store_matrix_sync(&C[outM * N + outN], c_frag, N, wmma::mem_row_major);
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
        matmul_bf16_fp32_low_register<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_low_register<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
