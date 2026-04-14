// iter014_larger_output.cu - 128x64 output tile, 8 warps
// More parallelism by using more warps with smaller tiles per warp

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;

constexpr int SMEM_PAD = 8;

// 8 warps: 4 in M, 2 in N
// Each warp handles 32x32 output (2x2 WMMA tiles)
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;

__global__ void matmul_bf16_fp32_larger_output(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;
    
    const int tileM = by * BLOCK_M;
    const int tileN = bx * BLOCK_N;
    const int warpBaseM = warpM * 32;
    const int warpBaseN = warpN * 32;
    
    __shared__ __nv_bfloat16 sA[BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __nv_bfloat16 sB[BLOCK_K][BLOCK_N + SMEM_PAD];
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    
    wmma::fill_fragment(c_frag[0][0], 0.0f);
    wmma::fill_fragment(c_frag[0][1], 0.0f);
    wmma::fill_fragment(c_frag[1][0], 0.0f);
    wmma::fill_fragment(c_frag[1][1], 0.0f);
    
    // K-loop
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A: 128x16 = 2048 elements, 256 threads = 8 per thread
        #pragma unroll 8
        for (int i = 0; i < 8; i++) {
            int idx = tid * 8 + i;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            int globalRow = tileM + row;
            int globalCol = k + col;
            sA[row][col] = (globalRow < M && globalCol < K) ? 
                           A[globalRow * K + globalCol] : __float2bfloat16(0.0f);
        }
        
        // Load B: 16x64 = 1024 elements, 256 threads = 4 per thread
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
        
        // Load and compute
        wmma::load_matrix_sync(a_frag[0], &sA[warpBaseM][0], BLOCK_K + SMEM_PAD);
        wmma::load_matrix_sync(a_frag[1], &sA[warpBaseM + WMMA_M][0], BLOCK_K + SMEM_PAD);
        wmma::load_matrix_sync(b_frag[0], &sB[0][warpBaseN], BLOCK_N + SMEM_PAD);
        wmma::load_matrix_sync(b_frag[1], &sB[0][warpBaseN + WMMA_N], BLOCK_N + SMEM_PAD);
        
        wmma::mma_sync(c_frag[0][0], a_frag[0], b_frag[0], c_frag[0][0]);
        wmma::mma_sync(c_frag[0][1], a_frag[0], b_frag[1], c_frag[0][1]);
        wmma::mma_sync(c_frag[1][0], a_frag[1], b_frag[0], c_frag[1][0]);
        wmma::mma_sync(c_frag[1][1], a_frag[1], b_frag[1], c_frag[1][1]);
        
        __syncthreads();
    }
    
    // Store
    int outM0 = tileM + warpBaseM;
    int outM1 = outM0 + WMMA_M;
    int outN0 = tileN + warpBaseN;
    int outN1 = outN0 + WMMA_N;
    
    if (outM0 < M && outN0 < N) wmma::store_matrix_sync(&C[outM0 * N + outN0], c_frag[0][0], N, wmma::mem_row_major);
    if (outM0 < M && outN1 < N) wmma::store_matrix_sync(&C[outM0 * N + outN1], c_frag[0][1], N, wmma::mem_row_major);
    if (outM1 < M && outN0 < N) wmma::store_matrix_sync(&C[outM1 * N + outN0], c_frag[1][0], N, wmma::mem_row_major);
    if (outM1 < M && outN1 < N) wmma::store_matrix_sync(&C[outM1 * N + outN1], c_frag[1][1], N, wmma::mem_row_major);
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
        matmul_bf16_fp32_larger_output<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_larger_output<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
