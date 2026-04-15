// iter017_warp_tile_64.cu - CUTLASS-style warp tile: 64x64 per warp
// Each warp handles a larger output tile (64x64 = 4096 elements)
// Uses 4 warps per block, each doing 4x4 WMMA tiles

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Each warp handles 64x64 output
constexpr int WARP_TILE_M = 64;
constexpr int WARP_TILE_N = 64;
constexpr int WMMA_TILES_M = WARP_TILE_M / WMMA_M;  // 4
constexpr int WMMA_TILES_N = WARP_TILE_N / WMMA_N;  // 4

// Single warp per block
constexpr int NUM_WARPS = 1;
constexpr int WARP_SIZE = 32;

constexpr int BLOCK_M = WARP_TILE_M;
constexpr int BLOCK_N = WARP_TILE_N;
constexpr int BLOCK_K = 16;
constexpr int SMEM_PAD = 8;

__global__ void matmul_bf16_fp32_warp_tile_64(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int tid = threadIdx.x;
    
    const int tileM = by * BLOCK_M;
    const int tileN = bx * BLOCK_N;
    
    __shared__ __nv_bfloat16 sA[BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __nv_bfloat16 sB[BLOCK_K][BLOCK_N + SMEM_PAD];
    
    // 4x4 WMMA tiles = 16 accumulators
    // Too many - will cause register spill
    // Let's try 2x2 at a time with loop
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[4][4];
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }
    
    // K-loop
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A: 64x16 = 1024 elements, 32 threads = 32 per thread
        #pragma unroll 32
        for (int i = 0; i < 32; i++) {
            int idx = tid * 32 + i;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            int globalRow = tileM + row;
            int globalCol = k + col;
            sA[row][col] = (globalRow < M && globalCol < K) ? 
                           A[globalRow * K + globalCol] : __float2bfloat16(0.0f);
        }
        
        // Load B: 16x64 = 1024 elements, 32 threads = 32 per thread
        #pragma unroll 32
        for (int i = 0; i < 32; i++) {
            int idx = tid * 32 + i;
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            int globalRow = k + row;
            int globalCol = tileN + col;
            sB[row][col] = (globalRow < K && globalCol < N) ? 
                           B[globalRow * N + globalCol] : __float2bfloat16(0.0f);
        }
        
        __syncthreads();
        
        // Compute 4x4 WMMA tiles
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
        
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            wmma::load_matrix_sync(a_frag, &sA[i * WMMA_M][0], BLOCK_K + SMEM_PAD);
            
            #pragma unroll 4
            for (int j = 0; j < 4; j++) {
                wmma::load_matrix_sync(b_frag, &sB[0][j * WMMA_N], BLOCK_N + SMEM_PAD);
                wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
            }
        }
        
        __syncthreads();
    }
    
    // Store 4x4 WMMA tiles
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        #pragma unroll 4
        for (int j = 0; j < 4; j++) {
            int outM = tileM + i * WMMA_M;
            int outN = tileN + j * WMMA_N;
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
    
    printf("Grid: %dx%d, Block: %d\n", grid.x, grid.y, block.x);
    
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_warp_tile_64<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_warp_tile_64<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
