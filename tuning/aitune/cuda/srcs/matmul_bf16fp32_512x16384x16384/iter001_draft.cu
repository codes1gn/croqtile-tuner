// iter001_draft.cu - First tuning iteration for matmul bf16->fp32 512x16384x16384
// Input: A (M x K) bf16, B (K x N) bf16
// Output: C (M x N) fp32
// Uses WMMA for bf16 tensor core operations with fp32 accumulation

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

// WMMA dimensions for bf16
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Tile dimensions
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;

// Warp configuration: 4x2 warps per block
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_SIZE = 32;

__global__ void matmul_bf16_fp32_wmma(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Block position
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Warp position within block
    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;
    
    // Lane within warp
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Global tile position
    const int tileM = by * BLOCK_M;
    const int tileN = bx * BLOCK_N;
    
    // WMMA fragment position within tile
    const int wmmaM = tileM + warpM * WMMA_M;
    const int wmmaN = tileN + warpN * WMMA_N;
    
    // Shared memory for tiles
    __shared__ __nv_bfloat16 sA[BLOCK_M][BLOCK_K];
    __shared__ __nv_bfloat16 sB[BLOCK_K][BLOCK_N];
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Loop over K dimension
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A tile to shared memory
        // Each thread loads multiple elements
        for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int globalRow = tileM + row;
            int globalCol = k + col;
            if (globalRow < M && globalCol < K) {
                sA[row][col] = A[globalRow * K + globalCol];
            } else {
                sA[row][col] = __float2bfloat16(0.0f);
            }
        }
        
        // Load B tile to shared memory
        for (int i = threadIdx.x; i < BLOCK_K * BLOCK_N; i += blockDim.x) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int globalRow = k + row;
            int globalCol = tileN + col;
            if (globalRow < K && globalCol < N) {
                sB[row][col] = B[globalRow * N + globalCol];
            } else {
                sB[row][col] = __float2bfloat16(0.0f);
            }
        }
        
        __syncthreads();
        
        // Load fragments from shared memory and compute
        // Only first warp does the computation for each WMMA tile
        if (warpM < (BLOCK_M / WMMA_M) && warpN < (BLOCK_N / WMMA_N)) {
            wmma::load_matrix_sync(a_frag, &sA[warpM * WMMA_M][0], BLOCK_K);
            wmma::load_matrix_sync(b_frag, &sB[0][warpN * WMMA_N], BLOCK_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        __syncthreads();
    }
    
    // Store result to global memory
    if (wmmaM < M && wmmaN < N) {
        wmma::store_matrix_sync(&C[wmmaM * N + wmmaN], c_frag, N, wmma::mem_row_major);
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
    
    size_t sizeA = M * K * sizeof(__nv_bfloat16);
    size_t sizeB = K * N * sizeof(__nv_bfloat16);
    size_t sizeC = M * N * sizeof(float);
    
    __nv_bfloat16 *d_A, *d_B;
    float *d_C;
    
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    // Initialize with random values (simplified)
    __nv_bfloat16* h_A = new __nv_bfloat16[M * K];
    __nv_bfloat16* h_B = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; i++) h_A[i] = __float2bfloat16((float)(rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2bfloat16((float)(rand() % 100) / 100.0f);
    
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    dim3 block(WARPS_M * WARPS_N * WARP_SIZE);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;
    
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;
    
    printf("Time: %.3f ms, TFLOPS: %.2f\n", avg_ms, tflops);
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
