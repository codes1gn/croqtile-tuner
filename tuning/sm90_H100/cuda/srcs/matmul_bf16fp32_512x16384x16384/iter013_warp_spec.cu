// iter013_warp_spec.cu - Warp specialization with producer-consumer pattern
// Warp 0-1: compute warps (WMMA)
// Warp 2-3: load warps (global -> shared)
// Use cp.async.wait_group for synchronization

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

constexpr int COMPUTE_WARPS = 2;
constexpr int LOAD_WARPS = 2;
constexpr int NUM_WARPS = COMPUTE_WARPS + LOAD_WARPS;
constexpr int WARP_SIZE = 32;

__global__ void matmul_bf16_fp32_warp_spec(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    
    const int tileM = by * BLOCK_M;
    const int tileN = bx * BLOCK_N;
    
    // Double buffer
    __shared__ __nv_bfloat16 sA[2][BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __nv_bfloat16 sB[2][BLOCK_K][BLOCK_N + SMEM_PAD];
    __shared__ int ready[2];  // Simple flag for synchronization
    
    // Initialize ready flags
    if (tid < 2) ready[tid] = 0;
    __syncthreads();
    
    const bool isComputeWarp = (warpId < COMPUTE_WARPS);
    
    if (isComputeWarp) {
        // Compute warp logic
        const int compWarpId = warpId;
        const int warpM = compWarpId;
        const int warpBaseM = warpM * 32;
        
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
        
        int buf = 0;
        const int numK = (K + BLOCK_K - 1) / BLOCK_K;
        
        for (int kk = 0; kk < numK; kk++) {
            // Wait for data to be ready
            while (atomicAdd(&ready[buf], 0) == 0) {}
            
            // Load fragments and compute
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                wmma::load_matrix_sync(a_frag[i], &sA[buf][warpBaseM + i * WMMA_M][0], BLOCK_K + SMEM_PAD);
            }
            
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::load_matrix_sync(b_frag[j], &sB[buf][0][j * WMMA_N], BLOCK_N + SMEM_PAD);
            }
            
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
            
            // Signal that we're done with this buffer
            __syncwarp();
            if (laneId == 0) atomicExch(&ready[buf], 0);
            
            buf = 1 - buf;
        }
        
        // Store results
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int outM = tileM + warpBaseM + i * WMMA_M;
                int outN = tileN + j * WMMA_N;
                if (outM < M && outN < N) {
                    wmma::store_matrix_sync(&C[outM * N + outN], c_frag[i][j], N, wmma::mem_row_major);
                }
            }
        }
    } else {
        // Load warp logic
        const int loadWarpId = warpId - COMPUTE_WARPS;
        
        int buf = 0;
        const int numK = (K + BLOCK_K - 1) / BLOCK_K;
        
        for (int kk = 0; kk < numK; kk++) {
            const int k = kk * BLOCK_K;
            
            // Load A (warp 2 loads first half, warp 3 loads second half)
            const int aRowStart = loadWarpId * 32;
            const int aRowEnd = aRowStart + 32;
            
            for (int row = aRowStart; row < aRowEnd; row++) {
                int col = laneId % BLOCK_K;
                int globalRow = tileM + row;
                int globalCol = k + col;
                if (laneId < BLOCK_K) {
                    sA[buf][row][col] = (globalRow < M && globalCol < K) ?
                                        A[globalRow * K + globalCol] : __float2bfloat16(0.0f);
                }
            }
            
            // Load B (each load warp loads 8 rows)
            const int bRowStart = loadWarpId * 8;
            const int bRowEnd = bRowStart + 8;
            
            for (int row = bRowStart; row < bRowEnd && row < BLOCK_K; row++) {
                // Each thread loads 2 columns
                int col1 = laneId * 2;
                int col2 = col1 + 1;
                int globalRow = k + row;
                
                if (col1 < BLOCK_N) {
                    int globalCol1 = tileN + col1;
                    sB[buf][row][col1] = (globalRow < K && globalCol1 < N) ?
                                         B[globalRow * N + globalCol1] : __float2bfloat16(0.0f);
                }
                if (col2 < BLOCK_N) {
                    int globalCol2 = tileN + col2;
                    sB[buf][row][col2] = (globalRow < K && globalCol2 < N) ?
                                         B[globalRow * N + globalCol2] : __float2bfloat16(0.0f);
                }
            }
            
            // Signal that data is ready
            __syncwarp();
            if (laneId == 0) atomicExch(&ready[buf], 1);
            
            // Wait for compute to finish before overwriting
            while (atomicAdd(&ready[1 - buf], 0) != 0) {}
            
            buf = 1 - buf;
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
        matmul_bf16_fp32_warp_spec<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_warp_spec<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
