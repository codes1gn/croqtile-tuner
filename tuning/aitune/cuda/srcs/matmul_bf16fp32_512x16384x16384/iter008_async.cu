// iter008_async.cu - Use cp.async for asynchronous global->shared memory loads
// Ampere (sm_80+) supports cp.async for overlapping loads with compute

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
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

__global__ void matmul_bf16_fp32_async(
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
    
    // Double buffer for async pipeline
    __shared__ __nv_bfloat16 sA[2][BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __nv_bfloat16 sB[2][BLOCK_K][BLOCK_N + SMEM_PAD];
    
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
    
    const int tid = threadIdx.x;
    int warpBaseM = warpM * 32;
    int warpBaseN = warpN * 32;
    
    const int numK = (K + BLOCK_K - 1) / BLOCK_K;
    
    // Prologue: async load first tile
    #pragma unroll
    for (int i = 0; i < (BLOCK_M * BLOCK_K) / (NUM_WARPS * WARP_SIZE); i++) {
        int idx = tid + i * NUM_WARPS * WARP_SIZE;
        int row = idx / BLOCK_K;
        int col = idx % BLOCK_K;
        int globalRow = tileM + row;
        int globalCol = col;
        
        if (globalRow < M && globalCol < K) {
            __pipeline_memcpy_async(&sA[0][row][col], &A[globalRow * K + globalCol], sizeof(__nv_bfloat16));
        } else {
            sA[0][row][col] = __float2bfloat16(0.0f);
        }
    }
    
    #pragma unroll
    for (int i = 0; i < (BLOCK_K * BLOCK_N) / (NUM_WARPS * WARP_SIZE); i++) {
        int idx = tid + i * NUM_WARPS * WARP_SIZE;
        int row = idx / BLOCK_N;
        int col = idx % BLOCK_N;
        int globalRow = row;
        int globalCol = tileN + col;
        
        if (globalRow < K && globalCol < N) {
            __pipeline_memcpy_async(&sB[0][row][col], &B[globalRow * N + globalCol], sizeof(__nv_bfloat16));
        } else {
            sB[0][row][col] = __float2bfloat16(0.0f);
        }
    }
    __pipeline_commit();
    
    int buf = 0;
    
    for (int kk = 0; kk < numK; kk++) {
        int nextBuf = 1 - buf;
        int nextK = (kk + 1) * BLOCK_K;
        
        // Start async load for next tile
        if (kk < numK - 1) {
            #pragma unroll
            for (int i = 0; i < (BLOCK_M * BLOCK_K) / (NUM_WARPS * WARP_SIZE); i++) {
                int idx = tid + i * NUM_WARPS * WARP_SIZE;
                int row = idx / BLOCK_K;
                int col = idx % BLOCK_K;
                int globalRow = tileM + row;
                int globalCol = nextK + col;
                
                if (globalRow < M && globalCol < K) {
                    __pipeline_memcpy_async(&sA[nextBuf][row][col], &A[globalRow * K + globalCol], sizeof(__nv_bfloat16));
                } else {
                    sA[nextBuf][row][col] = __float2bfloat16(0.0f);
                }
            }
            
            #pragma unroll
            for (int i = 0; i < (BLOCK_K * BLOCK_N) / (NUM_WARPS * WARP_SIZE); i++) {
                int idx = tid + i * NUM_WARPS * WARP_SIZE;
                int row = idx / BLOCK_N;
                int col = idx % BLOCK_N;
                int globalRow = nextK + row;
                int globalCol = tileN + col;
                
                if (globalRow < K && globalCol < N) {
                    __pipeline_memcpy_async(&sB[nextBuf][row][col], &B[globalRow * N + globalCol], sizeof(__nv_bfloat16));
                } else {
                    sB[nextBuf][row][col] = __float2bfloat16(0.0f);
                }
            }
            __pipeline_commit();
        }
        
        // Wait for current tile
        __pipeline_wait_prior(1);
        __syncthreads();
        
        // Compute on current buffer
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(a_frag[i], &sA[buf][warpBaseM + i * WMMA_M][0], BLOCK_K + SMEM_PAD);
        }
        
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::load_matrix_sync(b_frag[j], &sB[buf][0][warpBaseN + j * WMMA_N], BLOCK_N + SMEM_PAD);
        }
        
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
        
        buf = nextBuf;
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
    
    dim3 block(NUM_WARPS * WARP_SIZE);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_async<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_async<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
