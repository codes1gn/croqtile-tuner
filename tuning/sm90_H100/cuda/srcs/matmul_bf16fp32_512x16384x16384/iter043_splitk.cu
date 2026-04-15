// iter043_splitk.cu
// Split-K strategy: divide K dimension across multiple blocks
// Then atomically accumulate results
// May help with tall-skinny matrices where M is small

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
constexpr int SMEM_PAD = 4;

constexpr int WARPS_N = 2;
constexpr int WARP_SIZE = 32;
constexpr int SPLIT_K = 4;  // Split K into 4 parts

extern __shared__ __nv_bfloat16 smem[];

__global__ __launch_bounds__(128, 4)
void matmul_bf16_fp32_splitk(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K,
    const int k_split_idx
) {
    __nv_bfloat16* __restrict__ sA = smem;
    __nv_bfloat16* __restrict__ sB = smem + BLOCK_M * (BLOCK_K + SMEM_PAD);
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;
    
    const int tileM = bx * BLOCK_M;
    const int tileN = by * BLOCK_N;
    
    // Compute K range for this split
    const int k_per_split = (K + SPLIT_K - 1) / SPLIT_K;
    const int k_start = k_split_idx * k_per_split;
    const int k_end = min(k_start + k_per_split, K);
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }
    
    const int tid = threadIdx.x;
    const int warpBaseM = warpM * 32;
    const int warpBaseN = warpN * 32;
    const int sA_stride = BLOCK_K + SMEM_PAD;
    const int sB_stride = BLOCK_K + SMEM_PAD;
    
    for (int k = k_start; k < k_end; k += BLOCK_K) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 128;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            int gk = k + col;
            sA[row * sA_stride + col] = (gk < k_end) ? A[(tileM + row) * K + gk] : __float2bfloat16(0.0f);
        }
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 128;
            int srcRow = idx / BLOCK_N;
            int srcCol = idx % BLOCK_N;
            int gk = k + srcRow;
            sB[srcCol * sB_stride + srcRow] = (gk < k_end) ? B[gk * N + tileN + srcCol] : __float2bfloat16(0.0f);
        }
        
        __syncthreads();
        
        wmma::load_matrix_sync(a_frag[0], &sA[warpBaseM * sA_stride], sA_stride);
        wmma::load_matrix_sync(a_frag[1], &sA[(warpBaseM + WMMA_M) * sA_stride], sA_stride);
        wmma::load_matrix_sync(b_frag[0], &sB[warpBaseN * sB_stride], sB_stride);
        wmma::load_matrix_sync(b_frag[1], &sB[(warpBaseN + WMMA_N) * sB_stride], sB_stride);
        
        wmma::mma_sync(c_frag[0][0], a_frag[0], b_frag[0], c_frag[0][0]);
        wmma::mma_sync(c_frag[0][1], a_frag[0], b_frag[1], c_frag[0][1]);
        wmma::mma_sync(c_frag[1][0], a_frag[1], b_frag[0], c_frag[1][0]);
        wmma::mma_sync(c_frag[1][1], a_frag[1], b_frag[1], c_frag[1][1]);
        
        __syncthreads();
    }
    
    // Atomic add to output
    const int outRowBase = tileM + warpBaseM;
    const int outColBase = tileN + warpBaseN;
    const int laneId = threadIdx.x % 32;
    
    // Store to shared memory first, then atomic add
    __shared__ float sC[32][33];
    
    wmma::store_matrix_sync(&sC[0][0], c_frag[0][0], 33, wmma::mem_row_major);
    __syncwarp();
    #pragma unroll
    for (int i = laneId; i < 256; i += 32) {
        int r = i / 16;
        int c = i % 16;
        atomicAdd(&C[(outRowBase + r) * N + outColBase + c], sC[r][c]);
    }
    
    wmma::store_matrix_sync(&sC[0][0], c_frag[0][1], 33, wmma::mem_row_major);
    __syncwarp();
    #pragma unroll
    for (int i = laneId; i < 256; i += 32) {
        int r = i / 16;
        int c = i % 16;
        atomicAdd(&C[(outRowBase + r) * N + outColBase + WMMA_N + c], sC[r][c]);
    }
    
    wmma::store_matrix_sync(&sC[0][0], c_frag[1][0], 33, wmma::mem_row_major);
    __syncwarp();
    #pragma unroll
    for (int i = laneId; i < 256; i += 32) {
        int r = i / 16;
        int c = i % 16;
        atomicAdd(&C[(outRowBase + WMMA_M + r) * N + outColBase + c], sC[r][c]);
    }
    
    wmma::store_matrix_sync(&sC[0][0], c_frag[1][1], 33, wmma::mem_row_major);
    __syncwarp();
    #pragma unroll
    for (int i = laneId; i < 256; i += 32) {
        int r = i / 16;
        int c = i % 16;
        atomicAdd(&C[(outRowBase + WMMA_M + r) * N + outColBase + WMMA_N + c], sC[r][c]);
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
    
    size_t smemSize = (BLOCK_M * (BLOCK_K + SMEM_PAD) + BLOCK_N * (BLOCK_K + SMEM_PAD)) * sizeof(__nv_bfloat16);
    
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(128);
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        cudaMemset(dC, 0, sizeC);
        for (int s = 0; s < SPLIT_K; s++) {
            matmul_bf16_fp32_splitk<<<grid, block, smemSize>>>(dA, dB, dC, M, N, K, s);
        }
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cudaMemset(dC, 0, sizeC);
        for (int s = 0; s < SPLIT_K; s++) {
            matmul_bf16_fp32_splitk<<<grid, block, smemSize>>>(dA, dB, dC, M, N, K, s);
        }
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
