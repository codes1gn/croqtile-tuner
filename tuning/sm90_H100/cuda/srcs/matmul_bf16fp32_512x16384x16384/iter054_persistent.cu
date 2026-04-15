// iter054_persistent.cu
// Persistent kernel: each block processes multiple output tiles
// For shapes with small M (512), this can improve SM utilization

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

extern __shared__ __nv_bfloat16 smem[];

__device__ __forceinline__ void compute_tile(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    __nv_bfloat16* __restrict__ sA,
    __nv_bfloat16* __restrict__ sB,
    const int tileM, const int tileN,
    const int M, const int N, const int K,
    const int tid, const int warpM, const int warpN
) {
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
    
    const int warpBaseM = warpM * 32;
    const int warpBaseN = warpN * 32;
    const int sA_stride = BLOCK_K + SMEM_PAD;
    const int sB_stride = BLOCK_K + SMEM_PAD;
    
    for (int k = 0; k < K; k += BLOCK_K) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 128;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            sA[row * sA_stride + col] = A[(tileM + row) * K + k + col];
        }
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 128;
            int srcRow = idx / BLOCK_N;
            int srcCol = idx % BLOCK_N;
            sB[srcCol * sB_stride + srcRow] = B[(k + srcRow) * N + tileN + srcCol];
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
    
    const int outRowBase = tileM + warpBaseM;
    const int outColBase = tileN + warpBaseN;
    
    wmma::store_matrix_sync(&C[outRowBase * N + outColBase], c_frag[0][0], N, wmma::mem_row_major);
    wmma::store_matrix_sync(&C[outRowBase * N + outColBase + WMMA_N], c_frag[0][1], N, wmma::mem_row_major);
    wmma::store_matrix_sync(&C[(outRowBase + WMMA_M) * N + outColBase], c_frag[1][0], N, wmma::mem_row_major);
    wmma::store_matrix_sync(&C[(outRowBase + WMMA_M) * N + outColBase + WMMA_N], c_frag[1][1], N, wmma::mem_row_major);
}

__global__ __launch_bounds__(128, 4)
void matmul_bf16_fp32_persistent(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K,
    const int total_tiles
) {
    __nv_bfloat16* __restrict__ sA = smem;
    __nv_bfloat16* __restrict__ sB = smem + BLOCK_M * (BLOCK_K + SMEM_PAD);
    
    const int num_m_tiles = (M + BLOCK_M - 1) / BLOCK_M;
    const int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    
    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;
    const int tid = threadIdx.x;
    
    // Each block processes multiple tiles in a loop
    for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        int tile_m = tile_idx % num_m_tiles;
        int tile_n = tile_idx / num_m_tiles;
        
        int tileM = tile_m * BLOCK_M;
        int tileN = tile_n * BLOCK_N;
        
        compute_tile(A, B, C, sA, sB, tileM, tileN, M, N, K, tid, warpM, warpN);
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
    
    int num_m_tiles = (M + BLOCK_M - 1) / BLOCK_M;
    int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    int total_tiles = num_m_tiles * num_n_tiles;
    
    // Use fewer blocks than total tiles for persistent kernel
    int num_sms = 46;  // RTX 3070 has 46 SMs
    int blocks_per_sm = 4;
    int num_blocks = min(num_sms * blocks_per_sm, total_tiles);
    
    dim3 grid(num_blocks);
    dim3 block(128);
    
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_persistent<<<grid, block, smemSize>>>(dA, dB, dC, M, N, K, total_tiles);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_persistent<<<grid, block, smemSize>>>(dA, dB, dC, M, N, K, total_tiles);
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
