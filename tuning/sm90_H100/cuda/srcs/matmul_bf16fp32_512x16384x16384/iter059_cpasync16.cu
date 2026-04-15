// iter059_cpasync16.cu
// Use cp.async with 16-byte (8 bf16) vectorized loads

#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;
constexpr int SMEM_PAD = 0; // No pad needed with 16-byte aligned loads

constexpr int WARPS_N = 2;
constexpr int WARP_SIZE = 32;

extern __shared__ __nv_bfloat16 smem[];

__global__ __launch_bounds__(128, 4)
void matmul_bf16_fp32_cpasync16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    __nv_bfloat16* __restrict__ sA = smem;
    __nv_bfloat16* __restrict__ sB = smem + BLOCK_M * BLOCK_K;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;
    
    const int tileM = bx * BLOCK_M;
    const int tileN = by * BLOCK_N;
    
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
    
    // Each thread loads one 16-byte (8 bf16) chunk
    // For A: 64 rows x 16 cols = 1024 elements = 128 threads x 8 elements
    // For B: 16 rows x 64 cols = 1024 elements = 128 threads x 8 elements
    
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A: tid selects which 8-element row-chunk
        // 128 threads, each loads 8 elements = 1024 total
        // Row = tid / 2, col_start = (tid % 2) * 8
        int a_row = tid / 2;
        int a_col = (tid % 2) * 8;
        if (a_row < BLOCK_M) {
            __pipeline_memcpy_async(&sA[a_row * BLOCK_K + a_col],
                                   &A[(tileM + a_row) * K + k + a_col],
                                   16);  // 16 bytes = 8 bf16
        }
        
        // Load B and transpose: 16 rows x 64 cols
        // Each thread loads 8 cols from one row
        // Then we need to transpose for col-major sB
        // This is tricky - cp.async doesn't help with transpose
        // Fall back to regular loads for B
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 128;
            int srcRow = idx / BLOCK_N;
            int srcCol = idx % BLOCK_N;
            sB[srcCol * BLOCK_K + srcRow] = B[(k + srcRow) * N + tileN + srcCol];
        }
        
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        
        wmma::load_matrix_sync(a_frag[0], &sA[warpBaseM * BLOCK_K], BLOCK_K);
        wmma::load_matrix_sync(a_frag[1], &sA[(warpBaseM + WMMA_M) * BLOCK_K], BLOCK_K);
        wmma::load_matrix_sync(b_frag[0], &sB[warpBaseN * BLOCK_K], BLOCK_K);
        wmma::load_matrix_sync(b_frag[1], &sB[(warpBaseN + WMMA_N) * BLOCK_K], BLOCK_K);
        
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
    
    size_t smemSize = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(__nv_bfloat16);
    
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(128);
    
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_cpasync16<<<grid, block, smemSize>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_cpasync16<<<grid, block, smemSize>>>(dA, dB, dC, M, N, K);
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
