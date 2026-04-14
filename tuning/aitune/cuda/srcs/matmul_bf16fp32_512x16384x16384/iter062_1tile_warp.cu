// iter062_1tile_warp.cu
// Each warp computes only 1 WMMA tile (16x16)
// 4 warps = 128 threads, each warp = 1 tile
// Block covers 32x32 (2x2 warps)

#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;
constexpr int BLOCK_K = 16;
constexpr int SMEM_PAD = 4;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARP_SIZE = 32;

extern __shared__ __nv_bfloat16 smem[];

__global__ __launch_bounds__(128, 8)
void matmul_bf16_fp32_1tile(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
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
    
    // Each warp computes only 1 tile (16x16)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    const int tid = threadIdx.x;
    const int warpBaseM = warpM * WMMA_M;
    const int warpBaseN = warpN * WMMA_N;
    const int sA_stride = BLOCK_K + SMEM_PAD;
    const int sB_stride = BLOCK_K + SMEM_PAD;
    
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A: 32 rows x 16 cols = 512 elements
        // 128 threads, each loads 4 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 128;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            sA[row * sA_stride + col] = A[(tileM + row) * K + k + col];
        }
        
        // Load B: 16 rows x 32 cols = 512 elements
        // 128 threads, each loads 4 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 128;
            int srcRow = idx / BLOCK_N;
            int srcCol = idx % BLOCK_N;
            sB[srcCol * sB_stride + srcRow] = B[(k + srcRow) * N + tileN + srcCol];
        }
        
        __syncthreads();
        
        wmma::load_matrix_sync(a_frag, &sA[warpBaseM * sA_stride], sA_stride);
        wmma::load_matrix_sync(b_frag, &sB[warpBaseN * sB_stride], sB_stride);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        __syncthreads();
    }
    
    const int outRowBase = tileM + warpBaseM;
    const int outColBase = tileN + warpBaseN;
    
    wmma::store_matrix_sync(&C[outRowBase * N + outColBase], c_frag, N, wmma::mem_row_major);
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
    
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_1tile<<<grid, block, smemSize>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_1tile<<<grid, block, smemSize>>>(dA, dB, dC, M, N, K);
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
