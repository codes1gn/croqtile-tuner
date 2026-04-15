// iter042_min_smem.cu
// Minimize shared memory usage for higher occupancy
// Use 32x32 block instead of 64x64

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

constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = 64;  // 2 warps

extern __shared__ __nv_bfloat16 smem[];

__global__ __launch_bounds__(64, 16)  // Target high occupancy
void matmul_bf16_fp32_min_smem(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    __nv_bfloat16* __restrict__ sA = smem;
    __nv_bfloat16* __restrict__ sB = smem + BLOCK_M * (BLOCK_K + SMEM_PAD);
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int warpId = threadIdx.x / WARP_SIZE;  // 0 or 1
    
    const int tileM = bx * BLOCK_M;
    const int tileN = by * BLOCK_N;
    
    // Each warp handles 16x16
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    const int tid = threadIdx.x;
    // Warp 0: rows 0-15, Warp 1: rows 16-31 (different C output)
    // But both compute full 32x32 output, so we need 2x2 WMMA tiles per warp
    
    // Actually let each warp handle diagonal: warp0: (0,0) and (1,1), warp1: (0,1) and (1,0)
    // No, simpler: warp0 handles top-left 16x32, warp1 handles bottom-left 16x32
    const int warpBaseM = warpId * WMMA_M;
    
    const int sA_stride = BLOCK_K + SMEM_PAD;
    const int sB_stride = BLOCK_K + SMEM_PAD;
    
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag2[2];
    wmma::fill_fragment(c_frag2[0], 0.0f);
    wmma::fill_fragment(c_frag2[1], 0.0f);
    
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag2[2];
    
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A: 64 threads, 32*16 = 512 elements, 8 per thread
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * NUM_THREADS;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            sA[row * sA_stride + col] = A[(tileM + row) * K + k + col];
        }
        
        // Load B: 64 threads, 16*32 = 512 elements, 8 per thread
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * NUM_THREADS;
            int srcRow = idx / BLOCK_N;
            int srcCol = idx % BLOCK_N;
            sB[srcCol * sB_stride + srcRow] = B[(k + srcRow) * N + tileN + srcCol];
        }
        
        __syncthreads();
        
        // Each warp loads its A fragment and both B fragments
        wmma::load_matrix_sync(a_frag, &sA[warpBaseM * sA_stride], sA_stride);
        wmma::load_matrix_sync(b_frag2[0], &sB[0], sB_stride);
        wmma::load_matrix_sync(b_frag2[1], &sB[WMMA_N * sB_stride], sB_stride);
        
        wmma::mma_sync(c_frag2[0], a_frag, b_frag2[0], c_frag2[0]);
        wmma::mma_sync(c_frag2[1], a_frag, b_frag2[1], c_frag2[1]);
        
        __syncthreads();
    }
    
    const int outRowBase = tileM + warpBaseM;
    
    wmma::store_matrix_sync(&C[outRowBase * N + tileN], c_frag2[0], N, wmma::mem_row_major);
    wmma::store_matrix_sync(&C[outRowBase * N + tileN + WMMA_N], c_frag2[1], N, wmma::mem_row_major);
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
    dim3 block(NUM_THREADS);
    
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_min_smem<<<grid, block, smemSize>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_min_smem<<<grid, block, smemSize>>>(dA, dB, dC, M, N, K);
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
