// iter055_6warps.cu
// 64x96 block with 6 warps (3x2 configuration)
// Each warp handles 32x32, so 64x96 output per block

#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 96;
constexpr int BLOCK_K = 16;
constexpr int SMEM_PAD = 4;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 3;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 6 warps = 192 threads

extern __shared__ __nv_bfloat16 smem[];

__global__ __launch_bounds__(192, 3)
void matmul_bf16_fp32_6warps(
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
    const int warpM = warpId / WARPS_N;  // 0-1
    const int warpN = warpId % WARPS_N;  // 0-2
    
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
    const int sA_stride = BLOCK_K + SMEM_PAD;
    const int sB_stride = BLOCK_K + SMEM_PAD;
    
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A: 192 threads, 64*16 = 1024 elements, ~5.3 per thread
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            int idx = tid + i * 192;
            if (idx < BLOCK_M * BLOCK_K) {
                int row = idx / BLOCK_K;
                int col = idx % BLOCK_K;
                int gRow = tileM + row;
                int gCol = k + col;
                sA[row * sA_stride + col] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : __float2bfloat16(0.0f);
            }
        }
        
        // Load B: 192 threads, 16*96 = 1536 elements, 8 per thread
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 192;
            if (idx < BLOCK_K * BLOCK_N) {
                int srcRow = idx / BLOCK_N;
                int srcCol = idx % BLOCK_N;
                int gRow = k + srcRow;
                int gCol = tileN + srcCol;
                sB[srcCol * sB_stride + srcRow] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : __float2bfloat16(0.0f);
            }
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
    
    if (outRowBase < M && outColBase < N) {
        wmma::store_matrix_sync(&C[outRowBase * N + outColBase], c_frag[0][0], N, wmma::mem_row_major);
    }
    if (outRowBase < M && outColBase + WMMA_N < N) {
        wmma::store_matrix_sync(&C[outRowBase * N + outColBase + WMMA_N], c_frag[0][1], N, wmma::mem_row_major);
    }
    if (outRowBase + WMMA_M < M && outColBase < N) {
        wmma::store_matrix_sync(&C[(outRowBase + WMMA_M) * N + outColBase], c_frag[1][0], N, wmma::mem_row_major);
    }
    if (outRowBase + WMMA_M < M && outColBase + WMMA_N < N) {
        wmma::store_matrix_sync(&C[(outRowBase + WMMA_M) * N + outColBase + WMMA_N], c_frag[1][1], N, wmma::mem_row_major);
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
    dim3 block(192);
    
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_6warps<<<grid, block, smemSize>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_6warps<<<grid, block, smemSize>>>(dA, dB, dC, M, N, K);
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
