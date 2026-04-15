// iter022_b_col_major.cu
// Based on iter002 but with B stored col-major in shared memory
// This should improve B fragment loading efficiency

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
constexpr int SMEM_PAD = 8;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;

__global__ __launch_bounds__(128, 4)
void matmul_bf16_fp32_bcol(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int warpId = threadIdx.x / WARP_SIZE;
    int warpM = warpId / WARPS_N;
    int warpN = warpId % WARPS_N;
    
    int tileM = by * BLOCK_M;
    int tileN = bx * BLOCK_N;
    
    // sA: row-major [M][K]
    // sB: col-major [N][K] (transposed storage)
    __shared__ __nv_bfloat16 sA[BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __nv_bfloat16 sB[BLOCK_N][BLOCK_K + SMEM_PAD];
    
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
    
    int tid = threadIdx.x;
    
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A (row-major)
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 128;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            int gRow = tileM + row;
            int gCol = k + col;
            sA[row][col] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : __float2bfloat16(0.0f);
        }
        
        // Load B and transpose to col-major [N][K]
        // From B[k:k+16, tileN:tileN+64] to sB[0:64, 0:16]
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 128;
            int srcRow = idx / BLOCK_N;  // K dimension
            int srcCol = idx % BLOCK_N;  // N dimension
            int gRow = k + srcRow;
            int gCol = tileN + srcCol;
            // Store transposed: sB[N][K]
            sB[srcCol][srcRow] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : __float2bfloat16(0.0f);
        }
        
        __syncthreads();
        
        int warpBaseM = warpM * 32;
        int warpBaseN = warpN * 32;
        
        // Load A fragments (row-major)
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(a_frag[i], &sA[warpBaseM + i * WMMA_M][0], BLOCK_K + SMEM_PAD);
        }
        
        // Load B fragments (col-major from sB[N][K])
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::load_matrix_sync(b_frag[j], &sB[warpBaseN + j * WMMA_N][0], BLOCK_K + SMEM_PAD);
        }
        
        // Compute
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    int warpBaseM = warpM * 32;
    int warpBaseN = warpN * 32;
    
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
    
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(128);
    
    for (int i = 0; i < warmup; i++) {
        matmul_bf16_fp32_bcol<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_bf16_fp32_bcol<<<grid, block>>>(dA, dB, dC, M, N, K);
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
