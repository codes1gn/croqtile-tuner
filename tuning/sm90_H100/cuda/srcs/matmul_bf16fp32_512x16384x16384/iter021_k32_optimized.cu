#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

// iter002 + K-tile=32 for better compute/memory ratio
// Keep 64x64 block, 4 warps, but double K-tile
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;  // Doubled from 16
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int SMEM_PAD = 8;

__global__ __launch_bounds__(128, 4)
void matmul_k32(const __nv_bfloat16* __restrict__ A,
                const __nv_bfloat16* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K)
{
    __shared__ __nv_bfloat16 sA[BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __nv_bfloat16 sB[BLOCK_K][BLOCK_N + SMEM_PAD];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int warp_id = threadIdx.x / 32;
    int tid = threadIdx.x;
    
    // 4 warps: 2x2 layout
    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;
    
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
    
    int m_block = by * BLOCK_M;
    int n_block = bx * BLOCK_N;
    
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // Load A: 128 threads, 64*32 = 2048 elements, 16 elements per thread
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int idx = tid * 16 + i;
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            int g_row = m_block + row;
            int g_col = k_tile + col;
            sA[row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : __float2bfloat16(0.0f);
        }
        
        // Load B: 128 threads, 32*64 = 2048 elements, 16 elements per thread
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int idx = tid * 16 + i;
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            int g_row = k_tile + row;
            int g_col = n_block + col;
            sB[row][col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : __float2bfloat16(0.0f);
        }
        
        __syncthreads();
        
        // K-loop: 2 iterations for K=32
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k += WMMA_K) {
            // Load A fragments
            #pragma unroll
            for (int wm = 0; wm < 2; wm++) {
                int a_row = warp_m * 32 + wm * WMMA_M;
                wmma::load_matrix_sync(a_frag[wm], &sA[a_row][k], BLOCK_K + SMEM_PAD);
            }
            
            // Load B fragments
            #pragma unroll
            for (int wn = 0; wn < 2; wn++) {
                int b_col = warp_n * 32 + wn * WMMA_N;
                wmma::load_matrix_sync(b_frag[wn], &sB[k][b_col], BLOCK_N + SMEM_PAD);
            }
            
            // Compute
            #pragma unroll
            for (int wm = 0; wm < 2; wm++) {
                #pragma unroll
                for (int wn = 0; wn < 2; wn++) {
                    wmma::mma_sync(c_frag[wm][wn], a_frag[wm], b_frag[wn], c_frag[wm][wn]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int wm = 0; wm < 2; wm++) {
        #pragma unroll
        for (int wn = 0; wn < 2; wn++) {
            int c_row = m_block + warp_m * 32 + wm * WMMA_M;
            int c_col = n_block + warp_n * 32 + wn * WMMA_N;
            if (c_row < M && c_col < N) {
                wmma::store_matrix_sync(&C[c_row * N + c_col], c_frag[wm][wn], N, wmma::mem_row_major);
            }
        }
    }
}

int main(int argc, char** argv) {
    int M = 512, N = 16384, K = 16384;
    int warmup = (argc > 1) ? atoi(argv[1]) : 3;
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
        matmul_k32<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_k32<<<grid, block>>>(dA, dB, dC, M, N, K);
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
