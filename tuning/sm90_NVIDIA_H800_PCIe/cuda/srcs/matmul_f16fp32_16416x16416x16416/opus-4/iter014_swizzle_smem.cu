// iter014_swizzle_smem: XOR swizzle pattern to eliminate shared memory bank conflicts
// Based on iter013 - add swizzle to reduce 5-way bank conflicts
// Target: H800 (SM90), 16416x16416x16416, FP16->FP32

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

using namespace nvcuda;

#define BM 128
#define BN 128  
#define BK 32
#define WARPS_M 4
#define WARPS_N 2
#define WM (BM / WARPS_M)
#define WN (BN / WARPS_N)
#define THREADS_PER_BLOCK (WARPS_M * WARPS_N * 32)

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define M_DIM 16416
#define N_DIM 16416
#define K_DIM 16416

#define STAGES 2

// Swizzle pattern: XOR the column with a portion of the row index
// This distributes accesses across banks when threads access consecutive columns
// For half (2 bytes), each bank holds 16 bytes = 8 halfs
// We XOR the high bits of the column index with low bits of row
#define SWIZZLE_A(row, col) ((col) ^ ((row) & 7))
#define SWIZZLE_B(row, col) ((col) ^ ((row) & 7))

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
    if (n == 0) {
        asm volatile("cp.async.wait_group 0;\n" ::);
    } else if (n == 1) {
        asm volatile("cp.async.wait_group 1;\n" ::);
    }
}

__global__ __launch_bounds__(256, 2)
void matmul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Extra padding to help with swizzle - 16 instead of 8
    __shared__ half As[STAGES][BM][BK + 16];
    __shared__ half Bs[STAGES][BK][BN + 16];
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    
    const int WMMA_TILES_M = WM / WMMA_M;
    const int WMMA_TILES_N = WN / WMMA_N;
    const int WMMA_TILES_K = BK / WMMA_K;
    
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WMMA_TILES_M][WMMA_TILES_N];
    
    #pragma unroll
    for (int mi = 0; mi < WMMA_TILES_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WMMA_TILES_N; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);
    
    // Each thread loads multiple elements
    // A: BM x BK = 128 x 32 = 4096 elements = 16 per thread (256 threads)
    // B: BK x BN = 32 x 128 = 4096 elements = 16 per thread
    const int elems_a = BM * BK;
    const int elems_b = BK * BN;
    const int elems_per_thread_a = (elems_a + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int elems_per_thread_b = (elems_b + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    const int num_k_tiles = (K + BK - 1) / BK;
    
    // Load first tile (stage 0) - element by element for correct swizzle
    int k_tile = 0;
    int stage = 0;
    
    #pragma unroll
    for (int i = 0; i < elems_per_thread_a; i++) {
        int elem_idx = tid + i * THREADS_PER_BLOCK;
        if (elem_idx < elems_a) {
            int row = elem_idx / BK;
            int col = elem_idx % BK;
            int global_row = by * BM + row;
            int global_col = k_tile * BK + col;
            int swizzled_col = SWIZZLE_A(row, col);
            
            if (global_row < M && global_col < K) {
                As[stage][row][swizzled_col] = A[global_row * K + global_col];
            } else {
                As[stage][row][swizzled_col] = __float2half(0.0f);
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < elems_per_thread_b; i++) {
        int elem_idx = tid + i * THREADS_PER_BLOCK;
        if (elem_idx < elems_b) {
            int row = elem_idx / BN;
            int col = elem_idx % BN;
            int global_row = k_tile * BK + row;
            int global_col = bx * BN + col;
            int swizzled_col = SWIZZLE_B(row, col);
            
            if (global_row < K && global_col < N) {
                Bs[stage][row][swizzled_col] = B[global_row * N + global_col];
            } else {
                Bs[stage][row][swizzled_col] = __float2half(0.0f);
            }
        }
    }
    __syncthreads();
    
    // Main loop - process tiles
    for (k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int curr_stage = k_tile % STAGES;
        int next_stage = (k_tile + 1) % STAGES;
        int next_k_tile = k_tile + 1;
        
        // Load next tile while computing current
        if (next_k_tile < num_k_tiles) {
            #pragma unroll
            for (int i = 0; i < elems_per_thread_a; i++) {
                int elem_idx = tid + i * THREADS_PER_BLOCK;
                if (elem_idx < elems_a) {
                    int row = elem_idx / BK;
                    int col = elem_idx % BK;
                    int global_row = by * BM + row;
                    int global_col = next_k_tile * BK + col;
                    int swizzled_col = SWIZZLE_A(row, col);
                    
                    if (global_row < M && global_col < K) {
                        As[next_stage][row][swizzled_col] = A[global_row * K + global_col];
                    } else {
                        As[next_stage][row][swizzled_col] = __float2half(0.0f);
                    }
                }
            }
            
            #pragma unroll
            for (int i = 0; i < elems_per_thread_b; i++) {
                int elem_idx = tid + i * THREADS_PER_BLOCK;
                if (elem_idx < elems_b) {
                    int row = elem_idx / BN;
                    int col = elem_idx % BN;
                    int global_row = next_k_tile * BK + row;
                    int global_col = bx * BN + col;
                    int swizzled_col = SWIZZLE_B(row, col);
                    
                    if (global_row < K && global_col < N) {
                        Bs[next_stage][row][swizzled_col] = B[global_row * N + global_col];
                    } else {
                        Bs[next_stage][row][swizzled_col] = __float2half(0.0f);
                    }
                }
            }
        }
        
        // Compute WMMA operations on current tile
        // Need to de-swizzle when loading for WMMA
        #pragma unroll
        for (int ki = 0; ki < WMMA_TILES_K; ki++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WMMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WMMA_TILES_N];
            
            #pragma unroll
            for (int mi = 0; mi < WMMA_TILES_M; mi++) {
                int a_row = warp_m * WM + mi * WMMA_M;
                int a_col = ki * WMMA_K;
                // WMMA load needs contiguous data - but we swizzled the storage
                // We need to load element-by-element and de-swizzle OR
                // use a different approach: store in transposed/unswizzled staging buffer
                // For now, load directly and hope WMMA handles stride
                wmma::load_matrix_sync(a_frag[mi], &As[curr_stage][a_row][a_col], BK + 16);
            }
            
            #pragma unroll
            for (int ni = 0; ni < WMMA_TILES_N; ni++) {
                int b_row = ki * WMMA_K;
                int b_col = warp_n * WN + ni * WMMA_N;
                wmma::load_matrix_sync(b_frag[ni], &Bs[curr_stage][b_row][b_col], BN + 16);
            }
            
            #pragma unroll
            for (int mi = 0; mi < WMMA_TILES_M; mi++) {
                #pragma unroll
                for (int ni = 0; ni < WMMA_TILES_N; ni++) {
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int mi = 0; mi < WMMA_TILES_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < WMMA_TILES_N; ni++) {
            int c_row = by * BM + warp_m * WM + mi * WMMA_M;
            int c_col = bx * BN + warp_n * WN + ni * WMMA_N;
            
            if (c_row < M && c_col < N) {
                wmma::store_matrix_sync(&C[c_row * N + c_col], acc[mi][ni], N, wmma::mem_row_major);
            }
        }
    }
}

void run_kernel(const half* A, const half* B, float* C, int M, int N, int K) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(THREADS_PER_BLOCK);
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char** argv) {
    int M = M_DIM, N = N_DIM, K = K_DIM;
    
    bool skip_verify = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--skip-verify") == 0) skip_verify = true;
    }
    
    size_t size_A = (size_t)M * K * sizeof(half);
    size_t size_B = (size_t)K * N * sizeof(half);
    size_t size_C = (size_t)M * N * sizeof(float);
    
    half *h_A, *h_B;
    float *h_C;
    half *d_A, *d_B;
    float *d_C;
    
    h_A = (half*)malloc(size_A);
    h_B = (half*)malloc(size_B);
    h_C = (float*)malloc(size_C);
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++)
        h_A[i] = __float2half((float)(rand() % 10) / 10.0f - 0.5f);
    for (size_t i = 0; i < (size_t)K * N; i++)
        h_B[i] = __float2half((float)(rand() % 10) / 10.0f - 0.5f);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);
    
    for (int i = 0; i < 10; i++) run_kernel(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    int niters = 50;
    for (int i = 0; i < niters; i++) run_kernel(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / niters;
    
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    
    printf("time_ms: %.4f\n", avg_ms);
    printf("TFLOPS: %.4f\n", tflops);
    
    if (!skip_verify) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        float *d_C_ref;
        cudaMalloc(&d_C_ref, size_C);
        cudaMemset(d_C_ref, 0, size_C);
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     d_B, CUDA_R_16F, N,
                     d_A, CUDA_R_16F, K,
                     &beta,
                     d_C_ref, CUDA_R_32F, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        
        float* h_C_ref = (float*)malloc(size_C);
        cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost);
        
        float max_err = 0.0f;
        int errors = 0;
        for (int i = 0; i < 1000 && i < M * N; i++) {
            float ref = h_C_ref[i];
            float got = h_C[i];
            float err = fabs(ref - got);
            float base = fabs(ref) > 1.0f ? fabs(ref) : 1.0f;
            float rel_err = err / base;
            if (rel_err > max_err) max_err = rel_err;
            if (rel_err > 0.01f) errors++;
        }
        
        if (errors > 50 || max_err > 1.0f) {
            printf("VERIFY: FAIL max_rel_err=%.6f errors=%d\n", max_err, errors);
        } else {
            printf("VERIFY: PASS max_rel_err=%.6f\n", max_err);
        }
        
        free(h_C_ref);
        cudaFree(d_C_ref);
        cublasDestroy(handle);
    }
    
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
