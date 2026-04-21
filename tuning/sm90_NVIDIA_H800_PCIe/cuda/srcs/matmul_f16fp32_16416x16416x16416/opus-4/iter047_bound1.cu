// iter047_bound1: Tighter main loop with complete unroll of inner K iterations
// Based on iter013 - add #pragma unroll to main K loop and simplify control flow
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

__device__ __forceinline__ void cp_async_128(void* dst, const void* src) {
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_addr), "l"(src)
    );
}

__global__ __launch_bounds__(256, 1)
void matmul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ half As[STAGES][BM][BK + 8];
    __shared__ half Bs[STAGES][BK][BN + 8];
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    
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
    
    const int vec_per_row_a = BK / 8;
    const int vec_per_row_b = BN / 8;
    const int total_vec_a = BM * vec_per_row_a;
    const int total_vec_b = BK * vec_per_row_b;
    const int loads_per_thread = (total_vec_a + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    const int num_k_tiles = (K + BK - 1) / BK;
    
    // Load first tile (stage 0) with cp.async
    int k_tile = 0;
    int stage = 0;
    
    #pragma unroll
    for (int i = 0; i < loads_per_thread; i++) {
        int vec_idx = tid + i * THREADS_PER_BLOCK;
        if (vec_idx < total_vec_a) {
            int row = vec_idx / vec_per_row_a;
            int vec_col = vec_idx % vec_per_row_a;
            int global_row = by * BM + row;
            int global_col = k_tile * BK + vec_col * 8;
            
            if (global_row < M && global_col + 7 < K) {
                cp_async_128(&As[stage][row][vec_col * 8], &A[global_row * K + global_col]);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int gc = global_col + j;
                    if (global_row < M && gc < K)
                        As[stage][row][vec_col * 8 + j] = A[global_row * K + gc];
                    else
                        As[stage][row][vec_col * 8 + j] = __float2half(0.0f);
                }
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < loads_per_thread; i++) {
        int vec_idx = tid + i * THREADS_PER_BLOCK;
        if (vec_idx < total_vec_b) {
            int row = vec_idx / vec_per_row_b;
            int vec_col = vec_idx % vec_per_row_b;
            int global_row = k_tile * BK + row;
            int global_col = bx * BN + vec_col * 8;
            
            if (global_row < K && global_col + 7 < N) {
                cp_async_128(&Bs[stage][row][vec_col * 8], &B[global_row * N + global_col]);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int gc = global_col + j;
                    if (global_row < K && gc < N)
                        Bs[stage][row][vec_col * 8 + j] = B[global_row * N + gc];
                    else
                        Bs[stage][row][vec_col * 8 + j] = __float2half(0.0f);
                }
            }
        }
    }
    cp_async_commit();
    
    // Main loop - process tiles with unroll hint
    #pragma unroll 4
    for (k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int curr_stage = k_tile % STAGES;
        int next_stage = (k_tile + 1) % STAGES;
        int next_k_tile = k_tile + 1;
        
        // Load next tile while waiting for current
        if (next_k_tile < num_k_tiles) {
            #pragma unroll
            for (int i = 0; i < loads_per_thread; i++) {
                int vec_idx = tid + i * THREADS_PER_BLOCK;
                if (vec_idx < total_vec_a) {
                    int row = vec_idx / vec_per_row_a;
                    int vec_col = vec_idx % vec_per_row_a;
                    int global_row = by * BM + row;
                    int global_col = next_k_tile * BK + vec_col * 8;
                    
                    if (global_row < M && global_col + 7 < K) {
                        cp_async_128(&As[next_stage][row][vec_col * 8], &A[global_row * K + global_col]);
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            int gc = global_col + j;
                            if (global_row < M && gc < K)
                                As[next_stage][row][vec_col * 8 + j] = A[global_row * K + gc];
                            else
                                As[next_stage][row][vec_col * 8 + j] = __float2half(0.0f);
                        }
                    }
                }
            }
            
            #pragma unroll
            for (int i = 0; i < loads_per_thread; i++) {
                int vec_idx = tid + i * THREADS_PER_BLOCK;
                if (vec_idx < total_vec_b) {
                    int row = vec_idx / vec_per_row_b;
                    int vec_col = vec_idx % vec_per_row_b;
                    int global_row = next_k_tile * BK + row;
                    int global_col = bx * BN + vec_col * 8;
                    
                    if (global_row < K && global_col + 7 < N) {
                        cp_async_128(&Bs[next_stage][row][vec_col * 8], &B[global_row * N + global_col]);
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            int gc = global_col + j;
                            if (global_row < K && gc < N)
                                Bs[next_stage][row][vec_col * 8 + j] = B[global_row * N + gc];
                            else
                                Bs[next_stage][row][vec_col * 8 + j] = __float2half(0.0f);
                        }
                    }
                }
            }
            cp_async_commit();
        }
        
        // Wait for current tile
        cp_async_wait_group(next_k_tile < num_k_tiles ? 1 : 0);
        __syncthreads();
        
        // Compute WMMA operations on current tile
        #pragma unroll
        for (int ki = 0; ki < WMMA_TILES_K; ki++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WMMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WMMA_TILES_N];
            
            #pragma unroll
            for (int mi = 0; mi < WMMA_TILES_M; mi++) {
                int a_row = warp_m * WM + mi * WMMA_M;
                int a_col = ki * WMMA_K;
                wmma::load_matrix_sync(a_frag[mi], &As[curr_stage][a_row][a_col], BK + 8);
            }
            
            #pragma unroll
            for (int ni = 0; ni < WMMA_TILES_N; ni++) {
                int b_row = ki * WMMA_K;
                int b_col = warp_n * WN + ni * WMMA_N;
                wmma::load_matrix_sync(b_frag[ni], &Bs[curr_stage][b_row][b_col], BN + 8);
            }
            
            #pragma unroll
            for (int mi = 0; mi < WMMA_TILES_M; mi++) {
                #pragma unroll
                for (int ni = 0; ni < WMMA_TILES_N; ni++) {
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                }
            }
        }
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
