// iter007_ptx_mma: Raw PTX ldmatrix + mma.sync for lower overhead
// Change from iter004: Replace WMMA with direct PTX for m16n8k16 MMA
// Target: H800 (SM90), 16416x16416x16416, FP16->FP32

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// Same tile sizes as iter004
#define BM 128
#define BN 128  
#define BK 64
#define WARPS_M 4
#define WARPS_N 2
#define WM (BM / WARPS_M)  // 32
#define WN (BN / WARPS_N)  // 64
#define THREADS_PER_BLOCK (WARPS_M * WARPS_N * 32)  // 256

// MMA dimensions: m16n8k16
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

// Problem dimensions
#define M_DIM 16416
#define N_DIM 16416
#define K_DIM 16416

// ldmatrix loads 4 8x8 matrices (x4) or 2 8x8 matrices (x2)
__device__ __forceinline__ void ldmatrix_x4(uint32_t dst[4], const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t dst[2], const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(smem_addr)
    );
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
__device__ __forceinline__ void mma_m16n8k16_f32_f16(
    float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

__global__ void matmul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Shared memory with swizzle-friendly stride (multiple of 8 for ldmatrix)
    // Using stride of 72 (64+8) for A, 136 (128+8) for B
    __shared__ half As[BM][BK + 8];    // 128 x 72
    __shared__ half Bs[BK][BN + 8];    // 64 x 136
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    
    // Per warp: WM=32 x WN=64 output
    // MMA tile: 16x8, so 2x8 MMA tiles per warp
    const int MMA_TILES_M = WM / MMA_M;  // 32/16 = 2
    const int MMA_TILES_N = WN / MMA_N;  // 64/8 = 8
    const int MMA_TILES_K = BK / MMA_K;  // 64/16 = 4
    
    // Accumulators: 2x8 MMA tiles, 4 floats each
    float acc[MMA_TILES_M][MMA_TILES_N][4];
    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ni++)
            #pragma unroll
            for (int k = 0; k < 4; k++)
                acc[mi][ni][k] = 0.0f;
    
    // Vectorized load config
    const int vec_per_row_a = BK / 8;  // 8
    const int vec_per_row_b = BN / 8;  // 16
    const int total_vec_a = BM * vec_per_row_a;  // 1024
    const int total_vec_b = BK * vec_per_row_b;  // 1024
    const int loads_per_thread = (total_vec_a + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Main K-loop
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // Load A with vectorized loads
        #pragma unroll
        for (int i = 0; i < loads_per_thread; i++) {
            int vec_idx = tid + i * THREADS_PER_BLOCK;
            if (vec_idx < total_vec_a) {
                int row = vec_idx / vec_per_row_a;
                int vec_col = vec_idx % vec_per_row_a;
                int global_row = by * BM + row;
                int global_col = k_tile + vec_col * 8;
                
                if (global_row < M && global_col + 7 < K) {
                    int4 data = *reinterpret_cast<const int4*>(&A[global_row * K + global_col]);
                    *reinterpret_cast<int4*>(&As[row][vec_col * 8]) = data;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        int gc = global_col + j;
                        if (global_row < M && gc < K)
                            As[row][vec_col * 8 + j] = A[global_row * K + gc];
                        else
                            As[row][vec_col * 8 + j] = __float2half(0.0f);
                    }
                }
            }
        }
        
        // Load B
        #pragma unroll
        for (int i = 0; i < loads_per_thread; i++) {
            int vec_idx = tid + i * THREADS_PER_BLOCK;
            if (vec_idx < total_vec_b) {
                int row = vec_idx / vec_per_row_b;
                int vec_col = vec_idx % vec_per_row_b;
                int global_row = k_tile + row;
                int global_col = bx * BN + vec_col * 8;
                
                if (global_row < K && global_col + 7 < N) {
                    int4 data = *reinterpret_cast<const int4*>(&B[global_row * N + global_col]);
                    *reinterpret_cast<int4*>(&Bs[row][vec_col * 8]) = data;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        int gc = global_col + j;
                        if (global_row < K && gc < N)
                            Bs[row][vec_col * 8 + j] = B[global_row * N + gc];
                        else
                            Bs[row][vec_col * 8 + j] = __float2half(0.0f);
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Compute with PTX MMA
        #pragma unroll
        for (int ki = 0; ki < MMA_TILES_K; ki++) {
            // For m16n8k16.row.col:
            // A: 16 rows x 16 cols, row-major in smem
            // B: 16 rows x 8 cols, col-major access (transpose)
            
            uint32_t a_frag[MMA_TILES_M][4];
            uint32_t b_frag[MMA_TILES_N][2];
            
            // Load A fragments using ldmatrix.x4
            // Each ldmatrix.x4 loads 4 8x8 matrices = 16x16 elements
            // For row-major A, we need to load per warp_m tile
            #pragma unroll
            for (int mi = 0; mi < MMA_TILES_M; mi++) {
                // Row within the A tile for this thread's contribution to ldmatrix
                // ldmatrix: each of 32 threads provides address for one row of 8 elements
                int a_row_base = warp_m * WM + mi * MMA_M;
                int a_col_base = ki * MMA_K;
                
                // Thread layout for ldmatrix.x4: threads 0-7 load rows 0-7, threads 8-15 load rows 8-15, etc.
                int thread_row = lane_id % 16;
                int thread_group = lane_id / 16;  // 0 or 1
                
                // For x4, we need to load 16 rows of 16 elements (4 8x8 matrices)
                // Each thread provides address for its row
                const half* a_ptr = &As[a_row_base + thread_row][a_col_base + thread_group * 8];
                ldmatrix_x4(a_frag[mi], a_ptr);
            }
            
            // Load B fragments using ldmatrix.x2.trans
            // For col-major B access (mma expects B transposed conceptually)
            // B is stored row-major in smem, so we need transposed load
            #pragma unroll
            for (int ni = 0; ni < MMA_TILES_N; ni++) {
                int b_row_base = ki * MMA_K;
                int b_col_base = warp_n * WN + ni * MMA_N;
                
                // For ldmatrix.trans.x2: loads 2 8x8 matrices transposed
                int thread_row = lane_id % 16;
                const half* b_ptr = &Bs[b_row_base + thread_row][b_col_base];
                ldmatrix_x2_trans(b_frag[ni], b_ptr);
            }
            
            // Execute MMA
            #pragma unroll
            for (int mi = 0; mi < MMA_TILES_M; mi++) {
                #pragma unroll
                for (int ni = 0; ni < MMA_TILES_N; ni++) {
                    mma_m16n8k16_f32_f16(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    // MMA m16n8k16 output layout for FP32:
    // Each thread holds 4 values: (row0, col0-1), (row8, col0-1)
    // Thread layout: row = (lane_id % 4) * 2 or * 2 + 1 for the 8-row groups
    //                col = lane_id / 4
    // Actually for m16n8k16.row.col.f32: 
    //   acc[0],acc[1] are at (row, col), (row, col+1) where row = lane_id/4, col pair = lane_id%4
    //   acc[2],acc[3] are at (row+8, col), (row+8, col+1)
    
    #pragma unroll
    for (int mi = 0; mi < MMA_TILES_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < MMA_TILES_N; ni++) {
            int base_row = by * BM + warp_m * WM + mi * MMA_M;
            int base_col = bx * BN + warp_n * WN + ni * MMA_N;
            
            // m16n8k16 output mapping
            int row0 = base_row + (lane_id / 4);
            int row8 = base_row + (lane_id / 4) + 8;
            int col = base_col + (lane_id % 4) * 2;
            
            if (row0 < M && col < N)     C[row0 * N + col] = acc[mi][ni][0];
            if (row0 < M && col + 1 < N) C[row0 * N + col + 1] = acc[mi][ni][1];
            if (row8 < M && col < N)     C[row8 * N + col] = acc[mi][ni][2];
            if (row8 < M && col + 1 < N) C[row8 * N + col + 1] = acc[mi][ni][3];
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
