// iter056_ilp8.cu: Try ILP8 (process 8 k_groups = 32 K elements at once)
// Based on iter055 (10.59 TFLOPS)

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <random>
#include <iostream>
#include <cstdint>
#include <cstring>

constexpr int M = 4096;
constexpr int N = 8192;
constexpr int K = 8192;

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 64;

constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr int THREADS_PER_BLOCK = THREADS_X * THREADS_Y;

constexpr int OUTPUTS_PER_THREAD_M = TILE_M / THREADS_Y;
constexpr int OUTPUTS_PER_THREAD_N = TILE_N / THREADS_X;

#define H800_PCIE_PEAK_F16_TFLOPS 1513.0

__global__ __launch_bounds__(256, 4)
void sparse_gemm_ilp8_kernel(
    const __nv_fp8_e4m3* __restrict__ A_packed,
    const uint32_t* __restrict__ A_meta,
    const __nv_fp8_e4m3* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    int block_m = blockIdx.y * TILE_M;
    int block_n = blockIdx.x * TILE_N;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * THREADS_X + tx;
    
    __shared__ __nv_fp8_e4m3 As[TILE_M][TILE_K / 2 + 4];
    __shared__ uint32_t Am[TILE_M][TILE_K / 32 + 1];
    __shared__ __nv_fp8_e4m3 Bs[TILE_N][TILE_K + 4];
    
    float acc[OUTPUTS_PER_THREAD_M][OUTPUTS_PER_THREAD_N];
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < OUTPUTS_PER_THREAD_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    
    const int packed_k = k / 2;
    const int meta_k = k / 32;
    const int num_k_tiles = (k + TILE_K - 1) / TILE_K;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int k_offset = k_tile * TILE_K;
        
        #pragma unroll 4
        for (int i = tid; i < TILE_M * (TILE_K / 2); i += THREADS_PER_BLOCK) {
            int load_m = i / (TILE_K / 2);
            int load_k = i % (TILE_K / 2);
            int global_m = block_m + load_m;
            int global_k = k_offset / 2 + load_k;
            As[load_m][load_k] = (global_m < m && global_k < packed_k) 
                ? A_packed[global_m * packed_k + global_k] 
                : __nv_fp8_e4m3(0.0f);
        }
        
        #pragma unroll 2
        for (int i = tid; i < TILE_M * (TILE_K / 32); i += THREADS_PER_BLOCK) {
            int load_m = i / (TILE_K / 32);
            int load_k = i % (TILE_K / 32);
            int global_m = block_m + load_m;
            int global_k = k_offset / 32 + load_k;
            Am[load_m][load_k] = (global_m < m && global_k < meta_k) 
                ? A_meta[global_m * meta_k + global_k] 
                : 0;
        }
        
        #pragma unroll 4
        for (int i = tid; i < TILE_N * TILE_K; i += THREADS_PER_BLOCK) {
            int load_n = i / TILE_K;
            int load_k = i % TILE_K;
            int global_n = block_n + load_n;
            int global_k = k_offset + load_k;
            Bs[load_n][load_k] = (global_n < n && global_k < k) 
                ? B[global_n * k + global_k] 
                : __nv_fp8_e4m3(0.0f);
        }
        
        __syncthreads();
        
        // Compute ILP8: process all 8 k_groups from meta0, then all 8 from meta1
        #pragma unroll
        for (int i = 0; i < OUTPUTS_PER_THREAD_M; ++i) {
            int local_m = ty * OUTPUTS_PER_THREAD_M + i;
            
            // First half: k_groups 0-7 (meta0)
            uint32_t meta0 = Am[local_m][0];
            
            // Precompute all 8 index pairs for meta0
            int idx_0_0 = (meta0 >> 0) & 0x3, idx_0_1 = (meta0 >> 2) & 0x3;
            int idx_1_0 = (meta0 >> 4) & 0x3, idx_1_1 = (meta0 >> 6) & 0x3;
            int idx_2_0 = (meta0 >> 8) & 0x3, idx_2_1 = (meta0 >> 10) & 0x3;
            int idx_3_0 = (meta0 >> 12) & 0x3, idx_3_1 = (meta0 >> 14) & 0x3;
            int idx_4_0 = (meta0 >> 16) & 0x3, idx_4_1 = (meta0 >> 18) & 0x3;
            int idx_5_0 = (meta0 >> 20) & 0x3, idx_5_1 = (meta0 >> 22) & 0x3;
            int idx_6_0 = (meta0 >> 24) & 0x3, idx_6_1 = (meta0 >> 26) & 0x3;
            int idx_7_0 = (meta0 >> 28) & 0x3, idx_7_1 = (meta0 >> 30) & 0x3;
            
            // Load all A values for first 8 k_groups
            float a0_0 = float(As[local_m][0]),  a0_1 = float(As[local_m][1]);
            float a1_0 = float(As[local_m][2]),  a1_1 = float(As[local_m][3]);
            float a2_0 = float(As[local_m][4]),  a2_1 = float(As[local_m][5]);
            float a3_0 = float(As[local_m][6]),  a3_1 = float(As[local_m][7]);
            float a4_0 = float(As[local_m][8]),  a4_1 = float(As[local_m][9]);
            float a5_0 = float(As[local_m][10]), a5_1 = float(As[local_m][11]);
            float a6_0 = float(As[local_m][12]), a6_1 = float(As[local_m][13]);
            float a7_0 = float(As[local_m][14]), a7_1 = float(As[local_m][15]);
            
            #pragma unroll
            for (int j = 0; j < OUTPUTS_PER_THREAD_N; ++j) {
                int local_n = tx * OUTPUTS_PER_THREAD_N + j;
                
                acc[i][j] = __fmaf_rn(a0_0, float(Bs[local_n][idx_0_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a0_1, float(Bs[local_n][idx_0_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a1_0, float(Bs[local_n][4 + idx_1_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a1_1, float(Bs[local_n][4 + idx_1_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a2_0, float(Bs[local_n][8 + idx_2_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a2_1, float(Bs[local_n][8 + idx_2_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a3_0, float(Bs[local_n][12 + idx_3_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a3_1, float(Bs[local_n][12 + idx_3_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a4_0, float(Bs[local_n][16 + idx_4_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a4_1, float(Bs[local_n][16 + idx_4_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a5_0, float(Bs[local_n][20 + idx_5_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a5_1, float(Bs[local_n][20 + idx_5_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a6_0, float(Bs[local_n][24 + idx_6_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a6_1, float(Bs[local_n][24 + idx_6_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a7_0, float(Bs[local_n][28 + idx_7_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a7_1, float(Bs[local_n][28 + idx_7_1]), acc[i][j]);
            }
            
            // Second half: k_groups 8-15 (meta1)
            uint32_t meta1 = Am[local_m][1];
            
            idx_0_0 = (meta1 >> 0) & 0x3; idx_0_1 = (meta1 >> 2) & 0x3;
            idx_1_0 = (meta1 >> 4) & 0x3; idx_1_1 = (meta1 >> 6) & 0x3;
            idx_2_0 = (meta1 >> 8) & 0x3; idx_2_1 = (meta1 >> 10) & 0x3;
            idx_3_0 = (meta1 >> 12) & 0x3; idx_3_1 = (meta1 >> 14) & 0x3;
            idx_4_0 = (meta1 >> 16) & 0x3; idx_4_1 = (meta1 >> 18) & 0x3;
            idx_5_0 = (meta1 >> 20) & 0x3; idx_5_1 = (meta1 >> 22) & 0x3;
            idx_6_0 = (meta1 >> 24) & 0x3; idx_6_1 = (meta1 >> 26) & 0x3;
            idx_7_0 = (meta1 >> 28) & 0x3; idx_7_1 = (meta1 >> 30) & 0x3;
            
            a0_0 = float(As[local_m][16]); a0_1 = float(As[local_m][17]);
            a1_0 = float(As[local_m][18]); a1_1 = float(As[local_m][19]);
            a2_0 = float(As[local_m][20]); a2_1 = float(As[local_m][21]);
            a3_0 = float(As[local_m][22]); a3_1 = float(As[local_m][23]);
            a4_0 = float(As[local_m][24]); a4_1 = float(As[local_m][25]);
            a5_0 = float(As[local_m][26]); a5_1 = float(As[local_m][27]);
            a6_0 = float(As[local_m][28]); a6_1 = float(As[local_m][29]);
            a7_0 = float(As[local_m][30]); a7_1 = float(As[local_m][31]);
            
            #pragma unroll
            for (int j = 0; j < OUTPUTS_PER_THREAD_N; ++j) {
                int local_n = tx * OUTPUTS_PER_THREAD_N + j;
                
                acc[i][j] = __fmaf_rn(a0_0, float(Bs[local_n][32 + idx_0_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a0_1, float(Bs[local_n][32 + idx_0_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a1_0, float(Bs[local_n][36 + idx_1_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a1_1, float(Bs[local_n][36 + idx_1_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a2_0, float(Bs[local_n][40 + idx_2_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a2_1, float(Bs[local_n][40 + idx_2_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a3_0, float(Bs[local_n][44 + idx_3_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a3_1, float(Bs[local_n][44 + idx_3_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a4_0, float(Bs[local_n][48 + idx_4_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a4_1, float(Bs[local_n][48 + idx_4_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a5_0, float(Bs[local_n][52 + idx_5_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a5_1, float(Bs[local_n][52 + idx_5_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a6_0, float(Bs[local_n][56 + idx_6_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a6_1, float(Bs[local_n][56 + idx_6_1]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a7_0, float(Bs[local_n][60 + idx_7_0]), acc[i][j]);
                acc[i][j] = __fmaf_rn(a7_1, float(Bs[local_n][60 + idx_7_1]), acc[i][j]);
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD_M; ++i) {
        int global_m = block_m + ty * OUTPUTS_PER_THREAD_M + i;
        #pragma unroll
        for (int j = 0; j < OUTPUTS_PER_THREAD_N; ++j) {
            int global_n = block_n + tx * OUTPUTS_PER_THREAD_N + j;
            if (global_m < m && global_n < n) {
                C[global_m * n + global_n] = __float2half(acc[i][j]);
            }
        }
    }
}

void init_sparse_data(float* A_dense, __nv_fp8_e4m3* A_packed, uint32_t* A_meta, int m, int k, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pattern_dist(0, 5);
    const int patterns[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    
    for (int row = 0; row < m; ++row) {
        int packed_idx = 0;
        for (int k_group = 0; k_group < k / 4; ++k_group) {
            int pattern_idx = pattern_dist(gen);
            const int* pattern = patterns[pattern_idx];
            for (int i = 0; i < 4; ++i) A_dense[row * k + k_group * 4 + i] = 0.0f;
            float val0 = dist(gen), val1 = dist(gen);
            if (std::abs(val0) < 0.1f) val0 = (val0 < 0 ? -0.25f : 0.25f);
            if (std::abs(val1) < 0.1f) val1 = (val1 < 0 ? -0.25f : 0.25f);
            A_dense[row * k + k_group * 4 + pattern[0]] = val0;
            A_dense[row * k + k_group * 4 + pattern[1]] = val1;
            A_packed[row * (k / 2) + packed_idx++] = __nv_fp8_e4m3(val0);
            A_packed[row * (k / 2) + packed_idx++] = __nv_fp8_e4m3(val1);
        }
    }
    
    for (int row = 0; row < m; ++row) {
        for (int meta_col = 0; meta_col < k / 32; ++meta_col) {
            uint32_t meta_word = 0;
            for (int g = 0; g < 8; ++g) {
                int k_base = meta_col * 32 + g * 4;
                int nz0 = -1, nz1 = -1;
                for (int i = 0; i < 4; ++i) {
                    if (A_dense[row * k + k_base + i] != 0.0f) {
                        if (nz0 < 0) nz0 = i; else nz1 = i;
                    }
                }
                uint32_t idx0 = (nz0 >= 0) ? (nz0 & 0x3) : 0;
                uint32_t idx1 = (nz1 >= 0) ? (nz1 & 0x3) : 1;
                meta_word |= (idx0 << (g * 4));
                meta_word |= (idx1 << (g * 4 + 2));
            }
            A_meta[row * (k / 32) + meta_col] = meta_word;
        }
    }
}

void init_B(__nv_fp8_e4m3* B, int n, int k, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int j = 0; j < n; ++j) {
        for (int kk = 0; kk < k; ++kk) {
            float val = dist(gen);
            if (std::abs(val) < 0.1f) val = (val < 0 ? -0.25f : 0.25f);
            B[j * k + kk] = __nv_fp8_e4m3(val);
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "Sparse GEMM e4m3->fp16 (ILP8): " << M << "x" << N << "x" << K << "\n";
    
    size_t A_packed_bytes = M * (K / 2), A_meta_bytes = M * (K / 32) * sizeof(uint32_t);
    size_t B_bytes = N * K, C_bytes = M * N * sizeof(half);
    
    float* A_dense_h = (float*)malloc(M * K * sizeof(float));
    __nv_fp8_e4m3* A_packed_h = (__nv_fp8_e4m3*)malloc(A_packed_bytes);
    uint32_t* A_meta_h = (uint32_t*)malloc(A_meta_bytes);
    __nv_fp8_e4m3* B_h = (__nv_fp8_e4m3*)malloc(B_bytes);
    half* C_h = (half*)malloc(C_bytes);
    
    std::mt19937 gen(42);
    init_sparse_data(A_dense_h, A_packed_h, A_meta_h, M, K, gen);
    init_B(B_h, N, K, gen);
    
    __nv_fp8_e4m3 *A_packed_d, *B_d; uint32_t *A_meta_d; half *C_d;
    cudaMalloc(&A_packed_d, A_packed_bytes);
    cudaMalloc(&A_meta_d, A_meta_bytes);
    cudaMalloc(&B_d, B_bytes);
    cudaMalloc(&C_d, C_bytes);
    
    cudaMemcpy(A_packed_d, A_packed_h, A_packed_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(A_meta_d, A_meta_h, A_meta_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_bytes, cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, C_bytes);
    cudaDeviceSynchronize();
    
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    sparse_gemm_ilp8_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::cerr << "Kernel error\n"; return 1; }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    for (int i = 0; i < 10; ++i) sparse_gemm_ilp8_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 50; ++i) sparse_gemm_ilp8_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    
    float total_ms; cudaEventElapsedTime(&total_ms, start, stop);
    double tflops = (2.0 * M * N * K / (total_ms / 50.0 / 1000.0)) / 1e12;
    
    std::cout << "Timing avg ms: " << total_ms / 50.0 << "\nTFLOPS: " << tflops 
              << "\nHW efficiency: " << (tflops / 1513.0) * 100.0 << "%\n";
    
    cudaFree(A_packed_d); cudaFree(A_meta_d); cudaFree(B_d); cudaFree(C_d);
    free(A_dense_h); free(A_packed_h); free(A_meta_h); free(B_h); free(C_h);
    return 0;
}
