// iter038_reg_prefetch.cu: Prefetch A values into registers before k-loop
// Based on iter035 (6.71 TFLOPS best)

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

constexpr int OUTPUTS_PER_THREAD_M = TILE_M / THREADS_Y;  // 4
constexpr int OUTPUTS_PER_THREAD_N = TILE_N / THREADS_X;  // 4

constexpr int K_GROUPS_PER_TILE = TILE_K / 4;  // 16

#define H800_PCIE_PEAK_F16_TFLOPS 1513.0

__global__ void sparse_gemm_reg_prefetch_kernel(
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
    
    // Shared memory
    __shared__ __nv_fp8_e4m3 As[TILE_M][TILE_K / 2 + 4];
    __shared__ uint32_t Am[TILE_M][TILE_K / 32 + 1];
    __shared__ __nv_fp8_e4m3 Bs[TILE_N][TILE_K + 4];
    
    // Accumulators
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
        
        // Load A_packed
        #pragma unroll 4
        for (int i = tid; i < TILE_M * (TILE_K / 2); i += THREADS_PER_BLOCK) {
            int load_m = i / (TILE_K / 2);
            int load_k = i % (TILE_K / 2);
            int global_m = block_m + load_m;
            int global_k = k_offset / 2 + load_k;
            
            if (global_m < m && global_k < packed_k) {
                As[load_m][load_k] = A_packed[global_m * packed_k + global_k];
            } else {
                As[load_m][load_k] = __nv_fp8_e4m3(0.0f);
            }
        }
        
        // Load metadata
        #pragma unroll 2
        for (int i = tid; i < TILE_M * (TILE_K / 32); i += THREADS_PER_BLOCK) {
            int load_m = i / (TILE_K / 32);
            int load_k = i % (TILE_K / 32);
            int global_m = block_m + load_m;
            int global_k = k_offset / 32 + load_k;
            
            if (global_m < m && global_k < meta_k) {
                Am[load_m][load_k] = A_meta[global_m * meta_k + global_k];
            } else {
                Am[load_m][load_k] = 0;
            }
        }
        
        // Load B
        #pragma unroll 4
        for (int i = tid; i < TILE_N * TILE_K; i += THREADS_PER_BLOCK) {
            int load_n = i / TILE_K;
            int load_k = i % TILE_K;
            int global_n = block_n + load_n;
            int global_k = k_offset + load_k;
            
            if (global_n < n && global_k < k) {
                Bs[load_n][load_k] = B[global_n * k + global_k];
            } else {
                Bs[load_n][load_k] = __nv_fp8_e4m3(0.0f);
            }
        }
        
        __syncthreads();
        
        // Prefetch A values for this thread's output rows into registers
        float a_regs[OUTPUTS_PER_THREAD_M][K_GROUPS_PER_TILE * 2];  // 2 values per k-group
        int idx_regs[OUTPUTS_PER_THREAD_M][K_GROUPS_PER_TILE * 2];   // indices
        
        #pragma unroll
        for (int i = 0; i < OUTPUTS_PER_THREAD_M; ++i) {
            int local_m = ty * OUTPUTS_PER_THREAD_M + i;
            
            #pragma unroll
            for (int k_group = 0; k_group < K_GROUPS_PER_TILE; ++k_group) {
                int meta_col = k_group / 8;
                int meta_shift = (k_group % 8) * 4;
                
                uint32_t meta = Am[local_m][meta_col];
                int idx0 = (meta >> meta_shift) & 0x3;
                int idx1 = (meta >> (meta_shift + 2)) & 0x3;
                
                a_regs[i][k_group * 2] = float(As[local_m][k_group * 2]);
                a_regs[i][k_group * 2 + 1] = float(As[local_m][k_group * 2 + 1]);
                idx_regs[i][k_group * 2] = k_group * 4 + idx0;
                idx_regs[i][k_group * 2 + 1] = k_group * 4 + idx1;
            }
        }
        
        // Compute with prefetched values
        #pragma unroll
        for (int i = 0; i < OUTPUTS_PER_THREAD_M; ++i) {
            #pragma unroll
            for (int j = 0; j < OUTPUTS_PER_THREAD_N; ++j) {
                int local_n = tx * OUTPUTS_PER_THREAD_N + j;
                
                #pragma unroll
                for (int k_group = 0; k_group < K_GROUPS_PER_TILE; ++k_group) {
                    float a0 = a_regs[i][k_group * 2];
                    float a1 = a_regs[i][k_group * 2 + 1];
                    int b_idx0 = idx_regs[i][k_group * 2];
                    int b_idx1 = idx_regs[i][k_group * 2 + 1];
                    
                    float b0 = float(Bs[local_n][b_idx0]);
                    float b1 = float(Bs[local_n][b_idx1]);
                    
                    acc[i][j] = __fmaf_rn(a0, b0, acc[i][j]);
                    acc[i][j] = __fmaf_rn(a1, b1, acc[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
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

void init_sparse_data(
    float* A_dense, __nv_fp8_e4m3* A_packed, uint32_t* A_meta,
    int m, int k, std::mt19937& gen
) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pattern_dist(0, 5);
    
    const int patterns[6][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
    };
    
    for (int row = 0; row < m; ++row) {
        int packed_idx = 0;
        for (int k_group = 0; k_group < k / 4; ++k_group) {
            int pattern_idx = pattern_dist(gen);
            const int* pattern = patterns[pattern_idx];
            
            for (int i = 0; i < 4; ++i) {
                A_dense[row * k + k_group * 4 + i] = 0.0f;
            }
            
            float val0 = dist(gen);
            float val1 = dist(gen);
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
                        if (nz0 < 0) nz0 = i;
                        else nz1 = i;
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
    bool skip_verify = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--skip-verify") == 0) skip_verify = true;
    }
    
    std::cout << "Sparse GEMM e4m3->fp16 (reg prefetch): " << M << "x" << N << "x" << K << "\n";
    
    size_t A_dense_bytes = M * K * sizeof(float);
    size_t A_packed_bytes = M * (K / 2) * sizeof(__nv_fp8_e4m3);
    size_t A_meta_bytes = M * (K / 32) * sizeof(uint32_t);
    size_t B_bytes = N * K * sizeof(__nv_fp8_e4m3);
    size_t C_bytes = M * N * sizeof(half);
    
    float* A_dense_h = (float*)malloc(A_dense_bytes);
    __nv_fp8_e4m3* A_packed_h = (__nv_fp8_e4m3*)malloc(A_packed_bytes);
    uint32_t* A_meta_h = (uint32_t*)malloc(A_meta_bytes);
    __nv_fp8_e4m3* B_h = (__nv_fp8_e4m3*)malloc(B_bytes);
    half* C_h = (half*)malloc(C_bytes);
    
    std::mt19937 gen(42);
    init_sparse_data(A_dense_h, A_packed_h, A_meta_h, M, K, gen);
    init_B(B_h, N, K, gen);
    
    __nv_fp8_e4m3 *A_packed_d, *B_d;
    uint32_t *A_meta_d;
    half *C_d;
    
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
    
    sparse_gemm_reg_prefetch_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int warmup = 10;
    const int repeat = 50;
    
    for (int i = 0; i < warmup; ++i) {
        sparse_gemm_reg_prefetch_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
        sparse_gemm_reg_prefetch_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / repeat;
    
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    
    std::cout << "Timing avg ms: " << avg_ms << "\n";
    std::cout << "TFLOPS: " << tflops << "\n";
    std::cout << "HW efficiency: " << (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0 << "%\n";
    
    cudaFree(A_packed_d);
    cudaFree(A_meta_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(A_dense_h);
    free(A_packed_h);
    free(A_meta_h);
    free(B_h);
    free(C_h);
    
    return 0;
}
