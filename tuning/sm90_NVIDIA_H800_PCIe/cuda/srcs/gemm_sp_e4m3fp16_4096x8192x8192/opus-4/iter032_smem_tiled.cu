// iter032_smem_tiled.cu: Shared memory tiled sparse GEMM
// Scalar compute with optimized memory access patterns

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <random>
#include <iostream>
#include <cstdint>
#include <cstring>

namespace cg = cooperative_groups;

constexpr int M = 4096;
constexpr int N = 8192;
constexpr int K = 8192;

// Tile sizes
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 64;  // In dense K space, 32 in packed space

constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr int THREADS_PER_BLOCK = THREADS_X * THREADS_Y;

#define H800_PCIE_PEAK_F16_TFLOPS 1513.0

// Tiled sparse GEMM kernel
__global__ void sparse_gemm_tiled_kernel(
    const __nv_fp8_e4m3* __restrict__ A_packed,  // [M, K/2]
    const uint32_t* __restrict__ A_meta,          // [M, K/32]
    const __nv_fp8_e4m3* __restrict__ B,          // [N, K]
    half* __restrict__ C,                         // [M, N]
    int m, int n, int k
) {
    // Block indices
    int block_m = blockIdx.y * TILE_M;
    int block_n = blockIdx.x * TILE_N;
    
    // Thread indices within tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * THREADS_X + tx;
    
    // This thread computes C[row, col]
    int row = block_m + ty * (TILE_M / THREADS_Y) + (tid / (TILE_N / 4)) % (TILE_M / THREADS_Y);
    int col = block_n + tx * (TILE_N / THREADS_X) + (tid % (TILE_N / 4));
    
    // Actually, simpler: each thread computes multiple output elements
    // Let's do 4 outputs per thread
    constexpr int OUTPUTS_PER_THREAD_M = TILE_M / THREADS_Y;  // 4
    constexpr int OUTPUTS_PER_THREAD_N = TILE_N / THREADS_X;  // 4
    
    // Shared memory for tiles
    __shared__ __nv_fp8_e4m3 As[TILE_M][TILE_K / 2 + 4];  // Packed A (+padding for bank conflicts)
    __shared__ uint32_t Am[TILE_M][TILE_K / 32 + 1];       // Metadata (+padding)
    __shared__ __nv_fp8_e4m3 Bs[TILE_N][TILE_K + 4];       // Dense B (+padding)
    
    // Accumulator registers
    float acc[OUTPUTS_PER_THREAD_M][OUTPUTS_PER_THREAD_N];
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < OUTPUTS_PER_THREAD_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Loop over K tiles
    for (int k_tile = 0; k_tile < k; k_tile += TILE_K) {
        // Cooperative load of A_packed into shared memory
        // Each thread loads multiple elements
        int load_iters_a = (TILE_M * (TILE_K / 2)) / THREADS_PER_BLOCK;
        for (int i = 0; i < load_iters_a; ++i) {
            int idx = tid + i * THREADS_PER_BLOCK;
            int load_m = idx / (TILE_K / 2);
            int load_k = idx % (TILE_K / 2);
            int global_m = block_m + load_m;
            int global_k = k_tile / 2 + load_k;
            
            if (global_m < m && global_k < k / 2) {
                As[load_m][load_k] = A_packed[global_m * (k / 2) + global_k];
            } else {
                As[load_m][load_k] = __nv_fp8_e4m3(0.0f);
            }
        }
        
        // Cooperative load of metadata
        int load_iters_meta = (TILE_M * (TILE_K / 32)) / THREADS_PER_BLOCK + 1;
        for (int i = 0; i < load_iters_meta; ++i) {
            int idx = tid + i * THREADS_PER_BLOCK;
            if (idx < TILE_M * (TILE_K / 32)) {
                int load_m = idx / (TILE_K / 32);
                int load_k = idx % (TILE_K / 32);
                int global_m = block_m + load_m;
                int global_k = k_tile / 32 + load_k;
                
                if (global_m < m && global_k < k / 32) {
                    Am[load_m][load_k] = A_meta[global_m * (k / 32) + global_k];
                } else {
                    Am[load_m][load_k] = 0;
                }
            }
        }
        
        // Cooperative load of B into shared memory
        int load_iters_b = (TILE_N * TILE_K) / THREADS_PER_BLOCK;
        for (int i = 0; i < load_iters_b; ++i) {
            int idx = tid + i * THREADS_PER_BLOCK;
            int load_n = idx / TILE_K;
            int load_k = idx % TILE_K;
            int global_n = block_n + load_n;
            int global_k = k_tile + load_k;
            
            if (global_n < n && global_k < k) {
                Bs[load_n][load_k] = B[global_n * k + global_k];
            } else {
                Bs[load_n][load_k] = __nv_fp8_e4m3(0.0f);
            }
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int i = 0; i < OUTPUTS_PER_THREAD_M; ++i) {
            int local_m = ty * OUTPUTS_PER_THREAD_M + i;
            
            #pragma unroll
            for (int j = 0; j < OUTPUTS_PER_THREAD_N; ++j) {
                int local_n = tx * OUTPUTS_PER_THREAD_N + j;
                
                // Loop over K in groups of 4 (2:4 sparsity)
                #pragma unroll 4
                for (int kk = 0; kk < TILE_K; kk += 4) {
                    int k_group = kk / 4;
                    int meta_col = k_group / 8;
                    int meta_shift = (k_group % 8) * 4;
                    
                    uint32_t meta = Am[local_m][meta_col];
                    int idx0 = (meta >> meta_shift) & 0x3;
                    int idx1 = (meta >> (meta_shift + 2)) & 0x3;
                    
                    __nv_fp8_e4m3 a0 = As[local_m][k_group * 2];
                    __nv_fp8_e4m3 a1 = As[local_m][k_group * 2 + 1];
                    
                    __nv_fp8_e4m3 b0 = Bs[local_n][kk + idx0];
                    __nv_fp8_e4m3 b1 = Bs[local_n][kk + idx1];
                    
                    acc[i][j] += float(a0) * float(b0);
                    acc[i][j] += float(a1) * float(b1);
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

// Simple scalar kernel for reference
__global__ void sparse_gemm_scalar_kernel(
    const __nv_fp8_e4m3* __restrict__ A_packed,
    const uint32_t* __restrict__ A_meta,
    const __nv_fp8_e4m3* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= m || col >= n) return;
    
    float acc = 0.0f;
    
    for (int k_group = 0; k_group < k / 4; ++k_group) {
        int meta_col = k_group / 8;
        int meta_shift = (k_group % 8) * 4;
        
        uint32_t meta = A_meta[row * (k / 32) + meta_col];
        int idx0 = (meta >> meta_shift) & 0x3;
        int idx1 = (meta >> (meta_shift + 2)) & 0x3;
        
        __nv_fp8_e4m3 a0 = A_packed[row * (k / 2) + k_group * 2];
        __nv_fp8_e4m3 a1 = A_packed[row * (k / 2) + k_group * 2 + 1];
        
        __nv_fp8_e4m3 b0 = B[col * k + k_group * 4 + idx0];
        __nv_fp8_e4m3 b1 = B[col * k + k_group * 4 + idx1];
        
        acc += float(a0) * float(b0);
        acc += float(a1) * float(b1);
    }
    
    C[row * n + col] = __float2half(acc);
}

// Data initialization
void init_sparse_data(
    float* A_dense,
    __nv_fp8_e4m3* A_packed,
    uint32_t* A_meta,
    int m, int k,
    std::mt19937& gen
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
                    float val = A_dense[row * k + k_base + i];
                    if (val != 0.0f) {
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

void cpu_reference(
    const float* A_dense,
    const __nv_fp8_e4m3* B,
    half* C,
    int m, int n, int k
) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += A_dense[i * k + kk] * float(B[j * k + kk]);
            }
            C[i * n + j] = __float2half(acc);
        }
    }
}

int main(int argc, char** argv) {
    bool skip_verify = false;
    bool use_tiled = true;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--skip-verify") == 0) skip_verify = true;
        if (strcmp(argv[i], "--scalar") == 0) use_tiled = false;
    }
    
    std::cout << "Sparse GEMM e4m3->fp16 (tiled): " << M << "x" << N << "x" << K << "\n";
    
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
    half* C_ref_h = (half*)malloc(C_bytes);
    
    std::mt19937 gen(42);
    init_sparse_data(A_dense_h, A_packed_h, A_meta_h, M, K, gen);
    init_B(B_h, N, K, gen);
    
    if (!skip_verify) {
        std::cout << "Computing CPU reference..." << std::endl;
        cpu_reference(A_dense_h, B_h, C_ref_h, M, N, K);
    }
    
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
    
    dim3 block, grid;
    if (use_tiled) {
        std::cout << "Using tiled kernel..." << std::endl;
        block = dim3(THREADS_X, THREADS_Y);
        grid = dim3((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
        sparse_gemm_tiled_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
    } else {
        std::cout << "Using scalar kernel..." << std::endl;
        block = dim3(16, 16);
        grid = dim3((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        sparse_gemm_scalar_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
    }
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    cudaMemcpy(C_h, C_d, C_bytes, cudaMemcpyDeviceToHost);
    
    if (!skip_verify) {
        int errors = 0;
        float rel_tol = 0.1f;
        float abs_tol = 3.0f;
        
        for (int i = 0; i < M && errors < 20; ++i) {
            for (int j = 0; j < N && errors < 20; ++j) {
                float got = __half2float(C_h[i * N + j]);
                float ref = __half2float(C_ref_h[i * N + j]);
                float diff = std::abs(got - ref);
                float max_abs = std::max(std::abs(got), std::abs(ref));
                float tol = std::max(abs_tol, max_abs * rel_tol);
                
                if (diff > tol) {
                    std::cout << "Mismatch at (" << i << ", " << j << "): "
                              << "got=" << got << " ref=" << ref << " diff=" << diff << "\n";
                    errors++;
                }
            }
        }
        
        if (errors > 0) {
            std::cout << "VERIFICATION FAILED with " << errors << " errors\n";
        } else {
            std::cout << "VERIFICATION PASSED\n";
        }
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int warmup = 10;
    const int repeat = 50;
    
    for (int i = 0; i < warmup; ++i) {
        if (use_tiled) {
            sparse_gemm_tiled_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
        } else {
            sparse_gemm_scalar_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
        }
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
        if (use_tiled) {
            sparse_gemm_tiled_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
        } else {
            sparse_gemm_scalar_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
        }
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
    free(C_ref_h);
    
    return 0;
}
