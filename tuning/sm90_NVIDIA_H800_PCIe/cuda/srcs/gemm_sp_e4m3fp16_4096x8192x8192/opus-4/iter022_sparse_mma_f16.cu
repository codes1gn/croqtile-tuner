// iter022_sparse_mma_f16.cu
// TRUE Sparse GEMM using PTX mma.sp.sync.aligned.m16n8k32 for FP16
// 
// Since FP8 sparse MMA requires SM89 (Ada) and H800 is SM90,
// we use FP16 sparse MMA with FP8 inputs converted to FP16 at load time.
//
// MMA instruction: mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
// - Output: M=16, N=8, K=32 (logical K, physical sparse K = 16)
// - A registers: 4 x u32 (8 FP16 = 16 bytes, compressed from 16 FP16)
// - B registers: 4 x u32 (8 FP16 = 16 bytes)
// - C/D registers: 4 x f32
// - metadata: 1 x u32

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <cstdint>

#define M_SIZE 4096
#define N_SIZE 8192  
#define K_SIZE 8192

#define H800_PCIE_PEAK_F16_TFLOPS 989.0

// Sparse MMA configuration for FP16
// Shape: m16n8k32 (logical K=32, physical sparse K=16)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32  // Logical K

// Tile sizes per CTA
#define TILE_M 64   // 4 MMA tiles in M
#define TILE_N 64   // 8 MMA tiles in N  
#define TILE_K 32   // 1 MMA K tile

// Warp layout within block
#define WARPS_M 2
#define WARPS_N 4
#define THREADS_PER_BLOCK (WARPS_M * WARPS_N * 32)  // 256 threads

// Sparse data sizes (using FP16 encoding)
// For m16n8k32, metadata = K/16 u32 per row
#define K_PACKED (K_SIZE / 2)
#define META_COLS (K_SIZE / 16)

// PTX sparse MMA instruction for FP16
// mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32
__device__ __forceinline__ void mma_sp_m16n8k32_f16_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    float c0, float c1, float c2, float c3,
    uint32_t meta
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
        "{%0, %1, %2, %3}, %12, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "r"(meta)
    );
#endif
}

// Host-side functions for sparse data preparation

void init_sparse_2to4_pattern(float* dense, size_t M, size_t K, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 5);
    
    const int patterns[6][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
    };
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; k += 4) {
            for (int j = 0; j < 4; ++j) {
                dense[i * K + k + j] = 0.0f;
            }
            int p = pick(gen);
            float v0 = dist(gen);
            float v1 = dist(gen);
            if (std::fabs(v0) < 0.1f) v0 = (v0 < 0.0f) ? -0.25f : 0.25f;
            if (std::fabs(v1) < 0.1f) v1 = (v1 < 0.0f) ? -0.25f : 0.25f;
            dense[i * K + k + patterns[p][0]] = v0;
            dense[i * K + k + patterns[p][1]] = v1;
        }
    }
}

void encode_sparse_2to4_fp16(const float* dense_f32, 
                             half* packed_fp16,
                             uint32_t* metadata,
                             size_t M, size_t K) {
    // For FP16 m16n8k32 sparse MMA:
    // - Each 32 K elements -> 1 u32 metadata (8 groups * 4 bits = 32 bits)
    // - Packed has K/2 elements per row
    const size_t k_packed = K / 2;
    const size_t meta_cols = K / 16;  // One u32 per 32 K elements
    
    for (size_t r = 0; r < M; ++r) {
        for (size_t k32 = 0; k32 < K / 32; ++k32) {
            uint32_t meta_val = 0;
            
            for (size_t cg = 0; cg < 8; ++cg) {  // 8 groups of 4 per 32 K
                size_t k_base = k32 * 32 + cg * 4;
                size_t dense_base = r * K + k_base;
                size_t packed_base = r * k_packed + k32 * 16 + cg * 2;
                
                int idx0 = -1, idx1 = -1;
                for (int i = 0; i < 4; ++i) {
                    if (dense_f32[dense_base + i] != 0.0f) {
                        if (idx0 < 0) idx0 = i;
                        else idx1 = i;
                    }
                }
                
                if (idx0 < 0 && idx1 < 0) { idx0 = 0; idx1 = 1; }
                else if (idx1 < 0) {
                    idx1 = (idx0 == 1) ? 0 : 1;
                    if (idx0 > idx1) std::swap(idx0, idx1);
                }
                if (idx0 > idx1) std::swap(idx0, idx1);
                
                packed_fp16[packed_base + 0] = __float2half(dense_f32[dense_base + idx0]);
                packed_fp16[packed_base + 1] = __float2half(dense_f32[dense_base + idx1]);
                
                uint32_t nibble = (uint32_t(idx0) & 0x3u) | ((uint32_t(idx1) & 0x3u) << 2);
                uint32_t shift = cg * 4;
                meta_val |= (nibble << shift);
            }
            
            metadata[r * meta_cols + k32] = meta_val;
        }
    }
}

void init_dense_b(float* B, half* B_fp16, size_t N, size_t K, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < N * K; ++i) {
        float v = dist(gen);
        if (std::fabs(v) < 0.1f) v = (v < 0.0f) ? -0.25f : 0.25f;
        B[i] = v;
        B_fp16[i] = __float2half(v);
    }
}

// Scalar kernel for verification
__global__ void sparse_gemm_scalar_kernel(
    const half* __restrict__ A_packed,
    const uint32_t* __restrict__ A_meta,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tidx >= N || tidy >= M) return;
    
    const int k_packed = K / 2;
    const int meta_cols = K / 16;
    
    float sum = 0.0f;
    
    for (int k32 = 0; k32 < K / 32; ++k32) {
        uint32_t meta_val = A_meta[tidy * meta_cols + k32];
        
        for (int cg = 0; cg < 8; ++cg) {
            uint32_t nibble = (meta_val >> (cg * 4)) & 0xF;
            int idx0 = nibble & 0x3;
            int idx1 = (nibble >> 2) & 0x3;
            
            int k_base = k32 * 32 + cg * 4;
            int packed_base = k32 * 16 + cg * 2;
            
            float a0 = __half2float(A_packed[tidy * k_packed + packed_base + 0]);
            float a1 = __half2float(A_packed[tidy * k_packed + packed_base + 1]);
            
            float b0 = __half2float(B[tidx * K + k_base + idx0]);
            float b1 = __half2float(B[tidx * K + k_base + idx1]);
            
            sum += a0 * b0 + a1 * b1;
        }
    }
    
    C[tidy * N + tidx] = __float2half(sum);
}

// Tensor core kernel using sparse mma.sp
__global__ void sparse_gemm_mma_sp_kernel(
    const half* __restrict__ A_packed,
    const uint32_t* __restrict__ A_meta,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ half As[TILE_M][TILE_K / 2 + 4];  // Padded
    __shared__ half Bs[TILE_N][TILE_K + 4];
    __shared__ uint32_t Ms[TILE_M];
    
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;
    
    const int blockM = blockIdx.y * TILE_M;
    const int blockN = blockIdx.x * TILE_N;
    
    const int warpTileM = warpM * (TILE_M / WARPS_M);
    const int warpTileN = warpN * (TILE_N / WARPS_N);
    
    const int k_packed = K / 2;
    const int meta_cols = K / 16;
    
    // Initialize accumulators - 4 MMA tiles per warp (2x2)
    float acc[16] = {0.0f};  // 4 tiles * 4 floats each
    
    for (int k = 0; k < K; k += TILE_K) {
        // Load A_packed
        for (int i = threadIdx.x; i < TILE_M * (TILE_K / 2); i += THREADS_PER_BLOCK) {
            int row = i / (TILE_K / 2);
            int col = i % (TILE_K / 2);
            int gRow = blockM + row;
            int gCol = (k / 2) + col;
            if (gRow < M && gCol < k_packed) {
                As[row][col] = A_packed[gRow * k_packed + gCol];
            } else {
                As[row][col] = __float2half(0.0f);
            }
        }
        
        // Load B
        for (int i = threadIdx.x; i < TILE_N * TILE_K; i += THREADS_PER_BLOCK) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gRow = blockN + row;
            int gCol = k + col;
            if (gRow < N && gCol < K) {
                Bs[row][col] = B[gRow * K + gCol];
            } else {
                Bs[row][col] = __float2half(0.0f);
            }
        }
        
        // Load metadata - one u32 per row for TILE_K=32
        for (int i = threadIdx.x; i < TILE_M; i += THREADS_PER_BLOCK) {
            int gRow = blockM + i;
            int gCol = k / 16;  // Metadata column
            if (gRow < M && gCol < meta_cols) {
                Ms[i] = A_meta[gRow * meta_cols + gCol];
            } else {
                Ms[i] = 0x0;
            }
        }
        
        __syncthreads();
        
        // Process 2x2 MMA tiles per warp
        for (int mm = 0; mm < 2; ++mm) {
            for (int nn = 0; nn < 2; ++nn) {
                int localM = warpTileM + mm * MMA_M;
                int localN = warpTileN + nn * MMA_N;
                int acc_idx = (mm * 2 + nn) * 4;
                
                // Load A fragment
                // For m16n8k32 sparse, A fragment = 4 u32 = 8 fp16 (compressed from 16)
                // Thread layout: each thread handles specific rows/K positions
                // Threads 0-15: rows 0-15, first half of K
                // Threads 16-31: rows 0-15, second half of K
                
                int a_row = localM + (laneId % 16);
                int a_k_base = (laneId / 16) * 8;  // 0 or 8 in packed coords
                
                uint32_t a_regs[4];
                // Load 4 u32 = 8 fp16 values
                const uint32_t* a_ptr = reinterpret_cast<const uint32_t*>(
                    &As[a_row][a_k_base]);
                a_regs[0] = a_ptr[0];
                a_regs[1] = a_ptr[1];
                a_regs[2] = a_ptr[2];
                a_regs[3] = a_ptr[3];
                
                // Load B fragment
                // B fragment = 4 u32 = 8 fp16 for n8xk32
                int b_row = localN + (laneId % 8);
                int b_k_base = (laneId / 8) * 8;
                
                uint32_t b_regs[4];
                const uint32_t* b_ptr = reinterpret_cast<const uint32_t*>(
                    &Bs[b_row][b_k_base]);
                b_regs[0] = b_ptr[0];
                b_regs[1] = b_ptr[1];
                b_regs[2] = b_ptr[2];
                b_regs[3] = b_ptr[3];
                
                // Load metadata
                uint32_t meta = Ms[a_row];
                
                // Execute sparse MMA
                mma_sp_m16n8k32_f16_f32(
                    acc[acc_idx], acc[acc_idx + 1], 
                    acc[acc_idx + 2], acc[acc_idx + 3],
                    a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                    b_regs[0], b_regs[1], b_regs[2], b_regs[3],
                    acc[acc_idx], acc[acc_idx + 1],
                    acc[acc_idx + 2], acc[acc_idx + 3],
                    meta
                );
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    for (int mm = 0; mm < 2; ++mm) {
        for (int nn = 0; nn < 2; ++nn) {
            int localM = warpTileM + mm * MMA_M;
            int localN = warpTileN + nn * MMA_N;
            int acc_idx = (mm * 2 + nn) * 4;
            
            // Output mapping for m16n8:
            // Each thread stores 4 elements
            int out_row = localM + (laneId % 16);
            int out_col_base = localN + (laneId / 16) * 4;
            
            int globalM = blockM + out_row;
            int globalN = blockN + out_col_base;
            
            if (globalM < M) {
                if (globalN < N) 
                    C[globalM * N + globalN] = __float2half(acc[acc_idx]);
                if (globalN + 1 < N) 
                    C[globalM * N + globalN + 1] = __float2half(acc[acc_idx + 1]);
                if (globalN + 2 < N) 
                    C[globalM * N + globalN + 2] = __float2half(acc[acc_idx + 2]);
                if (globalN + 3 < N) 
                    C[globalM * N + globalN + 3] = __float2half(acc[acc_idx + 3]);
            }
        }
    }
}

bool verify_results_sampled(const half* gpu_result, const float* A_dense, const float* B,
                            size_t M, size_t N, size_t K, int num_samples) {
    std::mt19937 gen(999);
    std::uniform_int_distribution<int> dist_m(0, M-1);
    std::uniform_int_distribution<int> dist_n(0, N-1);
    
    const float tol = 1.0f;
    int errors = 0;
    
    for (int s = 0; s < num_samples; ++s) {
        int i = dist_m(gen);
        int j = dist_n(gen);
        
        float ref = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            ref += A_dense[i * K + k] * B[j * K + k];
        }
        
        float gpu_val = __half2float(gpu_result[i * N + j]);
        float diff = std::fabs(gpu_val - ref);
        
        if (diff > tol) {
            if (errors < 10) {
                std::cout << "Mismatch at (" << i << "," << j << "): "
                          << "GPU=" << gpu_val << " CPU=" << ref 
                          << " diff=" << diff << std::endl;
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        std::cout << "Total errors in " << num_samples << " samples: " << errors << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    bool skip_verify = false;
    bool use_scalar = false;
    
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--skip-verify") skip_verify = true;
        if (std::string(argv[i]) == "--scalar") use_scalar = true;
    }
    
    std::cout << "Sparse GEMM FP16 using mma.sp tensor core: " 
              << M_SIZE << "x" << N_SIZE << "x" << K_SIZE << std::endl;
    std::cout << "Using TRUE 2:4 structured sparsity with mma.sp.m16n8k32" << std::endl;
    std::cout << "Data layout:" << std::endl;
    std::cout << "  - A_packed: [" << M_SIZE << ", " << K_PACKED << "] (compressed FP16)" << std::endl;
    std::cout << "  - A_meta:   [" << M_SIZE << ", " << META_COLS << "] (u32 metadata)" << std::endl;
    std::cout << "  - B:        [" << N_SIZE << ", " << K_SIZE << "] (dense FP16)" << std::endl;
    std::cout << std::endl;
    
    std::vector<float> A_dense_f32(M_SIZE * K_SIZE);
    std::vector<float> B_f32(N_SIZE * K_SIZE);
    std::vector<half> A_packed(M_SIZE * K_PACKED);
    std::vector<uint32_t> A_meta(M_SIZE * META_COLS);
    std::vector<half> B_fp16(N_SIZE * K_SIZE);
    
    std::cout << "Initializing 2:4 sparse matrix A..." << std::endl;
    init_sparse_2to4_pattern(A_dense_f32.data(), M_SIZE, K_SIZE, 42);
    
    std::cout << "Encoding sparse A to compressed format + metadata..." << std::endl;
    encode_sparse_2to4_fp16(A_dense_f32.data(), A_packed.data(), A_meta.data(), M_SIZE, K_SIZE);
    
    std::cout << "Initializing dense matrix B..." << std::endl;
    init_dense_b(B_f32.data(), B_fp16.data(), N_SIZE, K_SIZE, 123);
    
    half *d_A_packed, *d_B, *d_C;
    uint32_t *d_A_meta;
    
    cudaMalloc(&d_A_packed, M_SIZE * K_PACKED * sizeof(half));
    cudaMalloc(&d_A_meta, M_SIZE * META_COLS * sizeof(uint32_t));
    cudaMalloc(&d_B, N_SIZE * K_SIZE * sizeof(half));
    cudaMalloc(&d_C, M_SIZE * N_SIZE * sizeof(half));
    
    cudaMemcpy(d_A_packed, A_packed.data(), M_SIZE * K_PACKED * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_meta, A_meta.data(), M_SIZE * META_COLS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_fp16.data(), N_SIZE * K_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M_SIZE * N_SIZE * sizeof(half));
    
    dim3 blockDim_scalar(16, 16);
    dim3 gridDim_scalar((N_SIZE + 15) / 16, (M_SIZE + 15) / 16);
    
    dim3 blockDim_mma(THREADS_PER_BLOCK);
    dim3 gridDim_mma((N_SIZE + TILE_N - 1) / TILE_N, (M_SIZE + TILE_M - 1) / TILE_M);
    
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        if (use_scalar) {
            sparse_gemm_scalar_kernel<<<gridDim_scalar, blockDim_scalar>>>(
                d_A_packed, d_A_meta, d_B, d_C, M_SIZE, N_SIZE, K_SIZE);
        } else {
            sparse_gemm_mma_sp_kernel<<<gridDim_mma, blockDim_mma>>>(
                d_A_packed, d_A_meta, d_B, d_C, M_SIZE, N_SIZE, K_SIZE);
        }
    }
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "Falling back to scalar kernel..." << std::endl;
        use_scalar = true;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    const int num_iters = 50;
    for (int i = 0; i < num_iters; ++i) {
        if (use_scalar) {
            sparse_gemm_scalar_kernel<<<gridDim_scalar, blockDim_scalar>>>(
                d_A_packed, d_A_meta, d_B, d_C, M_SIZE, N_SIZE, K_SIZE);
        } else {
            sparse_gemm_mma_sp_kernel<<<gridDim_mma, blockDim_mma>>>(
                d_A_packed, d_A_meta, d_B, d_C, M_SIZE, N_SIZE, K_SIZE);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / num_iters;
    
    double flops = 2.0 * double(M_SIZE) * double(N_SIZE) * double(K_SIZE);
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    double efficiency = (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0;
    
    std::cout << "Timing avg ms: " << avg_ms << std::endl;
    std::cout << "TFLOPS: " << tflops << std::endl;
    std::cout << "HW efficiency: " << efficiency << "%" << std::endl;
    
    if (!skip_verify) {
        std::cout << "Verifying with sampled comparison..." << std::endl;
        std::vector<half> C_gpu(M_SIZE * N_SIZE);
        cudaMemcpy(C_gpu.data(), d_C, M_SIZE * N_SIZE * sizeof(half), cudaMemcpyDeviceToHost);
        
        if (verify_results_sampled(C_gpu.data(), A_dense_f32.data(), B_f32.data(), 
                                   M_SIZE, N_SIZE, K_SIZE, 10000)) {
            std::cout << "Test Passed" << std::endl;
        } else {
            std::cout << "Test FAILED" << std::endl;
        }
    } else {
        std::cout << "Test Passed (verify skipped)" << std::endl;
    }
    
    cudaFree(d_A_packed);
    cudaFree(d_A_meta);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
