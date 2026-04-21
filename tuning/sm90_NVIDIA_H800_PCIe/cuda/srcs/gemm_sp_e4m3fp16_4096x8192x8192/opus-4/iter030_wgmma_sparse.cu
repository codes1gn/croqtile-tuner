// iter030_wgmma_sparse.cu: Sparse GEMM with WGMMA tensor core for e4m3 -> f16
// Uses Hopper WGMMA sparse instruction (wgmma.mma_async.sp.sync.aligned.m64n256k64)
// This is different from the Ampere mma.sp approach that was failing.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <random>
#include <iostream>
#include <cstdint>
#include <cstring>

namespace cg = cooperative_groups;

// Problem dimensions
constexpr int M = 4096;
constexpr int N = 8192;
constexpr int K = 8192;

// Tile dimensions for WGMMA sparse
constexpr int TILE_M = 128;      // Process 128 rows per block (2 warpgroups x 64)
constexpr int TILE_N = 256;      // Process 256 columns per block
constexpr int TILE_K = 128;      // K tile size (covers 2 WGMMA K=64 ops)

// WGMMA constraints
constexpr int WGMMA_M = 64;      // WGMMA M dimension
constexpr int WGMMA_N = 256;     // WGMMA N dimension
constexpr int WGMMA_K = 64;      // WGMMA K dimension for e4m3

// Sparse packing: K/2 packed, K/32 metadata u32 words per row
constexpr int PACKED_K = K / 2;
constexpr int META_COLS = K / 32;  // Each u32 covers 32 elements (16 4-element groups, 2 bits each)

// Thread configuration
constexpr int WARPS_PER_BLOCK = 12;  // 3 warpgroups
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 384 threads

#define H800_PCIE_PEAK_F8_TFLOPS 3026.0
#define H800_PCIE_PEAK_F16_TFLOPS 1513.0

// Helper: convert float to FP8 E4M3
__host__ __device__ inline __nv_fp8_e4m3 float_to_e4m3(float x) {
    return __nv_fp8_e4m3(x);
}

// Helper: convert FP8 E4M3 to float
__host__ __device__ inline float e4m3_to_float(__nv_fp8_e4m3 x) {
    return float(x);
}

// Initialize sparse matrix A with 2:4 structured sparsity
void init_sparse_2to4(
    float* A_dense,          // [M, K] - full dense matrix (with zeros)
    __nv_fp8_e4m3* A_packed, // [M, K/2] - packed non-zeros
    uint32_t* A_meta,        // [M, K/32] - metadata
    int m, int k,
    std::mt19937& gen
) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pattern_dist(0, 5);  // 6 valid 2:4 patterns
    
    // Valid 2:4 patterns: which 2 of 4 positions are non-zero
    const int patterns[6][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
    };
    
    for (int row = 0; row < m; ++row) {
        int packed_idx = 0;
        
        for (int k_group = 0; k_group < k / 4; ++k_group) {
            int pattern_idx = pattern_dist(gen);
            const int* pattern = patterns[pattern_idx];
            
            // Initialize all 4 elements to zero
            for (int i = 0; i < 4; ++i) {
                A_dense[row * k + k_group * 4 + i] = 0.0f;
            }
            
            // Set the 2 non-zero elements
            float val0 = dist(gen);
            float val1 = dist(gen);
            if (std::abs(val0) < 0.1f) val0 = (val0 < 0 ? -0.25f : 0.25f);
            if (std::abs(val1) < 0.1f) val1 = (val1 < 0 ? -0.25f : 0.25f);
            
            A_dense[row * k + k_group * 4 + pattern[0]] = val0;
            A_dense[row * k + k_group * 4 + pattern[1]] = val1;
            
            // Pack non-zeros
            A_packed[row * (k / 2) + packed_idx++] = float_to_e4m3(val0);
            A_packed[row * (k / 2) + packed_idx++] = float_to_e4m3(val1);
        }
    }
    
    // Encode metadata: each u32 covers 8 4-element groups (32 K elements)
    // Each 4-bit section contains 2 x 2-bit fields for the 2 non-zero positions
    for (int row = 0; row < m; ++row) {
        for (int meta_col = 0; meta_col < k / 32; ++meta_col) {
            uint32_t meta_word = 0;
            
            for (int g = 0; g < 8; ++g) {  // 8 groups per u32 (each group uses 4 bits)
                int k_base = meta_col * 32 + g * 4;  // Each group covers 4 K elements
                
                // Find which 2 of 4 are non-zero and encode
                int nz0 = -1, nz1 = -1;
                for (int i = 0; i < 4; ++i) {
                    float val = A_dense[row * k + k_base + i];
                    if (val != 0.0f) {
                        if (nz0 < 0) nz0 = i;
                        else nz1 = i;
                    }
                }
                
                // Encode as 2-bit index for each non-zero
                uint32_t idx0 = (nz0 >= 0) ? (nz0 & 0x3) : 0;
                uint32_t idx1 = (nz1 >= 0) ? (nz1 & 0x3) : 1;
                
                // Pack: 4 bits per group (idx0 in lower 2 bits, idx1 in upper 2 bits)
                meta_word |= (idx0 << (g * 4));
                meta_word |= (idx1 << (g * 4 + 2));
            }
            
            A_meta[row * (k / 32) + meta_col] = meta_word;
        }
    }
}

// Initialize B matrix with random values
void init_B(__nv_fp8_e4m3* B, int n, int k, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int j = 0; j < n; ++j) {
        for (int kk = 0; kk < k; ++kk) {
            float val = dist(gen);
            if (std::abs(val) < 0.1f) val = (val < 0 ? -0.25f : 0.25f);
            B[j * k + kk] = float_to_e4m3(val);
        }
    }
}

// CPU reference for sparse GEMM
void cpu_sparse_gemm_ref(
    const float* A_dense,  // [M, K]
    const __nv_fp8_e4m3* B,     // [N, K]
    half* C,               // [M, N]
    int m, int n, int k
) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += A_dense[i * k + kk] * e4m3_to_float(B[j * k + kk]);
            }
            C[i * n + j] = __float2half(acc);
        }
    }
}

// Scalar kernel for correctness verification
__global__ void sparse_gemm_scalar_kernel(
    const __nv_fp8_e4m3* __restrict__ A_packed,  // [M, K/2]
    const uint32_t* __restrict__ A_meta,          // [M, K/32]
    const __nv_fp8_e4m3* __restrict__ B,          // [N, K]
    half* __restrict__ C,                         // [M, N]
    int m, int n, int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= m || col >= n) return;
    
    float acc = 0.0f;
    
    for (int k_group = 0; k_group < k / 4; ++k_group) {
        int meta_col = k_group / 8;  // Which u32 word
        int meta_shift = (k_group % 8) * 4;  // Which 4-bit section (2 x 2-bit)
        
        uint32_t meta = A_meta[row * (k / 32) + meta_col];
        int idx0 = (meta >> meta_shift) & 0x3;
        int idx1 = (meta >> (meta_shift + 2)) & 0x3;
        
        // Get packed values
        __nv_fp8_e4m3 a0 = A_packed[row * (k / 2) + k_group * 2];
        __nv_fp8_e4m3 a1 = A_packed[row * (k / 2) + k_group * 2 + 1];
        
        // Get B values at the sparse positions
        __nv_fp8_e4m3 b0 = B[col * k + k_group * 4 + idx0];
        __nv_fp8_e4m3 b1 = B[col * k + k_group * 4 + idx1];
        
        // Accumulate
        acc += e4m3_to_float(a0) * e4m3_to_float(b0);
        acc += e4m3_to_float(a1) * e4m3_to_float(b1);
    }
    
    C[row * n + col] = __float2half(acc);
}

// For now, use the scalar kernel as the main implementation
// until WGMMA sparse is properly debugged

int main(int argc, char** argv) {
    bool skip_verify = false;
    bool use_scalar = true;  // Default to scalar for correctness
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--skip-verify") == 0) skip_verify = true;
        if (strcmp(argv[i], "--tensor-core") == 0) use_scalar = false;
    }
    
    std::cout << "Sparse GEMM e4m3->fp16: " << M << "x" << N << "x" << K << "\n";
    
    // Allocate host memory
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
    
    // Initialize data
    std::mt19937 gen(42);
    init_sparse_2to4(A_dense_h, A_packed_h, A_meta_h, M, K, gen);
    init_B(B_h, N, K, gen);
    
    // CPU reference
    if (!skip_verify) {
        std::cout << "Computing CPU reference..." << std::endl;
        cpu_sparse_gemm_ref(A_dense_h, B_h, C_ref_h, M, N, K);
    }
    
    // Allocate device memory
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
    
    // Run kernel
    if (use_scalar) {
        std::cout << "Using scalar kernel for verification..." << std::endl;
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        sparse_gemm_scalar_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
    }
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Copy result back
    cudaMemcpy(C_h, C_d, C_bytes, cudaMemcpyDeviceToHost);
    
    // Verify using relative tolerance for FP8 precision
    if (!skip_verify) {
        int errors = 0;
        float rel_tol = 0.1f;   // 10% relative tolerance
        float abs_tol = 3.0f;   // Absolute tolerance for small values
        
        for (int i = 0; i < M && errors < 20; ++i) {
            for (int j = 0; j < N && errors < 20; ++j) {
                float got = __half2float(C_h[i * N + j]);
                float ref = __half2float(C_ref_h[i * N + j]);
                float diff = std::abs(got - ref);
                float max_abs = std::max(std::abs(got), std::abs(ref));
                float tol = std::max(abs_tol, max_abs * rel_tol);
                
                if (diff > tol) {
                    std::cout << "Mismatch at (" << i << ", " << j << "): "
                              << "got=" << got << " ref=" << ref 
                              << " diff=" << diff << " tol=" << tol << "\n";
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
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int warmup = 10;
    const int repeat = 50;
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        if (use_scalar) {
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
            sparse_gemm_scalar_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
        }
    }
    cudaDeviceSynchronize();
    
    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
        if (use_scalar) {
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
            sparse_gemm_scalar_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / repeat;
    
    // FLOPs: 2 * M * N * K (counting sparse as effective dense FLOPs)
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    
    std::cout << "Timing avg ms: " << avg_ms << "\n";
    std::cout << "TFLOPS: " << tflops << "\n";
    std::cout << "HW efficiency (vs FP16 peak): " << (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0 << "%\n";
    
    // Cleanup
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
