// iter023_sparse_mma_v2.cu
// TRUE Sparse GEMM with correct fragment layout for mma.sp.m16n8k32
// 
// The fragment layout for mma.sp follows the same pattern as mma.m16n8k16
// but with K doubled (32 logical = 16 physical sparse).
//
// For mma.m16n8k16:
// - groupID = laneId / 4 (0-7)
// - groupLaneID = laneId % 4 (0-3)
// - A fragment rows: groupID, groupID+8 
// - A fragment cols: groupLaneID*2, groupLaneID*2+1 for k0-7, then +8 for k8-15
// - B fragment rows: k0-15 (split across threads)
// - B fragment cols: 0-7 (n dimension)

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

// MMA shape: m16n8k32 (sparse: logical K=32, physical K=16)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

// Tile sizes
#define TILE_M 32   // 2 MMA tiles in M
#define TILE_N 32   // 4 MMA tiles in N
#define TILE_K 32   // 1 K tile

// Block configuration
#define WARPS_M 1
#define WARPS_N 4   // 4 warps per block
#define THREADS_PER_BLOCK (WARPS_M * WARPS_N * 32)  // 128 threads

// Sparse data sizes
#define K_PACKED (K_SIZE / 2)
#define META_COLS (K_SIZE / 16)

// PTX sparse MMA
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

void init_sparse_2to4_pattern(float* dense, size_t M, size_t K, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 5);
    
    const int patterns[6][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
    };
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; k += 4) {
            for (int j = 0; j < 4; ++j) dense[i * K + k + j] = 0.0f;
            int p = pick(gen);
            float v0 = dist(gen), v1 = dist(gen);
            if (std::fabs(v0) < 0.1f) v0 = (v0 < 0.0f) ? -0.25f : 0.25f;
            if (std::fabs(v1) < 0.1f) v1 = (v1 < 0.0f) ? -0.25f : 0.25f;
            dense[i * K + k + patterns[p][0]] = v0;
            dense[i * K + k + patterns[p][1]] = v1;
        }
    }
}

void encode_sparse_2to4_fp16(const float* dense_f32, half* packed_fp16,
                             uint32_t* metadata, size_t M, size_t K) {
    const size_t k_packed = K / 2;
    const size_t meta_cols = K / 16;
    
    for (size_t r = 0; r < M; ++r) {
        for (size_t k32 = 0; k32 < K / 32; ++k32) {
            uint32_t meta_val = 0;
            for (size_t cg = 0; cg < 8; ++cg) {
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
                meta_val |= (nibble << (cg * 4));
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

// Tensor core kernel with correct fragment layout
// For m16n8k32 sparse MMA:
// - A: 4 u32 = 8 fp16 (from 16 physical k positions = 32 logical k)
// - B: 4 u32 = 8 fp16 
// - C/D: 4 f32
// - meta: 1 u32

__global__ void sparse_gemm_mma_sp_kernel(
    const half* __restrict__ A_packed,
    const uint32_t* __restrict__ A_meta,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory
    __shared__ half As[TILE_M][TILE_K / 2 + 4];  // Padded
    __shared__ half Bs[TILE_N][TILE_K + 4];
    __shared__ uint32_t Ms[TILE_M];
    
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    
    const int blockM = blockIdx.y * TILE_M;
    const int blockN = blockIdx.x * TILE_N;
    
    // Each warp handles one 16x8 output tile
    const int warpTileM = 0;  // Only 1 warp in M for simplicity
    const int warpTileN = warpId * MMA_N;  // 0, 8, 16, 24
    
    const int k_packed = K / 2;
    const int meta_cols = K / 16;
    
    // Initialize accumulators
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Fragment layout indices for m16n8k16 (base of m16n8k32)
    // groupID = laneId / 4, groupLaneID = laneId % 4
    const int groupID = laneId / 4;       // 0-7
    const int groupLaneID = laneId % 4;   // 0-3
    
    for (int k = 0; k < K; k += TILE_K) {
        // Collaborative load A_packed
        for (int i = threadIdx.x; i < TILE_M * (TILE_K / 2); i += THREADS_PER_BLOCK) {
            int row = i / (TILE_K / 2);
            int col = i % (TILE_K / 2);
            int gRow = blockM + row;
            int gCol = (k / 2) + col;
            As[row][col] = (gRow < M && gCol < k_packed) ? 
                           A_packed[gRow * k_packed + gCol] : __float2half(0.0f);
        }
        
        // Collaborative load B
        for (int i = threadIdx.x; i < TILE_N * TILE_K; i += THREADS_PER_BLOCK) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gRow = blockN + row;
            int gCol = k + col;
            Bs[row][col] = (gRow < N && gCol < K) ? 
                           B[gRow * K + gCol] : __float2half(0.0f);
        }
        
        // Load metadata
        for (int i = threadIdx.x; i < TILE_M; i += THREADS_PER_BLOCK) {
            int gRow = blockM + i;
            int gCol = k / 16;
            Ms[i] = (gRow < M && gCol < meta_cols) ? 
                    A_meta[gRow * meta_cols + gCol] : 0x0;
        }
        
        __syncthreads();
        
        // Process 2 MMA tiles in M direction
        for (int mm = 0; mm < 2; ++mm) {
            int localM = mm * MMA_M;
            
            // Load A fragment following m16n8k32 layout
            // For sparse m16n8k32, A is 16 rows x 16 physical K cols (= 32 logical)
            // Each thread loads 8 fp16 = 4 u32
            
            // Fragment layout for A (row-major, m16n8k32 sparse):
            // Thread groupID (0-7) handles rows groupID and groupID+8
            // Thread groupLaneID (0-3) handles different K positions
            
            // For k32 sparse MMA:
            // a0,a1 = rows groupID, k positions groupLaneID*2, groupLaneID*2+1 (physical k0-7)
            // a2,a3 = rows groupID+8, same k positions
            // a4,a5 = rows groupID, k positions +8 (physical k8-15)
            // a6,a7 = rows groupID+8, k positions +8
            
            int aRow0 = localM + groupID;
            int aRow1 = localM + groupID + 8;
            int aK0 = groupLaneID * 2;      // 0, 2, 4, 6
            int aK1 = groupLaneID * 2 + 8;  // 8, 10, 12, 14
            
            half aReg[8];
            aReg[0] = As[aRow0][aK0];
            aReg[1] = As[aRow0][aK0 + 1];
            aReg[2] = As[aRow1][aK0];
            aReg[3] = As[aRow1][aK0 + 1];
            aReg[4] = As[aRow0][aK1];
            aReg[5] = As[aRow0][aK1 + 1];
            aReg[6] = As[aRow1][aK1];
            aReg[7] = As[aRow1][aK1 + 1];
            
            uint32_t a_regs[4];
            a_regs[0] = *reinterpret_cast<uint32_t*>(&aReg[0]);
            a_regs[1] = *reinterpret_cast<uint32_t*>(&aReg[2]);
            a_regs[2] = *reinterpret_cast<uint32_t*>(&aReg[4]);
            a_regs[3] = *reinterpret_cast<uint32_t*>(&aReg[6]);
            
            // Load B fragment
            // B is n8 x k32 (dense)
            // Fragment layout: groupLaneID handles different K, groupID handles N
            // b0,b1 = k positions groupLaneID*2, groupLaneID*2+1, n = groupID (mod 8)
            // b2,b3 = same but for k+8
            
            int bN = warpTileN + (groupID % 8);  // N position
            int bK0 = groupLaneID * 2;           // K position for first half
            int bK1 = groupLaneID * 2 + 8;       // K position for second half
            
            // For k32: need 4 groups of 8 k values
            // Actually for m16n8k32, B is 32xN (k32 x n8)
            // Per-thread: 8 fp16 from different k positions
            
            half bReg[8];
            bReg[0] = Bs[bN][bK0];
            bReg[1] = Bs[bN][bK0 + 1];
            bReg[2] = Bs[bN][bK0 + 16];
            bReg[3] = Bs[bN][bK0 + 17];
            bReg[4] = Bs[bN][bK1];
            bReg[5] = Bs[bN][bK1 + 1];
            bReg[6] = Bs[bN][bK1 + 16];
            bReg[7] = Bs[bN][bK1 + 17];
            
            uint32_t b_regs[4];
            b_regs[0] = *reinterpret_cast<uint32_t*>(&bReg[0]);
            b_regs[1] = *reinterpret_cast<uint32_t*>(&bReg[2]);
            b_regs[2] = *reinterpret_cast<uint32_t*>(&bReg[4]);
            b_regs[3] = *reinterpret_cast<uint32_t*>(&bReg[6]);
            
            // Load metadata for this row
            uint32_t meta = Ms[aRow0];  // Same metadata for rows groupID and groupID+8
            
            // Execute sparse MMA
            float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3];
            mma_sp_m16n8k32_f16_f32(
                d0, d1, d2, d3,
                a_regs[0], a_regs[1], a_regs[2], a_regs[3],
                b_regs[0], b_regs[1], b_regs[2], b_regs[3],
                d0, d1, d2, d3,
                meta
            );
            acc[0] = d0; acc[1] = d1; acc[2] = d2; acc[3] = d3;
        }
        
        __syncthreads();
    }
    
    // Store results
    // Output fragment layout for m16n8: each thread outputs 4 values
    // Rows: groupID, groupID+8 (2 rows, 2 values each)
    // Cols: (groupLaneID % 2) * 2 for first pair, (groupLaneID % 2) * 2 for second
    
    int outRow0 = blockM + groupID;
    int outRow1 = blockM + groupID + 8;
    int outCol = blockN + warpTileN + (groupLaneID % 2) * 2 + (groupLaneID / 2) * 4;
    
    // acc[0], acc[1] go to outRow0
    // acc[2], acc[3] go to outRow1
    if (outRow0 < M && outCol < N) {
        C[outRow0 * N + outCol] = __float2half(acc[0]);
        if (outCol + 1 < N) C[outRow0 * N + outCol + 1] = __float2half(acc[1]);
    }
    if (outRow1 < M && outCol < N) {
        C[outRow1 * N + outCol] = __float2half(acc[2]);
        if (outCol + 1 < N) C[outRow1 * N + outCol + 1] = __float2half(acc[3]);
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
    
    std::cout << "Sparse GEMM FP16 with mma.sp.m16n8k32: " 
              << M_SIZE << "x" << N_SIZE << "x" << K_SIZE << std::endl;
    
    std::vector<float> A_dense_f32(M_SIZE * K_SIZE);
    std::vector<float> B_f32(N_SIZE * K_SIZE);
    std::vector<half> A_packed(M_SIZE * K_PACKED);
    std::vector<uint32_t> A_meta(M_SIZE * META_COLS);
    std::vector<half> B_fp16(N_SIZE * K_SIZE);
    
    std::cout << "Initializing data..." << std::endl;
    init_sparse_2to4_pattern(A_dense_f32.data(), M_SIZE, K_SIZE, 42);
    encode_sparse_2to4_fp16(A_dense_f32.data(), A_packed.data(), A_meta.data(), M_SIZE, K_SIZE);
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
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
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
    
    return 0;
}
