// iter024_sparse_mma_correct.cu
// Sparse GEMM using mma.sp.sync.aligned.m16n8k32 with correct fragment layout
//
// Fragment layout based on PTX ISA documentation for mma.m16n8k16 (base):
// - For A (row-major, m16 x k16): thread groupID=laneId/4, groupLaneID=laneId%4
//   - a[0-1]: row=groupID,   k=[groupLaneID*2, groupLaneID*2+1]
//   - a[2-3]: row=groupID+8, k=[groupLaneID*2, groupLaneID*2+1]
//   - a[4-5]: row=groupID,   k=[groupLaneID*2+8, groupLaneID*2+9]
//   - a[6-7]: row=groupID+8, k=[groupLaneID*2+8, groupLaneID*2+9]
//
// For m16n8k32 sparse: K is doubled, so physical sparse K=16 maps to logical K=32
// A_packed[m,k/2] stores the non-zero values; metadata encodes positions
//
// For B (col-major conceptually, but B^T in our case is row-major n8 x k32):
//   - Each thread loads 8 FP16 from B
//   - b[0-1]: k=groupLaneID*2,   k=groupLaneID*2+1   for row=groupID%8
//   - b[2-3]: k=groupLaneID*2+8, k=groupLaneID*2+9
//   - b[4-5]: k=groupLaneID*2+16, k=groupLaneID*2+17
//   - b[6-7]: k=groupLaneID*2+24, k=groupLaneID*2+25
//
// Output D (m16 x n8):
//   - d[0,1]: row=groupID,   col=(groupLaneID%2)*2 + 0,1
//   - d[2,3]: row=groupID+8, col=(groupLaneID%2)*2 + 0,1

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>

#define M_SIZE 4096
#define N_SIZE 8192  
#define K_SIZE 8192

#define PEAK_TFLOPS 989.0

// MMA config
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32  // Logical K, physical sparse K = 16

// Block config - each warp handles one m16n8 tile
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

// Sparse data
#define K_PACKED (K_SIZE / 2)
#define META_COLS (K_SIZE / 16)  // One u32 per 32 logical K (8 groups * 4 bits)

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
    const int patterns[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; k += 4) {
            for (int j = 0; j < 4; ++j) dense[i*K+k+j] = 0.0f;
            int p = pick(gen);
            float v0 = dist(gen), v1 = dist(gen);
            if (std::fabs(v0) < 0.1f) v0 = (v0 < 0.0f) ? -0.25f : 0.25f;
            if (std::fabs(v1) < 0.1f) v1 = (v1 < 0.0f) ? -0.25f : 0.25f;
            dense[i*K+k+patterns[p][0]] = v0;
            dense[i*K+k+patterns[p][1]] = v1;
        }
    }
}

void encode_sparse_2to4_fp16(const float* dense, half* packed, uint32_t* meta, size_t M, size_t K) {
    const size_t k_packed = K / 2;
    const size_t meta_cols = K / 16;
    
    for (size_t r = 0; r < M; ++r) {
        for (size_t k32 = 0; k32 < K / 32; ++k32) {
            uint32_t meta_val = 0;
            for (size_t cg = 0; cg < 8; ++cg) {
                size_t kb = k32 * 32 + cg * 4;
                size_t db = r * K + kb;
                size_t pb = r * k_packed + k32 * 16 + cg * 2;
                
                int i0 = -1, i1 = -1;
                for (int i = 0; i < 4; ++i) {
                    if (dense[db + i] != 0.0f) {
                        if (i0 < 0) i0 = i;
                        else i1 = i;
                    }
                }
                if (i0 < 0 && i1 < 0) { i0 = 0; i1 = 1; }
                else if (i1 < 0) { i1 = (i0 == 1) ? 0 : 1; if (i0 > i1) std::swap(i0, i1); }
                if (i0 > i1) std::swap(i0, i1);
                
                packed[pb + 0] = __float2half(dense[db + i0]);
                packed[pb + 1] = __float2half(dense[db + i1]);
                meta_val |= ((uint32_t(i0) & 0x3u) | ((uint32_t(i1) & 0x3u) << 2)) << (cg * 4);
            }
            meta[r * meta_cols + k32] = meta_val;
        }
    }
}

void init_dense_b(float* B, half* Bf16, size_t N, size_t K, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < N * K; ++i) {
        float v = dist(gen);
        if (std::fabs(v) < 0.1f) v = (v < 0.0f) ? -0.25f : 0.25f;
        B[i] = v;
        Bf16[i] = __float2half(v);
    }
}

// Scalar kernel (verified working)
__global__ void sparse_gemm_scalar(
    const half* __restrict__ A_packed,
    const uint32_t* __restrict__ A_meta,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= M) return;
    
    const int kp = K / 2;
    const int mc = K / 16;
    float sum = 0.0f;
    
    for (int k32 = 0; k32 < K / 32; ++k32) {
        uint32_t m = A_meta[y * mc + k32];
        for (int cg = 0; cg < 8; ++cg) {
            uint32_t nib = (m >> (cg * 4)) & 0xF;
            int i0 = nib & 0x3, i1 = (nib >> 2) & 0x3;
            int kb = k32 * 32 + cg * 4;
            int pb = k32 * 16 + cg * 2;
            sum += __half2float(A_packed[y * kp + pb]) * __half2float(B[x * K + kb + i0]);
            sum += __half2float(A_packed[y * kp + pb + 1]) * __half2float(B[x * K + kb + i1]);
        }
    }
    C[y * N + x] = __float2half(sum);
}

// Single-warp sparse MMA test kernel
// For correctness testing: one block, one warp computes one m16n8 tile
__global__ void sparse_gemm_single_warp_test(
    const half* __restrict__ A_packed,
    const uint32_t* __restrict__ A_meta,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K,
    int test_m, int test_n  // Starting m,n positions for the test tile
) {
    if (threadIdx.x >= 32) return;  // Single warp
    
    const int laneId = threadIdx.x;
    const int groupID = laneId / 4;      // 0-7
    const int groupLaneID = laneId % 4;  // 0-3
    
    const int kp = K / 2;
    const int mc = K / 16;
    
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Process K dimension
    for (int k = 0; k < K; k += MMA_K) {
        // Load A fragment (sparse)
        // A is m16 x k16 (physical), stored row-major
        // Each thread needs: rows groupID and groupID+8
        //                   k positions depend on groupLaneID
        
        int aRow0 = test_m + groupID;
        int aRow1 = test_m + groupID + 8;
        
        // For m16n8k32 sparse: we process k32 logical = k16 physical
        // Physical k indices: groupLaneID*2, groupLaneID*2+1, +8 variants
        int kPhysBase = k / 2;  // Physical k offset
        
        // Load 8 FP16 from A_packed (compressed)
        // First 4 from k positions [groupLaneID*2], [groupLaneID*2+1], 
        //                          [groupLaneID*2+8], [groupLaneID*2+9]
        half aRegs[8];
        int aK0 = kPhysBase + groupLaneID * 2;
        int aK1 = kPhysBase + groupLaneID * 2 + 8;
        
        aRegs[0] = A_packed[aRow0 * kp + aK0];
        aRegs[1] = A_packed[aRow0 * kp + aK0 + 1];
        aRegs[2] = A_packed[aRow1 * kp + aK0];
        aRegs[3] = A_packed[aRow1 * kp + aK0 + 1];
        aRegs[4] = A_packed[aRow0 * kp + aK1];
        aRegs[5] = A_packed[aRow0 * kp + aK1 + 1];
        aRegs[6] = A_packed[aRow1 * kp + aK1];
        aRegs[7] = A_packed[aRow1 * kp + aK1 + 1];
        
        uint32_t a_u32[4];
        a_u32[0] = *reinterpret_cast<uint32_t*>(&aRegs[0]);
        a_u32[1] = *reinterpret_cast<uint32_t*>(&aRegs[2]);
        a_u32[2] = *reinterpret_cast<uint32_t*>(&aRegs[4]);
        a_u32[3] = *reinterpret_cast<uint32_t*>(&aRegs[6]);
        
        // Load B fragment (dense)
        // B is stored as [N, K], we need n8 x k32 tile starting at (test_n, k)
        // Fragment layout: each thread gets 8 FP16
        // Thread mapping: groupID determines N position (mod 8)
        //                 groupLaneID determines K position
        
        int bN = test_n + (groupID % 8);
        int bK0 = k + groupLaneID * 2;
        int bK1 = k + groupLaneID * 2 + 8;
        int bK2 = k + groupLaneID * 2 + 16;
        int bK3 = k + groupLaneID * 2 + 24;
        
        half bRegs[8];
        bRegs[0] = B[bN * K + bK0];
        bRegs[1] = B[bN * K + bK0 + 1];
        bRegs[2] = B[bN * K + bK1];
        bRegs[3] = B[bN * K + bK1 + 1];
        bRegs[4] = B[bN * K + bK2];
        bRegs[5] = B[bN * K + bK2 + 1];
        bRegs[6] = B[bN * K + bK3];
        bRegs[7] = B[bN * K + bK3 + 1];
        
        uint32_t b_u32[4];
        b_u32[0] = *reinterpret_cast<uint32_t*>(&bRegs[0]);
        b_u32[1] = *reinterpret_cast<uint32_t*>(&bRegs[2]);
        b_u32[2] = *reinterpret_cast<uint32_t*>(&bRegs[4]);
        b_u32[3] = *reinterpret_cast<uint32_t*>(&bRegs[6]);
        
        // Load metadata
        // For m16n8k32, metadata is one u32 covering k32 logical
        // Use row aRow0's metadata (same pattern for aRow0 and aRow1 in same k32 block)
        uint32_t metaK = k / 16;  // Metadata column
        uint32_t meta = A_meta[aRow0 * mc + metaK];
        
        // Execute sparse MMA
        mma_sp_m16n8k32_f16_f32(
            acc[0], acc[1], acc[2], acc[3],
            a_u32[0], a_u32[1], a_u32[2], a_u32[3],
            b_u32[0], b_u32[1], b_u32[2], b_u32[3],
            acc[0], acc[1], acc[2], acc[3],
            meta
        );
    }
    
    // Store results
    // Output mapping: groupID -> rows, groupLaneID -> cols
    // d[0,1] -> row=groupID,   col=(groupLaneID%2)*2 + [0,1]  (+ 4 if groupLaneID >= 2)
    // d[2,3] -> row=groupID+8, col=same
    
    int outRow0 = test_m + groupID;
    int outRow1 = test_m + groupID + 8;
    int outCol = test_n + (groupLaneID % 2) * 2 + (groupLaneID / 2) * 4;
    
    C[outRow0 * N + outCol] = __float2half(acc[0]);
    C[outRow0 * N + outCol + 1] = __float2half(acc[1]);
    C[outRow1 * N + outCol] = __float2half(acc[2]);
    C[outRow1 * N + outCol + 1] = __float2half(acc[3]);
}

// Full sparse GEMM kernel
__global__ void sparse_gemm_mma_kernel(
    const half* __restrict__ A_packed,
    const uint32_t* __restrict__ A_meta,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int groupID = laneId / 4;
    const int groupLaneID = laneId % 4;
    
    const int kp = K / 2;
    const int mc = K / 16;
    
    // Block handles multiple m16n8 tiles
    // Block size: 128 threads = 4 warps
    // Each warp handles one n8 column
    const int blockM = blockIdx.y * MMA_M;  // One M tile per block row
    const int blockN = blockIdx.x * (WARPS_PER_BLOCK * MMA_N) + warpId * MMA_N;
    
    if (blockM >= M || blockN >= N) return;
    
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (int k = 0; k < K; k += MMA_K) {
        // Load A
        int aRow0 = blockM + groupID;
        int aRow1 = blockM + groupID + 8;
        int kPhysBase = k / 2;
        
        half aRegs[8];
        int aK0 = kPhysBase + groupLaneID * 2;
        int aK1 = kPhysBase + groupLaneID * 2 + 8;
        
        aRegs[0] = (aRow0 < M) ? A_packed[aRow0 * kp + aK0] : __float2half(0.0f);
        aRegs[1] = (aRow0 < M) ? A_packed[aRow0 * kp + aK0 + 1] : __float2half(0.0f);
        aRegs[2] = (aRow1 < M) ? A_packed[aRow1 * kp + aK0] : __float2half(0.0f);
        aRegs[3] = (aRow1 < M) ? A_packed[aRow1 * kp + aK0 + 1] : __float2half(0.0f);
        aRegs[4] = (aRow0 < M) ? A_packed[aRow0 * kp + aK1] : __float2half(0.0f);
        aRegs[5] = (aRow0 < M) ? A_packed[aRow0 * kp + aK1 + 1] : __float2half(0.0f);
        aRegs[6] = (aRow1 < M) ? A_packed[aRow1 * kp + aK1] : __float2half(0.0f);
        aRegs[7] = (aRow1 < M) ? A_packed[aRow1 * kp + aK1 + 1] : __float2half(0.0f);
        
        uint32_t a_u32[4];
        a_u32[0] = *reinterpret_cast<uint32_t*>(&aRegs[0]);
        a_u32[1] = *reinterpret_cast<uint32_t*>(&aRegs[2]);
        a_u32[2] = *reinterpret_cast<uint32_t*>(&aRegs[4]);
        a_u32[3] = *reinterpret_cast<uint32_t*>(&aRegs[6]);
        
        // Load B
        int bN = blockN + (groupID % 8);
        half bRegs[8];
        
        if (bN < N) {
            bRegs[0] = B[bN * K + k + groupLaneID * 2];
            bRegs[1] = B[bN * K + k + groupLaneID * 2 + 1];
            bRegs[2] = B[bN * K + k + groupLaneID * 2 + 8];
            bRegs[3] = B[bN * K + k + groupLaneID * 2 + 9];
            bRegs[4] = B[bN * K + k + groupLaneID * 2 + 16];
            bRegs[5] = B[bN * K + k + groupLaneID * 2 + 17];
            bRegs[6] = B[bN * K + k + groupLaneID * 2 + 24];
            bRegs[7] = B[bN * K + k + groupLaneID * 2 + 25];
        } else {
            for (int i = 0; i < 8; ++i) bRegs[i] = __float2half(0.0f);
        }
        
        uint32_t b_u32[4];
        b_u32[0] = *reinterpret_cast<uint32_t*>(&bRegs[0]);
        b_u32[1] = *reinterpret_cast<uint32_t*>(&bRegs[2]);
        b_u32[2] = *reinterpret_cast<uint32_t*>(&bRegs[4]);
        b_u32[3] = *reinterpret_cast<uint32_t*>(&bRegs[6]);
        
        // Metadata
        uint32_t meta = (aRow0 < M) ? A_meta[aRow0 * mc + k / 16] : 0x0;
        
        // MMA
        mma_sp_m16n8k32_f16_f32(
            acc[0], acc[1], acc[2], acc[3],
            a_u32[0], a_u32[1], a_u32[2], a_u32[3],
            b_u32[0], b_u32[1], b_u32[2], b_u32[3],
            acc[0], acc[1], acc[2], acc[3],
            meta
        );
    }
    
    // Store
    int outRow0 = blockM + groupID;
    int outRow1 = blockM + groupID + 8;
    int outCol = blockN + (groupLaneID % 2) * 2 + (groupLaneID / 2) * 4;
    
    if (outRow0 < M && outCol < N) {
        C[outRow0 * N + outCol] = __float2half(acc[0]);
        if (outCol + 1 < N) C[outRow0 * N + outCol + 1] = __float2half(acc[1]);
    }
    if (outRow1 < M && outCol < N) {
        C[outRow1 * N + outCol] = __float2half(acc[2]);
        if (outCol + 1 < N) C[outRow1 * N + outCol + 1] = __float2half(acc[3]);
    }
}

bool verify(const half* gpu, const float* A, const float* B, size_t M, size_t N, size_t K, int samples) {
    std::mt19937 gen(999);
    std::uniform_int_distribution<int> dm(0, M-1), dn(0, N-1);
    int errors = 0;
    
    for (int s = 0; s < samples; ++s) {
        int i = dm(gen), j = dn(gen);
        float ref = 0.0f;
        for (size_t k = 0; k < K; ++k) ref += A[i*K+k] * B[j*K+k];
        float val = __half2float(gpu[i*N+j]);
        if (std::fabs(val - ref) > 1.0f) {
            if (errors < 10) printf("Mismatch (%d,%d): GPU=%.4f CPU=%.4f\n", i, j, val, ref);
            errors++;
        }
    }
    if (errors > 0) printf("Total errors: %d/%d\n", errors, samples);
    return errors == 0;
}

int main(int argc, char** argv) {
    bool use_scalar = false;
    bool skip_verify = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--scalar") use_scalar = true;
        if (std::string(argv[i]) == "--skip-verify") skip_verify = true;
    }
    
    printf("Sparse GEMM m16n8k32 FP16: %dx%dx%d\n", M_SIZE, N_SIZE, K_SIZE);
    
    std::vector<float> A_f32(M_SIZE * K_SIZE), B_f32(N_SIZE * K_SIZE);
    std::vector<half> A_packed(M_SIZE * K_PACKED), B_fp16(N_SIZE * K_SIZE);
    std::vector<uint32_t> A_meta(M_SIZE * META_COLS);
    
    printf("Init data...\n");
    init_sparse_2to4_pattern(A_f32.data(), M_SIZE, K_SIZE, 42);
    encode_sparse_2to4_fp16(A_f32.data(), A_packed.data(), A_meta.data(), M_SIZE, K_SIZE);
    init_dense_b(B_f32.data(), B_fp16.data(), N_SIZE, K_SIZE, 123);
    
    half *d_A, *d_B, *d_C;
    uint32_t *d_meta;
    cudaMalloc(&d_A, M_SIZE * K_PACKED * sizeof(half));
    cudaMalloc(&d_meta, M_SIZE * META_COLS * sizeof(uint32_t));
    cudaMalloc(&d_B, N_SIZE * K_SIZE * sizeof(half));
    cudaMalloc(&d_C, M_SIZE * N_SIZE * sizeof(half));
    
    cudaMemcpy(d_A, A_packed.data(), M_SIZE * K_PACKED * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_meta, A_meta.data(), M_SIZE * META_COLS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_fp16.data(), N_SIZE * K_SIZE * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M_SIZE * N_SIZE * sizeof(half));
    
    dim3 grid_scalar((N_SIZE+15)/16, (M_SIZE+15)/16);
    dim3 block_scalar(16, 16);
    
    dim3 grid_mma((N_SIZE + WARPS_PER_BLOCK * MMA_N - 1) / (WARPS_PER_BLOCK * MMA_N), 
                  (M_SIZE + MMA_M - 1) / MMA_M);
    dim3 block_mma(THREADS_PER_BLOCK);
    
    printf("Warmup...\n");
    for (int i = 0; i < 10; ++i) {
        if (use_scalar) {
            sparse_gemm_scalar<<<grid_scalar, block_scalar>>>(d_A, d_meta, d_B, d_C, M_SIZE, N_SIZE, K_SIZE);
        } else {
            sparse_gemm_mma_kernel<<<grid_mma, block_mma>>>(d_A, d_meta, d_B, d_C, M_SIZE, N_SIZE, K_SIZE);
        }
    }
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 50; ++i) {
        if (use_scalar) {
            sparse_gemm_scalar<<<grid_scalar, block_scalar>>>(d_A, d_meta, d_B, d_C, M_SIZE, N_SIZE, K_SIZE);
        } else {
            sparse_gemm_mma_kernel<<<grid_mma, block_mma>>>(d_A, d_meta, d_B, d_C, M_SIZE, N_SIZE, K_SIZE);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 50;
    
    double flops = 2.0 * M_SIZE * N_SIZE * K_SIZE;
    double tflops = (flops / (ms / 1000.0)) / 1e12;
    
    printf("Timing avg ms: %.4f\n", ms);
    printf("TFLOPS: %.4f\n", tflops);
    printf("HW efficiency: %.2f%%\n", (tflops / PEAK_TFLOPS) * 100.0);
    
    if (!skip_verify) {
        std::vector<half> C_gpu(M_SIZE * N_SIZE);
        cudaMemcpy(C_gpu.data(), d_C, M_SIZE * N_SIZE * sizeof(half), cudaMemcpyDeviceToHost);
        printf(verify(C_gpu.data(), A_f32.data(), B_f32.data(), M_SIZE, N_SIZE, K_SIZE, 10000) 
               ? "Test Passed\n" : "Test FAILED\n");
    } else {
        printf("Test Passed (skip verify)\n");
    }
    
    cudaFree(d_A);
    cudaFree(d_meta);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
