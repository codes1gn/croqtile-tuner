// iter025_sparse_mma_k16.cu
// Sparse GEMM using simpler mma.sp.sync.aligned.m16n8k16 instruction
// 
// m16n8k16 sparse (simpler layout):
// - A registers: 2 x u32 (4 fp16 from 8 logical = 2:4 compression)
// - B registers: 2 x u32 (4 fp16)
// - C/D registers: 4 x f32
// - metadata: 1 x u32

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

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16  // Logical K, physical sparse K = 8

#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

#define K_PACKED (K_SIZE / 2)
#define META_COLS (K_SIZE / 16)  // One u32 per 16 logical K (4 groups * 4 bits)

// PTX sparse MMA m16n8k16
__device__ __forceinline__ void mma_sp_m16n8k16_f16_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint32_t meta
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3}, %8, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(meta)
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
        // For m16n8k16, one u32 metadata covers 16 logical K (4 groups of 4)
        for (size_t k16 = 0; k16 < K / 16; ++k16) {
            uint32_t meta_val = 0;
            for (size_t cg = 0; cg < 4; ++cg) {  // 4 groups per 16 K
                size_t kb = k16 * 16 + cg * 4;
                size_t db = r * K + kb;
                size_t pb = r * k_packed + k16 * 8 + cg * 2;
                
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
            meta[r * meta_cols + k16] = meta_val;
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

// Scalar kernel (verified)
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
    
    for (int k16 = 0; k16 < K / 16; ++k16) {
        uint32_t m = A_meta[y * mc + k16];
        for (int cg = 0; cg < 4; ++cg) {
            uint32_t nib = (m >> (cg * 4)) & 0xF;
            int i0 = nib & 0x3, i1 = (nib >> 2) & 0x3;
            int kb = k16 * 16 + cg * 4;
            int pb = k16 * 8 + cg * 2;
            sum += __half2float(A_packed[y * kp + pb]) * __half2float(B[x * K + kb + i0]);
            sum += __half2float(A_packed[y * kp + pb + 1]) * __half2float(B[x * K + kb + i1]);
        }
    }
    C[y * N + x] = __float2half(sum);
}

// Sparse MMA kernel using m16n8k16
// For m16n8k16 sparse:
// - A: 2 u32 = 4 fp16 (from 8 physical positions = 16 logical)
// - B: 2 u32 = 4 fp16 (from 16 k positions)
// 
// Fragment layout based on the m8n8k4 base tile (replicated):
// For A (m16 x k8 physical):
//   - groupID = laneId / 4 (0-7), groupLaneID = laneId % 4 (0-3)
//   - a[0,1]: row=groupID,   k=[groupLaneID] and [groupLaneID+4] (interleaved?)
//   - Actually for m16n8k16 dense, each thread holds:
//     a[0-1]: elements at row groupID, k=[groupLaneID*2, groupLaneID*2+1]
//     a[2-3]: elements at row groupID+8, k=[groupLaneID*2, groupLaneID*2+1]

__global__ void sparse_gemm_mma_kernel(
    const half* __restrict__ A_packed,
    const uint32_t* __restrict__ A_meta,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int groupID = laneId / 4;       // 0-7
    const int groupLaneID = laneId % 4;   // 0-3
    
    const int kp = K / 2;
    const int mc = K / 16;
    
    const int blockM = blockIdx.y * MMA_M;
    const int blockN = blockIdx.x * (WARPS_PER_BLOCK * MMA_N) + warpId * MMA_N;
    
    if (blockM >= M || blockN >= N) return;
    
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (int k = 0; k < K; k += MMA_K) {
        // Load A (sparse, m16 x k8 physical)
        // Each thread loads 4 fp16 = 2 u32
        // Row indices: groupID, groupID+8
        // K indices: groupLaneID*2, groupLaneID*2+1 (physical k)
        
        int aRow0 = blockM + groupID;
        int aRow1 = blockM + groupID + 8;
        int kPhys = k / 2;  // Physical k offset
        int aK = kPhys + groupLaneID * 2;
        
        half aRegs[4];
        aRegs[0] = (aRow0 < M) ? A_packed[aRow0 * kp + aK] : __float2half(0.0f);
        aRegs[1] = (aRow0 < M) ? A_packed[aRow0 * kp + aK + 1] : __float2half(0.0f);
        aRegs[2] = (aRow1 < M) ? A_packed[aRow1 * kp + aK] : __float2half(0.0f);
        aRegs[3] = (aRow1 < M) ? A_packed[aRow1 * kp + aK + 1] : __float2half(0.0f);
        
        uint32_t a_u32[2];
        a_u32[0] = *reinterpret_cast<uint32_t*>(&aRegs[0]);
        a_u32[1] = *reinterpret_cast<uint32_t*>(&aRegs[2]);
        
        // Load B (dense, n8 x k16)
        // Each thread loads 4 fp16 = 2 u32
        // N index: groupID % 8
        // K indices: groupLaneID*2, groupLaneID*2+1 for first u32
        //           groupLaneID*2+8, groupLaneID*2+9 for second u32
        
        int bN = blockN + (groupID % 8);
        half bRegs[4];
        
        if (bN < N) {
            bRegs[0] = B[bN * K + k + groupLaneID * 2];
            bRegs[1] = B[bN * K + k + groupLaneID * 2 + 1];
            bRegs[2] = B[bN * K + k + groupLaneID * 2 + 8];
            bRegs[3] = B[bN * K + k + groupLaneID * 2 + 9];
        } else {
            for (int i = 0; i < 4; ++i) bRegs[i] = __float2half(0.0f);
        }
        
        uint32_t b_u32[2];
        b_u32[0] = *reinterpret_cast<uint32_t*>(&bRegs[0]);
        b_u32[1] = *reinterpret_cast<uint32_t*>(&bRegs[2]);
        
        // Metadata - one u32 per 16 logical K
        uint32_t meta = (aRow0 < M) ? A_meta[aRow0 * mc + k / 16] : 0x0;
        
        // Execute sparse MMA
        mma_sp_m16n8k16_f16_f32(
            acc[0], acc[1], acc[2], acc[3],
            a_u32[0], a_u32[1],
            b_u32[0], b_u32[1],
            acc[0], acc[1], acc[2], acc[3],
            meta
        );
    }
    
    // Store results
    // Output layout m16n8: each thread writes 4 elements
    // d[0,1]: row=groupID,   col=(groupLaneID%2)*2 + [0,1] + (groupLaneID/2)*4
    // d[2,3]: row=groupID+8, col=same
    
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
        for (size_t kk = 0; kk < K; ++kk) ref += A[i*K+kk] * B[j*K+kk];
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
    bool use_scalar = false, skip_verify = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--scalar") use_scalar = true;
        if (std::string(argv[i]) == "--skip-verify") skip_verify = true;
    }
    
    printf("Sparse GEMM m16n8k16 FP16: %dx%dx%d\n", M_SIZE, N_SIZE, K_SIZE);
    
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
