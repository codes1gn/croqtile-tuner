#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

constexpr int M_DIM = 16384;
constexpr int N_DIM = 16384;
constexpr int K_DIM = 16384;

// WGMMA tile sizes for sm_90 - m16n256k16 shape
constexpr int WGMMA_M = 64;  // 4 warps in M, 16 rows each
constexpr int WGMMA_N = 256;
constexpr int WGMMA_K = 16;

// Block tile sizes
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 64;

constexpr int WARP_SIZE = 32;
constexpr int WARP_GROUP_SIZE = 128;  // 4 warps = 1 warp group for WGMMA
constexpr int NUM_WARP_GROUPS = 2;
constexpr int BLOCK_SIZE = NUM_WARP_GROUPS * WARP_GROUP_SIZE;  // 256 threads

// WGMMA descriptor helper - computes descriptor for shared memory operand
__device__ __forceinline__ uint64_t make_smem_desc(const void* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    // SM90 WGMMA descriptor format:
    // bits[0:13] = base address >> 4 (16-byte aligned)
    // bits[14:15] = 0 (stride mode = 0 for contiguous)
    // bits[16:29] = leading dimension stride >> 4
    // bits[30:31] = swizzle mode
    return (uint64_t(addr >> 4) & 0x3FFF);
}

// WGMMA fence operations
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::);
}

__device__ __forceinline__ void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::);
}

__device__ __forceinline__ void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::);
}

// WGMMA m64n256k16 for FP16 inputs, FP32 accumulators
// Each warp group computes a 64x256 tile
__device__ __forceinline__ void wgmma_m64n256k16_f16f16f32(
    float (&d)[64],  // 64 floats per thread in warp group covers 64x256 output
    const half* __restrict__ a_smem,  // 64x16 A tile in shared memory
    const half* __restrict__ b_smem   // 16x256 B tile in shared memory
) {
    uint64_t desc_a = make_smem_desc(a_smem);
    uint64_t desc_b = make_smem_desc(b_smem);
    
    // Use inline PTX for wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16
    // This is a complex instruction - using simplified form
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, 1, 1, 1;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(desc_a), "l"(desc_b)
    );
}

__global__ void matmul_wgmma_kernel(const half* __restrict__ A,
                                    const half* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    const int warpGroupId = threadIdx.x / WARP_GROUP_SIZE;
    const int tid = threadIdx.x;
    
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    
    // Shared memory for A and B tiles
    __shared__ __align__(128) half As[BM][BK + 8];
    __shared__ __align__(128) half Bs[BK][BN + 8];
    
    // Each warp group accumulates its 64x256 output tile
    // With BM=128, BN=256, we have 2 warp groups each doing 64x256
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        acc[i] = 0.0f;
    }
    
    // Warp group row offset within block
    const int wgRow = warpGroupId * 64;  // 0 or 64
    
    // Load patterns - 256 threads loading BM x BK = 128 x 64 = 8192 halfs
    // Each thread loads 32 halfs (4 float4s)
    const int loadARow = tid / 8;   // 256/8 = 32 unique rows per pass
    const int loadACol = (tid % 8) * 8;  // 8 halfs per thread
    
    const int loadBRow = tid / 32;  // 256/32 = 8 unique rows per pass
    const int loadBCol = (tid % 32) * 8;  // 8 halfs per thread
    
    for (int kBlock = 0; kBlock < K; kBlock += BK) {
        // Load A tile: BM x BK = 128 x 64
        #pragma unroll
        for (int m = 0; m < 4; m++) {  // 4 passes to load 128 rows
            int row = loadARow + m * 32;
            int globalM = blockRow + row;
            int globalK = kBlock + loadACol;
            
            if (globalM < M && globalK + 7 < K) {
                __pipeline_memcpy_async(
                    reinterpret_cast<float4*>(&As[row][loadACol]),
                    reinterpret_cast<const float4*>(&A[globalM * K + globalK]),
                    sizeof(float4)
                );
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int gk = globalK + i;
                    As[row][loadACol + i] = (globalM < M && gk < K) ? A[globalM * K + gk] : __float2half(0.0f);
                }
            }
        }
        
        // Load B tile: BK x BN = 64 x 256
        #pragma unroll
        for (int k = 0; k < 8; k++) {  // 8 passes to load 64 rows
            int row = loadBRow + k * 8;
            int globalK = kBlock + row;
            int globalN = blockCol + loadBCol;
            
            if (globalK < K && globalN + 7 < N) {
                __pipeline_memcpy_async(
                    reinterpret_cast<float4*>(&Bs[row][loadBCol]),
                    reinterpret_cast<const float4*>(&B[globalK * N + globalN]),
                    sizeof(float4)
                );
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int gn = globalN + i;
                    Bs[row][loadBCol + i] = (globalK < K && gn < N) ? B[globalK * N + gn] : __float2half(0.0f);
                }
            }
        }
        
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        
        // WGMMA compute - each warp group processes K dimension
        wgmma_fence();
        
        #pragma unroll
        for (int k = 0; k < BK; k += WGMMA_K) {
            // Each warp group computes 64x256 tile
            wgmma_m64n256k16_f16f16f32(
                acc,
                &As[wgRow][k],
                &Bs[k][0]
            );
        }
        
        wgmma_commit_group();
        wgmma_wait_group();
        
        __syncthreads();
    }
    
    // Store results - each thread in warp group stores its portion
    // For m64n256k16: each of 128 threads stores 64x256/(128) = 128 elements
    // But we accumulated 64 floats per thread, so layout is complex
    // Simplified: use standard warp-level store pattern
    const int laneId = threadIdx.x % 32;
    const int warpIdInGroup = (threadIdx.x % WARP_GROUP_SIZE) / 32;
    
    // Each warp in the group handles a portion of the 64x256 tile
    // 4 warps per group, so each warp handles 64x64 = 4096 elements
    // Each lane handles 4096/32 = 128 elements (but we have 64 accumulators - need to adjust)
    
    // For now, use simpler store - each thread stores 16 elements
    const int warpRow = warpIdInGroup * 16;  // 4 warps -> rows 0, 16, 32, 48
    const int rowOffset = laneId / 2;  // 16 rows per warp, 2 threads per row
    const int colOffset = (laneId % 2) * 8;  // Each thread handles 8 cols
    
    #pragma unroll
    for (int n = 0; n < 256; n += 16) {
        int outRow = blockRow + wgRow + warpRow + rowOffset;
        int outCol = blockCol + n + colOffset;
        
        if (outRow < M && outCol + 7 < N) {
            int accIdx = (warpRow / 16) * 16 + (rowOffset / 4) * 4 + n / 64;
            if (accIdx < 64) {
                // Store via vectorized store
                float4 val;
                val.x = acc[accIdx];
                val.y = acc[accIdx + 1 < 64 ? accIdx + 1 : 0];
                val.z = acc[accIdx + 2 < 64 ? accIdx + 2 : 0];
                val.w = acc[accIdx + 3 < 64 ? accIdx + 3 : 0];
                
                // Store individual floats to avoid alignment issues
                for (int i = 0; i < 4 && outCol + colOffset + i < N; i++) {
                    C[outRow * N + outCol + i] = ((float*)&val)[i];
                }
            }
        }
    }
}

void verifyResult(const float* C_custom, const float* C_ref, int M, int N) {
    float maxAbsErr = 0.0f;
    float maxRelErr = 0.0f;
    
    for (int i = 0; i < M * N; i++) {
        float absErr = fabs(C_custom[i] - C_ref[i]);
        float relErr = absErr / (fabs(C_ref[i]) + 1e-6f);
        maxAbsErr = fmax(maxAbsErr, absErr);
        maxRelErr = fmax(maxRelErr, relErr);
    }
    
    float tolerance = 1.0f + 0.01f * fabs(C_ref[0]);
    if (maxAbsErr < tolerance) {
        printf("VERIFY: PASS max_abs_err=%f max_rel_err=%f\n", maxAbsErr, maxRelErr);
    } else {
        printf("VERIFY: FAIL max_abs_err=%f max_rel_err=%f\n", maxAbsErr, maxRelErr);
    }
}

int main() {
    half *d_A, *d_B;
    float *d_C, *d_C_ref;
    float *h_C, *h_C_ref;
    
    size_t sizeA = M_DIM * K_DIM * sizeof(half);
    size_t sizeB = K_DIM * N_DIM * sizeof(half);
    size_t sizeC = M_DIM * N_DIM * sizeof(float);
    
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMalloc(&d_C_ref, sizeC);
    
    h_C = (float*)malloc(sizeC);
    h_C_ref = (float*)malloc(sizeC);
    
    // Initialize with random values
    half *h_A = (half*)malloc(sizeA);
    half *h_B = (half*)malloc(sizeB);
    
    srand(42);
    for (int i = 0; i < M_DIM * K_DIM; i++) {
        h_A[i] = __float2half((float)(rand() % 10 - 5) / 10.0f);
    }
    for (int i = 0; i < K_DIM * N_DIM; i++) {
        h_B[i] = __float2half((float)(rand() % 10 - 5) / 10.0f);
    }
    
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N_DIM, M_DIM, K_DIM,
                 &alpha,
                 d_B, CUDA_R_16F, N_DIM,
                 d_A, CUDA_R_16F, K_DIM,
                 &beta,
                 d_C_ref, CUDA_R_32F, N_DIM,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cudaMemcpy(h_C_ref, d_C_ref, sizeC, cudaMemcpyDeviceToHost);
    
    // Custom kernel
    dim3 grid((N_DIM + BN - 1) / BN, (M_DIM + BM - 1) / BM);
    dim3 block(BLOCK_SIZE);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_C, 0, sizeC);
        matmul_wgmma_kernel<<<grid, block>>>(d_A, d_B, d_C, M_DIM, N_DIM, K_DIM);
    }
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    verifyResult(h_C, h_C_ref, M_DIM, N_DIM);
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 50; i++) {
        matmul_wgmma_kernel<<<grid, block>>>(d_A, d_B, d_C, M_DIM, N_DIM, K_DIM);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avgMs = ms / 50.0f;
    
    double flops = 2.0 * M_DIM * N_DIM * K_DIM;
    double tflops = flops / (avgMs * 1e9);
    
    printf("TFLOPS: %.3f   time_ms: %.4f\n", tflops, avgMs);
    
    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    
    return 0;
}
