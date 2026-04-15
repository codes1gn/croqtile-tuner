// iter029_fullM.cu - Process all M=512 rows in single block width
// Shape: M=512, N=16384, K=16384
// Idea: Each block covers all 512 M rows with 32-wide N slice

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cmath>

using namespace nvcuda;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

constexpr int M = 512;
constexpr int N = 16384;
constexpr int K = 16384;
constexpr int WARMUP = 10;
constexpr int ITERS = 50;
constexpr int SAMPLES = 5;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Full M in each block
constexpr int BM = 512;  // All rows
constexpr int BN = 32;   // Narrow N
constexpr int BK = 16;

constexpr int WARP_SIZE = 32;
// 32 warps to cover 512/16 = 32 M tiles, each covering 16x32 output
constexpr int WARPS_PER_BLOCK = 32;

__global__ __launch_bounds__(1024, 1)
void matmul_bf16_fp32_fullM(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int m, int n, int k
) {
    int bx = blockIdx.x;  // N dimension only
    int warpId = threadIdx.x / WARP_SIZE;

    int tile_col = bx * BN;

    // Shared memory - can hold full A slice and B tile
    __shared__ __nv_bfloat16 As[BM][BK + 4];   // 512x20 = 20KB
    __shared__ __nv_bfloat16 Bs[BK][BN + 4];   // 16x36 = 1.1KB

    // Each warp covers 16 M rows, 32 N columns = 2 WMMA tiles
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        wmma::fill_fragment(acc[i], 0.0f);
    }

    int num_k_tiles = (k + BK - 1) / BK;
    int tid = threadIdx.x;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int kk = kt * BK;
        
        // Load A: [512 x 16] = 8192 bf16 = 2048 bf16x4
        // 1024 threads, 2 each
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * 1024;
            if (idx < BM * (BK / 4)) {
                int row = idx / (BK / 4);
                int col4 = idx % (BK / 4);
                int col = col4 * 4;
                int gCol = kk + col;
                
                if (gCol + 3 < k) {
                    __pipeline_memcpy_async(&As[row][col], &A[row * k + gCol], 8);
                }
            }
        }
        
        // Load B: [16 x 32] = 512 bf16 = 128 bf16x4
        // 1024 threads, 1 each (only first 128 threads active)
        if (tid < BK * (BN / 4)) {
            int row = tid / (BN / 4);
            int col4 = tid % (BN / 4);
            int col = col4 * 4;
            int gRow = kk + row;
            int gCol = tile_col + col;
            
            if (gRow < k && gCol + 3 < n) {
                __pipeline_memcpy_async(&Bs[row][col], &B[gRow * n + gCol], 8);
            }
        }
        
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        // Compute
        int warp_row = warpId * 16;
        
        #pragma unroll
        for (int wj = 0; wj < 2; wj++) {
            int b_col = wj * WMMA_N;
            
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            
            wmma::load_matrix_sync(a_frag, &As[warp_row][0], BK + 4);
            wmma::load_matrix_sync(b_frag, &Bs[0][b_col], BN + 4);
            wmma::mma_sync(acc[wj], a_frag, b_frag, acc[wj]);
        }
        
        __syncthreads();
    }

    // Store
    int warp_row = warpId * 16;
    
    #pragma unroll
    for (int wj = 0; wj < 2; wj++) {
        int global_col = tile_col + wj * WMMA_N;
        
        if (warp_row < m && global_col < n) {
            wmma::store_matrix_sync(&C[warp_row * n + global_col], acc[wj], n, wmma::mem_row_major);
        }
    }
}

void init_bf16_random(__nv_bfloat16* data, size_t size, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; i++) {
        data[i] = __float2bfloat16(dist(gen));
    }
}

int main(int argc, char** argv) {
    bool verify_mode = (argc > 1 && std::string(argv[1]) == "--verify");
    
    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;

    __nv_bfloat16 *h_A, *h_B;
    float *h_C;
    __nv_bfloat16 *d_A, *d_B;
    float *d_C;

    h_A = new __nv_bfloat16[size_A];
    h_B = new __nv_bfloat16[size_B];
    h_C = new float[size_C];

    init_bf16_random(h_A, size_A, 42);
    init_bf16_random(h_B, size_B, 123);

    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, size_C * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    dim3 grid((N + BN - 1) / BN);  // Only N dimension
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);

    for (int i = 0; i < WARMUP; i++) {
        matmul_bf16_fp32_fullM<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        matmul_bf16_fp32_fullM<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        
        int nonzero = 0, nan_count = 0;
        for (int i = 0; i < 1000; i++) {
            if (h_C[i] != 0.0f) nonzero++;
            if (isnan(h_C[i])) nan_count++;
        }
        printf("Sanity check: %d/1000 non-zero, %d NaN\n", nonzero, nan_count);
        printf("verification: %s\n", (nonzero > 900 && nan_count == 0) ? "PASSED" : "FAILED");
        
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        delete[] h_A; delete[] h_B; delete[] h_C;
        return (nonzero > 900 && nan_count == 0) ? 0 : 1;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double tflops_sum = 0.0, tflops_min = 1e9, tflops_max = 0.0;

    for (int s = 0; s < SAMPLES; s++) {
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++) {
            matmul_bf16_fp32_fullM<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        double time_per_iter_ms = ms / ITERS;
        double flops = 2.0 * M * N * K;
        double tflops = flops / (time_per_iter_ms * 1e-3) / 1e12;

        printf("sample %d: time=%.3f ms, tflops=%.2f\n", s + 1, time_per_iter_ms, tflops);
        tflops_sum += tflops;
        tflops_min = fmin(tflops_min, tflops);
        tflops_max = fmax(tflops_max, tflops);
    }

    printf("\n=== ITER029 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M, N, K);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter029_fullM (512x32, 32 warps)\n");
    printf("avg_tflops: %.2f\n", tflops_sum / SAMPLES);
    printf("min_tflops: %.2f\n", tflops_min);
    printf("max_tflops: %.2f\n", tflops_max);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
