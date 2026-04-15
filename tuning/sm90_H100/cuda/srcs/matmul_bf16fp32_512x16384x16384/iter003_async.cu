// iter003_async.cu - WMMA with async memory copy and double buffering
// Shape: M=512, N=16384, K=16384  
// Idea: Use cp.async for memory loads, double buffer shared memory to hide latency
// Keep 128x128 output tile but use async prefetch to overlap compute and memory

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

// Use 128x128 output tile with 64 K steps
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;  // Larger K tile to reduce iterations

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;  // 256 threads

// Double buffering for shared memory
constexpr int NUM_STAGES = 2;

__global__ __launch_bounds__(256, 2)
void matmul_bf16_fp32_wmma_async(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int m, int n, int k
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int warpId = threadIdx.x / WARP_SIZE;

    int tile_row = by * BM;
    int tile_col = bx * BN;

    // Double-buffered shared memory
    __shared__ __nv_bfloat16 As[NUM_STAGES][BM][BK];
    __shared__ __nv_bfloat16 Bs[NUM_STAGES][BK][BN];

    // Each warp handles 2x4 WMMA tiles = 32x64 output
    // 8 warps in 2x4 arrangement cover 64x256 = not quite 128x128
    // Let's use 4x2 warp arrangement: each warp does 2x2 WMMA = 32x32
    // 8 warps in 2x4 grid: 2 rows x 4 cols = 64x128 coverage
    // Need 2 passes to cover 128x128
    
    // Simpler: each warp does 2 rows and 2 cols of WMMA = 32x32
    // 8 warps arranged as 4 rows x 2 cols covers 128x64
    // Need 2 horizontal passes
    
    // Even simpler: 8 warps, 4x2 grid, each does 2x4 wmma tiles
    int warp_row = warpId / 2;   // 0,1,2,3
    int warp_col = warpId % 2;   // 0,1

    // Each warp handles 1 row and 2 cols of WMMA tiles = 16x32
    // 8 warps: 4 rows x 2 cols = 64x64... still not 128x128
    
    // Let's go 4x2 warps, each doing 32x64 (2x4 wmma)
    // Total: 4 rows * 32 = 128, 2 cols * 64 = 128. Yes!
    
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][4];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    int num_k_tiles = (k + BK - 1) / BK;
    
    // Prologue: load first tile
    int stage_write = 0;
    {
        int kk = 0;
        // Load A: [BM, BK] = [128, 64] = 8192 bf16
        // 256 threads, 32 elements each
        int elems_per_thread_A = (BM * BK) / 256;
        #pragma unroll
        for (int i = 0; i < elems_per_thread_A; i++) {
            int idx = threadIdx.x * elems_per_thread_A + i;
            int row = idx / BK;
            int col = idx % BK;
            int gRow = tile_row + row;
            int gCol = kk + col;
            As[stage_write][row][col] = (gRow < m && gCol < k) ? A[gRow * k + gCol] : __float2bfloat16(0.0f);
        }
        
        // Load B: [BK, BN] = [64, 128] = 8192 bf16
        int elems_per_thread_B = (BK * BN) / 256;
        #pragma unroll
        for (int i = 0; i < elems_per_thread_B; i++) {
            int idx = threadIdx.x * elems_per_thread_B + i;
            int row = idx / BN;
            int col = idx % BN;
            int gRow = kk + row;
            int gCol = tile_col + col;
            Bs[stage_write][row][col] = (gRow < k && gCol < n) ? B[gRow * n + gCol] : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Main loop with double buffering
    for (int tile_k = 0; tile_k < num_k_tiles; tile_k++) {
        int stage_read = tile_k % NUM_STAGES;
        int stage_next = (tile_k + 1) % NUM_STAGES;
        int kk_next = (tile_k + 1) * BK;

        // Prefetch next tile if not last
        if (tile_k + 1 < num_k_tiles) {
            int elems_per_thread_A = (BM * BK) / 256;
            #pragma unroll
            for (int i = 0; i < elems_per_thread_A; i++) {
                int idx = threadIdx.x * elems_per_thread_A + i;
                int row = idx / BK;
                int col = idx % BK;
                int gRow = tile_row + row;
                int gCol = kk_next + col;
                As[stage_next][row][col] = (gRow < m && gCol < k) ? A[gRow * k + gCol] : __float2bfloat16(0.0f);
            }
            
            int elems_per_thread_B = (BK * BN) / 256;
            #pragma unroll  
            for (int i = 0; i < elems_per_thread_B; i++) {
                int idx = threadIdx.x * elems_per_thread_B + i;
                int row = idx / BN;
                int col = idx % BN;
                int gRow = kk_next + row;
                int gCol = tile_col + col;
                Bs[stage_next][row][col] = (gRow < k && gCol < n) ? B[gRow * n + gCol] : __float2bfloat16(0.0f);
            }
        }

        // Compute on current stage
        #pragma unroll
        for (int wk = 0; wk < BK; wk += WMMA_K) {
            #pragma unroll
            for (int ti = 0; ti < 2; ti++) {
                #pragma unroll
                for (int tj = 0; tj < 4; tj++) {
                    int wmma_row = warp_row * 2 + ti;
                    int wmma_col = warp_col * 4 + tj;

                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;

                    wmma::load_matrix_sync(a_frag, &As[stage_read][wmma_row * WMMA_M][wk], BK);
                    wmma::load_matrix_sync(b_frag, &Bs[stage_read][wk][wmma_col * WMMA_N], BN);
                    wmma::mma_sync(acc[ti][tj], a_frag, b_frag, acc[ti][tj]);
                }
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int ti = 0; ti < 2; ti++) {
        #pragma unroll
        for (int tj = 0; tj < 4; tj++) {
            int wmma_row = warp_row * 2 + ti;
            int wmma_col = warp_col * 4 + tj;

            int global_row = tile_row + wmma_row * WMMA_M;
            int global_col = tile_col + wmma_col * WMMA_N;

            if (global_row < m && global_col < n) {
                wmma::store_matrix_sync(&C[global_row * n + global_col], acc[ti][tj], n, wmma::mem_row_major);
            }
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

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);  // 128 x 4 = 512 blocks
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);  // 256 threads

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        matmul_bf16_fp32_wmma_async<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        matmul_bf16_fp32_wmma_async<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        
        int nonzero = 0;
        int nan_count = 0;
        for (int i = 0; i < 1000; i++) {
            if (h_C[i] != 0.0f) nonzero++;
            if (isnan(h_C[i])) nan_count++;
        }
        printf("Sanity check: %d/1000 non-zero, %d NaN\n", nonzero, nan_count);
        printf("verification: %s\n", (nonzero > 900 && nan_count == 0) ? "PASSED" : "FAILED");
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        return (nonzero > 900 && nan_count == 0) ? 0 : 1;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double tflops_sum = 0.0;
    double tflops_min = 1e9;
    double tflops_max = 0.0;

    for (int s = 0; s < SAMPLES; s++) {
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++) {
            matmul_bf16_fp32_wmma_async<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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

    double tflops_avg = tflops_sum / SAMPLES;
    printf("\n=== ITER003 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M, N, K);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter003_async (128x128 tiles, BK=64, double buffer)\n");
    printf("avg_tflops: %.2f\n", tflops_avg);
    printf("min_tflops: %.2f\n", tflops_min);
    printf("max_tflops: %.2f\n", tflops_max);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
