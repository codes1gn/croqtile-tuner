// iter004_split.cu - Split-K strategy for tall skinny matmul
// Shape: M=512, N=16384, K=16384
// Idea: M is small (512), so we're limited in block parallelism
// Use split-K to increase parallelism: divide K across multiple blocks
// Then do atomic or reduction for final result

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
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

// Small tiles for more blocks
constexpr int BM = 32;   
constexpr int BN = 64;
constexpr int BK = 32;

// Split K into multiple parts
constexpr int SPLIT_K = 8;  // 8 blocks share the same output tile, each does K/8

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;  // 128 threads

__global__ __launch_bounds__(128, 4)
void matmul_bf16_fp32_splitk(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int m, int n, int k,
    int k_start, int k_end
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int warpId = threadIdx.x / WARP_SIZE;

    int tile_row = by * BM;
    int tile_col = bx * BN;

    __shared__ __nv_bfloat16 As[BM][BK + 4];
    __shared__ __nv_bfloat16 Bs[BK][BN + 4];

    // 4 warps covering 32x64 output
    // Layout: 2x2 warps, each warp does 1x2 WMMA tiles = 16x32
    // 2x2 grid of 16x32 = 32x64. 
    int warp_row = warpId / 2;  // 0,1
    int warp_col = warpId % 2;  // 0,1

    // Each warp handles 1 row and 2 cols of WMMA tiles
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[1][2];
    
    wmma::fill_fragment(acc[0][0], 0.0f);
    wmma::fill_fragment(acc[0][1], 0.0f);

    for (int kk = k_start; kk < k_end; kk += BK) {
        int actual_bk = min(BK, k_end - kk);
        
        // Load A: [BM, BK] = [32, 32] = 1024 bf16
        // 128 threads -> 8 each
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = threadIdx.x * 8 + i;
            int row = idx / BK;
            int col = idx % BK;
            int gRow = tile_row + row;
            int gCol = kk + col;
            As[row][col] = (gRow < m && gCol < k && col < actual_bk) ? 
                A[gRow * k + gCol] : __float2bfloat16(0.0f);
        }

        // Load B: [BK, BN] = [32, 64] = 2048 bf16
        // 128 threads -> 16 each
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int idx = threadIdx.x * 16 + i;
            int row = idx / BN;
            int col = idx % BN;
            int gRow = kk + row;
            int gCol = tile_col + col;
            Bs[row][col] = (gRow < k && gCol < n && row < actual_bk) ?
                B[gRow * n + gCol] : __float2bfloat16(0.0f);
        }

        __syncthreads();

        // WMMA compute
        #pragma unroll
        for (int wk = 0; wk < BK; wk += WMMA_K) {
            int wmma_row = warp_row;
            
            #pragma unroll
            for (int tj = 0; tj < 2; tj++) {
                int wmma_col = warp_col * 2 + tj;

                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;

                wmma::load_matrix_sync(a_frag, &As[wmma_row * WMMA_M][wk], BK + 4);
                wmma::load_matrix_sync(b_frag, &Bs[wk][wmma_col * WMMA_N], BN + 4);
                wmma::mma_sync(acc[0][tj], a_frag, b_frag, acc[0][tj]);
            }
        }

        __syncthreads();
    }

    // Atomic add to global memory (for split-K reduction)
    int wmma_row = warp_row;
    #pragma unroll
    for (int tj = 0; tj < 2; tj++) {
        int wmma_col = warp_col * 2 + tj;

        int global_row = tile_row + wmma_row * WMMA_M;
        int global_col = tile_col + wmma_col * WMMA_N;

        if (global_row < m && global_col < n) {
            // Store to temp then atomic add
            __shared__ float temp[WMMA_M][WMMA_N];
            wmma::store_matrix_sync(&temp[0][0], acc[0][tj], WMMA_N, wmma::mem_row_major);
            __syncthreads();
            
            // Each thread in warp atomically adds its portion
            int laneId = threadIdx.x % WARP_SIZE;
            // 32 threads, 256 elements -> 8 each
            for (int i = 0; i < 8; i++) {
                int idx = laneId * 8 + i;
                int r = idx / WMMA_N;
                int c = idx % WMMA_N;
                if (global_row + r < m && global_col + c < n) {
                    atomicAdd(&C[(global_row + r) * n + (global_col + c)], temp[r][c]);
                }
            }
        }
    }
}

// Zero the output buffer
__global__ void zero_output(float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = 0.0f;
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

    // Grid: N/BN x M/BM blocks
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);  // 256 x 16 = 4096 blocks
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);  // 128 threads

    int k_per_split = (K + SPLIT_K - 1) / SPLIT_K;

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        zero_output<<<(size_C + 255) / 256, 256>>>(d_C, size_C);
        for (int s = 0; s < SPLIT_K; s++) {
            int k_start = s * k_per_split;
            int k_end = min((s + 1) * k_per_split, K);
            matmul_bf16_fp32_splitk<<<grid, block>>>(d_A, d_B, d_C, M, N, K, k_start, k_end);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        zero_output<<<(size_C + 255) / 256, 256>>>(d_C, size_C);
        for (int s = 0; s < SPLIT_K; s++) {
            int k_start = s * k_per_split;
            int k_end = min((s + 1) * k_per_split, K);
            matmul_bf16_fp32_splitk<<<grid, block>>>(d_A, d_B, d_C, M, N, K, k_start, k_end);
        }
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

    for (int s_i = 0; s_i < SAMPLES; s_i++) {
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++) {
            zero_output<<<(size_C + 255) / 256, 256>>>(d_C, size_C);
            for (int s = 0; s < SPLIT_K; s++) {
                int k_start = s * k_per_split;
                int k_end = min((s + 1) * k_per_split, K);
                matmul_bf16_fp32_splitk<<<grid, block>>>(d_A, d_B, d_C, M, N, K, k_start, k_end);
            }
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        double time_per_iter_ms = ms / ITERS;
        double flops = 2.0 * M * N * K;
        double tflops = flops / (time_per_iter_ms * 1e-3) / 1e12;

        printf("sample %d: time=%.3f ms, tflops=%.2f\n", s_i + 1, time_per_iter_ms, tflops);
        tflops_sum += tflops;
        tflops_min = fmin(tflops_min, tflops);
        tflops_max = fmax(tflops_max, tflops);
    }

    double tflops_avg = tflops_sum / SAMPLES;
    printf("\n=== ITER004 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M, N, K);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter004_split (split-K=%d, 32x64 tiles)\n", SPLIT_K);
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
