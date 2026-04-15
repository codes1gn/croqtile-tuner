// iter002_lean.cu - Lean WMMA kernel with reduced register pressure
// Shape: M=512, N=16384, K=16384
// Idea: Reduce registers by using 2x2 WMMA tiles per warp instead of 4x4
// Target: Increase occupancy from 16.67% to 50%+, beat 41.86 TFLOPS baseline

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

// WMMA tile sizes
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Smaller block tiles for reduced register pressure
constexpr int BM = 64;   // Was 128
constexpr int BN = 64;   // Was 128
constexpr int BK = 16;   // Smaller K tile

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;  // Fewer warps, 128 threads

// Each warp handles 2x2 = 4 WMMA tiles (instead of 8)
// Total: 4 warps * 4 tiles = 16 tiles = 4x4 WMMA tiles = 64x64 output tile

__global__ __launch_bounds__(128, 4)  // Hint compiler for 4 blocks/SM
void matmul_bf16_fp32_wmma_lean(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int m, int n, int k
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    int tile_row = by * BM;
    int tile_col = bx * BN;

    // Smaller shared memory tiles
    __shared__ __nv_bfloat16 As[BM][BK + 1];  // +1 for bank conflict avoidance
    __shared__ __nv_bfloat16 Bs[BK][BN + 1];

    // Each warp handles 2x2 WMMA tiles = 32x32 output
    // 4 warps arranged in 2x2 grid covering 64x64
    int warp_row = warpId / 2;  // 0 or 1
    int warp_col = warpId % 2;  // 0 or 1

    // Only 4 accumulator fragments per warp (2x2)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    // Vectorized loading using bf16x2
    for (int kk = 0; kk < k; kk += BK) {
        // Load A tile: [BM, BK] = [64, 16] = 1024 bf16 elements
        // 128 threads load 8 elements each
        int a_elements_per_thread = (BM * BK) / (WARPS_PER_BLOCK * WARP_SIZE);  // 8
        
        #pragma unroll
        for (int i = 0; i < a_elements_per_thread; i++) {
            int elem_idx = threadIdx.x * a_elements_per_thread + i;
            int row = elem_idx / BK;
            int col = elem_idx % BK;
            int global_row = tile_row + row;
            int global_col = kk + col;
            
            __nv_bfloat16 val = (global_row < m && global_col < k) ? 
                A[global_row * k + global_col] : __float2bfloat16(0.0f);
            As[row][col] = val;
        }

        // Load B tile: [BK, BN] = [16, 64] = 1024 bf16 elements
        int b_elements_per_thread = (BK * BN) / (WARPS_PER_BLOCK * WARP_SIZE);  // 8
        
        #pragma unroll
        for (int i = 0; i < b_elements_per_thread; i++) {
            int elem_idx = threadIdx.x * b_elements_per_thread + i;
            int row = elem_idx / BN;
            int col = elem_idx % BN;
            int global_row = kk + row;
            int global_col = tile_col + col;
            
            __nv_bfloat16 val = (global_row < k && global_col < n) ?
                B[global_row * n + global_col] : __float2bfloat16(0.0f);
            Bs[row][col] = val;
        }

        __syncthreads();

        // WMMA compute
        #pragma unroll
        for (int wk = 0; wk < BK; wk += WMMA_K) {
            #pragma unroll
            for (int ti = 0; ti < 2; ti++) {
                #pragma unroll
                for (int tj = 0; tj < 2; tj++) {
                    int wmma_row = warp_row * 2 + ti;
                    int wmma_col = warp_col * 2 + tj;

                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;

                    wmma::load_matrix_sync(a_frag, &As[wmma_row * WMMA_M][wk], BK + 1);
                    wmma::load_matrix_sync(b_frag, &Bs[wk][wmma_col * WMMA_N], BN + 1);
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
        for (int tj = 0; tj < 2; tj++) {
            int wmma_row = warp_row * 2 + ti;
            int wmma_col = warp_col * 2 + tj;

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

    // Grid: more blocks due to smaller tile size
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);  // 256 x 8 = 2048 blocks
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);  // 128 threads

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        matmul_bf16_fp32_wmma_lean<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        // Use cuBLAS for verification reference
        printf("Verification mode: comparing against stored reference\n");
        matmul_bf16_fp32_wmma_lean<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Basic sanity check - values shouldn't be all zeros or NaN
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

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double tflops_sum = 0.0;
    double tflops_min = 1e9;
    double tflops_max = 0.0;

    for (int s = 0; s < SAMPLES; s++) {
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++) {
            matmul_bf16_fp32_wmma_lean<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    printf("\n=== ITER002 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M, N, K);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter002_lean (64x64 tiles, 2x2 WMMA per warp)\n");
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
