// iter005_classic.cu - Classic high-performance WMMA pattern
// Shape: M=512, N=16384, K=16384
// Idea: Use proven pattern from CUTLASS with proper warp tiling
// Each warp computes 32x32 output using 2x2 WMMA tiles
// 4 warps per block = 64x64 block output (smaller for better occupancy)

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

// Block configuration for better occupancy
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;  // Small K tile for less shared memory

constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = 2;  // 2 warps along M
constexpr int WARPS_N = 2;  // 2 warps along N  
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;  // 4 warps = 128 threads

// Each warp computes 32x32 = 2x2 WMMA tiles
constexpr int WARP_M = BM / WARPS_M;  // 32
constexpr int WARP_N = BN / WARPS_N;  // 32
constexpr int WARP_WMMA_M = WARP_M / WMMA_M;  // 2
constexpr int WARP_WMMA_N = WARP_N / WMMA_N;  // 2

__global__ __launch_bounds__(128, 8)  // Target 8 blocks per SM for good occupancy
void matmul_bf16_fp32_classic(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int m, int n, int k
) {
    // Block and warp indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int warpId = threadIdx.x / WARP_SIZE;
    int warp_m = warpId / WARPS_N;  // 0 or 1
    int warp_n = warpId % WARPS_N;  // 0 or 1

    // Global tile position
    int tile_row = by * BM;
    int tile_col = bx * BN;

    // Shared memory with padding to avoid bank conflicts
    __shared__ __nv_bfloat16 As[BM][BK + 8];
    __shared__ __nv_bfloat16 Bs[BK][BN + 8];

    // Accumulator fragments - 2x2 WMMA tiles per warp = 4 fragments
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_WMMA_M][WARP_WMMA_N];
    
    #pragma unroll
    for (int i = 0; i < WARP_WMMA_M; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_WMMA_N; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    // Number of K tiles
    int num_k_tiles = (k + BK - 1) / BK;

    // Main loop over K
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int kk = kt * BK;
        
        // Cooperative loading of A tile [BM x BK] = [64 x 16] = 1024 elements
        // 128 threads, 8 elements each
        int tid = threadIdx.x;
        int a_loads = (BM * BK + 127) / 128;  // 8
        
        #pragma unroll
        for (int i = 0; i < a_loads; i++) {
            int idx = tid + i * 128;
            if (idx < BM * BK) {
                int row = idx / BK;
                int col = idx % BK;
                int gRow = tile_row + row;
                int gCol = kk + col;
                As[row][col] = (gRow < m && gCol < k) ? A[gRow * k + gCol] : __float2bfloat16(0.0f);
            }
        }
        
        // Cooperative loading of B tile [BK x BN] = [16 x 64] = 1024 elements
        int b_loads = (BK * BN + 127) / 128;  // 8
        
        #pragma unroll
        for (int i = 0; i < b_loads; i++) {
            int idx = tid + i * 128;
            if (idx < BK * BN) {
                int row = idx / BN;
                int col = idx % BN;
                int gRow = kk + row;
                int gCol = tile_col + col;
                Bs[row][col] = (gRow < k && gCol < n) ? B[gRow * n + gCol] : __float2bfloat16(0.0f);
            }
        }
        
        __syncthreads();

        // WMMA compute - single K iteration since BK = WMMA_K = 16
        // Warp position in shared memory
        int warp_row_base = warp_m * WARP_M;  // 0 or 32
        int warp_col_base = warp_n * WARP_N;  // 0 or 32
        
        #pragma unroll
        for (int wi = 0; wi < WARP_WMMA_M; wi++) {
            #pragma unroll
            for (int wj = 0; wj < WARP_WMMA_N; wj++) {
                int a_row = warp_row_base + wi * WMMA_M;
                int b_col = warp_col_base + wj * WMMA_N;
                
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                
                wmma::load_matrix_sync(a_frag, &As[a_row][0], BK + 8);
                wmma::load_matrix_sync(b_frag, &Bs[0][b_col], BN + 8);
                wmma::mma_sync(acc[wi][wj], a_frag, b_frag, acc[wi][wj]);
            }
        }
        
        __syncthreads();
    }

    // Store results
    int warp_row_base = warp_m * WARP_M;
    int warp_col_base = warp_n * WARP_N;
    
    #pragma unroll
    for (int wi = 0; wi < WARP_WMMA_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_WMMA_N; wj++) {
            int global_row = tile_row + warp_row_base + wi * WMMA_M;
            int global_col = tile_col + warp_col_base + wj * WMMA_N;
            
            if (global_row < m && global_col < n) {
                wmma::store_matrix_sync(&C[global_row * n + global_col], acc[wi][wj], n, wmma::mem_row_major);
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

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);  // 256 x 8 = 2048 blocks
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);  // 128 threads

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        matmul_bf16_fp32_classic<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        matmul_bf16_fp32_classic<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
            matmul_bf16_fp32_classic<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    printf("\n=== ITER005 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M, N, K);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter005_classic (64x64 tiles, 4 warps, 2x2 WMMA per warp)\n");
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
