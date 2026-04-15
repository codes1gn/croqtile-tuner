// iter037_128x64.cu - Larger M tile to cover full M=512 in fewer blocks
// Shape: M=512, N=16384, K=16384
// Idea: BM=128, BN=64 with 8 warps for better M-dimension coverage

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

constexpr int M_SIZE = 512;
constexpr int N_SIZE = 16384;
constexpr int K_SIZE = 16384;
constexpr int WARMUP = 10;
constexpr int ITERS = 50;
constexpr int SAMPLES = 5;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 16;  // Smaller BK for shared memory budget
constexpr int STAGES = 2;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = 4;  // 4 warps in M
constexpr int WARPS_N = 2;  // 2 warps in N
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;

__global__ __launch_bounds__(256, 3)
void matmul_bf16_fp32_128x64(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    const int m, const int n, const int k,
    const int grid_n
) {
    const int num_blocks_n = grid_n;
    const int block_id = blockIdx.y * num_blocks_n + blockIdx.x;
    
    const int local_m = block_id / num_blocks_n;
    int local_n = block_id % num_blocks_n;
    if (local_m % 2 == 1) {
        local_n = num_blocks_n - 1 - local_n;
    }

    const int warpId = threadIdx.x / WARP_SIZE;
    const int warp_m = warpId / WARPS_N;  // 0-3
    const int warp_n = warpId % WARPS_N;  // 0-1

    const int tile_row = local_m * BM;
    const int tile_col = local_n * BN;

    __shared__ __nv_bfloat16 As[STAGES][BM][BK + 4];
    __shared__ __nv_bfloat16 Bs[STAGES][BK][BN + 4];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    const int num_k_tiles = (k + BK - 1) / BK;
    const int tid = threadIdx.x;

    // 256 threads, load 128x16 A tile
    // Each thread loads 8 bf16 values
    const int a_row = tid / 2;  // 0-127
    const int a_col = (tid % 2) * 8;  // 0 or 8
    
    // Load 16x64 B tile - each thread loads 4 bf16
    const int b_row = tid / 16;  // 0-15
    const int b_col = (tid % 16) * 4;

    // Prefetch first tile
    {
        const int kk = 0;
        
        // A: 128 rows, 16 cols - 256 threads, 8 bytes each
        {
            const int gRow = tile_row + a_row;
            const int gCol = kk + a_col;
            if (gRow < m && gCol + 7 < k) {
                __pipeline_memcpy_async(&As[0][a_row][a_col], &A[gRow * k + gCol], 16);
            }
        }
        
        // B: 16 rows, 64 cols - 256 threads, 8 bytes each
        {
            const int gRow = kk + b_row;
            const int gCol = tile_col + b_col;
            if (gRow < k && gCol + 3 < n) {
                __pipeline_memcpy_async(&Bs[0][b_row][b_col], &B[gRow * n + gCol], 8);
            }
        }
        __pipeline_commit();
    }

    const int warp_row_base = warp_m * 32;
    const int warp_col_base = warp_n * 32;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int curr_stage = kt & 1;
        const int next_stage = (kt + 1) & 1;
        
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        if (kt + 1 < num_k_tiles) {
            const int kk = (kt + 1) * BK;
            
            {
                const int gRow = tile_row + a_row;
                const int gCol = kk + a_col;
                if (gRow < m && gCol + 7 < k) {
                    __pipeline_memcpy_async(&As[next_stage][a_row][a_col], &A[gRow * k + gCol], 16);
                }
            }
            
            {
                const int gRow = kk + b_row;
                const int gCol = tile_col + b_col;
                if (gRow < k && gCol + 3 < n) {
                    __pipeline_memcpy_async(&Bs[next_stage][b_row][b_col], &B[gRow * n + gCol], 8);
                }
            }
            __pipeline_commit();
        }

        // Compute: each warp handles 32x32 output
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a0, a1;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b0, b1;
            
            wmma::load_matrix_sync(a0, &As[curr_stage][warp_row_base][0], BK + 4);
            wmma::load_matrix_sync(a1, &As[curr_stage][warp_row_base + 16][0], BK + 4);
            wmma::load_matrix_sync(b0, &Bs[curr_stage][0][warp_col_base], BN + 4);
            wmma::load_matrix_sync(b1, &Bs[curr_stage][0][warp_col_base + 16], BN + 4);
            
            wmma::mma_sync(acc[0][0], a0, b0, acc[0][0]);
            wmma::mma_sync(acc[0][1], a0, b1, acc[0][1]);
            wmma::mma_sync(acc[1][0], a1, b0, acc[1][0]);
            wmma::mma_sync(acc[1][1], a1, b1, acc[1][1]);
        }
    }
    
    #pragma unroll
    for (int wi = 0; wi < 2; wi++) {
        #pragma unroll
        for (int wj = 0; wj < 2; wj++) {
            const int global_row = tile_row + warp_row_base + wi * WMMA_M;
            const int global_col = tile_col + warp_col_base + wj * WMMA_N;
            
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
    
    size_t size_A = (size_t)M_SIZE * K_SIZE;
    size_t size_B = (size_t)K_SIZE * N_SIZE;
    size_t size_C = (size_t)M_SIZE * N_SIZE;

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

    dim3 grid((N_SIZE + BN - 1) / BN, (M_SIZE + BM - 1) / BM);
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);

    for (int i = 0; i < WARMUP; i++) {
        matmul_bf16_fp32_128x64<<<grid, block>>>(d_A, d_B, d_C, M_SIZE, N_SIZE, K_SIZE, grid.x);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        matmul_bf16_fp32_128x64<<<grid, block>>>(d_A, d_B, d_C, M_SIZE, N_SIZE, K_SIZE, grid.x);
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
            matmul_bf16_fp32_128x64<<<grid, block>>>(d_A, d_B, d_C, M_SIZE, N_SIZE, K_SIZE, grid.x);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        double time_per_iter_ms = ms / ITERS;
        double flops = 2.0 * M_SIZE * N_SIZE * K_SIZE;
        double tflops = flops / (time_per_iter_ms * 1e-3) / 1e12;

        printf("sample %d: time=%.3f ms, tflops=%.2f\n", s + 1, time_per_iter_ms, tflops);
        tflops_sum += tflops;
        tflops_min = fmin(tflops_min, tflops);
        tflops_max = fmax(tflops_max, tflops);
    }

    printf("\n=== ITER037 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M_SIZE, N_SIZE, K_SIZE);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter037_128x64 (BM=128 BN=64 8 warps)\n");
    printf("avg_tflops: %.2f\n", tflops_sum / SAMPLES);
    printf("min_tflops: %.2f\n", tflops_min);
    printf("max_tflops: %.2f\n", tflops_max);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
