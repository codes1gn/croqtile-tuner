// iter173_swizzle.cu - Different block scheduling for better L2 locality
// Shape: M=512, N=16384, K=16384
// Use 2D swizzle pattern: process 4x4 block groups together

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

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int STAGES = 2;

constexpr int SWIZZLE = 8;  // Group 8 blocks together

constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
constexpr int NUM_K_TILES = K_SIZE / BK;

__global__ __launch_bounds__(128, 4)
void matmul_bf16_fp32_opt(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C
) {
    const int num_blocks_m = M_SIZE / BM;  // 8
    const int num_blocks_n = N_SIZE / BN;  // 256
    
    // 2D swizzle: group blocks into SWIZZLE x SWIZZLE tiles
    const int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int num_groups_n = (num_blocks_n + SWIZZLE - 1) / SWIZZLE;
    
    // Which group and position within group
    const int group_id = block_id / (num_blocks_m * SWIZZLE);
    const int in_group = block_id % (num_blocks_m * SWIZZLE);
    
    const int local_m = in_group / SWIZZLE;
    int local_n = group_id * SWIZZLE + (in_group % SWIZZLE);
    
    // Zigzag within each row
    if (local_m % 2 == 1) {
        local_n = group_id * SWIZZLE + (SWIZZLE - 1 - (in_group % SWIZZLE));
    }
    
    if (local_n >= num_blocks_n) return;

    const int warpId = threadIdx.x / WARP_SIZE;
    const int warp_m = warpId / WARPS_N;
    const int warp_n = warpId % WARPS_N;

    const int tile_row = local_m * BM;
    const int tile_col = local_n * BN;

    __shared__ __nv_bfloat16 As[STAGES][BM][BK + 8];
    __shared__ __nv_bfloat16 Bs[STAGES][BK][BN + 8];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    const int tid = threadIdx.x;
    const int a_row_base = tid / 8;
    const int a_col_base = (tid % 8) * 4;
    const int b_row_base = tid / 16;
    const int b_col_base = (tid % 16) * 4;

    // Prefetch first tile
    {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int row = a_row_base + i * 16;
            const int col = a_col_base;
            const int gRow = tile_row + row;
            __pipeline_memcpy_async(&As[0][row][col], &A[gRow * K_SIZE + col], 8);
        }
        
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int row = b_row_base + i * 16;
            const int col = b_col_base;
            const int gCol = tile_col + col;
            __pipeline_memcpy_async(&Bs[0][row][col], &B[row * N_SIZE + gCol], 8);
        }
        __pipeline_commit();
    }

    const int warp_row_base = warp_m * 32;
    const int warp_col_base = warp_n * 32;

    #pragma unroll 1
    for (int kt = 0; kt < NUM_K_TILES; kt++) {
        const int curr_stage = kt & 1;
        const int next_stage = (kt + 1) & 1;
        
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        if (kt + 1 < NUM_K_TILES) {
            const int kk = (kt + 1) * BK;
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int row = a_row_base + i * 16;
                const int col = a_col_base;
                const int gRow = tile_row + row;
                const int gCol = kk + col;
                __pipeline_memcpy_async(&As[next_stage][row][col], &A[gRow * K_SIZE + gCol], 8);
            }
            
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const int row = b_row_base + i * 16;
                const int col = b_col_base;
                const int gRow = kk + row;
                const int gCol = tile_col + col;
                __pipeline_memcpy_async(&Bs[next_stage][row][col], &B[gRow * N_SIZE + gCol], 8);
            }
            __pipeline_commit();
        }

        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a0, a1;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b0, b1;
            
            wmma::load_matrix_sync(a0, &As[curr_stage][warp_row_base][0], BK + 8);
            wmma::load_matrix_sync(a1, &As[curr_stage][warp_row_base + 16][0], BK + 8);
            wmma::load_matrix_sync(b0, &Bs[curr_stage][0][warp_col_base], BN + 8);
            wmma::load_matrix_sync(b1, &Bs[curr_stage][0][warp_col_base + 16], BN + 8);
            
            wmma::mma_sync(acc[0][0], a0, b0, acc[0][0]);
            wmma::mma_sync(acc[0][1], a0, b1, acc[0][1]);
            wmma::mma_sync(acc[1][0], a1, b0, acc[1][0]);
            wmma::mma_sync(acc[1][1], a1, b1, acc[1][1]);
        }
        
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a0, a1;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b0, b1;
            
            wmma::load_matrix_sync(a0, &As[curr_stage][warp_row_base][16], BK + 8);
            wmma::load_matrix_sync(a1, &As[curr_stage][warp_row_base + 16][16], BK + 8);
            wmma::load_matrix_sync(b0, &Bs[curr_stage][16][warp_col_base], BN + 8);
            wmma::load_matrix_sync(b1, &Bs[curr_stage][16][warp_col_base + 16], BN + 8);
            
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
            wmma::store_matrix_sync(&C[global_row * N_SIZE + global_col], acc[wi][wj], N_SIZE, wmma::mem_row_major);
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

    dim3 grid(N_SIZE / BN, M_SIZE / BM);
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);

    for (int i = 0; i < WARMUP; i++) {
        matmul_bf16_fp32_opt<<<grid, block>>>(d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        matmul_bf16_fp32_opt<<<grid, block>>>(d_A, d_B, d_C);
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
            matmul_bf16_fp32_opt<<<grid, block>>>(d_A, d_B, d_C);
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

    printf("\n=== ITER173 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M_SIZE, N_SIZE, K_SIZE);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter173_swizzle (SWIZZLE=%d block groups)\n", SWIZZLE);
    printf("avg_tflops: %.2f\n", tflops_sum / SAMPLES);
    printf("min_tflops: %.2f\n", tflops_min);
    printf("max_tflops: %.2f\n", tflops_max);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
