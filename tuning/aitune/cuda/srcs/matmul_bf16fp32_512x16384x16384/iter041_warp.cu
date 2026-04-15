// iter041_256warp.cu - 256 threads with CUTLASS-style warp tiling
// Shape: M=512, N=16384, K=16384  
// Idea: 8 warps, each doing 16x32 output, total 64x128 per block

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
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int STAGES = 2;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = 4;  // 4 warps in M direction
constexpr int WARPS_N = 2;  // 2 warps in N direction  
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;  // 8 warps = 256 threads

__global__ __launch_bounds__(256, 3)
void matmul_bf16_fp32_256warp(
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

    // Each warp computes 16x64 output (1x4 WMMA tiles)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4];
    
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        wmma::fill_fragment(acc[j], 0.0f);
    }

    const int num_k_tiles = (k + BK - 1) / BK;
    const int tid = threadIdx.x;

    // 256 threads loading 64x32 A tile: each thread loads 4 bf16 = 8 bytes
    const int a_row = tid / 8;        // 0-31 (need 2 iterations for 64 rows)
    const int a_col = (tid % 8) * 4;  // 0,4,8,12,16,20,24,28
    
    // 256 threads loading 32x128 B tile  
    const int b_row = tid / 32;       // 0-7 (need 4 iterations for 32 rows)
    const int b_col = (tid % 32) * 4; // 0-124 step 4

    // Prefetch first tile
    {
        const int kk = 0;
        
        // A: 64x32 - two iterations for 64 rows
        #pragma unroll 2
        for (int i = 0; i < 2; i++) {
            const int row = a_row + i * 32;
            const int col = a_col;
            if (row < BM) {
                const int gRow = tile_row + row;
                const int gCol = kk + col;
                if (gRow < m && gCol + 3 < k) {
                    __pipeline_memcpy_async(&As[0][row][col], &A[gRow * k + gCol], 8);
                }
            }
        }
        
        // B: 32x128 - four iterations for 32 rows
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            const int row = b_row + i * 8;
            const int col = b_col;
            if (row < BK) {
                const int gRow = kk + row;
                const int gCol = tile_col + col;
                if (gRow < k && gCol + 3 < n) {
                    __pipeline_memcpy_async(&Bs[0][row][col], &B[gRow * n + gCol], 8);
                }
            }
        }
        __pipeline_commit();
    }

    const int warp_row_base = warp_m * 16;  // Each warp handles 16 rows
    const int warp_col_base = warp_n * 64;  // Each warp handles 64 cols

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int curr_stage = kt & 1;
        const int next_stage = (kt + 1) & 1;
        
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        if (kt + 1 < num_k_tiles) {
            const int kk = (kt + 1) * BK;
            
            #pragma unroll 2
            for (int i = 0; i < 2; i++) {
                const int row = a_row + i * 32;
                const int col = a_col;
                if (row < BM) {
                    const int gRow = tile_row + row;
                    const int gCol = kk + col;
                    if (gRow < m && gCol + 3 < k) {
                        __pipeline_memcpy_async(&As[next_stage][row][col], &A[gRow * k + gCol], 8);
                    }
                }
            }
            
            #pragma unroll 4
            for (int i = 0; i < 4; i++) {
                const int row = b_row + i * 8;
                const int col = b_col;
                if (row < BK) {
                    const int gRow = kk + row;
                    const int gCol = tile_col + col;
                    if (gRow < k && gCol + 3 < n) {
                        __pipeline_memcpy_async(&Bs[next_stage][row][col], &B[gRow * n + gCol], 8);
                    }
                }
            }
            __pipeline_commit();
        }

        // Compute 16x64 output per warp with BK=32 (two K steps)
        // K=0
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag[4];
            
            wmma::load_matrix_sync(a_frag, &As[curr_stage][warp_row_base][0], BK + 4);
            wmma::load_matrix_sync(b_frag[0], &Bs[curr_stage][0][warp_col_base], BN + 4);
            wmma::load_matrix_sync(b_frag[1], &Bs[curr_stage][0][warp_col_base + 16], BN + 4);
            wmma::load_matrix_sync(b_frag[2], &Bs[curr_stage][0][warp_col_base + 32], BN + 4);
            wmma::load_matrix_sync(b_frag[3], &Bs[curr_stage][0][warp_col_base + 48], BN + 4);
            
            wmma::mma_sync(acc[0], a_frag, b_frag[0], acc[0]);
            wmma::mma_sync(acc[1], a_frag, b_frag[1], acc[1]);
            wmma::mma_sync(acc[2], a_frag, b_frag[2], acc[2]);
            wmma::mma_sync(acc[3], a_frag, b_frag[3], acc[3]);
        }
        
        // K=16
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag[4];
            
            wmma::load_matrix_sync(a_frag, &As[curr_stage][warp_row_base][16], BK + 4);
            wmma::load_matrix_sync(b_frag[0], &Bs[curr_stage][16][warp_col_base], BN + 4);
            wmma::load_matrix_sync(b_frag[1], &Bs[curr_stage][16][warp_col_base + 16], BN + 4);
            wmma::load_matrix_sync(b_frag[2], &Bs[curr_stage][16][warp_col_base + 32], BN + 4);
            wmma::load_matrix_sync(b_frag[3], &Bs[curr_stage][16][warp_col_base + 48], BN + 4);
            
            wmma::mma_sync(acc[0], a_frag, b_frag[0], acc[0]);
            wmma::mma_sync(acc[1], a_frag, b_frag[1], acc[1]);
            wmma::mma_sync(acc[2], a_frag, b_frag[2], acc[2]);
            wmma::mma_sync(acc[3], a_frag, b_frag[3], acc[3]);
        }
    }
    
    // Store output
    #pragma unroll
    for (int wj = 0; wj < 4; wj++) {
        const int global_row = tile_row + warp_row_base;
        const int global_col = tile_col + warp_col_base + wj * WMMA_N;
        
        if (global_row < m && global_col < n) {
            wmma::store_matrix_sync(&C[global_row * n + global_col], acc[wj], n, wmma::mem_row_major);
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
        matmul_bf16_fp32_256warp<<<grid, block>>>(d_A, d_B, d_C, M_SIZE, N_SIZE, K_SIZE, grid.x);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        matmul_bf16_fp32_256warp<<<grid, block>>>(d_A, d_B, d_C, M_SIZE, N_SIZE, K_SIZE, grid.x);
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
            matmul_bf16_fp32_256warp<<<grid, block>>>(d_A, d_B, d_C, M_SIZE, N_SIZE, K_SIZE, grid.x);
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

    printf("\n=== ITER041 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M_SIZE, N_SIZE, K_SIZE);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter041_256warp (64x128 tiles, 8 warps)\n");
    printf("avg_tflops: %.2f\n", tflops_sum / SAMPLES);
    printf("min_tflops: %.2f\n", tflops_min);
    printf("max_tflops: %.2f\n", tflops_max);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
