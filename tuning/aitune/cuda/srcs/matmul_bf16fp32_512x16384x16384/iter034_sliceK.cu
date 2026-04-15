// iter034_sliceK.cu - Warp-level sliced K for better parallelism
// Shape: M=512, N=16384, K=16384
// Idea: Each warp processes different K slices, reduce at end

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

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int STAGES = 2;

// 8 warps: 2 for spatial, 4 for K slicing
constexpr int WARP_SIZE = 32;
constexpr int WARPS_SPATIAL = 4;  // 2x2 spatial
constexpr int WARPS_K = 2;  // K slicing factor
constexpr int WARPS_PER_BLOCK = WARPS_SPATIAL * WARPS_K;

__global__ __launch_bounds__(256, 2)
void matmul_bf16_fp32_sliceK(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int m, int n, int k,
    int grid_n
) {
    int num_blocks_n = grid_n;
    int num_blocks_m = (M + BM - 1) / BM;
    int block_id = blockIdx.y * num_blocks_n + blockIdx.x;
    
    int within_group = block_id % (num_blocks_m * num_blocks_n);
    int local_m = within_group / num_blocks_n;
    int local_n = within_group % num_blocks_n;
    
    if (local_m % 2 == 1) {
        local_n = num_blocks_n - 1 - local_n;
    }
    
    int by = local_m;
    int bx = local_n;

    int warpId = threadIdx.x / WARP_SIZE;
    int k_slice = warpId / WARPS_SPATIAL;  // 0 or 1
    int spatial_warp = warpId % WARPS_SPATIAL;
    int warp_m = spatial_warp / 2;
    int warp_n = spatial_warp % 2;

    int tile_row = by * BM;
    int tile_col = bx * BN;

    __shared__ __nv_bfloat16 As[STAGES][BM][BK + 8];
    __shared__ __nv_bfloat16 Bs[STAGES][BK][BN + 8];
    __shared__ float reduction[WARPS_K][BM][BN];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    int num_k_tiles = (k + BK - 1) / BK;
    int tiles_per_slice = (num_k_tiles + WARPS_K - 1) / WARPS_K;
    int k_start = k_slice * tiles_per_slice;
    int k_end = min(k_start + tiles_per_slice, num_k_tiles);
    
    int tid = threadIdx.x;

    for (int kt = k_start; kt < k_end; kt++) {
        int kk = kt * BK;
        
        // All threads load cooperatively
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * 256;
            if (idx < BM * (BK / 4)) {
                int row = idx / (BK / 4);
                int col4 = idx % (BK / 4);
                int col = col4 * 4;
                int gRow = tile_row + row;
                int gCol = kk + col;
                
                if (gRow < m && gCol + 3 < k) {
                    __pipeline_memcpy_async(&As[kt % STAGES][row][col], &A[gRow * k + gCol], 8);
                }
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * 256;
            if (idx < BK * (BN / 4)) {
                int row = idx / (BN / 4);
                int col4 = idx % (BN / 4);
                int col = col4 * 4;
                int gRow = kk + row;
                int gCol = tile_col + col;
                
                if (gRow < k && gCol + 3 < n) {
                    __pipeline_memcpy_async(&Bs[kt % STAGES][row][col], &B[gRow * n + gCol], 8);
                }
            }
        }
        
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        int warp_row_base = warp_m * 32;
        int warp_col_base = warp_n * 32;
        
        #pragma unroll
        for (int wk = 0; wk < BK; wk += WMMA_K) {
            #pragma unroll
            for (int wi = 0; wi < 2; wi++) {
                #pragma unroll
                for (int wj = 0; wj < 2; wj++) {
                    int a_row = warp_row_base + wi * WMMA_M;
                    int b_col = warp_col_base + wj * WMMA_N;
                    
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                    
                    wmma::load_matrix_sync(a_frag, &As[kt % STAGES][a_row][wk], BK + 8);
                    wmma::load_matrix_sync(b_frag, &Bs[kt % STAGES][wk][b_col], BN + 8);
                    wmma::mma_sync(acc[wi][wj], a_frag, b_frag, acc[wi][wj]);
                }
            }
        }
        
        __syncthreads();
    }

    // Store partial results to shared memory
    int warp_row_base = warp_m * 32;
    int warp_col_base = warp_n * 32;
    
    #pragma unroll
    for (int wi = 0; wi < 2; wi++) {
        #pragma unroll
        for (int wj = 0; wj < 2; wj++) {
            int local_row = warp_row_base + wi * WMMA_M;
            int local_col = warp_col_base + wj * WMMA_N;
            wmma::store_matrix_sync(&reduction[k_slice][local_row][local_col], acc[wi][wj], BN, wmma::mem_row_major);
        }
    }
    
    __syncthreads();
    
    // Reduce and store to global
    // Only first WARPS_SPATIAL warps do reduction
    if (warpId < WARPS_SPATIAL) {
        int warp_row = (spatial_warp / 2) * 32;
        int warp_col = (spatial_warp % 2) * 32;
        
        for (int wi = 0; wi < 2; wi++) {
            for (int wj = 0; wj < 2; wj++) {
                int lane = threadIdx.x % 32;
                int frag_row = wi * WMMA_M + lane / 8;
                int frag_col = wj * WMMA_N + (lane % 8) * 2;
                
                int local_row = warp_row + frag_row;
                int local_col = warp_col + frag_col;
                
                float sum = 0.0f;
                #pragma unroll
                for (int s = 0; s < WARPS_K; s++) {
                    sum += reduction[s][local_row][local_col];
                    sum += reduction[s][local_row][local_col + 1];
                }
                
                int global_row = tile_row + local_row;
                int global_col = tile_col + local_col;
                
                if (global_row < m && global_col < n) {
                    C[global_row * n + global_col] = sum / 2;
                }
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

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);

    for (int i = 0; i < WARMUP; i++) {
        matmul_bf16_fp32_sliceK<<<grid, block>>>(d_A, d_B, d_C, M, N, K, grid.x);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        matmul_bf16_fp32_sliceK<<<grid, block>>>(d_A, d_B, d_C, M, N, K, grid.x);
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
            matmul_bf16_fp32_sliceK<<<grid, block>>>(d_A, d_B, d_C, M, N, K, grid.x);
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

    printf("\n=== ITER034 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M, N, K);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter034_sliceK (warp-level K slicing)\n");
    printf("avg_tflops: %.2f\n", tflops_sum / SAMPLES);
    printf("min_tflops: %.2f\n", tflops_min);
    printf("max_tflops: %.2f\n", tflops_max);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
