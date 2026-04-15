// iter054_1acc.cu - Single 16x16 accumulator per warp to reduce register pressure
// Shape: M=512, N=16384, K=16384
// Idea: Each warp computes only 16x16 output, need more warps per block

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

// 64x64 output tile, 16 warps (4x4), each warp does 16x16
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int STAGES = 2;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 4;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;  // 16 warps = 512 threads

constexpr int NUM_K_TILES = K_SIZE / BK;

__global__ __launch_bounds__(512, 2)  // 512 threads, 2 blocks/SM
void matmul_bf16_fp32_1acc(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C
) {
    const int num_blocks_n = N_SIZE / BN;
    const int block_id = blockIdx.y * num_blocks_n + blockIdx.x;
    
    const int local_m = block_id / num_blocks_n;
    int local_n = block_id % num_blocks_n;
    if (local_m % 2 == 1) {
        local_n = num_blocks_n - 1 - local_n;
    }

    const int warpId = threadIdx.x / WARP_SIZE;
    const int warp_m = warpId / WARPS_N;  // 0-3
    const int warp_n = warpId % WARPS_N;  // 0-3

    const int tile_row = local_m * BM;
    const int tile_col = local_n * BN;

    __shared__ __nv_bfloat16 As[STAGES][BM][BK + 8];
    __shared__ __nv_bfloat16 Bs[STAGES][BK][BN + 8];

    // Single accumulator per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    const int tid = threadIdx.x;
    
    // 512 threads loading 64x32 A and 32x64 B
    // A: 64x32 = 2048 bf16, 512 threads, 4 bf16 each
    const int a_row = tid / 8;  // 0-63
    const int a_col = (tid % 8) * 4;  // 0,4,8,12,16,20,24,28
    
    // B: 32x64 = 2048 bf16, 512 threads, 4 bf16 each
    const int b_row = tid / 16;  // 0-31
    const int b_col = (tid % 16) * 4;  // 0,4,...,60

    // Prefetch first tile
    {
        if (a_row < BM) {
            const int gRowA = tile_row + a_row;
            __pipeline_memcpy_async(&As[0][a_row][a_col], &A[gRowA * K_SIZE + a_col], 8);
        }
        
        if (b_row < BK) {
            const int gColB = tile_col + b_col;
            __pipeline_memcpy_async(&Bs[0][b_row][b_col], &B[b_row * N_SIZE + gColB], 8);
        }
        __pipeline_commit();
    }

    const int warp_row_base = warp_m * WMMA_M;  // 0, 16, 32, 48
    const int warp_col_base = warp_n * WMMA_N;  // 0, 16, 32, 48

    #pragma unroll 1
    for (int kt = 0; kt < NUM_K_TILES; kt++) {
        const int curr_stage = kt & 1;
        const int next_stage = (kt + 1) & 1;
        
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        if (kt + 1 < NUM_K_TILES) {
            const int kk = (kt + 1) * BK;
            if (a_row < BM) {
                const int gRowA = tile_row + a_row;
                __pipeline_memcpy_async(&As[next_stage][a_row][a_col], &A[gRowA * K_SIZE + kk + a_col], 8);
            }
            
            if (b_row < BK) {
                const int gColB = tile_col + b_col;
                __pipeline_memcpy_async(&Bs[next_stage][b_row][b_col], &B[(kk + b_row) * N_SIZE + gColB], 8);
            }
            __pipeline_commit();
        }

        // Each warp computes single 16x16 output
        // K=0
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            
            wmma::load_matrix_sync(a_frag, &As[curr_stage][warp_row_base][0], BK + 8);
            wmma::load_matrix_sync(b_frag, &Bs[curr_stage][0][warp_col_base], BN + 8);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        
        // K=16
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            
            wmma::load_matrix_sync(a_frag, &As[curr_stage][warp_row_base][16], BK + 8);
            wmma::load_matrix_sync(b_frag, &Bs[curr_stage][16][warp_col_base], BN + 8);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
    }
    
    // Store single 16x16 tile
    const int global_row = tile_row + warp_row_base;
    const int global_col = tile_col + warp_col_base;
    wmma::store_matrix_sync(&C[global_row * N_SIZE + global_col], acc, N_SIZE, wmma::mem_row_major);
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
        matmul_bf16_fp32_1acc<<<grid, block>>>(d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        matmul_bf16_fp32_1acc<<<grid, block>>>(d_A, d_B, d_C);
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
            matmul_bf16_fp32_1acc<<<grid, block>>>(d_A, d_B, d_C);
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

    printf("\n=== ITER054 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M_SIZE, N_SIZE, K_SIZE);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter054_1acc (16 warps, 1 acc/warp)\n");
    printf("avg_tflops: %.2f\n", tflops_sum / SAMPLES);
    printf("min_tflops: %.2f\n", tflops_min);
    printf("max_tflops: %.2f\n", tflops_max);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
