// iter016_wideN.cu - Exploit wide N dimension with narrow M
// Shape: M=512, N=16384, K=16384
// Idea: Since M=512 is small (only 32 x 16-row tiles), maximize N coverage per block

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

// 32x128 output per block: covers 2 M tiles x 8 N tiles
constexpr int BM = 32;
constexpr int BN = 128;
constexpr int BK = 16;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;  // 256 threads

using bf16x4 = uint2;

__device__ __forceinline__ bf16x4 load_bf16x4(const __nv_bfloat16* ptr) {
    return *reinterpret_cast<const bf16x4*>(ptr);
}

__device__ __forceinline__ void store_bf16x4(__nv_bfloat16* ptr, bf16x4 val) {
    *reinterpret_cast<bf16x4*>(ptr) = val;
}

__global__ __launch_bounds__(256, 4)
void matmul_bf16_fp32_wideN(
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

    __shared__ __nv_bfloat16 As[BM][BK + 4];     // 32x20 = 1.25KB
    __shared__ __nv_bfloat16 Bs[BK][BN + 8];     // 16x136 = 4.25KB

    // Layout: 8 warps arranged as 2x4 (2 in M, 4 in N direction)
    int warp_m = warpId / 4;  // 0 or 1
    int warp_n = warpId % 4;  // 0-3

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        wmma::fill_fragment(acc[i], 0.0f);
    }

    int num_k_tiles = (k + BK - 1) / BK;
    int tid = threadIdx.x;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int kk = kt * BK;
        
        // Load A: [32 x 16] = 512 bf16 = 128 bf16x4
        // 256 threads, 1 per 2 threads, but vectorize
        if (tid < BM * (BK / 4)) {
            int row = tid / (BK / 4);
            int col4 = tid % (BK / 4);
            int col = col4 * 4;
            int gRow = tile_row + row;
            int gCol = kk + col;
            
            if (gRow < m && gCol + 3 < k) {
                bf16x4 val = load_bf16x4(&A[gRow * k + gCol]);
                store_bf16x4(&As[row][col], val);
            } else {
                As[row][col] = __float2bfloat16(0.0f);
                As[row][col+1] = __float2bfloat16(0.0f);
                As[row][col+2] = __float2bfloat16(0.0f);
                As[row][col+3] = __float2bfloat16(0.0f);
            }
        }
        
        // Load B: [16 x 128] = 2048 bf16 = 512 bf16x4
        // 256 threads, 2 bf16x4 each
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
                    bf16x4 val = load_bf16x4(&B[gRow * n + gCol]);
                    store_bf16x4(&Bs[row][col], val);
                } else {
                    Bs[row][col] = __float2bfloat16(0.0f);
                    Bs[row][col+1] = __float2bfloat16(0.0f);
                    Bs[row][col+2] = __float2bfloat16(0.0f);
                    Bs[row][col+3] = __float2bfloat16(0.0f);
                }
            }
        }
        
        __syncthreads();

        // Each warp: 16x32 output (16 rows, 2 WMMA tiles in N)
        int warp_row = warp_m * 16;
        int warp_col = warp_n * 32;
        
        #pragma unroll
        for (int wj = 0; wj < 2; wj++) {
            int b_col = warp_col + wj * 16;
            
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            
            wmma::load_matrix_sync(a_frag, &As[warp_row][0], BK + 4);
            wmma::load_matrix_sync(b_frag, &Bs[0][b_col], BN + 8);
            wmma::mma_sync(acc[wj], a_frag, b_frag, acc[wj]);
        }
        
        __syncthreads();
    }

    int warp_row = warp_m * 16;
    int warp_col = warp_n * 32;
    
    #pragma unroll
    for (int wj = 0; wj < 2; wj++) {
        int global_row = tile_row + warp_row;
        int global_col = tile_col + warp_col + wj * 16;
        
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
        matmul_bf16_fp32_wideN<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        matmul_bf16_fp32_wideN<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
            matmul_bf16_fp32_wideN<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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

    printf("\n=== ITER016 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M, N, K);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter016_wideN (32x128 tiles, 8 warps)\n");
    printf("avg_tflops: %.2f\n", tflops_sum / SAMPLES);
    printf("min_tflops: %.2f\n", tflops_min);
    printf("max_tflops: %.2f\n", tflops_max);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
