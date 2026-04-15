// iter001_draft.cu - First draft custom bf16->fp32 matmul kernel
// Shape: M=512, N=16384, K=16384
// Strategy: Basic tiled matmul with shared memory, wmma for bf16 accumulation to fp32
// Target: Beat cuBLAS baseline of ~42 TFLOPS

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
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

// Tile sizes for WMMA
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block tile sizes
constexpr int BM = 128;  // Block tile M
constexpr int BN = 128;  // Block tile N
constexpr int BK = 32;   // Block tile K (loop step)

// Thread block dimensions
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK_M = BM / WMMA_M;  // 8
constexpr int WARPS_PER_BLOCK_N = BN / WMMA_N;  // 8
constexpr int WARPS_PER_BLOCK = 8;  // Use 8 warps = 256 threads

__global__ void matmul_bf16_fp32_wmma(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [K, N] row-major
    float* __restrict__ C,                 // [M, N] row-major
    int m, int n, int k
) {
    // Block position
    int bx = blockIdx.x;  // along N
    int by = blockIdx.y;  // along M

    // Warp position within block
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    // Each warp handles a WMMA_M x WMMA_N tile
    // We have 8 warps handling 4x2 WMMA tiles
    int warp_row = warpId / 2;  // 0-3
    int warp_col = warpId % 2;  // 0-1

    // Global tile offsets
    int tile_row = by * BM;
    int tile_col = bx * BN;

    // Each warp processes multiple WMMA tiles to cover BM x BN
    // With 8 warps and 8x8 WMMA tiles needed (128/16 = 8), each warp does 8 tiles
    
    // Shared memory for A and B tiles
    __shared__ __nv_bfloat16 As[BM][BK];
    __shared__ __nv_bfloat16 Bs[BK][BN];

    // WMMA fragments for accumulation
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4][4];
    
    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    // Loop over K dimension
    for (int kk = 0; kk < k; kk += BK) {
        // Cooperative load A tile [BM, BK]
        // Each thread loads multiple elements
        int load_idx = threadIdx.x;
        int loads_per_thread = (BM * BK) / (WARPS_PER_BLOCK * WARP_SIZE);
        
        #pragma unroll
        for (int i = 0; i < loads_per_thread; i++) {
            int elem_idx = load_idx + i * (WARPS_PER_BLOCK * WARP_SIZE);
            int row = elem_idx / BK;
            int col = elem_idx % BK;
            int global_row = tile_row + row;
            int global_col = kk + col;
            
            if (global_row < m && global_col < k) {
                As[row][col] = A[global_row * k + global_col];
            } else {
                As[row][col] = __float2bfloat16(0.0f);
            }
        }

        // Cooperative load B tile [BK, BN]
        int loads_per_thread_B = (BK * BN) / (WARPS_PER_BLOCK * WARP_SIZE);
        
        #pragma unroll
        for (int i = 0; i < loads_per_thread_B; i++) {
            int elem_idx = load_idx + i * (WARPS_PER_BLOCK * WARP_SIZE);
            int row = elem_idx / BN;
            int col = elem_idx % BN;
            int global_row = kk + row;
            int global_col = tile_col + col;
            
            if (global_row < k && global_col < n) {
                Bs[row][col] = B[global_row * n + global_col];
            } else {
                Bs[row][col] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        // WMMA compute: each warp processes its assigned tiles
        // With 8 warps, we assign each warp to process a 2x4 grid of WMMA tiles
        // Total: 8 warps * 2 * 4 = 64 tiles = 8x8 coverage for BM=128, BN=128
        
        #pragma unroll
        for (int wk = 0; wk < BK; wk += WMMA_K) {
            // Each warp handles 4 rows and 4 cols of WMMA tiles
            int base_row = (warpId / 2) * 2;  // 0, 2, 4, 6
            int base_col = (warpId % 2) * 4;  // 0, 4

            #pragma unroll
            for (int ti = 0; ti < 2; ti++) {
                #pragma unroll
                for (int tj = 0; tj < 4; tj++) {
                    int wmma_row = base_row + ti;
                    int wmma_col = base_col + tj;

                    // Load A fragment from shared memory
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;

                    // A fragment: rows [wmma_row*16, wmma_row*16+16), cols [wk, wk+16)
                    wmma::load_matrix_sync(a_frag, &As[wmma_row * WMMA_M][wk], BK);
                    
                    // B fragment: rows [wk, wk+16), cols [wmma_col*16, wmma_col*16+16)
                    wmma::load_matrix_sync(b_frag, &Bs[wk][wmma_col * WMMA_N], BN);

                    // Accumulate
                    wmma::mma_sync(acc[ti][tj], a_frag, b_frag, acc[ti][tj]);
                }
            }
        }

        __syncthreads();
    }

    // Store results
    int base_row = (warpId / 2) * 2;
    int base_col = (warpId % 2) * 4;

    #pragma unroll
    for (int ti = 0; ti < 2; ti++) {
        #pragma unroll
        for (int tj = 0; tj < 4; tj++) {
            int wmma_row = base_row + ti;
            int wmma_col = base_col + tj;

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

void compute_reference_fp32(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += __bfloat162float(A[i * k + l]) * __bfloat162float(B[l * n + j]);
            }
            C[i * n + j] = sum;
        }
    }
}

bool verify_output(const float* ref, const float* out, int size, float atol = 1e-1f, float rtol = 1e-2f) {
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int errors = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(ref[i] - out[i]);
        float rel = (fabsf(ref[i]) > 1e-6f) ? diff / fabsf(ref[i]) : diff;
        max_abs_err = fmaxf(max_abs_err, diff);
        max_rel_err = fmaxf(max_rel_err, rel);
        if (diff > atol + rtol * fabsf(ref[i])) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at %d: ref=%f, out=%f, diff=%f\n", i, ref[i], out[i], diff);
            }
        }
    }
    printf("max_abs_err=%e max_rel_err=%e errors=%d/%d\n", max_abs_err, max_rel_err, errors, size);
    return errors < size / 100;  // Allow <1% errors due to accumulation order differences
}

int main(int argc, char** argv) {
    bool verify_mode = (argc > 1 && std::string(argv[1]) == "--verify");
    
    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;

    __nv_bfloat16 *h_A, *h_B;
    float *h_C, *h_ref;
    __nv_bfloat16 *d_A, *d_B;
    float *d_C;

    h_A = new __nv_bfloat16[size_A];
    h_B = new __nv_bfloat16[size_B];
    h_C = new float[size_C];
    h_ref = new float[size_C];

    init_bf16_random(h_A, size_A, 42);
    init_bf16_random(h_B, size_B, 123);

    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, size_C * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Grid dimensions
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);  // 256 threads

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        matmul_bf16_fp32_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        // Compute CPU reference
        compute_reference_fp32(h_A, h_B, h_ref, M, N, K);
        
        // Run kernel
        matmul_bf16_fp32_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        
        bool passed = verify_output(h_ref, h_C, size_C);
        printf("verification: %s\n", passed ? "PASSED" : "FAILED");
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_ref;
        
        return passed ? 0 : 1;
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
            matmul_bf16_fp32_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    printf("\n=== ITER001 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M, N, K);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter001_draft (WMMA bf16->fp32)\n");
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
    delete[] h_ref;

    return 0;
}
