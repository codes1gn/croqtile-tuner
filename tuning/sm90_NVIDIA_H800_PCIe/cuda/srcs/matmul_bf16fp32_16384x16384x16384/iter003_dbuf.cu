// iter003_dbuf.cu - Double-buffered shared memory matmul
// Shape: M=16384, N=16384, K=16384
// Changes from iter001: double buffering to overlap next tile load with compute,
// using regular loads (not cp.async), BK=32

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
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

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

constexpr int M_DIM = 16384;
constexpr int N_DIM = 16384;
constexpr int K_DIM = 16384;
constexpr int WARMUP = 10;
constexpr int ITERS = 50;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int NUM_WARPS = 8;
constexpr int BLOCK_SIZE = NUM_WARPS * 32;

constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 4;
constexpr int WARPS_N = 2;

constexpr int AS_STRIDE = BK + 8;
constexpr int BS_STRIDE = BN + 8;

// Register-cached loads: preload next tile into registers while computing on smem
__global__ void matmul_bf16_fp32_v3(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int warpId = threadIdx.x / 32;
    const int warp_m = warpId / WARPS_N;
    const int warp_n = warpId % WARPS_N;

    __shared__ __nv_bfloat16 As[2][BM][AS_STRIDE];
    __shared__ __nv_bfloat16 Bs[2][BK][BS_STRIDE];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int tile_row = by * BM;
    const int tile_col = bx * BN;
    const int num_k_tiles = K_DIM / BK;

    // Load first tile into buffer 0
    for (int idx = threadIdx.x; idx < BM * BK; idx += BLOCK_SIZE) {
        int r = idx / BK;
        int c = idx % BK;
        As[0][r][c] = A[(tile_row + r) * K_DIM + c];
    }
    for (int idx = threadIdx.x; idx < BK * BN; idx += BLOCK_SIZE) {
        int r = idx / BN;
        int c = idx % BN;
        Bs[0][r][c] = B[r * N_DIM + tile_col + c];
    }
    __syncthreads();

    for (int tile_k = 0; tile_k < num_k_tiles; tile_k++) {
        int cur = tile_k & 1;
        int nxt = 1 - cur;

        // Start loading next tile into other buffer (if not last)
        if (tile_k + 1 < num_k_tiles) {
            int next_kk = (tile_k + 1) * BK;
            for (int idx = threadIdx.x; idx < BM * BK; idx += BLOCK_SIZE) {
                int r = idx / BK;
                int c = idx % BK;
                As[nxt][r][c] = A[(tile_row + r) * K_DIM + next_kk + c];
            }
            for (int idx = threadIdx.x; idx < BK * BN; idx += BLOCK_SIZE) {
                int r = idx / BN;
                int c = idx % BN;
                Bs[nxt][r][c] = B[(next_kk + r) * N_DIM + tile_col + c];
            }
        }

        // Compute on current buffer (reads from smem, writes to register)
        #pragma unroll
        for (int wk = 0; wk < BK; wk += WMMA_K) {
            #pragma unroll
            for (int ti = 0; ti < WARP_TILES_M; ti++) {
                #pragma unroll
                for (int tj = 0; tj < WARP_TILES_N; tj++) {
                    int sm_row = (warp_m * WARP_TILES_M + ti) * WMMA_M;
                    int sm_col = (warp_n * WARP_TILES_N + tj) * WMMA_N;

                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                   __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                   __nv_bfloat16, wmma::row_major> b_frag;

                    wmma::load_matrix_sync(a_frag, &As[cur][sm_row][wk], AS_STRIDE);
                    wmma::load_matrix_sync(b_frag, &Bs[cur][wk][sm_col], BS_STRIDE);
                    wmma::mma_sync(acc[ti][tj], a_frag, b_frag, acc[ti][tj]);
                }
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int ti = 0; ti < WARP_TILES_M; ti++) {
        #pragma unroll
        for (int tj = 0; tj < WARP_TILES_N; tj++) {
            int out_row = tile_row + (warp_m * WARP_TILES_M + ti) * WMMA_M;
            int out_col = tile_col + (warp_n * WARP_TILES_N + tj) * WMMA_N;
            wmma::store_matrix_sync(&C[out_row * N_DIM + out_col],
                                    acc[ti][tj], N_DIM, wmma::mem_row_major);
        }
    }
}

void init_bf16_random(__nv_bfloat16* data, size_t size, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < size; i++)
        data[i] = __float2bfloat16(dist(gen));
}

int main(int argc, char** argv) {
    bool verify = (argc > 1 && std::string(argv[1]) == "--verify");

    size_t sA = (size_t)M_DIM * K_DIM;
    size_t sB = (size_t)K_DIM * N_DIM;
    size_t sC = (size_t)M_DIM * N_DIM;

    __nv_bfloat16 *h_A = new __nv_bfloat16[sA];
    __nv_bfloat16 *h_B = new __nv_bfloat16[sB];
    float *h_C = new float[sC];
    init_bf16_random(h_A, sA, 42);
    init_bf16_random(h_B, sB, 123);

    __nv_bfloat16 *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sA * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, sB * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, sC * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    dim3 grid((N_DIM + BN - 1) / BN, (M_DIM + BM - 1) / BM);
    dim3 block(BLOCK_SIZE);

    if (verify) {
        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));
        float alpha = 1.0f, beta = 0.0f;
        float *d_C_ref;
        CHECK_CUDA(cudaMalloc(&d_C_ref, sC * sizeof(float)));
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N_DIM, M_DIM, K_DIM, &alpha,
            d_B, CUDA_R_16BF, N_DIM, d_A, CUDA_R_16BF, K_DIM,
            &beta, d_C_ref, CUDA_R_32F, N_DIM,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK_CUDA(cudaDeviceSynchronize());

        matmul_bf16_fp32_v3<<<grid, block>>>(d_A, d_B, d_C);
        CHECK_CUDA(cudaDeviceSynchronize());

        constexpr int CR = 256, CC = 256;
        float *h_ours = new float[(size_t)CR * N_DIM];
        float *h_ref = new float[(size_t)CR * N_DIM];
        CHECK_CUDA(cudaMemcpy(h_ours, d_C, (size_t)CR * N_DIM * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_ref, d_C_ref, (size_t)CR * N_DIM * sizeof(float), cudaMemcpyDeviceToHost));

        float max_abs = 0.0f;
        int errs = 0;
        for (int i = 0; i < CR; i++)
            for (int j = 0; j < CC; j++) {
                float diff = fabsf(h_ref[i * N_DIM + j] - h_ours[i * N_DIM + j]);
                max_abs = fmaxf(max_abs, diff);
                if (diff > 1.0f) errs++;
            }
        bool pass = (max_abs < 5.0f) && (errs < CR * CC / 100);
        printf("VERIFY: %s max_abs_err=%.4f errors=%d/%d\n",
               pass ? "PASS" : "FAIL", max_abs, errs, CR * CC);

        delete[] h_ours; delete[] h_ref;
        cudaFree(d_C_ref); cublasDestroy(handle);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        delete[] h_A; delete[] h_B; delete[] h_C;
        return pass ? 0 : 1;
    }

    for (int i = 0; i < WARMUP; i++)
        matmul_bf16_fp32_v3<<<grid, block>>>(d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        matmul_bf16_fp32_v3<<<grid, block>>>(d_A, d_B, d_C);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    double time_ms = ms / ITERS;
    double tflops = 2.0 * M_DIM * N_DIM * K_DIM / (time_ms * 1e-3) / 1e12;
    printf("TFLOPS: %.2f   time_ms: %.3f\n", tflops, time_ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
