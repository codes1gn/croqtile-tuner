// iter005_mmasync.cu - mma.sync PTX + ldmatrix matmul kernel
// Shape: M=16384, N=16384, K=16384
// Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// with ldmatrix for efficient smem->register transfer
// BM=128, BN=128, BK=32, 8 warps (256 threads)
// Each warp: 4x2 mma tiles (64x16 output), covering 128x128 with 8 warps

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cmath>
#include <cstdint>

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
constexpr int NUM_WARPS = 8;
constexpr int BLOCK_SIZE = NUM_WARPS * 32;

// mma.sync m16n8k16 tile dimensions
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// Warp tile: each warp computes WARP_M x WARP_N of output
// 8 warps: 4 in M-dim, 2 in N-dim
// Each warp: 2 MMA tiles in M (32 rows), 8 MMA tiles in N (64 cols) = 32x64
// Total: 4*32=128 rows, 2*64=128 cols = 128x128 block
constexpr int WARP_M = 32;
constexpr int WARP_N = 64;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_MMA_M = WARP_M / MMA_M; // 2
constexpr int WARP_MMA_N = WARP_N / MMA_N; // 8

// Shared memory: A is row-major [BM][BK], B is col-major [BK][BN]
// B stored col-major so ldmatrix.trans works properly
constexpr int AS_STRIDE = BK;   // 32 bf16 = 64 bytes per row (no padding needed for 32)
constexpr int BS_STRIDE = BK;   // B col-major: BK rows x BN cols, stride=BK

__global__ void matmul_bf16_fp32_v5(
    const __nv_bfloat16* __restrict__ A,  // row-major [M, K]
    const __nv_bfloat16* __restrict__ B,  // row-major [K, N]
    float* __restrict__ C                  // row-major [M, N]
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warp_m = warpId / WARPS_N;
    const int warp_n = warpId % WARPS_N;

    // A: row-major in smem [BM][BK]
    // B: col-major in smem [BN][BK] (transposed during load for ldmatrix compatibility)
    __shared__ __nv_bfloat16 As[BM][AS_STRIDE];
    __shared__ __nv_bfloat16 Bs[BN][BS_STRIDE];

    // Accumulator registers: 2 MMA-M tiles x 8 MMA-N tiles, each m16n8k16 produces 4 f32 per thread
    float acc[WARP_MMA_M][WARP_MMA_N][4];
    #pragma unroll
    for (int i = 0; i < WARP_MMA_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_MMA_N; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int tile_row = by * BM;
    const int tile_col = bx * BN;

    for (int kk = 0; kk < K_DIM; kk += BK) {
        // Load A tile [BM, BK] in row-major
        for (int idx = threadIdx.x; idx < BM * BK; idx += BLOCK_SIZE) {
            int r = idx / BK;
            int c = idx % BK;
            As[r][c] = A[(tile_row + r) * K_DIM + kk + c];
        }
        // Load B tile [BK, BN] but store transposed as [BN, BK] for ldmatrix col access
        for (int idx = threadIdx.x; idx < BK * BN; idx += BLOCK_SIZE) {
            int r = idx / BN;  // K dimension
            int c = idx % BN;  // N dimension
            Bs[c][r] = B[(kk + r) * N_DIM + tile_col + c]; // transpose: Bs[n][k]
        }
        __syncthreads();

        // MMA compute loop
        #pragma unroll
        for (int wk = 0; wk < BK; wk += MMA_K) {
            // Load A fragments from smem: each mma needs 4 bf16x2 = 4 uint32_t per thread
            // A layout in smem: row-major [BM][BK], stride=BK=32
            // For m16n8k16, fragment A is m16 x k16
            uint32_t a_frag[WARP_MMA_M][4];
            #pragma unroll
            for (int mi = 0; mi < WARP_MMA_M; mi++) {
                int sm_row = warp_m * WARP_M + mi * MMA_M;
                // ldmatrix loads 8 rows x 16 cols (8x16 bf16 = 8x32 bytes)
                // Each thread in a warp provides address for one 16-byte row
                // lane % 16 selects which of the 16 rows, lane / 16 for the 2nd 8x16 half
                int row_in_tile = laneId % 16;
                uint32_t smem_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&As[sm_row + row_in_tile][wk]));

                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[mi][0]), "=r"(a_frag[mi][1]),
                      "=r"(a_frag[mi][2]), "=r"(a_frag[mi][3])
                    : "r"(smem_addr)
                );
            }

            // Load B fragments from smem: B is stored as [BN][BK] (col-major for the original matrix)
            // For m16n8k16: fragment B is k16 x n8
            // ldmatrix.trans loads 8 rows x 8 cols, transposing
            uint32_t b_frag[WARP_MMA_N][2];
            #pragma unroll
            for (int ni = 0; ni < WARP_MMA_N; ni++) {
                int sm_col = warp_n * WARP_N + ni * MMA_N;
                int row_in_tile = laneId % 16;
                uint32_t smem_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&Bs[sm_col + (laneId / 16) * 8 + (laneId % 8)][wk + (laneId / 8) % 2 * 8]));

                // For B fragment (k16 x n8), we need 2 uint32 per thread
                // Simpler: just load directly from smem
                // B[n][k] in smem, we need k-major for mma
                const __nv_bfloat16* b_ptr = &Bs[sm_col][wk];
                int k_idx = laneId % 16;
                int n_idx = laneId / 16;
                __nv_bfloat16 bv0 = b_ptr[n_idx * BS_STRIDE + k_idx];
                __nv_bfloat16 bv1 = b_ptr[(n_idx + 2) * BS_STRIDE + k_idx];
                __nv_bfloat16 bv2 = b_ptr[(n_idx + 4) * BS_STRIDE + k_idx];
                __nv_bfloat16 bv3 = b_ptr[(n_idx + 6) * BS_STRIDE + k_idx];

                // Pack into uint32 (2 bf16 per uint32)
                b_frag[ni][0] = *reinterpret_cast<uint32_t*>(&bv0);
                b_frag[ni][1] = *reinterpret_cast<uint32_t*>(&bv2);
            }

            // Execute MMA
            #pragma unroll
            for (int mi = 0; mi < WARP_MMA_M; mi++) {
                #pragma unroll
                for (int ni = 0; ni < WARP_MMA_N; ni++) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};\n"
                        : "=f"(acc[mi][ni][0]), "=f"(acc[mi][ni][1]),
                          "=f"(acc[mi][ni][2]), "=f"(acc[mi][ni][3])
                        : "r"(a_frag[mi][0]), "r"(a_frag[mi][1]),
                          "r"(a_frag[mi][2]), "r"(a_frag[mi][3]),
                          "r"(b_frag[ni][0]), "r"(b_frag[ni][1]),
                          "f"(acc[mi][ni][0]), "f"(acc[mi][ni][1]),
                          "f"(acc[mi][ni][2]), "f"(acc[mi][ni][3])
                    );
                }
            }
        }
        __syncthreads();
    }

    // Store results — mma.sync m16n8k16 output layout:
    // Thread (lane) owns: 4 f32 values at positions determined by lane ID
    // Row = (lane / 4), Col within 8-wide = (lane % 4) * 2 and (lane % 4) * 2 + 1
    // Plus row offset of 8 for results [2] and [3]
    #pragma unroll
    for (int mi = 0; mi < WARP_MMA_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < WARP_MMA_N; ni++) {
            int base_row = tile_row + warp_m * WARP_M + mi * MMA_M;
            int base_col = tile_col + warp_n * WARP_N + ni * MMA_N;

            int row0 = base_row + (laneId / 4);
            int col0 = base_col + (laneId % 4) * 2;

            if (row0 < M_DIM && col0 < N_DIM)
                C[row0 * N_DIM + col0] = acc[mi][ni][0];
            if (row0 < M_DIM && col0 + 1 < N_DIM)
                C[row0 * N_DIM + col0 + 1] = acc[mi][ni][1];

            int row1 = base_row + 8 + (laneId / 4);
            if (row1 < M_DIM && col0 < N_DIM)
                C[row1 * N_DIM + col0] = acc[mi][ni][2];
            if (row1 < M_DIM && col0 + 1 < N_DIM)
                C[row1 * N_DIM + col0 + 1] = acc[mi][ni][3];
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

        matmul_bf16_fp32_v5<<<grid, block>>>(d_A, d_B, d_C);
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
        matmul_bf16_fp32_v5<<<grid, block>>>(d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++)
        matmul_bf16_fp32_v5<<<grid, block>>>(d_A, d_B, d_C);
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
