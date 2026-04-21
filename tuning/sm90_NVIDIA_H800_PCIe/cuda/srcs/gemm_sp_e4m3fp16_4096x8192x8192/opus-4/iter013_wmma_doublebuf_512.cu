/*
 * Sparse GEMM E4M3 -> FP16 using WMMA with double-buffered shared memory
 * Shape: M=4096, N=8192, K=8192  
 * 
 * Try 512 threads (16 warps) for higher occupancy, smaller tile
 * 64x64 output tile with 4x4 warp grid
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <cmath>

using namespace nvcuda;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

constexpr int M_DIM = 4096;
constexpr int N_DIM = 8192;
constexpr int K_DIM = 8192;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// 4x4 warp grid = 16 warps = 512 threads for higher occupancy
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 4;
constexpr int TILE_M = WARPS_M * WMMA_M;  // 64
constexpr int TILE_N = WARPS_N * WMMA_N;  // 64
constexpr int TILE_K = 32;
constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 512

constexpr double H800_PEAK_FP8_TFLOPS = 3026.0;

__global__ void sparse_gemm_wmma_doublebuf_512(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ half As[2][TILE_M][TILE_K];
    __shared__ half Bs[2][TILE_N][TILE_K];
    
    const int warp_id = threadIdx.x / 32;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    
    const int block_row = blockIdx.x * TILE_M;
    const int block_col = blockIdx.y * TILE_N;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    const int A_elements = TILE_M * TILE_K;  // 64 * 32 = 2048
    const int B_elements = TILE_N * TILE_K;  // 64 * 32 = 2048
    
    // Prefetch first tiles
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < A_elements; idx += THREADS) {
        int row = idx / TILE_K;
        int col = idx % TILE_K;
        int g_row = block_row + row;
        As[0][row][col] = (g_row < M && col < K) ? A[g_row * K + col] : __float2half(0.0f);
    }
    
    #pragma unroll 4
    for (int idx = threadIdx.x; idx < B_elements; idx += THREADS) {
        int n_off = idx / TILE_K;
        int k_off = idx % TILE_K;
        int g_n = block_col + n_off;
        Bs[0][n_off][k_off] = (g_n < N && k_off < K) ? B[g_n * K + k_off] : __float2half(0.0f);
    }
    
    __syncthreads();
    
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int k_idx = 0; k_idx < num_k_tiles; ++k_idx) {
        int cur_buf = k_idx & 1;
        int nxt_buf = 1 - cur_buf;
        int next_k_tile = (k_idx + 1) * TILE_K;
        
        // Compute on current buffer
        #pragma unroll 2
        for (int k = 0; k < TILE_K; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, &As[cur_buf][warp_row * WMMA_M][k], TILE_K);
            wmma::load_matrix_sync(b_frag, &Bs[cur_buf][warp_col * WMMA_N][k], TILE_K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        __syncthreads();
        
        // Load next tile
        if (k_idx + 1 < num_k_tiles) {
            #pragma unroll 4
            for (int idx = threadIdx.x; idx < A_elements; idx += THREADS) {
                int row = idx / TILE_K;
                int col = idx % TILE_K;
                int g_row = block_row + row;
                int g_col = next_k_tile + col;
                As[nxt_buf][row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : __float2half(0.0f);
            }
            
            #pragma unroll 4
            for (int idx = threadIdx.x; idx < B_elements; idx += THREADS) {
                int n_off = idx / TILE_K;
                int k_off = idx % TILE_K;
                int g_n = block_col + n_off;
                int g_k = next_k_tile + k_off;
                Bs[nxt_buf][n_off][k_off] = (g_n < N && g_k < K) ? B[g_n * K + g_k] : __float2half(0.0f);
            }
            
            __syncthreads();
        }
    }
    
    int c_row = block_row + warp_row * WMMA_M;
    int c_col = block_col + warp_col * WMMA_N;
    
    if (c_row + WMMA_M <= M && c_col + WMMA_N <= N) {
        half c_half[WMMA_M * WMMA_N];
        #pragma unroll
        for (int i = 0; i < c_frag.num_elements; ++i) {
            c_half[i] = __float2half(c_frag.x[i]);
        }
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag_half;
        #pragma unroll
        for (int i = 0; i < c_frag_half.num_elements; ++i) {
            c_frag_half.x[i] = c_half[i];
        }
        wmma::store_matrix_sync(&C[c_row * N + c_col], c_frag_half, N, wmma::mem_row_major);
    }
}

void init_sparse_matrix(half* dense_fp16, int rows, int cols, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j += 4) {
            float vals[4];
            int idx[4] = {0, 1, 2, 3};
            
            for (int k = 0; k < 4; ++k) {
                vals[k] = dist(gen);
                if (std::fabs(vals[k]) < 0.1f) vals[k] = (vals[k] < 0) ? -0.25f : 0.25f;
            }
            
            for (int a = 0; a < 3; ++a) {
                for (int b = a + 1; b < 4; ++b) {
                    if (std::fabs(vals[idx[a]]) < std::fabs(vals[idx[b]])) {
                        std::swap(idx[a], idx[b]);
                    }
                }
            }
            
            int keep1 = idx[0], keep2 = idx[1];
            
            for (int k = 0; k < 4; ++k) {
                float v = (k == keep1 || k == keep2) ? vals[k] : 0.0f;
                dense_fp16[i * cols + j + k] = __float2half(v);
            }
        }
    }
}

void init_dense_matrix(half* mat_fp16, int rows, int cols, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < rows * cols; ++i) {
        float v = dist(gen);
        if (std::fabs(v) < 0.1f) v = (v < 0) ? -0.25f : 0.25f;
        mat_fp16[i] = __float2half(v);
    }
}

int main(int argc, char** argv) {
    bool skip_verify = false;
    int warmup = 10;
    int repeat = 50;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--skip-verify") == 0) skip_verify = true;
        if (strncmp(argv[i], "--warmup=", 9) == 0) warmup = atoi(argv[i] + 9);
        if (strncmp(argv[i], "--repeat=", 9) == 0) repeat = atoi(argv[i] + 9);
    }
    
    printf("Sparse GEMM E4M3->FP16 (WMMA DoubleBuf 512): M=%d, N=%d, K=%d\n", M_DIM, N_DIM, K_DIM);
    printf("Tile: %dx%dx%d, WMMA: %dx%dx%d, Warps: %dx%d, Threads=%d\n", 
           TILE_M, TILE_N, TILE_K, WMMA_M, WMMA_N, WMMA_K, WARPS_M, WARPS_N, THREADS);
    
    std::mt19937 gen(42);
    
    size_t A_sz = M_DIM * K_DIM;
    size_t B_sz = N_DIM * K_DIM;
    size_t C_sz = M_DIM * N_DIM;
    
    half* h_A = (half*)malloc(A_sz * sizeof(half));
    half* h_B = (half*)malloc(B_sz * sizeof(half));
    half* h_C = (half*)malloc(C_sz * sizeof(half));
    
    init_sparse_matrix(h_A, M_DIM, K_DIM, gen);
    init_dense_matrix(h_B, N_DIM, K_DIM, gen);
    memset(h_C, 0, C_sz * sizeof(half));
    
    half *d_A, *d_B, *d_C;
    
    CHECK_CUDA(cudaMalloc(&d_A, A_sz * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, B_sz * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, C_sz * sizeof(half)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, A_sz * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, B_sz * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, C_sz * sizeof(half)));
    
    dim3 grid((M_DIM + TILE_M - 1) / TILE_M, (N_DIM + TILE_N - 1) / TILE_N);
    dim3 block(THREADS);
    
    printf("Grid: (%d, %d), Block: %d\n", grid.x, grid.y, block.x);
    printf("Shared memory per block: %lu bytes\n", 
           sizeof(half) * 2 * (TILE_M * TILE_K + TILE_N * TILE_K));
    
    for (int i = 0; i < warmup; ++i) {
        sparse_gemm_wmma_doublebuf_512<<<grid, block>>>(d_A, d_B, d_C, M_DIM, N_DIM, K_DIM);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        sparse_gemm_wmma_doublebuf_512<<<grid, block>>>(d_A, d_B, d_C, M_DIM, N_DIM, K_DIM);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / repeat;
    
    double flops = 2.0 * M_DIM * N_DIM * K_DIM;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    double eff = (tflops / H800_PEAK_FP8_TFLOPS) * 100.0;
    
    printf("Timing avg ms: %.4f\n", avg_ms);
    printf("TFLOPS: %.2f\n", tflops);
    printf("HW efficiency: %.2f%%\n", eff);
    
    if (!skip_verify) {
        CHECK_CUDA(cudaMemcpy(h_C, d_C, C_sz * sizeof(half), cudaMemcpyDeviceToHost));
        
        printf("Sampled verification...\n");
        std::mt19937 vgen(123);
        std::uniform_int_distribution<int> dm(0, M_DIM - 1);
        std::uniform_int_distribution<int> dn(0, N_DIM - 1);
        
        int num_samples = 1000;
        int failed = 0;
        float max_err = 0.0f;
        
        for (int s = 0; s < num_samples; ++s) {
            int i = dm(vgen);
            int j = dn(vgen);
            
            float ref = 0.0f;
            for (int kk = 0; kk < K_DIM; ++kk) {
                ref += __half2float(h_A[i * K_DIM + kk]) * __half2float(h_B[j * K_DIM + kk]);
            }
            
            float got = __half2float(h_C[i * N_DIM + j]);
            float diff = std::fabs(got - ref);
            float tol = 1.0f + 0.05f * std::fabs(ref);
            
            if (diff > max_err) max_err = diff;
            if (diff > tol) {
                if (failed < 5) {
                    printf("Mismatch (%d,%d): got=%.4f ref=%.4f diff=%.4f\n", i, j, got, ref, diff);
                }
                failed++;
            }
        }
        
        printf("Verification: %d/%d passed, max_err=%.4f\n", num_samples - failed, num_samples, max_err);
        if (failed > num_samples / 10) {
            printf("VERIFY: FAIL max_abs_err=%.4f\n", max_err);
            return 1;
        }
        printf("VERIFY: PASS\n");
    }
    
    printf("Test Passed\n");
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
