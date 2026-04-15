// iter036_profile.cu - Profiling version with single run

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
constexpr int WARMUP = 0;
constexpr int ITERS = 1;
constexpr int SAMPLES = 1;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int STAGES = 2;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;

__global__ __launch_bounds__(128, 6)
void matmul_bf16_fp32_regopt(
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

    const int num_k_tiles = (k + BK - 1) / BK;
    const int tid = threadIdx.x;

    const int a_row_base = tid / 8;
    const int a_col_base = (tid % 8) * 4;
    const int b_row_base = tid / 16;
    const int b_col_base = (tid % 16) * 4;

    {
        const int kk = 0;
        
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            const int row = a_row_base + i * 16;
            const int col = a_col_base;
            if (row < BM) {
                const int gRow = tile_row + row;
                const int gCol = kk + col;
                if (gRow < m && gCol + 3 < k) {
                    __pipeline_memcpy_async(&As[0][row][col], &A[gRow * k + gCol], 8);
                }
            }
        }
        
        #pragma unroll 2
        for (int i = 0; i < 2; i++) {
            const int row = b_row_base + i * 16;
            const int col = b_col_base;
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

    const int warp_row_base = warp_m * 32;
    const int warp_col_base = warp_n * 32;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int curr_stage = kt & 1;
        const int next_stage = (kt + 1) & 1;
        
        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        if (kt + 1 < num_k_tiles) {
            const int kk = (kt + 1) * BK;
            
            #pragma unroll 4
            for (int i = 0; i < 4; i++) {
                const int row = a_row_base + i * 16;
                const int col = a_col_base;
                if (row < BM) {
                    const int gRow = tile_row + row;
                    const int gCol = kk + col;
                    if (gRow < m && gCol + 3 < k) {
                        __pipeline_memcpy_async(&As[next_stage][row][col], &A[gRow * k + gCol], 8);
                    }
                }
            }
            
            #pragma unroll 2
            for (int i = 0; i < 2; i++) {
                const int row = b_row_base + i * 16;
                const int col = b_col_base;
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
    size_t size_A = (size_t)M_SIZE * K_SIZE;
    size_t size_B = (size_t)K_SIZE * N_SIZE;
    size_t size_C = (size_t)M_SIZE * N_SIZE;

    __nv_bfloat16 *h_A, *h_B;
    __nv_bfloat16 *d_A, *d_B;
    float *d_C;

    h_A = new __nv_bfloat16[size_A];
    h_B = new __nv_bfloat16[size_B];

    init_bf16_random(h_A, size_A, 42);
    init_bf16_random(h_B, size_B, 123);

    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, size_C * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    dim3 grid((N_SIZE + BN - 1) / BN, (M_SIZE + BM - 1) / BM);
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);

    matmul_bf16_fp32_regopt<<<grid, block>>>(d_A, d_B, d_C, M_SIZE, N_SIZE, K_SIZE, grid.x);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("Profile run complete\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B;
    return 0;
}
