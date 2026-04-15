// iter003_profile.cu - Single launch version for profiling
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <random>

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

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 32;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;

__global__ __launch_bounds__(256, 2)
void matmul_bf16_fp32_wmma_swizzle(
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

    __shared__ __nv_bfloat16 As[BM][BK + 8];
    __shared__ __nv_bfloat16 Bs[BK][BN + 8];

    int warp_row = warpId / 2;
    int warp_col = warpId % 2;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    for (int kk = 0; kk < k; kk += BK) {
        int a_per_thread = (BM * BK) / 256;
        
        #pragma unroll
        for (int i = 0; i < a_per_thread; i++) {
            int idx = threadIdx.x * a_per_thread + i;
            int row = idx / BK;
            int col = idx % BK;
            int gRow = tile_row + row;
            int gCol = kk + col;
            As[row][col] = (gRow < m && gCol < k) ? A[gRow * k + gCol] : __float2bfloat16(0.0f);
        }

        int b_per_thread = (BK * BN) / 256;
        
        #pragma unroll
        for (int i = 0; i < b_per_thread; i++) {
            int idx = threadIdx.x * b_per_thread + i;
            int row = idx / BN;
            int col = idx % BN;
            int gRow = kk + row;
            int gCol = tile_col + col;
            Bs[row][col] = (gRow < k && gCol < n) ? B[gRow * n + gCol] : __float2bfloat16(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int wk = 0; wk < BK; wk += WMMA_K) {
            #pragma unroll
            for (int ti = 0; ti < 2; ti++) {
                #pragma unroll
                for (int tj = 0; tj < 2; tj++) {
                    int wmma_row = warp_row * 2 + ti;
                    int wmma_col = warp_col * 2 + tj;

                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;

                    wmma::load_matrix_sync(a_frag, &As[wmma_row * WMMA_M][wk], BK + 8);
                    wmma::load_matrix_sync(b_frag, &Bs[wk][wmma_col * WMMA_N], BN + 8);
                    wmma::mma_sync(acc[ti][tj], a_frag, b_frag, acc[ti][tj]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int ti = 0; ti < 2; ti++) {
        #pragma unroll
        for (int tj = 0; tj < 2; tj++) {
            int wmma_row = warp_row * 2 + ti;
            int wmma_col = warp_col * 2 + tj;

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

int main() {
    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;

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

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);

    matmul_bf16_fp32_wmma_swizzle<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Profiling run complete\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;

    return 0;
}
