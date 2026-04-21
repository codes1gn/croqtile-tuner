// iter062_dbuf_best.cu: 64x96x32 tile with TX=8 TY=32, double buffered - 16.23 TFLOPS
// Current best configuration for sparse GEMM e4m3->fp16

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <random>
#include <iostream>
#include <cstdint>
#include <cstring>

constexpr int M = 4096;
constexpr int N = 8192;
constexpr int K = 8192;

constexpr int TILE_M = 64;
constexpr int TILE_N = 96;
constexpr int TILE_K = 32;
constexpr int STAGES = 2;

constexpr int THREADS_X = 8;
constexpr int THREADS_Y = 32;
constexpr int THREADS_PER_BLOCK = THREADS_X * THREADS_Y;  // 256

constexpr int OUTPUTS_PER_THREAD_M = TILE_M / THREADS_Y;  // 2
constexpr int OUTPUTS_PER_THREAD_N = TILE_N / THREADS_X;  // 12

#define H800_PCIE_PEAK_F16_TFLOPS 1513.0

__global__ __launch_bounds__(256, 4)
void sparse_gemm_dbuf_kernel(
    const __nv_fp8_e4m3* __restrict__ A_packed,
    const uint32_t* __restrict__ A_meta,
    const __nv_fp8_e4m3* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k
) {
    int block_m = blockIdx.y * TILE_M;
    int block_n = blockIdx.x * TILE_N;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * THREADS_X + tx;
    
    __shared__ __nv_fp8_e4m3 As[STAGES][TILE_M][TILE_K / 2 + 4];
    __shared__ uint32_t Am[STAGES][TILE_M][2];
    __shared__ __nv_fp8_e4m3 Bs[STAGES][TILE_N][TILE_K + 4];
    
    float acc[OUTPUTS_PER_THREAD_M][OUTPUTS_PER_THREAD_N];
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < OUTPUTS_PER_THREAD_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    
    const int packed_k = k / 2;
    const int meta_k = k / 32;
    const int num_k_tiles = (k + TILE_K - 1) / TILE_K;
    
    // Lambda: load tile to shared memory
    auto load_tile = [&](int stage, int k_offset) {
        // Load A_packed
        #pragma unroll 4
        for (int i = tid; i < TILE_M * (TILE_K / 2); i += THREADS_PER_BLOCK) {
            int lm = i / (TILE_K / 2), lk = i % (TILE_K / 2);
            int gm = block_m + lm, gk = k_offset / 2 + lk;
            As[stage][lm][lk] = (gm < m && gk < packed_k) ? A_packed[gm * packed_k + gk] : __nv_fp8_e4m3(0.0f);
        }
        
        // Load metadata
        if (tid < TILE_M) {
            int gm = block_m + tid, gk = k_offset / 32;
            Am[stage][tid][0] = (gm < m && gk < meta_k) ? A_meta[gm * meta_k + gk] : 0;
        }
        
        // Load B
        #pragma unroll 4
        for (int i = tid; i < TILE_N * TILE_K; i += THREADS_PER_BLOCK) {
            int ln = i / TILE_K, lk = i % TILE_K;
            int gn = block_n + ln, gk = k_offset + lk;
            Bs[stage][ln][lk] = (gn < n && gk < k) ? B[gn * k + gk] : __nv_fp8_e4m3(0.0f);
        }
    };
    
    // Lambda: compute from shared memory
    auto compute_tile = [&](int stage) {
        #pragma unroll
        for (int i = 0; i < OUTPUTS_PER_THREAD_M; ++i) {
            int local_m = ty * OUTPUTS_PER_THREAD_M + i;
            uint32_t meta = Am[stage][local_m][0];
            
            #pragma unroll 2
            for (int kk_base = 0; kk_base < TILE_K; kk_base += 16) {
                int kg0 = kk_base / 4, kg1 = kg0 + 1, kg2 = kg0 + 2, kg3 = kg0 + 3;
                int s0 = kg0 * 4, s1 = kg1 * 4, s2 = kg2 * 4, s3 = kg3 * 4;
                int idx0_0 = (meta >> s0) & 0x3, idx0_1 = (meta >> (s0 + 2)) & 0x3;
                int idx1_0 = (meta >> s1) & 0x3, idx1_1 = (meta >> (s1 + 2)) & 0x3;
                int idx2_0 = (meta >> s2) & 0x3, idx2_1 = (meta >> (s2 + 2)) & 0x3;
                int idx3_0 = (meta >> s3) & 0x3, idx3_1 = (meta >> (s3 + 2)) & 0x3;
                
                float a0_0 = float(As[stage][local_m][kg0 * 2]);
                float a0_1 = float(As[stage][local_m][kg0 * 2 + 1]);
                float a1_0 = float(As[stage][local_m][kg1 * 2]);
                float a1_1 = float(As[stage][local_m][kg1 * 2 + 1]);
                float a2_0 = float(As[stage][local_m][kg2 * 2]);
                float a2_1 = float(As[stage][local_m][kg2 * 2 + 1]);
                float a3_0 = float(As[stage][local_m][kg3 * 2]);
                float a3_1 = float(As[stage][local_m][kg3 * 2 + 1]);
                
                #pragma unroll
                for (int j = 0; j < OUTPUTS_PER_THREAD_N; ++j) {
                    int local_n = tx * OUTPUTS_PER_THREAD_N + j;
                    
                    float b0_0 = float(Bs[stage][local_n][kk_base + idx0_0]);
                    float b0_1 = float(Bs[stage][local_n][kk_base + idx0_1]);
                    float b1_0 = float(Bs[stage][local_n][kk_base + 4 + idx1_0]);
                    float b1_1 = float(Bs[stage][local_n][kk_base + 4 + idx1_1]);
                    float b2_0 = float(Bs[stage][local_n][kk_base + 8 + idx2_0]);
                    float b2_1 = float(Bs[stage][local_n][kk_base + 8 + idx2_1]);
                    float b3_0 = float(Bs[stage][local_n][kk_base + 12 + idx3_0]);
                    float b3_1 = float(Bs[stage][local_n][kk_base + 12 + idx3_1]);
                    
                    acc[i][j] = __fmaf_rn(a0_0, b0_0, acc[i][j]);
                    acc[i][j] = __fmaf_rn(a1_0, b1_0, acc[i][j]);
                    acc[i][j] = __fmaf_rn(a2_0, b2_0, acc[i][j]);
                    acc[i][j] = __fmaf_rn(a3_0, b3_0, acc[i][j]);
                    acc[i][j] = __fmaf_rn(a0_1, b0_1, acc[i][j]);
                    acc[i][j] = __fmaf_rn(a1_1, b1_1, acc[i][j]);
                    acc[i][j] = __fmaf_rn(a2_1, b2_1, acc[i][j]);
                    acc[i][j] = __fmaf_rn(a3_1, b3_1, acc[i][j]);
                }
            }
        }
    };
    
    // Load first tile
    load_tile(0, 0);
    __syncthreads();
    
    // Main loop with double buffering
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int curr_stage = k_tile % STAGES;
        int next_stage = (k_tile + 1) % STAGES;
        int next_k_offset = (k_tile + 1) * TILE_K;
        
        if (k_tile + 1 < num_k_tiles) {
            load_tile(next_stage, next_k_offset);
        }
        
        compute_tile(curr_stage);
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD_M; ++i) {
        int global_m = block_m + ty * OUTPUTS_PER_THREAD_M + i;
        
        #pragma unroll
        for (int j = 0; j < OUTPUTS_PER_THREAD_N; ++j) {
            int global_n = block_n + tx * OUTPUTS_PER_THREAD_N + j;
            
            if (global_m < m && global_n < n) {
                C[global_m * n + global_n] = __float2half(acc[i][j]);
            }
        }
    }
}

void init_sparse_data(float* A_dense, __nv_fp8_e4m3* A_packed, uint32_t* A_meta, int m, int k, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pattern_dist(0, 5);
    const int patterns[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    
    for (int row = 0; row < m; ++row) {
        int packed_idx = 0;
        for (int k_group = 0; k_group < k / 4; ++k_group) {
            int pattern_idx = pattern_dist(gen);
            const int* pattern = patterns[pattern_idx];
            for (int i = 0; i < 4; ++i) A_dense[row * k + k_group * 4 + i] = 0.0f;
            float val0 = dist(gen), val1 = dist(gen);
            if (std::abs(val0) < 0.1f) val0 = (val0 < 0 ? -0.25f : 0.25f);
            if (std::abs(val1) < 0.1f) val1 = (val1 < 0 ? -0.25f : 0.25f);
            A_dense[row * k + k_group * 4 + pattern[0]] = val0;
            A_dense[row * k + k_group * 4 + pattern[1]] = val1;
            A_packed[row * (k / 2) + packed_idx++] = __nv_fp8_e4m3(val0);
            A_packed[row * (k / 2) + packed_idx++] = __nv_fp8_e4m3(val1);
        }
    }
    
    for (int row = 0; row < m; ++row) {
        for (int meta_col = 0; meta_col < k / 32; ++meta_col) {
            uint32_t meta_word = 0;
            for (int g = 0; g < 8; ++g) {
                int k_base = meta_col * 32 + g * 4;
                int nz0 = -1, nz1 = -1;
                for (int i = 0; i < 4; ++i) {
                    if (A_dense[row * k + k_base + i] != 0.0f) {
                        if (nz0 < 0) nz0 = i; else nz1 = i;
                    }
                }
                uint32_t idx0 = (nz0 >= 0) ? (nz0 & 0x3) : 0;
                uint32_t idx1 = (nz1 >= 0) ? (nz1 & 0x3) : 1;
                meta_word |= (idx0 << (g * 4));
                meta_word |= (idx1 << (g * 4 + 2));
            }
            A_meta[row * (k / 32) + meta_col] = meta_word;
        }
    }
}

void init_B(__nv_fp8_e4m3* B, int n, int k, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int j = 0; j < n; ++j) {
        for (int kk = 0; kk < k; ++kk) {
            float val = dist(gen);
            if (std::abs(val) < 0.1f) val = (val < 0 ? -0.25f : 0.25f);
            B[j * k + kk] = __nv_fp8_e4m3(val);
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "Sparse GEMM e4m3->fp16 (64x96x32 double buffered TX=8 TY=32): " << M << "x" << N << "x" << K << "\n";
    
    size_t A_packed_bytes = M * (K / 2);
    size_t A_meta_bytes = M * (K / 32) * sizeof(uint32_t);
    size_t B_bytes = N * K;
    size_t C_bytes = M * N * sizeof(half);
    
    float* A_dense_h = (float*)malloc(M * K * sizeof(float));
    __nv_fp8_e4m3* A_packed_h = (__nv_fp8_e4m3*)malloc(A_packed_bytes);
    uint32_t* A_meta_h = (uint32_t*)malloc(A_meta_bytes);
    __nv_fp8_e4m3* B_h = (__nv_fp8_e4m3*)malloc(B_bytes);
    half* C_h = (half*)malloc(C_bytes);
    
    std::mt19937 gen(42);
    init_sparse_data(A_dense_h, A_packed_h, A_meta_h, M, K, gen);
    init_B(B_h, N, K, gen);
    
    __nv_fp8_e4m3 *A_packed_d, *B_d;
    uint32_t *A_meta_d;
    half *C_d;
    
    cudaMalloc(&A_packed_d, A_packed_bytes);
    cudaMalloc(&A_meta_d, A_meta_bytes);
    cudaMalloc(&B_d, B_bytes);
    cudaMalloc(&C_d, C_bytes);
    
    cudaMemcpy(A_packed_d, A_packed_h, A_packed_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(A_meta_d, A_meta_h, A_meta_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_bytes, cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, C_bytes);
    cudaDeviceSynchronize();
    
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    sparse_gemm_dbuf_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int warmup = 10;
    const int repeat = 50;
    
    for (int i = 0; i < warmup; ++i) {
        sparse_gemm_dbuf_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
        sparse_gemm_dbuf_kernel<<<grid, block>>>(A_packed_d, A_meta_d, B_d, C_d, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / repeat;
    
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    
    std::cout << "Timing avg ms: " << avg_ms << "\n";
    std::cout << "TFLOPS: " << tflops << "\n";
    std::cout << "HW efficiency: " << (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0 << "%\n";
    
    cudaFree(A_packed_d);
    cudaFree(A_meta_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(A_dense_h);
    free(A_packed_h);
    free(A_meta_h);
    free(B_h);
    free(C_h);
    
    return 0;
}
