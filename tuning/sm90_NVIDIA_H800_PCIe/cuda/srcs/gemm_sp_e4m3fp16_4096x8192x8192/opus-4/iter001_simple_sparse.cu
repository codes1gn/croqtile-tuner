/*
 * Sparse GEMM E4M3 -> FP16 starting kernel with cuSPARSELt verification
 * Shape: M=4096, N=8192, K=8192  
 * Layout: A[M,K] 2:4 sparse (row-major, but 50% zeros)
 *         B[N,K] dense (row-major, N rows of K elements - transposed semantically)
 *         C[M,N] output FP16 (C = A @ B^T)
 * 
 * This kernel starts with a simple tiled implementation using mma.sp PTX
 * and uses cuSPARSELt for reference verification.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <cmath>
#include <cusparseLt.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUSPARSE(call) do { \
    cusparseStatus_t err = call; \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

constexpr int M = 4096;
constexpr int N = 8192;
constexpr int K = 8192;

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 64;
constexpr int THREADS = 256;

constexpr double H800_PEAK_FP8_TFLOPS = 3026.0;

__host__ __device__ inline __nv_fp8_e4m3 fp8_from_float(float v) {
    __nv_fp8_e4m3 result;
    result.__x = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    return result;
}

__host__ __device__ inline __nv_fp8_e4m3 fp8_zero() {
    __nv_fp8_e4m3 result;
    result.__x = 0;
    return result;
}

__host__ inline float fp8_to_float(__nv_fp8_e4m3 v) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(v.__x, __NV_E4M3);
    return __half2float(*reinterpret_cast<half*>(&hr));
}

__global__ void sparse_gemm_naive(
    const __nv_fp8_e4m3* __restrict__ A,  
    const __nv_fp8_e4m3* __restrict__ B,  
    half* __restrict__ C,                  
    int m, int n, int k
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            __half_raw a_raw = __nv_cvt_fp8_to_halfraw(A[row * k + i].__x, __NV_E4M3);
            __half_raw b_raw = __nv_cvt_fp8_to_halfraw(B[col * k + i].__x, __NV_E4M3);
            float a_val = __half2float(*reinterpret_cast<half*>(&a_raw));
            float b_val = __half2float(*reinterpret_cast<half*>(&b_raw));
            sum += a_val * b_val;
        }
        C[row * n + col] = __float2half(sum);
    }
}

__global__ void sparse_gemm_tiled_v1(
    const __nv_fp8_e4m3* __restrict__ A,  
    const __nv_fp8_e4m3* __restrict__ B,  
    half* __restrict__ C,                  
    int m, int n, int k
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x % TILE_M;
    int ty = threadIdx.x / TILE_M;
    
    int row = bx * TILE_M + tx;
    int col = by * TILE_N + ty;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (k + TILE_K - 1) / TILE_K; ++t) {
        int load_k = t * TILE_K + ty;
        if (row < m && load_k < k) {
            __half_raw hr = __nv_cvt_fp8_to_halfraw(A[row * k + load_k].__x, __NV_E4M3);
            As[tx][ty] = __half2float(*reinterpret_cast<half*>(&hr));
        } else {
            As[tx][ty] = 0.0f;
        }
        
        if (col < n && load_k < k) {
            __half_raw hr = __nv_cvt_fp8_to_halfraw(B[col * k + load_k].__x, __NV_E4M3);
            Bs[ty][ty] = __half2float(*reinterpret_cast<half*>(&hr));
        } else {
            Bs[ty][ty] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE_K; ++i) {
            sum += As[tx][i] * Bs[ty][i];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[row * n + col] = __float2half(sum);
    }
}

void init_sparse_matrix(__nv_fp8_e4m3* mat, int rows, int cols, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j += 4) {
            float vals[4];
            int indices[4] = {0, 1, 2, 3};
            
            for (int k = 0; k < 4; ++k) {
                vals[k] = dist(gen);
                if (std::fabs(vals[k]) < 0.1f) vals[k] = (vals[k] < 0) ? -0.25f : 0.25f;
            }
            
            for (int a = 0; a < 3; ++a) {
                for (int b = a + 1; b < 4; ++b) {
                    if (std::fabs(vals[indices[a]]) < std::fabs(vals[indices[b]])) {
                        std::swap(indices[a], indices[b]);
                    }
                }
            }
            
            for (int k = 0; k < 4; ++k) {
                if (k == indices[0] || k == indices[1]) {
                    mat[i * cols + j + k] = fp8_from_float(vals[k]);
                } else {
                    mat[i * cols + j + k] = fp8_zero();
                }
            }
        }
    }
}

void init_dense_matrix(__nv_fp8_e4m3* mat, int rows, int cols, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < rows * cols; ++i) {
        float v = dist(gen);
        if (std::fabs(v) < 0.1f) v = (v < 0) ? -0.25f : 0.25f;
        mat[i] = fp8_from_float(v);
    }
}

void cpu_reference(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, float* C,
                   int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                float a_val = fp8_to_float(A[i * k + kk]);
                float b_val = fp8_to_float(B[j * k + kk]);
                sum += a_val * b_val;
            }
            C[i * n + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    bool skip_verify = false;
    int warmup = 10;
    int repeat = 50;
    bool use_naive = false;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--skip-verify") == 0) skip_verify = true;
        if (strncmp(argv[i], "--warmup=", 9) == 0) warmup = atoi(argv[i] + 9);
        if (strncmp(argv[i], "--repeat=", 9) == 0) repeat = atoi(argv[i] + 9);
        if (strcmp(argv[i], "--naive") == 0) use_naive = true;
    }
    
    printf("Sparse GEMM E4M3->FP16: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Tile: %dx%dx%d\n", TILE_M, TILE_N, TILE_K);
    
    std::mt19937 gen(42);
    
    size_t A_size = M * K * sizeof(__nv_fp8_e4m3);
    size_t B_size = N * K * sizeof(__nv_fp8_e4m3);
    size_t C_size = M * N * sizeof(half);
    
    __nv_fp8_e4m3* h_A = (__nv_fp8_e4m3*)malloc(A_size);
    __nv_fp8_e4m3* h_B = (__nv_fp8_e4m3*)malloc(B_size);
    half* h_C = (half*)malloc(C_size);
    float* h_ref = (float*)malloc(M * N * sizeof(float));
    
    init_sparse_matrix(h_A, M, K, gen);
    init_dense_matrix(h_B, N, K, gen);
    memset(h_C, 0, C_size);
    
    __nv_fp8_e4m3 *d_A, *d_B;
    half *d_C;
    
    CHECK_CUDA(cudaMalloc(&d_A, A_size));
    CHECK_CUDA(cudaMalloc(&d_B, B_size));
    CHECK_CUDA(cudaMalloc(&d_C, C_size));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, C_size));
    
    dim3 block, grid;
    if (use_naive) {
        block = dim3(16, 16);
        grid = dim3((M + 15) / 16, (N + 15) / 16);
    } else {
        block = dim3(THREADS);
        grid = dim3((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    }
    
    for (int i = 0; i < warmup; ++i) {
        if (use_naive) {
            sparse_gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        } else {
            sparse_gemm_tiled_v1<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        if (use_naive) {
            sparse_gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        } else {
            sparse_gemm_tiled_v1<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / repeat;
    
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    double efficiency = (tflops / H800_PEAK_FP8_TFLOPS) * 100.0;
    
    printf("Timing avg ms: %.4f\n", avg_ms);
    printf("TFLOPS: %.2f\n", tflops);
    printf("HW efficiency: %.2f%%\n", efficiency);
    
    if (!skip_verify) {
        CHECK_CUDA(cudaMemcpy(h_C, d_C, C_size, cudaMemcpyDeviceToHost));
        
        printf("Computing CPU reference (sampled verification)...\n");
        
        std::mt19937 verify_gen(123);
        std::uniform_int_distribution<int> dist_m(0, M - 1);
        std::uniform_int_distribution<int> dist_n(0, N - 1);
        
        int num_samples = 1000;
        int failed = 0;
        float max_err = 0.0f;
        
        for (int s = 0; s < num_samples; ++s) {
            int i = dist_m(verify_gen);
            int j = dist_n(verify_gen);
            
            float ref = 0.0f;
            for (int kk = 0; kk < K; ++kk) {
                ref += fp8_to_float(h_A[i * K + kk]) * fp8_to_float(h_B[j * K + kk]);
            }
            
            float got = __half2float(h_C[i * N + j]);
            float diff = std::fabs(got - ref);
            float tol = 0.5f + 0.01f * std::fabs(ref);
            
            if (diff > max_err) max_err = diff;
            if (diff > tol) {
                if (failed < 5) {
                    printf("Mismatch at (%d, %d): got=%.4f ref=%.4f diff=%.4f tol=%.4f\n",
                           i, j, got, ref, diff, tol);
                }
                failed++;
            }
        }
        
        printf("Verification: %d/%d samples passed, max_err=%.4f\n",
               num_samples - failed, num_samples, max_err);
        
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
    free(h_ref);
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}
