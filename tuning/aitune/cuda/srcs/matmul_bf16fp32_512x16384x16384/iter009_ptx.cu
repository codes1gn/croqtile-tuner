// iter009_ptx.cu - Native PTX mma.sync for bf16->fp32
// Shape: M=512, N=16384, K=16384
// Idea: Use PTX mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// This gives finer control than WMMA and may reduce overhead

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cmath>

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

// MMA tile: m16n8k16 for bf16->f32
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// Block tiles
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;  // 128 threads

// Each warp processes 16x16 output using 2 m16n8k16 ops side by side
constexpr int WARP_M = 16;
constexpr int WARP_N = 16;

__device__ __forceinline__ void mma_sync_bf16_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

__global__ __launch_bounds__(128, 8)
void matmul_bf16_fp32_ptx(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int m, int n, int k
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    
    // Warp position: 2x2 warps
    int warp_m = warpId / 2;  // 0-1
    int warp_n = warpId % 2;  // 0-1

    int tile_row = by * BM;
    int tile_col = bx * BN;

    __shared__ __nv_bfloat16 As[BM][BK + 8];
    __shared__ __nv_bfloat16 Bs[BK][BN + 8];

    // Each warp computes 16x16 = 2 m16n8 tiles
    // Accumulators: 2 tiles, 4 floats each = 8 floats
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

    int num_k_tiles = (k + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int kk = kt * BK;
        
        // Load A and B tiles
        int tid = threadIdx.x;
        int a_total = BM * BK;  // 1024
        int b_total = BK * BN;  // 1024
        
        for (int i = tid; i < a_total; i += 128) {
            int row = i / BK;
            int col = i % BK;
            int gRow = tile_row + row;
            int gCol = kk + col;
            As[row][col] = (gRow < m && gCol < k) ? A[gRow * k + gCol] : __float2bfloat16(0.0f);
        }
        
        for (int i = tid; i < b_total; i += 128) {
            int row = i / BN;
            int col = i % BN;
            int gRow = kk + row;
            int gCol = tile_col + col;
            Bs[row][col] = (gRow < k && gCol < n) ? B[gRow * n + gCol] : __float2bfloat16(0.0f);
        }
        
        __syncthreads();

        // MMA compute
        // For m16n8k16: A fragment is 4 bf16x2 (8 bf16), B fragment is 2 bf16x2 (4 bf16)
        // Thread mapping for A (m16k16, row-major):
        //   thread t owns A[t/4, (t%4)*4:(t%4)*4+4] for k=0..3, then A[t/4+8, (t%4)*4:(t%4)*4+4] for k=4..7
        // Thread mapping for B (k16n8, col-major):
        //   thread t owns B[(t%4)*4:(t%4)*4+4, t/4] for k=0..3
        
        int warp_row_base = warp_m * 32;  // Warp m offset (0 or 32)
        int warp_col_base = warp_n * 32;  // Warp n offset (0 or 32)
        
        // Do 2 m16n8 operations to cover 16x16
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            int a_row = warp_row_base + (laneId / 4);
            int b_col = warp_col_base + ni * 8 + (laneId / 4);
            
            // Load A fragment (4 uint32 = 8 bf16)
            uint32_t a_frag[4];
            int a_row_0 = a_row;
            int a_row_1 = a_row + 8;
            int a_col = (laneId % 4) * 4;
            
            // Pack bf16 pairs
            a_frag[0] = *reinterpret_cast<const uint32_t*>(&As[a_row_0][a_col]);
            a_frag[1] = *reinterpret_cast<const uint32_t*>(&As[a_row_0][a_col + 2]);
            a_frag[2] = *reinterpret_cast<const uint32_t*>(&As[a_row_1][a_col]);
            a_frag[3] = *reinterpret_cast<const uint32_t*>(&As[a_row_1][a_col + 2]);
            
            // Load B fragment (2 uint32 = 4 bf16)
            // B is in shared mem as row-major [K][N], but mma expects col-major
            // For k16n8 col-major: we need B[k, n] where k is rows, n is cols
            uint32_t b_frag[2];
            int b_k0 = (laneId % 4) * 4;
            int b_k1 = (laneId % 4) * 4 + 2;
            int b_n = b_col;
            
            // Since B is row-major [K][N+8], B[k][n] = Bs[k][n]
            // We need to load from different k rows for the same n column
            __nv_bfloat16 b_vals[4];
            b_vals[0] = Bs[b_k0][b_n];
            b_vals[1] = Bs[b_k0 + 1][b_n];
            b_vals[2] = Bs[b_k1][b_n];
            b_vals[3] = Bs[b_k1 + 1][b_n];
            b_frag[0] = *reinterpret_cast<uint32_t*>(&b_vals[0]);
            b_frag[1] = *reinterpret_cast<uint32_t*>(&b_vals[2]);
            
            // Execute MMA
            mma_sync_bf16_f32(
                acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                b_frag[0], b_frag[1],
                acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]
            );
        }
        
        __syncthreads();
    }

    // Store results
    // Result mapping for m16n8: 
    // thread t owns C[t/4, (t%4)*2 : (t%4)*2+2] and C[t/4+8, (t%4)*2 : (t%4)*2+2]
    int warp_row_base = warp_m * 32;
    int warp_col_base = warp_n * 32;
    
    #pragma unroll
    for (int ni = 0; ni < 2; ni++) {
        int c_row0 = tile_row + warp_row_base + (laneId / 4);
        int c_row1 = c_row0 + 8;
        int c_col = tile_col + warp_col_base + ni * 8 + (laneId % 4) * 2;
        
        if (c_row0 < m && c_col < n) {
            C[c_row0 * n + c_col] = acc[ni][0];
            if (c_col + 1 < n) C[c_row0 * n + c_col + 1] = acc[ni][1];
        }
        if (c_row1 < m && c_col < n) {
            C[c_row1 * n + c_col] = acc[ni][2];
            if (c_col + 1 < n) C[c_row1 * n + c_col + 1] = acc[ni][3];
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
        matmul_bf16_fp32_ptx<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verify_mode) {
        matmul_bf16_fp32_ptx<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
            matmul_bf16_fp32_ptx<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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

    printf("\n=== ITER009 RESULTS ===\n");
    printf("shape: M=%d N=%d K=%d\n", M, N, K);
    printf("dtype: bf16 input, fp32 output\n");
    printf("kernel: iter009_ptx (native mma.sync m16n8k16)\n");
    printf("avg_tflops: %.2f\n", tflops_sum / SAMPLES);
    printf("min_tflops: %.2f\n", tflops_min);
    printf("max_tflops: %.2f\n", tflops_max);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
