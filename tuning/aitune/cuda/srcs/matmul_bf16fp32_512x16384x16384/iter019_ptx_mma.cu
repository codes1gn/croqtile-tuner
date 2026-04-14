#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// M16N8K16 mma.sync for bf16->fp32
// Each warp computes 16x8 output
// We tile warps to cover larger areas

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;
constexpr int WARP_M = 16;
constexpr int WARP_N = 16;  // 2 mma.sync calls horizontally
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int SMEM_PAD = 8;

// PTX mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
__device__ __forceinline__ void mma_m16n8k16_bf16_f32(
    float* d, const unsigned* a, const unsigned* b, const float* c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// Load 8x8 bf16 tile from shared memory to registers
// Each thread loads 4 bf16 values (2 per register)
__device__ __forceinline__ void load_a_fragment(
    unsigned* regs, const __nv_bfloat16* sA, int k_offset, int lane_id)
{
    // For m16k16 matrix A (row-major in shared):
    // lane 0-3: row 0-3, col 0-7
    // lane 4-7: row 4-7, col 0-7
    // etc.
    int row_in_warp = (lane_id / 4) * 2 + (lane_id % 4) / 2;
    int col_start = (lane_id % 2) * 8 + k_offset;
    
    // A fragment: 4 registers, each holding 2 bf16
    for (int i = 0; i < 4; i++) {
        int row = row_in_warp + (i / 2) * 8;
        int col = col_start + (i % 2) * 2;
        const __nv_bfloat16* ptr = &sA[row * (BLOCK_K + SMEM_PAD) + col];
        regs[i] = *reinterpret_cast<const unsigned*>(ptr);
    }
}

// Load 8x8 bf16 tile for B from shared memory
__device__ __forceinline__ void load_b_fragment(
    unsigned* regs, const __nv_bfloat16* sB, int k_offset, int n_offset, int lane_id)
{
    // For n8k16 matrix B (col-major in shared):
    // Each register holds 2 bf16 from the same column
    int col_in_tile = (lane_id % 4) + n_offset;
    int row_start = (lane_id / 4) * 2 + k_offset;
    
    for (int i = 0; i < 2; i++) {
        int row = row_start + i * 8;
        const __nv_bfloat16* ptr = &sB[col_in_tile * (BLOCK_K + SMEM_PAD) + row];
        regs[i] = *reinterpret_cast<const unsigned*>(ptr);
    }
}

__global__ __launch_bounds__(256, 2)
void matmul_ptx_mma(const __nv_bfloat16* __restrict__ A,
                    const __nv_bfloat16* __restrict__ B,
                    float* __restrict__ C,
                    int M, int N, int K)
{
    __shared__ __nv_bfloat16 sA[BLOCK_M][BLOCK_K + SMEM_PAD];
    __shared__ __nv_bfloat16 sB[BLOCK_N][BLOCK_K + SMEM_PAD];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int tid = threadIdx.x;
    
    // 8 warps per block: 4 in M dim, 2 in N dim
    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;
    
    int m_base = by * BLOCK_M + warp_m * WARP_M;
    int n_base = bx * BLOCK_N + warp_n * 32;
    
    // Initialize accumulators
    float acc[4][4] = {{0}};  // 4 mma tiles: 2 in M x 2 in N
    
    unsigned a_frag[4];
    unsigned b_frag[2];
    
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // Cooperative load A
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += 256) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int g_row = by * BLOCK_M + row;
            int g_col = k_tile + col;
            sA[row][col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : __float2bfloat16(0.0f);
        }
        
        // Cooperative load B (transpose to col-major)
        for (int i = tid; i < BLOCK_N * BLOCK_K; i += 256) {
            int col = i / BLOCK_K;
            int row = i % BLOCK_K;
            int g_row = k_tile + row;
            int g_col = bx * BLOCK_N + col;
            sB[col][row] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : __float2bfloat16(0.0f);
        }
        
        __syncthreads();
        
        // Compute: each warp does 2x2 grid of mma.sync
        for (int mma_m = 0; mma_m < 2; mma_m++) {
            for (int mma_n = 0; mma_n < 2; mma_n++) {
                // Load A fragment for this mma tile
                load_a_fragment(a_frag, &sA[warp_m * WARP_M + mma_m * MMA_M][0], 0, lane_id);
                
                // Load B fragment  
                load_b_fragment(b_frag, &sB[warp_n * 32 + mma_n * MMA_N][0], 0, 0, lane_id);
                
                // Execute mma
                mma_m16n8k16_bf16_f32(&acc[mma_m * 2 + mma_n][0], a_frag, b_frag, &acc[mma_m * 2 + mma_n][0]);
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    // Each thread in warp stores 4 values from its accumulators
    for (int mma_m_idx = 0; mma_m_idx < 2; mma_m_idx++) {
        for (int mma_n_idx = 0; mma_n_idx < 2; mma_n_idx++) {
            int acc_idx = mma_m_idx * 2 + mma_n_idx;
            
            // mma.sync output mapping:
            // Each thread holds 4 values from 2 rows x 2 columns
            int row_in_mma = (lane_id / 4) * 2;
            int col_in_mma = ((lane_id % 4) * 2);
            
            int g_row = m_base + mma_m_idx * MMA_M + row_in_mma;
            int g_col = n_base + mma_n_idx * MMA_N + col_in_mma;
            
            if (g_row < M && g_col < N) {
                C[g_row * N + g_col] = acc[acc_idx][0];
            }
            if (g_row < M && g_col + 1 < N) {
                C[g_row * N + g_col + 1] = acc[acc_idx][1];
            }
            if (g_row + 1 < M && g_col < N) {
                C[(g_row + 1) * N + g_col] = acc[acc_idx][2];
            }
            if (g_row + 1 < M && g_col + 1 < N) {
                C[(g_row + 1) * N + g_col + 1] = acc[acc_idx][3];
            }
        }
    }
}

int main(int argc, char** argv) {
    int M = 512, N = 16384, K = 16384;
    int warmup = (argc > 1) ? atoi(argv[1]) : 3;
    int iters = (argc > 2) ? atoi(argv[2]) : 10;
    
    size_t sizeA = M * K * sizeof(__nv_bfloat16);
    size_t sizeB = K * N * sizeof(__nv_bfloat16);
    size_t sizeC = M * N * sizeof(float);
    
    __nv_bfloat16 *dA, *dB;
    float *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);
    
    // Init random
    __nv_bfloat16* hA = new __nv_bfloat16[M * K];
    __nv_bfloat16* hB = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; i++) hA[i] = __float2bfloat16((rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) hB[i] = __float2bfloat16((rand() % 100) / 100.0f);
    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);
    delete[] hA;
    delete[] hB;
    
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(256);
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        matmul_ptx_mma<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();
    
    // Timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        matmul_ptx_mma<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    ms /= iters;
    
    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("Time: %.3f ms, TFLOPS: %.2f\n", ms, tflops);
    
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
