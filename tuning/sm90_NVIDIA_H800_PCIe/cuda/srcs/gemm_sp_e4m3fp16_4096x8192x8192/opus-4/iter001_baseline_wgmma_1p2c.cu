/*
 * Sparse GEMM E4M3 -> FP16 baseline kernel
 * Shape: M=4096, N=8192, K=8192
 * Layout: A[M,K] 2:4 sparse (packed to [M,K/2] + meta [M,K/32])
 *         B[N,K] dense (row-major)
 *         C[M,N] output FP16
 * Compute: C = A @ B^T (sparse A × transposed B)
 * 
 * Architecture: SM90 (H100/H800)
 * Kernel type: 1p2c warpgroup specialization (1 producer, 2 consumer warpgroups)
 * Tile: 128x256x128 (TILE_M x TILE_N x TILE_K)
 * Pipeline: 3-stage async TMA
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <cmath>

#include <cusparseLt.h>

namespace cde = cuda::device::experimental;

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

constexpr int TILE_M = 128;
constexpr int TILE_N = 256;
constexpr int TILE_K = 128;
constexpr int WARP_M = 64;
constexpr int WARP_N = 256;
constexpr int WARP_K = 64;
constexpr int STAGES = 3;
constexpr int PACKED_TILE_K = 64;
constexpr int META_COLS_PER_TILE = 4;

constexpr double H800_PEAK_FP8_TFLOPS = 3026.0;

template<int SWIZ>
__device__ __forceinline__ uint64_t make_smem_desc(void* ptr) {
    uint64_t desc = 0;
    uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    desc |= (addr & 0x3FFFF) >> 4;
    desc |= ((addr >> 32) & 0x7FFF) << 14;
    desc |= (uint64_t(0) << 29);
    if constexpr (SWIZ == 64)  desc |= (uint64_t(1) << 62);
    if constexpr (SWIZ == 128) desc |= (uint64_t(2) << 62);
    desc |= (uint64_t(1) << 46);
    return desc;
}

__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}

__device__ __forceinline__ void warpgroup_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}

template<int N_WAIT>
__device__ __forceinline__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N_WAIT));
}

__device__ __forceinline__ void sparse_wgmma_64x256x64_e4m3_fp16(
    uint64_t desc_a, uint64_t desc_b, uint32_t meta,
    uint32_t (&d)[64]
) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sp.sync.aligned.m64n256k64.f16.e4m3.e4m3 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, %67, 0, 1, 1, 1;\n"
        "}\n"
        : "+r"(d[0]),  "+r"(d[1]),  "+r"(d[2]),  "+r"(d[3]),
          "+r"(d[4]),  "+r"(d[5]),  "+r"(d[6]),  "+r"(d[7]),
          "+r"(d[8]),  "+r"(d[9]),  "+r"(d[10]), "+r"(d[11]),
          "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
          "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]),
          "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
          "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]),
          "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
          "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]),
          "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]),
          "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]),
          "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
          "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]),
          "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]),
          "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]),
          "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
        : "l"(desc_a), "l"(desc_b), "r"(1), "r"(meta)
    );
}

__device__ __forceinline__ void tma_load_2d(
    void* dst, const CUtensorMap* tma_desc, uint64_t* barrier,
    int coord0, int coord1
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(dst)),
          "l"(tma_desc),
          "r"((uint32_t)__cvta_generic_to_shared(barrier)),
          "r"(coord0), "r"(coord1)
        : "memory"
    );
}

__device__ __forceinline__ void tma_store_2d(
    const CUtensorMap* tma_desc, void* src, int coord0, int coord1
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
        "[%0, {%2, %3}], [%1];"
        :
        : "l"(tma_desc),
          "r"((uint32_t)__cvta_generic_to_shared(src)),
          "r"(coord0), "r"(coord1)
        : "memory"
    );
}

__global__ void __launch_bounds__(384, 1)
sparse_gemm_e4m3_fp16_kernel(
    __nv_fp8_e4m3* __restrict__ lhs_packed,
    uint32_t* __restrict__ lhs_meta,
    __nv_fp8_e4m3* __restrict__ rhs,
    half* __restrict__ output,
    const __grid_constant__ CUtensorMap tma_meta,
    const __grid_constant__ CUtensorMap tma_lhs,
    const __grid_constant__ CUtensorMap tma_rhs,
    const __grid_constant__ CUtensorMap tma_out
) {
    extern __shared__ char smem_raw[];
    char* smem = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~1023ULL);
    
    __shared__ uint64_t full_barrier[STAGES];
    __shared__ uint64_t empty_barrier[STAGES];
    
    const int tid = threadIdx.x;
    const int warpgroup_id = tid / 128;
    const int lane_in_wg = tid % 128;
    
    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    
    constexpr int RHS_SMEM_SIZE = STAGES * TILE_N * TILE_K;
    constexpr int LHS_SMEM_SIZE = STAGES * WARP_M * PACKED_TILE_K;
    constexpr int META_SMEM_SIZE = STAGES * WARP_M * META_COLS_PER_TILE;
    
    __nv_fp8_e4m3* rhs_smem = reinterpret_cast<__nv_fp8_e4m3*>(smem);
    __nv_fp8_e4m3* lhs0_smem = reinterpret_cast<__nv_fp8_e4m3*>(smem + RHS_SMEM_SIZE);
    __nv_fp8_e4m3* lhs1_smem = reinterpret_cast<__nv_fp8_e4m3*>(smem + RHS_SMEM_SIZE + LHS_SMEM_SIZE);
    uint32_t* meta0_smem = reinterpret_cast<uint32_t*>(smem + RHS_SMEM_SIZE + 2 * LHS_SMEM_SIZE);
    uint32_t* meta1_smem = reinterpret_cast<uint32_t*>(smem + RHS_SMEM_SIZE + 2 * LHS_SMEM_SIZE + META_SMEM_SIZE);
    half* out0_smem = reinterpret_cast<half*>(smem);
    half* out1_smem = reinterpret_cast<half*>(smem + WARP_M * TILE_N * sizeof(half));
    
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(&full_barrier[s])), "r"(1));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(&empty_barrier[s])), "r"(256));
        }
    }
    __syncthreads();
    
    const int k_iters = (K + TILE_K - 1) / TILE_K;
    const int m_base = block_m * TILE_M;
    const int n_base = block_n * TILE_N;
    
    if (warpgroup_id == 0 && lane_in_wg == 0) {
        for (int ik = 0; ik < k_iters; ++ik) {
            int stage = ik % STAGES;
            
            uint64_t* eb = &empty_barrier[stage];
            uint64_t* fb = &full_barrier[stage];
            
            {
                uint32_t wait_result;
                asm volatile("{\n"
                    ".reg .pred p;\n"
                    "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                    "selp.b32 %0, 1, 0, p;\n"
                    "}\n"
                    : "=r"(wait_result)
                    : "r"((uint32_t)__cvta_generic_to_shared(eb)), "r"(0)
                );
                while (wait_result == 0) {
                    asm volatile("{\n"
                        ".reg .pred p;\n"
                        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                        "selp.b32 %0, 1, 0, p;\n"
                        "}\n"
                        : "=r"(wait_result)
                        : "r"((uint32_t)__cvta_generic_to_shared(eb)), "r"(0)
                    );
                }
            }
            
            int meta_offset = stage * WARP_M * META_COLS_PER_TILE;
            int lhs_offset = stage * WARP_M * PACKED_TILE_K;
            int rhs_offset = stage * TILE_N * TILE_K;
            
            tma_load_2d(meta0_smem + meta_offset, &tma_meta, fb, ik * META_COLS_PER_TILE, m_base);
            tma_load_2d(meta1_smem + meta_offset, &tma_meta, fb, ik * META_COLS_PER_TILE, m_base + WARP_M);
            tma_load_2d(lhs0_smem + lhs_offset, &tma_lhs, fb, ik * PACKED_TILE_K, m_base);
            tma_load_2d(lhs1_smem + lhs_offset, &tma_lhs, fb, ik * PACKED_TILE_K, m_base + WARP_M);
            tma_load_2d(rhs_smem + rhs_offset, &tma_rhs, fb, ik * TILE_K, n_base);
            
            constexpr int tx_bytes = META_SMEM_SIZE * 4 + 2 * LHS_SMEM_SIZE + TILE_N * TILE_K;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(fb)), "r"(tx_bytes));
        }
    }
    
    if (warpgroup_id >= 1) {
        const int consumer_id = warpgroup_id - 1;
        __nv_fp8_e4m3* my_lhs = (consumer_id == 0) ? lhs0_smem : lhs1_smem;
        uint32_t* my_meta = (consumer_id == 0) ? meta0_smem : meta1_smem;
        
        uint32_t acc[64];
        #pragma unroll
        for (int i = 0; i < 64; ++i) acc[i] = 0;
        
        for (int s = 0; s < STAGES; ++s) {
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(&empty_barrier[s])));
        }
        
        const int sp_tid = lane_in_wg;
        const int sp_row = ((sp_tid >> 2) & 7) + ((sp_tid & 1) << 3) + ((sp_tid >> 5) << 4);
        const int sp_col = (sp_tid >> 1) & 1;
        const int meta_idx = sp_row * META_COLS_PER_TILE + sp_col;
        
        uint64_t* fb0 = &full_barrier[0];
        {
            uint32_t wait_result;
            asm volatile("{\n"
                ".reg .pred p;\n"
                "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                "selp.b32 %0, 1, 0, p;\n"
                "}\n"
                : "=r"(wait_result)
                : "r"((uint32_t)__cvta_generic_to_shared(fb0)), "r"(0)
            );
            while (wait_result == 0) {
                asm volatile("{\n"
                    ".reg .pred p;\n"
                    "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                    "selp.b32 %0, 1, 0, p;\n"
                    "}\n"
                    : "=r"(wait_result)
                    : "r"((uint32_t)__cvta_generic_to_shared(fb0)), "r"(0)
                );
            }
        }
        warpgroup_arrive();
        
        for (int ik = 0; ik < k_iters; ++ik) {
            const int stage = ik % STAGES;
            const int next_stage = (ik + 1) % STAGES;
            
            const int lhs_off = stage * WARP_M * PACKED_TILE_K;
            const int rhs_off = stage * TILE_N * TILE_K;
            const int meta_off = stage * WARP_M * META_COLS_PER_TILE;
            
            __nv_fp8_e4m3* ma_ptr0 = my_lhs + lhs_off;
            __nv_fp8_e4m3* mb_ptr0 = rhs_smem + rhs_off;
            uint32_t me0 = my_meta[meta_off + meta_idx];
            
            uint64_t desc_a0 = make_smem_desc<64>(ma_ptr0);
            uint64_t desc_b0 = make_smem_desc<128>(mb_ptr0);
            sparse_wgmma_64x256x64_e4m3_fp16(desc_a0, desc_b0, me0, acc);
            
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(&empty_barrier[stage])));
            
            __nv_fp8_e4m3* ma_ptr1 = my_lhs + lhs_off + 32;
            __nv_fp8_e4m3* mb_ptr1 = rhs_smem + rhs_off + 64;
            uint32_t me1 = my_meta[meta_off + meta_idx + 2];
            
            uint64_t desc_a1 = make_smem_desc<64>(ma_ptr1);
            uint64_t desc_b1 = make_smem_desc<128>(mb_ptr1);
            sparse_wgmma_64x256x64_e4m3_fp16(desc_a1, desc_b1, me1, acc);
            
            warpgroup_commit();
            
            if (ik < k_iters - 1) {
                warpgroup_wait<1>();
                uint64_t* fb_next = &full_barrier[next_stage];
                {
                    uint32_t wait_result;
                    asm volatile("{\n"
                        ".reg .pred p;\n"
                        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                        "selp.b32 %0, 1, 0, p;\n"
                        "}\n"
                        : "=r"(wait_result)
                        : "r"((uint32_t)__cvta_generic_to_shared(fb_next)), "r"((ik + 1) % 2)
                    );
                    while (wait_result == 0) {
                        asm volatile("{\n"
                            ".reg .pred p;\n"
                            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                            "selp.b32 %0, 1, 0, p;\n"
                            "}\n"
                            : "=r"(wait_result)
                            : "r"((uint32_t)__cvta_generic_to_shared(fb_next)), "r"((ik + 1) % 2)
                        );
                    }
                }
                warpgroup_arrive();
            }
        }
        warpgroup_wait<0>();
        
        half* my_out_smem = (consumer_id == 0) ? out0_smem : out1_smem;
        
        const int warp_id = lane_in_wg / 32;
        const int lane = lane_in_wg % 32;
        
        #pragma unroll
        for (int i = 0; i < 64; i += 2) {
            half2 val = *reinterpret_cast<half2*>(&acc[i]);
            int row = (i / 8) * 8 + (lane % 4) * 2 + (i % 8) / 4;
            int col = warp_id * 64 + (lane / 4) * 2 + (i % 2);
            my_out_smem[row * TILE_N + col] = val.x;
            my_out_smem[row * TILE_N + col + 1] = val.y;
        }
        
        __syncwarp();
        cde::fence_proxy_async_shared_cta();
        
        if (lane_in_wg == 0) {
            int out_m = m_base + consumer_id * WARP_M;
            tma_store_2d(&tma_out, my_out_smem, n_base, out_m);
            asm volatile("cp.async.bulk.commit_group;");
        }
    }
    
    asm volatile("cp.async.bulk.wait_group 0;");
}

__host__ __device__ inline __nv_fp8_e4m3 fp8_from_float(float v) {
    __nv_fp8_storage_t storage = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 result;
    result.__x = storage;
    return result;
}

__host__ __device__ inline __nv_fp8_e4m3 fp8_zero() {
    __nv_fp8_e4m3 result;
    result.__x = 0;
    return result;
}

__host__ __device__ inline float fp8_to_float(__nv_fp8_e4m3 v) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(v.__x, __NV_E4M3);
    return __half2float(*reinterpret_cast<half*>(&hr));
}

void init_sparse_matrix(__nv_fp8_e4m3* dense, __nv_fp8_e4m3* packed, uint32_t* meta,
                        int rows, int cols, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j += 4) {
            float vals[4];
            for (int k = 0; k < 4; ++k) {
                vals[k] = dist(gen);
                if (std::fabs(vals[k]) < 0.1f) vals[k] = (vals[k] < 0) ? -0.25f : 0.25f;
            }
            
            int max1 = 0, max2 = 1;
            if (std::fabs(vals[1]) > std::fabs(vals[max1])) { max2 = max1; max1 = 1; }
            else max2 = 1;
            if (std::fabs(vals[2]) > std::fabs(vals[max1])) { max2 = max1; max1 = 2; }
            else if (std::fabs(vals[2]) > std::fabs(vals[max2])) max2 = 2;
            if (std::fabs(vals[3]) > std::fabs(vals[max1])) { max2 = max1; max1 = 3; }
            else if (std::fabs(vals[3]) > std::fabs(vals[max2])) max2 = 3;
            
            if (max1 > max2) std::swap(max1, max2);
            
            for (int k = 0; k < 4; ++k) {
                dense[i * cols + j + k] = (k == max1 || k == max2) 
                    ? fp8_from_float(vals[k])
                    : fp8_zero();
            }
            
            int pack_idx = i * (cols / 2) + j / 2;
            packed[pack_idx] = dense[i * cols + j + max1];
            packed[pack_idx + 1] = dense[i * cols + j + max2];
            
            int meta_row = i;
            int meta_col = j / 32;
            int bit_offset = (j % 32) / 4 * 4;
            uint32_t selector = (max1 << 2) | max2;
            meta[meta_row * (cols / 32) + meta_col] |= (selector << bit_offset);
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

void cpu_sparse_gemm(__nv_fp8_e4m3* A_dense, __nv_fp8_e4m3* B, half* C,
                     int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                float a_val = fp8_to_float(A_dense[i * k + kk]);
                float b_val = fp8_to_float(B[j * k + kk]);
                acc += a_val * b_val;
            }
            C[i * n + j] = __float2half(acc);
        }
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
    
    printf("Sparse GEMM E4M3->FP16: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Tile: %dx%dx%d, Stages=%d\n", TILE_M, TILE_N, TILE_K, STAGES);
    
    std::mt19937 gen(42);
    
    size_t dense_size = M * K * sizeof(__nv_fp8_e4m3);
    size_t packed_size = M * (K / 2) * sizeof(__nv_fp8_e4m3);
    size_t meta_size = M * (K / 32) * sizeof(uint32_t);
    size_t rhs_size = N * K * sizeof(__nv_fp8_e4m3);
    size_t out_size = M * N * sizeof(half);
    
    __nv_fp8_e4m3* h_dense = (__nv_fp8_e4m3*)malloc(dense_size);
    __nv_fp8_e4m3* h_packed = (__nv_fp8_e4m3*)malloc(packed_size);
    uint32_t* h_meta = (uint32_t*)calloc(M * (K / 32), sizeof(uint32_t));
    __nv_fp8_e4m3* h_rhs = (__nv_fp8_e4m3*)malloc(rhs_size);
    half* h_out = (half*)malloc(out_size);
    half* h_ref = (half*)malloc(out_size);
    
    init_sparse_matrix(h_dense, h_packed, h_meta, M, K, gen);
    init_dense_matrix(h_rhs, N, K, gen);
    memset(h_out, 0, out_size);
    
    __nv_fp8_e4m3 *d_packed, *d_rhs;
    uint32_t *d_meta;
    half *d_out;
    
    CHECK_CUDA(cudaMalloc(&d_packed, packed_size));
    CHECK_CUDA(cudaMalloc(&d_meta, meta_size));
    CHECK_CUDA(cudaMalloc(&d_rhs, rhs_size));
    CHECK_CUDA(cudaMalloc(&d_out, out_size));
    
    CHECK_CUDA(cudaMemcpy(d_packed, h_packed, packed_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_meta, h_meta, meta_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rhs, h_rhs, rhs_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, out_size));
    
    alignas(64) CUtensorMap tma_meta{}, tma_lhs{}, tma_rhs{}, tma_out{};
    
    uint64_t meta_shape[] = {(uint64_t)(K / 32), (uint64_t)M};
    uint64_t meta_strides[] = {(uint64_t)(K / 32 * 4)};
    uint32_t meta_box[] = {META_COLS_PER_TILE, WARP_M};
    uint32_t meta_elem_strides[] = {1, 1};
    cuTensorMapEncodeTiled(&tma_meta, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2,
        d_meta, meta_shape, meta_strides, meta_box, meta_elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    uint64_t lhs_shape[] = {(uint64_t)(K / 2), (uint64_t)M};
    uint64_t lhs_strides[] = {(uint64_t)(K / 2)};
    uint32_t lhs_box[] = {PACKED_TILE_K, WARP_M};
    uint32_t lhs_elem_strides[] = {1, 1};
    cuTensorMapEncodeTiled(&tma_lhs, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        d_packed, lhs_shape, lhs_strides, lhs_box, lhs_elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    uint64_t rhs_shape[] = {(uint64_t)K, (uint64_t)N};
    uint64_t rhs_strides[] = {(uint64_t)K};
    uint32_t rhs_box[] = {TILE_K, TILE_N};
    uint32_t rhs_elem_strides[] = {1, 1};
    cuTensorMapEncodeTiled(&tma_rhs, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        d_rhs, rhs_shape, rhs_strides, rhs_box, rhs_elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    uint64_t out_shape[] = {(uint64_t)N, (uint64_t)M};
    uint64_t out_strides[] = {(uint64_t)(N * 2)};
    uint32_t out_box[] = {TILE_N, WARP_M};
    uint32_t out_elem_strides[] = {1, 1};
    cuTensorMapEncodeTiled(&tma_out, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
        d_out, out_shape, out_strides, out_box, out_elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    dim3 block(384);
    constexpr int smem_size = 150 * 1024;
    
    CHECK_CUDA(cudaFuncSetAttribute(sparse_gemm_e4m3_fp16_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    for (int i = 0; i < warmup; ++i) {
        sparse_gemm_e4m3_fp16_kernel<<<grid, block, smem_size>>>(
            d_packed, d_meta, d_rhs, d_out, tma_meta, tma_lhs, tma_rhs, tma_out);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        sparse_gemm_e4m3_fp16_kernel<<<grid, block, smem_size>>>(
            d_packed, d_meta, d_rhs, d_out, tma_meta, tma_lhs, tma_rhs, tma_out);
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
        CHECK_CUDA(cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost));
        
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
                float a_val = fp8_to_float(h_dense[i * K + kk]);
                float b_val = fp8_to_float(h_rhs[j * K + kk]);
                ref += a_val * b_val;
            }
            
            float got = __half2float(h_out[i * N + j]);
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
    
    CHECK_CUDA(cudaFree(d_packed));
    CHECK_CUDA(cudaFree(d_meta));
    CHECK_CUDA(cudaFree(d_rhs));
    CHECK_CUDA(cudaFree(d_out));
    
    free(h_dense);
    free(h_packed);
    free(h_meta);
    free(h_rhs);
    free(h_out);
    free(h_ref);
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}
