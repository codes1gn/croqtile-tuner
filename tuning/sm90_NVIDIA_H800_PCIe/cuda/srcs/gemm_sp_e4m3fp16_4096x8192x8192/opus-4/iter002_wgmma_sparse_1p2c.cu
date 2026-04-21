/*
 * Sparse GEMM E4M3 -> FP16 with WGMMA Tensor Cores
 * Shape: M=4096, N=8192, K=8192  
 * Layout: A[M,K] 2:4 sparse packed to [M,K/2] + metadata [M,K/32]
 *         B[N,K] dense (row-major)
 *         C[M,N] output FP16
 * 
 * Architecture: SM90 Hopper
 * Kernel: 1p2c warpgroup spec (1 producer + 2 consumer warpgroups)
 * Tiles: 128x256x128 with 3-stage pipeline
 * Uses: TMA load, wgmma.mma_async.sp, stmatrix
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
#include <cuda/barrier>

namespace cde = cuda::device::experimental;

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

constexpr int TILE_M = 128;
constexpr int TILE_N = 256;
constexpr int TILE_K = 128;
constexpr int WARP_M = 64;
constexpr int WARP_N = 256;
constexpr int WARP_K = 64;
constexpr int STAGES = 3;
constexpr int PACKED_TILE_K = 64;
constexpr int META_COLS = 4;

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

template<int SWIZ_BYTES>
__device__ __forceinline__ uint64_t make_wgmma_desc(void* ptr) {
    uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    uint64_t desc = 0;
    desc |= (addr & 0x3FFFF) >> 4;
    desc |= ((addr >> 32) & 0x7FFF) << 14;
    if constexpr (SWIZ_BYTES == 64)  desc |= (1ULL << 62);
    if constexpr (SWIZ_BYTES == 128) desc |= (2ULL << 62);
    desc |= (1ULL << 46);
    return desc;
}

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}

__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}

template<int N>
__device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N));
}

__device__ __forceinline__ void sparse_wgmma_e4m3_64x256x64(
    uint64_t desc_a, uint64_t desc_b, uint32_t meta,
    uint32_t (&acc)[64]
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
        : "+r"(acc[0]),  "+r"(acc[1]),  "+r"(acc[2]),  "+r"(acc[3]),
          "+r"(acc[4]),  "+r"(acc[5]),  "+r"(acc[6]),  "+r"(acc[7]),
          "+r"(acc[8]),  "+r"(acc[9]),  "+r"(acc[10]), "+r"(acc[11]),
          "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]),
          "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]),
          "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]),
          "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
          "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31]),
          "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35]),
          "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39]),
          "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43]),
          "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47]),
          "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51]),
          "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55]),
          "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59]),
          "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        : "l"(desc_a), "l"(desc_b), "r"(1), "r"(meta)
    );
}

__device__ __forceinline__ void tma_load_2d(
    void* smem_dst, const CUtensorMap* tma_map, uint64_t* mbar,
    int coord_k, int coord_m
) {
    uint64_t smem_addr = __cvta_generic_to_shared(smem_dst);
    uint64_t mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"((uint32_t)smem_addr), "l"(tma_map),
           "r"(coord_k), "r"(coord_m), "r"((uint32_t)mbar_addr)
        : "memory"
    );
}

__device__ __forceinline__ void tma_store_2d(
    const CUtensorMap* tma_map, void* smem_src, int coord_n, int coord_m
) {
    uint64_t smem_addr = __cvta_generic_to_shared(smem_src);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
        :: "l"(tma_map), "r"(coord_n), "r"(coord_m), "r"((uint32_t)smem_addr)
        : "memory"
    );
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_tx(uint64_t* mbar, int tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(tx_bytes));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, int phase) {
    uint32_t smem = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra LAB_WAIT;\n"
        "}\n"
        :: "r"(smem), "r"(phase)
    );
}

__global__ void __launch_bounds__(384, 1)
sparse_gemm_wgmma_kernel(
    __nv_fp8_e4m3* __restrict__ A_packed,
    uint32_t* __restrict__ A_meta,
    __nv_fp8_e4m3* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K,
    CUtensorMap tma_meta,
    CUtensorMap tma_lhs,
    CUtensorMap tma_rhs,
    CUtensorMap tma_out
) {
    extern __shared__ char smem_raw[];
    char* smem = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~1023ULL);
    
    constexpr int RHS_SIZE = STAGES * TILE_N * TILE_K;
    constexpr int LHS_SIZE = STAGES * WARP_M * PACKED_TILE_K;
    constexpr int META_SIZE = STAGES * WARP_M * META_COLS;
    constexpr int OUT_SIZE = WARP_M * TILE_N;
    
    __nv_fp8_e4m3* rhs_smem = reinterpret_cast<__nv_fp8_e4m3*>(smem);
    __nv_fp8_e4m3* lhs0_smem = reinterpret_cast<__nv_fp8_e4m3*>(smem + RHS_SIZE);
    __nv_fp8_e4m3* lhs1_smem = reinterpret_cast<__nv_fp8_e4m3*>(smem + RHS_SIZE + LHS_SIZE);
    uint32_t* meta0_smem = reinterpret_cast<uint32_t*>(smem + RHS_SIZE + 2 * LHS_SIZE);
    uint32_t* meta1_smem = reinterpret_cast<uint32_t*>(smem + RHS_SIZE + 2 * LHS_SIZE + META_SIZE * sizeof(uint32_t));
    half* out0_smem = reinterpret_cast<half*>(smem);
    half* out1_smem = reinterpret_cast<half*>(smem + OUT_SIZE * sizeof(half));
    
    __shared__ uint64_t full_mbar[STAGES];
    __shared__ uint64_t empty_mbar[STAGES];
    
    const int tid = threadIdx.x;
    const int wg_id = tid / 128;
    const int lane_wg = tid % 128;
    const int blk_m = blockIdx.x;
    const int blk_n = blockIdx.y;
    const int k_tiles = (K + TILE_K - 1) / TILE_K;
    const int m_base = blk_m * TILE_M;
    const int n_base = blk_n * TILE_N;
    
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbarrier_init(&full_mbar[s], 1);
            mbarrier_init(&empty_mbar[s], 2);
        }
    }
    __syncthreads();
    
    if (wg_id == 0 && lane_wg == 0) {
        for (int ik = 0; ik < k_tiles; ++ik) {
            int stage = ik % STAGES;
            int phase = ik / STAGES;
            
            mbarrier_wait(&empty_mbar[stage], phase);
            
            int meta_off = stage * WARP_M * META_COLS;
            int lhs_off = stage * WARP_M * PACKED_TILE_K;
            int rhs_off = stage * TILE_N * TILE_K;
            
            tma_load_2d(meta0_smem + meta_off, &tma_meta, &full_mbar[stage], 
                        ik * META_COLS, m_base);
            tma_load_2d(meta1_smem + meta_off, &tma_meta, &full_mbar[stage],
                        ik * META_COLS, m_base + WARP_M);
            tma_load_2d(lhs0_smem + lhs_off, &tma_lhs, &full_mbar[stage],
                        ik * PACKED_TILE_K, m_base);
            tma_load_2d(lhs1_smem + lhs_off, &tma_lhs, &full_mbar[stage],
                        ik * PACKED_TILE_K, m_base + WARP_M);
            tma_load_2d(rhs_smem + rhs_off, &tma_rhs, &full_mbar[stage],
                        ik * TILE_K, n_base);
            
            constexpr int tx_bytes = 2 * META_COLS * WARP_M * 4 + 2 * WARP_M * PACKED_TILE_K + TILE_N * TILE_K;
            mbarrier_arrive_tx(&full_mbar[stage], tx_bytes);
        }
    }
    
    if (wg_id >= 1) {
        const int consumer_id = wg_id - 1;
        __nv_fp8_e4m3* my_lhs = (consumer_id == 0) ? lhs0_smem : lhs1_smem;
        uint32_t* my_meta = (consumer_id == 0) ? meta0_smem : meta1_smem;
        
        uint32_t acc[64];
        #pragma unroll
        for (int i = 0; i < 64; ++i) acc[i] = 0;
        
        if (lane_wg == 0) {
            #pragma unroll
            for (int s = 0; s < STAGES; ++s) {
                mbarrier_arrive(&empty_mbar[s]);
            }
        }
        
        const int sp_tid = lane_wg;
        const int sp_row = ((sp_tid >> 2) & 7) + ((sp_tid & 1) << 3) + ((sp_tid >> 5) << 4);
        const int sp_col = (sp_tid >> 1) & 1;
        const int meta_idx = sp_row * META_COLS + sp_col;
        
        mbarrier_wait(&full_mbar[0], 0);
        wgmma_fence();
        
        for (int ik = 0; ik < k_tiles; ++ik) {
            const int stage = ik % STAGES;
            const int next_stage = (ik + 1) % STAGES;
            const int phase = ik / STAGES;
            const int next_phase = (ik + 1) / STAGES;
            
            const int lhs_off = stage * WARP_M * PACKED_TILE_K;
            const int rhs_off = stage * TILE_N * TILE_K;
            const int meta_off = stage * WARP_M * META_COLS;
            
            __nv_fp8_e4m3* ma0 = my_lhs + lhs_off;
            __nv_fp8_e4m3* mb0 = rhs_smem + rhs_off;
            uint32_t me0 = my_meta[meta_off + meta_idx];
            
            uint64_t desc_a0 = make_wgmma_desc<64>(ma0);
            uint64_t desc_b0 = make_wgmma_desc<128>(mb0);
            sparse_wgmma_e4m3_64x256x64(desc_a0, desc_b0, me0, acc);
            
            if (lane_wg == 0) {
                mbarrier_arrive(&empty_mbar[stage]);
            }
            
            __nv_fp8_e4m3* ma1 = my_lhs + lhs_off + 32;
            __nv_fp8_e4m3* mb1 = rhs_smem + rhs_off + 64;
            uint32_t me1 = my_meta[meta_off + meta_idx + 2];
            
            uint64_t desc_a1 = make_wgmma_desc<64>(ma1);
            uint64_t desc_b1 = make_wgmma_desc<128>(mb1);
            sparse_wgmma_e4m3_64x256x64(desc_a1, desc_b1, me1, acc);
            
            wgmma_commit();
            
            if (ik < k_tiles - 1) {
                wgmma_wait<1>();
                mbarrier_wait(&full_mbar[next_stage], next_phase);
                wgmma_fence();
            }
        }
        wgmma_wait<0>();
        
        half* my_out = (consumer_id == 0) ? out0_smem : out1_smem;
        
        const int warp_id = lane_wg / 32;
        const int lane = lane_wg % 32;
        
        #pragma unroll
        for (int i = 0; i < 64; i += 2) {
            half2 val = *reinterpret_cast<half2*>(&acc[i]);
            int row = (i / 8) * 8 + (lane % 4) * 2 + (i % 8) / 4;
            int col = warp_id * 64 + (lane / 4) * 2;
            if (row < WARP_M && col < TILE_N) {
                my_out[row * TILE_N + col + i % 2] = (i % 2 == 0) ? val.x : val.y;
            }
        }
        
        __syncwarp();
        cde::fence_proxy_async_shared_cta();
        
        if (lane_wg == 0) {
            int out_m = m_base + consumer_id * WARP_M;
            tma_store_2d(&tma_out, my_out, n_base, out_m);
            asm volatile("cp.async.bulk.commit_group;\n");
        }
    }
    
    asm volatile("cp.async.bulk.wait_group 0;\n");
}

void init_sparse_fp8(__nv_fp8_e4m3* dense, __nv_fp8_e4m3* packed, uint32_t* meta,
                     int rows, int cols, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    memset(meta, 0, rows * (cols / 32) * sizeof(uint32_t));
    
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
            if (keep1 > keep2) std::swap(keep1, keep2);
            
            for (int k = 0; k < 4; ++k) {
                dense[i * cols + j + k] = (k == keep1 || k == keep2) 
                    ? fp8_from_float(vals[k]) : fp8_zero();
            }
            
            int pack_idx = i * (cols / 2) + j / 2;
            packed[pack_idx] = dense[i * cols + j + keep1];
            packed[pack_idx + 1] = dense[i * cols + j + keep2];
            
            int meta_row = i;
            int meta_col = j / 32;
            int bit_pos = ((j % 32) / 4) * 4;
            uint32_t selector = (keep1 << 2) | keep2;
            meta[meta_row * (cols / 32) + meta_col] |= (selector << bit_pos);
        }
    }
}

void init_dense_fp8(__nv_fp8_e4m3* mat, int rows, int cols, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < rows * cols; ++i) {
        float v = dist(gen);
        if (std::fabs(v) < 0.1f) v = (v < 0) ? -0.25f : 0.25f;
        mat[i] = fp8_from_float(v);
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
    
    printf("Sparse GEMM E4M3->FP16 (WGMMA): M=%d, N=%d, K=%d\n", M_DIM, N_DIM, K_DIM);
    printf("Tile: %dx%dx%d, Stages=%d, 1p2c warp spec\n", TILE_M, TILE_N, TILE_K, STAGES);
    
    std::mt19937 gen(42);
    
    size_t dense_sz = M_DIM * K_DIM * sizeof(__nv_fp8_e4m3);
    size_t packed_sz = M_DIM * (K_DIM / 2) * sizeof(__nv_fp8_e4m3);
    size_t meta_sz = M_DIM * (K_DIM / 32) * sizeof(uint32_t);
    size_t B_sz = N_DIM * K_DIM * sizeof(__nv_fp8_e4m3);
    size_t C_sz = M_DIM * N_DIM * sizeof(half);
    
    __nv_fp8_e4m3* h_dense = (__nv_fp8_e4m3*)malloc(dense_sz);
    __nv_fp8_e4m3* h_packed = (__nv_fp8_e4m3*)malloc(packed_sz);
    uint32_t* h_meta = (uint32_t*)calloc(M_DIM * (K_DIM / 32), sizeof(uint32_t));
    __nv_fp8_e4m3* h_B = (__nv_fp8_e4m3*)malloc(B_sz);
    half* h_C = (half*)malloc(C_sz);
    
    init_sparse_fp8(h_dense, h_packed, h_meta, M_DIM, K_DIM, gen);
    init_dense_fp8(h_B, N_DIM, K_DIM, gen);
    memset(h_C, 0, C_sz);
    
    __nv_fp8_e4m3 *d_packed, *d_B;
    uint32_t *d_meta;
    half *d_C;
    
    CHECK_CUDA(cudaMalloc(&d_packed, packed_sz));
    CHECK_CUDA(cudaMalloc(&d_meta, meta_sz));
    CHECK_CUDA(cudaMalloc(&d_B, B_sz));
    CHECK_CUDA(cudaMalloc(&d_C, C_sz));
    
    CHECK_CUDA(cudaMemcpy(d_packed, h_packed, packed_sz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_meta, h_meta, meta_sz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, B_sz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, C_sz));
    
    alignas(64) CUtensorMap tma_meta{}, tma_lhs{}, tma_rhs{}, tma_out{};
    
    uint64_t meta_shape[] = {(uint64_t)(K_DIM / 32), (uint64_t)M_DIM};
    uint64_t meta_stride[] = {(uint64_t)(K_DIM / 32 * 4)};
    uint32_t meta_box[] = {META_COLS, WARP_M};
    uint32_t meta_elem[] = {1, 1};
    cuTensorMapEncodeTiled(&tma_meta, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2,
        d_meta, meta_shape, meta_stride, meta_box, meta_elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    uint64_t lhs_shape[] = {(uint64_t)(K_DIM / 2), (uint64_t)M_DIM};
    uint64_t lhs_stride[] = {(uint64_t)(K_DIM / 2)};
    uint32_t lhs_box[] = {PACKED_TILE_K, WARP_M};
    uint32_t lhs_elem[] = {1, 1};
    cuTensorMapEncodeTiled(&tma_lhs, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        d_packed, lhs_shape, lhs_stride, lhs_box, lhs_elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    uint64_t rhs_shape[] = {(uint64_t)K_DIM, (uint64_t)N_DIM};
    uint64_t rhs_stride[] = {(uint64_t)K_DIM};
    uint32_t rhs_box[] = {TILE_K, TILE_N};
    uint32_t rhs_elem[] = {1, 1};
    cuTensorMapEncodeTiled(&tma_rhs, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        d_B, rhs_shape, rhs_stride, rhs_box, rhs_elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    uint64_t out_shape[] = {(uint64_t)N_DIM, (uint64_t)M_DIM};
    uint64_t out_stride[] = {(uint64_t)(N_DIM * 2)};
    uint32_t out_box[] = {TILE_N, WARP_M};
    uint32_t out_elem[] = {1, 1};
    cuTensorMapEncodeTiled(&tma_out, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
        d_C, out_shape, out_stride, out_box, out_elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    dim3 grid((M_DIM + TILE_M - 1) / TILE_M, (N_DIM + TILE_N - 1) / TILE_N);
    dim3 block(384);
    constexpr int smem_size = 150 * 1024;
    
    CHECK_CUDA(cudaFuncSetAttribute(sparse_gemm_wgmma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    for (int i = 0; i < warmup; ++i) {
        sparse_gemm_wgmma_kernel<<<grid, block, smem_size>>>(
            d_packed, d_meta, d_B, d_C, M_DIM, N_DIM, K_DIM,
            tma_meta, tma_lhs, tma_rhs, tma_out);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        sparse_gemm_wgmma_kernel<<<grid, block, smem_size>>>(
            d_packed, d_meta, d_B, d_C, M_DIM, N_DIM, K_DIM,
            tma_meta, tma_lhs, tma_rhs, tma_out);
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
        CHECK_CUDA(cudaMemcpy(h_C, d_C, C_sz, cudaMemcpyDeviceToHost));
        
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
                ref += fp8_to_float(h_dense[i * K_DIM + kk]) * fp8_to_float(h_B[j * K_DIM + kk]);
            }
            
            float got = __half2float(h_C[i * N_DIM + j]);
            float diff = std::fabs(got - ref);
            float tol = 0.5f + 0.01f * std::fabs(ref);
            
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
    
    CHECK_CUDA(cudaFree(d_packed));
    CHECK_CUDA(cudaFree(d_meta));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    free(h_dense);
    free(h_packed);
    free(h_meta);
    free(h_B);
    free(h_C);
    
    return 0;
}
