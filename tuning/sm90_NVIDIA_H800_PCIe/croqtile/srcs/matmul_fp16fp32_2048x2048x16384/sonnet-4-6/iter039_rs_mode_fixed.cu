// matmul_fp16fp32_2048x2048x16384 iter039: Fixed RS mode kernel
// 
// Key optimization: Load A directly from global memory to registers
// - No TMA/SMEM staging for A
// - Only B uses TMA to SMEM
// - WGMMA RS mode: A from registers, B from SMEM descriptor
//
// Fixes from iter038:
// 1. Fixed barrier deadlock - removed double arrive/wait
// 2. Corrected register layout for ALayout_64x16
// 3. Simplified pipeline - single warpgroup, all threads compute
//
// Register Layout for ALayout_64x16 (from CuTe):
// Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,  _2>>,
//        Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>
//
// For thread t (0-127), the 8 f16 values (4 uint32_t) are:
//   reg[v] where v in [0,8) maps to matrix element A[m,k] via:
//   Linear coord in [0, 64*16) = 128*m0 + 1*k0 + 16*m1 + 64*v0 + 8*v1 + 512*v2
//   where t = (m0,k0,m1) and v = (v0,v1,v2)
//
// Simpler interpretation: 
//   t % 8 = k_base (within K=16: elements 0-7 for lane 0-7)
//   t / 8 = m_base (within M=64: 16 warps of 8 threads each)

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "choreo.h"
#include <cute/arch/mma_sm90_gmma.hpp>
namespace cde = cuda::device::experimental;
#include <cooperative_groups.h>
using namespace choreo;

#define H800_PCIE_PEAK_F16_TFLOPS 1513

#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 128
#define MATMUL_TILE_M 64      // Single WGMMA M
#define MATMUL_TILE_N 128     // Full N tile
#define MATMUL_TILE_K 64      // K tile for pipeline
#define MATMUL_WARP_K 16      // K per WGMMA instruction
#define MATMUL_SWIZ 128
#define MATMUL_STAGES 4       // Can afford more stages without A in SMEM

#define MATMUL_DEFAULT_M 2048
#define MATMUL_DEFAULT_N 2048
#define MATMUL_DEFAULT_K 16384

// ALayout_64x16 interpretation:
// For MMA_64x128x16_F32F16F16_RS with K-major A:
// 
// The 64x16 A matrix is distributed across 128 threads.
// Each thread holds 4 uint32_t = 8 f16 values.
// 
// Thread layout within warpgroup (128 threads):
// - 4 warps × 32 threads = 128 threads
// - Within M=64: each warp handles M=16 rows
// - Within K=16: distributed across lanes
//
// For K-major layout, consecutive K elements are contiguous.
// Thread t owns: row_base = (t/4) % 16 within warp's M=16 section
//                col_base determined by (t%4) and warp_id
//
// Specifically for m64n128k16 RS with f16:
// The 4 registers per thread pack 8 f16 values as:
//   a0 = A[row0, k0:k1], a1 = A[row0, k2:k3]
//   a2 = A[row1, k0:k1], a3 = A[row1, k2:k3]
// where row0, row1 are 8 rows apart within the thread's M section

__device__ __forceinline__ void load_a_gmem_to_reg_fixed(
    const f16* __restrict__ gmem_a,  // Global memory pointer [M, K] row-major
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    int M_offset,                     // M offset for this CTA's tile
    int K_offset,                     // K offset for this WGMMA K16 iteration
    int K_stride) {                   // Full K dimension stride

  // Thread ID within warpgroup (0-127)
  int tid = threadIdx.x % 128;
  
  // Decode ALayout_64x16:
  // Shape:  ((4,8,4), (2,2,2))
  // Stride: ((128,1,16), (64,8,512))
  //
  // Thread index t maps to (m0, k0, m1) where:
  //   t = m0 * 128 + k0 * 1 + m1 * 16 (mod 128)
  //   But this is inverted - we need to find which elements thread t owns.
  //
  // Actually, for RS mode, the layout says:
  //   element at (m,k) is owned by thread t = (m/16)*16 + (k%8) + ((m%16)/4)*4
  //   where each thread owns values at specific (m,k) positions.
  //
  // Let's use the standard RS mode mapping for M64K16:
  // Each warp (32 threads) handles 16 rows of A.
  // Lane within warp determines which K elements.
  
  int warp_id = tid / 32;        // 0-3, each warp handles M=16 rows
  int lane_id = tid % 32;        // 0-31
  
  // For K-major RS mode m64n128k16:
  // Lane determines K position, grouped in pairs
  int k_pair = lane_id % 4;      // Which K pair (0-3) -> K positions 0,4,8,12
  int m_group = lane_id / 4;     // Which M group (0-7)
  
  // Each thread owns 2 rows, 4 K-elements each
  // Row positions: row0 = warp_id*16 + m_group, row1 = row0 + 8
  int row0 = warp_id * 16 + m_group;
  int row1 = row0 + 8;
  
  // K positions: 4 consecutive starting at k_pair*4
  int k_base = k_pair * 4;
  
  // Global addresses
  int global_row0 = M_offset + row0;
  int global_row1 = M_offset + row1;
  int global_k = K_offset + k_base;
  
  // Load 4 f16 values per row (2 uint32_t per row)
  const f16* ptr_r0 = gmem_a + global_row0 * K_stride + global_k;
  const f16* ptr_r1 = gmem_a + global_row1 * K_stride + global_k;
  
  // Pack into uint32_t (2 f16 per uint32_t)
  a0 = *reinterpret_cast<const uint32_t*>(ptr_r0);      // row0, k+0:k+1
  a1 = *reinterpret_cast<const uint32_t*>(ptr_r0 + 2);  // row0, k+2:k+3
  a2 = *reinterpret_cast<const uint32_t*>(ptr_r1);      // row1, k+0:k+1
  a3 = *reinterpret_cast<const uint32_t*>(ptr_r1 + 2);  // row1, k+2:k+3
}

__global__ void __launch_bounds__(128, 2) matmul_rs_mode_kernel(
    const f16* __restrict__ lhs,    // A: [M, K]
    const f16* __restrict__ rhs,    // B: [N, K] row-major (K-major for WGMMA)
    float* __restrict__ output,      // C: [M, N]
    int M, int N, int K) {
  
  // Grid: [cdiv(M, TILE_M), cdiv(N, TILE_N)]
  int block_m = blockIdx.x;
  int block_n = blockIdx.y;
  
  extern __shared__ char smem[];
  // B in shared memory: STAGES * 128 * 64 * 2B = STAGES * 16KB
  f16* rhs_smem = reinterpret_cast<f16*>(smem);
  // Output staging: 64 * 128 * 4B = 32KB
  float* output_smem = reinterpret_cast<float*>(smem + MATMUL_STAGES * MATMUL_TILE_N * MATMUL_TILE_K * sizeof(f16));
  
  // Initialize accumulators
  float mc[64];
  #pragma unroll
  for (int i = 0; i < 64; ++i) mc[i] = 0.0f;
  
  // Global offsets
  int m_base = block_m * MATMUL_TILE_M;
  int n_base = block_n * MATMUL_TILE_N;
  int k_tiles = (K + MATMUL_TILE_K - 1) / MATMUL_TILE_K;
  
  int tid = threadIdx.x;
  
  // Main loop - simplified without producer/consumer split for now
  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    int stage = k_tile % MATMUL_STAGES;
    int k_base = k_tile * MATMUL_TILE_K;
    
    // Load B tile to SMEM (cooperative load, simplified)
    f16* b_smem_stage = rhs_smem + stage * MATMUL_TILE_N * MATMUL_TILE_K;
    int elements_per_thread = (MATMUL_TILE_N * MATMUL_TILE_K) / 128;
    
    #pragma unroll 4
    for (int i = 0; i < elements_per_thread; ++i) {
      int idx = tid * elements_per_thread + i;
      int b_n = idx / MATMUL_TILE_K;
      int b_k = idx % MATMUL_TILE_K;
      if (n_base + b_n < N && k_base + b_k < K) {
        b_smem_stage[idx] = rhs[(n_base + b_n) * K + k_base + b_k];
      } else {
        b_smem_stage[idx] = __float2half(0.0f);
      }
    }
    __syncthreads();
    
    // K-loop: 4 WGMMA iterations per K tile (64/16 = 4)
    #pragma unroll
    for (int k_iter = 0; k_iter < 4; ++k_iter) {
      int k_offset = k_base + k_iter * MATMUL_WARP_K;
      
      // Load A from global memory to registers
      uint32_t a0, a1, a2, a3;
      load_a_gmem_to_reg_fixed(lhs, a0, a1, a2, a3, m_base, k_offset, K);
      
      // Create B descriptor for SMEM
      // B is stored K-major: [N, K] with K contiguous
      // For WGMMA, we need N×K tile at current k_iter position
      f16* b_smem_ptr = b_smem_stage + k_iter * MATMUL_WARP_K;
      
      // Make SMEM descriptor for B
      // B layout in SMEM: [128, 64] K-contiguous, we take [128, 16] slice
      // Swizzle pattern B128 for optimal access
      uint64_t desc_b = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(b_smem_ptr);
      
      // WGMMA RS mode
      warpgroup_arrive();
      cute::SM90::GMMA::MMA_64x128x16_F32F16F16_RS<
          cute::SM90::GMMA::Major::K,   // A is K-major in registers
          cute::SM90::GMMA::Major::K    // B is K-major in SMEM
      >::fma(a0, a1, a2, a3, desc_b,
             mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7],
             mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15],
             mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23],
             mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31],
             mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39],
             mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47],
             mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55],
             mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63]);
      warpgroup_commit_batch();
    }
    
    warpgroup_wait<0>();
    __syncthreads();  // Ensure B is consumed before next stage overwrites
  }
  
  // Store output via shared memory
  auto shape_out = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
  auto stride_out = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
  auto layout_out = cute::make_layout(shape_out, stride_out);
  auto tensor_out = cute::make_tensor(cute::make_smem_ptr<float>(output_smem), layout_out);
  store_fragment_d<CUTE_WGMMA_M64K16, 128>(tensor_out, mc);
  __syncthreads();
  
  // Copy from SMEM to global
  int elements_per_thread_out = (MATMUL_TILE_M * MATMUL_TILE_N) / 128;
  #pragma unroll 4
  for (int i = 0; i < elements_per_thread_out; ++i) {
    int idx = tid * elements_per_thread_out + i;
    int out_m = idx / MATMUL_TILE_N;
    int out_n = idx % MATMUL_TILE_N;
    if (m_base + out_m < M && n_base + out_n < N) {
      output[(m_base + out_m) * N + n_base + out_n] = output_smem[idx];
    }
  }
}

int main(int argc, char** argv) {
  bool enable_timing = true;
  bool skip_verify = false;

  int M = MATMUL_DEFAULT_M;
  int N = MATMUL_DEFAULT_N;
  int K = MATMUL_DEFAULT_K;

  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--disable-timing", 16) == 0) { enable_timing = false; continue; }
    if (std::strncmp(argv[i], "--skip-verify", 13) == 0)    { skip_verify = true; continue; }
    if (std::strncmp(argv[i], "--m=", 4) == 0) { M = std::atoi(argv[i] + 4); continue; }
    if (std::strncmp(argv[i], "--n=", 4) == 0) { N = std::atoi(argv[i] + 4); continue; }
    if (std::strncmp(argv[i], "--k=", 4) == 0) { K = std::atoi(argv[i] + 4); continue; }
  }

  const char* sv = std::getenv("CHOREO_SKIP_VERIFY");
  if (sv && sv[0] == '1' && sv[1] == '\0') skip_verify = true;

  // Allocate memory
  std::vector<half> lhs_h(M * K), rhs_h(N * K);
  std::vector<float> res_h(M * N, 0.0f);
  
  // Initialize with random values
  srand(42);
  for (auto& v : lhs_h) v = __float2half((rand() / float(RAND_MAX) - 0.5f) * 2.0f);
  for (auto& v : rhs_h) v = __float2half((rand() / float(RAND_MAX) - 0.5f) * 2.0f);

  half *a_d, *b_d;
  float *c_d;
  cudaMalloc(&a_d, M * K * sizeof(half));
  cudaMalloc(&b_d, N * K * sizeof(half));
  cudaMalloc(&c_d, M * N * sizeof(float));
  cudaMemcpy(a_d, lhs_h.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, rhs_h.data(), N * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemset(c_d, 0, M * N * sizeof(float));

  // Grid and block dimensions
  dim3 grid((M + MATMUL_TILE_M - 1) / MATMUL_TILE_M,
            (N + MATMUL_TILE_N - 1) / MATMUL_TILE_N);
  dim3 block(128);
  
  // SMEM size: B stages + output
  int smem_size = MATMUL_STAGES * MATMUL_TILE_N * MATMUL_TILE_K * sizeof(f16) +
                  MATMUL_TILE_M * MATMUL_TILE_N * sizeof(float);
  
  cudaFuncSetAttribute(matmul_rs_mode_kernel, 
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  // Run kernel
  matmul_rs_mode_kernel<<<grid, block, smem_size>>>(
      reinterpret_cast<f16*>(a_d),
      reinterpret_cast<f16*>(b_d),
      c_d, M, N, K);
  cudaDeviceSynchronize();

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  if (!skip_verify) {
    cudaMemcpy(res_h.data(), c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Sample verification
    size_t sample_step = std::max(1UL, (size_t)(M * N) / 2000UL);
    for (size_t idx = 0; idx < M * N; idx += sample_step) {
      int i = idx / N, j = idx % N;
      float ref = 0.0f;
      for (int k = 0; k < K; ++k)
        ref += __half2float(lhs_h[i * K + k]) * __half2float(rhs_h[j * K + k]);
      float got = res_h[idx];
      float tol = 1.0f + 0.01f * std::abs(ref);
      if (std::abs(got - ref) > tol) {
        std::cout << "[" << i << "," << j << "] ref=" << ref << " got=" << got << "\n";
        std::cout << "Verification FAILED\n";
        return 1;
      }
    }
    std::cout << "Verification passed (sampled " << (M * N / sample_step) << " elements)\n";
  }

  if (enable_timing) {
    int warmup = 10, repeat = 50;
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
      matmul_rs_mode_kernel<<<grid, block, smem_size>>>(
          reinterpret_cast<f16*>(a_d),
          reinterpret_cast<f16*>(b_d),
          c_d, M, N, K);
    }
    cudaDeviceSynchronize();
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
      matmul_rs_mode_kernel<<<grid, block, smem_size>>>(
          reinterpret_cast<f16*>(a_d),
          reinterpret_cast<f16*>(b_d),
          c_d, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / repeat;
    
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    double eff = tflops / H800_PCIE_PEAK_F16_TFLOPS * 100.0;
    
    std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "Timing avg ms: " << avg_ms << "\n";
    std::cout << "TFLOPS: " << tflops << "\n";
    std::cout << "HW efficiency: " << eff << "%\n";
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  return 0;
}
