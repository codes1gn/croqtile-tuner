// matmul_fp16fp32_2048x2048x16384 iter038: Direct GMEM load for A + RS WGMMA
// 
// Key optimization: Load A directly from global memory to registers
// - No TMA/SMEM staging for A
// - Only B uses TMA to SMEM
// - WGMMA RS mode: A from registers, B from SMEM descriptor
//
// SMEM reduction:
// - Old: A(256×64×2B×2stages) + B(128×64×2B×2stages) + output = 230KB
// - New: B(128×64×2B×4stages) + output = ~160KB, allows 4 stages
//
// Register pressure:
// - Each thread loads 4 uint32_t (8 f16) for A per WGMMA K16 iteration
// - 64 accumulator registers per MMA

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
#define MATMUL_TILE_M 64      // Only 64 rows (single WGMMA M)
#define MATMUL_TILE_N 128     // Full N tile
#define MATMUL_TILE_K 64      // K tile for pipeline
#define MATMUL_WARP_K 16      // K per WGMMA instruction
#define MATMUL_SWIZ 128
#define MATMUL_STAGES 4       // Can afford more stages without A in SMEM

#define MATMUL_DEFAULT_M 2048
#define MATMUL_DEFAULT_N 2048
#define MATMUL_DEFAULT_K 16384

// Load A fragment directly from global memory for RS mode
// For M64N128K16: need 4 uint32_t (8 f16) per thread
// A layout: [M, K] row-major
__device__ __forceinline__ void load_a_gmem_to_reg(
    const f16* __restrict__ gmem_a,  // Global memory pointer
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    int M_offset,                     // M offset for this tile
    int K_offset,                     // K offset for this iteration
    int K_stride) {                   // K dimension stride
  
  int tid = threadIdx.x % 128;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  
  // RS mode register layout for M64N128K16:
  // Each warp handles M=16 rows
  // Lane determines which elements within those rows
  // Thread (warp_id, lane_id) owns:
  //   row0 = warp_id * 16 + lane_id / 4
  //   row1 = row0 + 8
  //   col_pair_base = (lane_id % 4) * 4
  
  int row0 = warp_id * 16 + lane_id / 4;
  int row1 = row0 + 8;
  int col_base = (lane_id % 4) * 4;  // Each thread handles 4 consecutive K elements
  
  // Global memory addresses
  const f16* ptr_r0_c0 = gmem_a + (M_offset + row0) * K_stride + K_offset + col_base;
  const f16* ptr_r0_c2 = ptr_r0_c0 + 2;
  const f16* ptr_r1_c0 = gmem_a + (M_offset + row1) * K_stride + K_offset + col_base;
  const f16* ptr_r1_c2 = ptr_r1_c0 + 2;
  
  // Load 2 f16 pairs per row (4 f16 per row, 2 rows = 8 f16 = 4 uint32_t)
  a0 = *reinterpret_cast<const uint32_t*>(ptr_r0_c0);
  a1 = *reinterpret_cast<const uint32_t*>(ptr_r0_c2);
  a2 = *reinterpret_cast<const uint32_t*>(ptr_r1_c0);
  a3 = *reinterpret_cast<const uint32_t*>(ptr_r1_c2);
}

__global__ void __launch_bounds__(128, 2) matmul_gmem_a_rs_kernel(
    const f16* __restrict__ lhs,    // A: [M, K]
    const f16* __restrict__ rhs,    // B: [N, K] row-major (K-major for WGMMA)
    float* __restrict__ output,      // C: [M, N]
    int M, int N, int K,
    const __grid_constant__ CUtensorMap rhs_tma_desc) {
  
  // Grid: [cdiv(M, TILE_M), cdiv(N, TILE_N)]
  int block_m = blockIdx.x;
  int block_n = blockIdx.y;
  
  extern __shared__ char smem[];
  // Only B in shared memory: STAGES * 128 * 64 * 2B
  f16* rhs_smem = reinterpret_cast<f16*>(smem);
  // Output staging: 64 * 128 * 4B
  float* output_smem = reinterpret_cast<float*>(smem + MATMUL_STAGES * MATMUL_TILE_N * MATMUL_TILE_K * sizeof(f16));
  
  // Pipeline barriers
  __shared__ cuda::barrier<cuda::thread_scope_block> full[MATMUL_STAGES];
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[MATMUL_STAGES];
  
  if (threadIdx.x == 0) {
    for (int s = 0; s < MATMUL_STAGES; ++s) {
      init(&full[s], 128);
      init(&empty[s], 128);
    }
  }
  __syncthreads();
  
  // Initialize accumulators
  float mc[64];
  #pragma unroll
  for (int i = 0; i < 64; ++i) mc[i] = 0.0f;
  
  // Global offsets
  int m_base = block_m * MATMUL_TILE_M;
  int n_base = block_n * MATMUL_TILE_N;
  int k_tiles = (K + MATMUL_TILE_K - 1) / MATMUL_TILE_K;
  
  // Producer: load B tiles using TMA
  // (Simplified: all threads participate in loading for now)
  
  // Main loop
  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    int stage = k_tile % MATMUL_STAGES;
    int k_base = k_tile * MATMUL_TILE_K;
    
    // Wait for SMEM stage to be available
    if (k_tile >= MATMUL_STAGES) {
      empty[stage].arrive_and_wait();
    }
    
    // Load B tile to SMEM (simplified: cooperative load)
    // In real impl, use TMA
    f16* b_smem_stage = rhs_smem + stage * MATMUL_TILE_N * MATMUL_TILE_K;
    int tid = threadIdx.x;
    int elements_per_thread = (MATMUL_TILE_N * MATMUL_TILE_K) / 128;
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
    
    // Signal B is ready
    full[stage].arrive();
    
    // Wait for B to be ready
    full[stage].arrive_and_wait();
    
    // K-loop: 4 WGMMA iterations per K tile (64/16 = 4)
    #pragma unroll
    for (int k_iter = 0; k_iter < 4; ++k_iter) {
      int k_offset = k_base + k_iter * MATMUL_WARP_K;
      
      // Load A from global memory to registers
      uint32_t a0, a1, a2, a3;
      load_a_gmem_to_reg(lhs, a0, a1, a2, a3, m_base, k_offset, K);
      
      // Create B descriptor for SMEM
      f16* b_smem_ptr = b_smem_stage + k_iter * MATMUL_WARP_K;
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
    
    // Signal we're done with this stage
    empty[stage].arrive();
  }
  
  // Store output
  // Store to shared memory first, then to global
  auto shape_out = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
  auto stride_out = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
  auto layout_out = cute::make_layout(shape_out, stride_out);
  auto tensor_out = cute::make_tensor(cute::make_smem_ptr<float>(output_smem), layout_out);
  store_fragment_d<CUTE_WGMMA_M64K16, 128>(tensor_out, mc);
  __syncthreads();
  
  // Copy from SMEM to global
  int tid = threadIdx.x;
  int elements_per_thread = (MATMUL_TILE_M * MATMUL_TILE_N) / 128;
  for (int i = 0; i < elements_per_thread; ++i) {
    int idx = tid * elements_per_thread + i;
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
  
  cudaFuncSetAttribute(matmul_gmem_a_rs_kernel, 
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  // Create TMA descriptor (placeholder - not used in simplified version)
  CUtensorMap rhs_tma_desc{};

  // Run kernel
  matmul_gmem_a_rs_kernel<<<grid, block, smem_size>>>(
      reinterpret_cast<f16*>(a_d),
      reinterpret_cast<f16*>(b_d),
      c_d, M, N, K, rhs_tma_desc);
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
      matmul_gmem_a_rs_kernel<<<grid, block, smem_size>>>(
          reinterpret_cast<f16*>(a_d),
          reinterpret_cast<f16*>(b_d),
          c_d, M, N, K, rhs_tma_desc);
    }
    cudaDeviceSynchronize();
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
      matmul_gmem_a_rs_kernel<<<grid, block, smem_size>>>(
          reinterpret_cast<f16*>(a_d),
          reinterpret_cast<f16*>(b_d),
          c_d, M, N, K, rhs_tma_desc);
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
