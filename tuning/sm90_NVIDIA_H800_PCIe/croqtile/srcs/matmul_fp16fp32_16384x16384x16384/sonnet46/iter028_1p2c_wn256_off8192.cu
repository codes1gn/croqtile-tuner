
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "choreo.h"
namespace cde = cuda::device::experimental;
#include <cooperative_groups.h>
using namespace choreo;

#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ 90

static inline void __choreo_check_cuda_environment__() {
  static bool already_checked = false;
  if (already_checked) return;
  already_checked = true;
  auto decode_cuda_version = [](int v, int& major, int& minor, int& patch) {
    major = v / 1000; minor = (v % 1000) / 10; patch = v % 10;
  };
  int runtime_ver = 0;
  cudaError_t err = cudaRuntimeGetVersion(&runtime_ver);
  if (err != cudaSuccess) { std::fprintf(stderr, "[choreo] CUDA runtime not available: %s\n", cudaGetErrorString(err)); std::exit(EXIT_FAILURE); }
  int driver_ver = 0;
  err = cudaDriverGetVersion(&driver_ver);
  if (err != cudaSuccess) { std::fprintf(stderr, "[choreo] CUDA driver not available: %s\n", cudaGetErrorString(err)); std::exit(EXIT_FAILURE); }
  int rMaj, rMin, rPat, dMaj, dMin, dPat, reqMaj, reqMin, reqPat;
  decode_cuda_version(runtime_ver, rMaj, rMin, rPat);
  decode_cuda_version(driver_ver, dMaj, dMin, dPat);
  decode_cuda_version(CUDART_VERSION, reqMaj, reqMin, reqPat);
  if (runtime_ver < CUDART_VERSION) {
    std::fprintf(stderr, "[choreo] CUDA runtime too old\n"); std::exit(EXIT_FAILURE);
  }
  int device_count = 0;
  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) { std::fprintf(stderr, "[choreo] No CUDA devices.\n"); std::exit(EXIT_FAILURE); }
  int device_id = 0;
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) { std::fprintf(stderr, "[choreo] cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err)); std::exit(EXIT_FAILURE); }
  int sm = prop.major * 10 + prop.minor;
  if (sm < __CHOREO_REQUIRED_GPU_DEVICE_SM__) {
    std::fprintf(stderr, "[choreo] Compute capability too low: found SM %d.%d, required SM >= %d\n", prop.major, prop.minor, __CHOREO_REQUIRED_GPU_DEVICE_SM__);
    std::exit(EXIT_FAILURE);
  }
}

// iter027: 1p2c WARP_N=256 STAGES=2 with depth-1 WGMMA pipeline
// Producer loads A(128x64) + B(256x64) per stage.
// Consumer1: rows 0..63 of A tile, full B tile -> output[blockIdx.x*128 rows]
// Consumer2: rows 64..127 of A tile, full B tile -> output[blockIdx.x*128+64 rows]
// Grid: (M/128, N/256). Block: 384 threads (3 warpgroups).
// SMEM: 65536(rhs) + 32768(lhs) + 65536(out1) + 65536(out2) = 229376B (224KB < 228KB H800 limit)
// Barrier init: 257 = 1 (producer) + 128 (consumer1) + 128 (consumer2)

#include <cstring>

#define H800_PCIE_PEAK_F16_TFLOPS 1513

#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 256
#define MATMUL_TILE_M 128
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_STAGES 2

#define MATMUL_DEFAULT_M 16384
#define MATMUL_DEFAULT_N 16384
#define MATMUL_DEFAULT_K 16384

__global__ void __choreo_device_matmul(
    f16 * lhs, f16 * rhs, float * output, unsigned K, unsigned M, unsigned N,
    const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map,  // A: 128x64 per stage
    const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map,  // B: 256x64 per stage
    const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map,  // C1: 64x256 output (consumer1)
    const __grid_constant__ CUtensorMap __choreo_tma_3_tensor_map   // C2: 64x256 output (consumer2)
) {
  extern __shared__ char __choreo_device_matmul__runtime_shared_buffer__raw[];
  auto __choreo_device_matmul__runtime_shared_buffer__ = reinterpret_cast<char*>(
      aligned_up_ptr<1024 * 8>(__choreo_device_matmul__runtime_shared_buffer__raw));

  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());

  // TMA barrier atoms for output stores
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) { init(&choreo_copy_atom_t_0_barrier, 1); cde::fence_proxy_async_shared_cta(); }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_1_barrier;
  if (__CHOREO_BLOCK_SINGLE__) { init(&choreo_copy_atom_t_1_barrier, 1); cde::fence_proxy_async_shared_cta(); }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_1{&choreo_copy_atom_t_1_barrier};

  auto anon_1 = (unsigned char*)__choreo_device_matmul__runtime_shared_buffer__;
  // SMEM layout (STAGES=2):
  //   rhs_load_s  @ +0:      2 * 256*64*2 = 65536B  (stride 32768B/stage)
  //   lhs_load_s  @ +65536:  2 * 8192 + 16384 = 32768B  (stride 8192B/stage, tile=16384B)
  //   output_s_1  @ +98304:  64*256*4 = 65536B (consumer1)
  //   output_s_2  @ +163840: 64*256*4 = 65536B (consumer2)
  //   Total: 229376B = 224KB

  __shared__ cuda::barrier<cuda::thread_scope_block> full[2];
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&full[0], 257);  // 1 producer + 2 * 128 consumers
    init(&full[1], 257);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[2];
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&empty[0], 257);
    init(&empty[1], 257);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  f16*   rhs_load_s = (f16*)(anon_1 + 0);
  f16*   lhs_load_s = (f16*)(anon_1 + 65536);
  float* output_s_1 = (float*)(anon_1 + 98304);
  float* output_s_2 = (float*)(anon_1 + 163840);

  const auto __choreo_vg4id_x = threadIdx.x / 128;
  const auto __choreo_vtid_x  = threadIdx.x % 128;

  // ----- PRODUCER (warpgroup 0, thread 0 only) -----
  if (__choreo_vg4id_x == 0 && __choreo_vtid_x == 0) {
    for (int k = 0; k < ((K + 63) / 64); ++k) {
      int stage = k % 2;
      empty[stage].wait(empty[stage].arrive());
      const unsigned lhs_k_off = (k * 64);
      const unsigned lhs_m_off = (blockIdx.x * 128);
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          (lhs_load_s + ((k % 2) * 8192)), &__choreo_tma_0_tensor_map, lhs_k_off, lhs_m_off, full[stage]);
      const unsigned rhs_k_off = (k * 64);
      const unsigned rhs_n_off = (blockIdx.y * 256);
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          (rhs_load_s + ((k % 2) * 32768)), &__choreo_tma_1_tensor_map, rhs_k_off, rhs_n_off, full[stage]);
      // LHS tile = 128*64*2 = 16384B, RHS tile = 256*64*2 = 32768B
      (void)cuda::device::barrier_arrive_tx(full[stage], 1, 16384 + 32768);
    }
  }

  // ----- CONSUMER 1 (warpgroup 1): rows 0..63 of A -----
  if (__choreo_vg4id_x == 1) {
    float mc[128];
    for (int i = 0; i < 128; ++i) mc[i] = 0.0f;

    // Prime: signal all stages empty
    for (int s = 0; s < 2; ++s) (void)empty[s].arrive();

    const int K_TILES = (K + 63) / 64;
    auto issue_c1 = [&](int tile_k) {
      int s = tile_k % 2;
      warpgroup_arrive();
      for (int w = 0; w < 4; ++w) {
        f16* ma = (f16*)((w * 16 + s * 8192 + lhs_load_s));
        f16* mb = (f16*)((w * 16 + s * 32768 + rhs_load_s));
        uint64_t dma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma);
        uint64_t dmb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb);
        cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<
            static_cast<cute::SM90::GMMA::Major>(0),
            static_cast<cute::SM90::GMMA::Major>(0)>::fma(
            dma, dmb,
            mc[0],mc[1],mc[2],mc[3],mc[4],mc[5],mc[6],mc[7],mc[8],mc[9],mc[10],mc[11],mc[12],mc[13],mc[14],mc[15],
            mc[16],mc[17],mc[18],mc[19],mc[20],mc[21],mc[22],mc[23],mc[24],mc[25],mc[26],mc[27],mc[28],mc[29],mc[30],mc[31],
            mc[32],mc[33],mc[34],mc[35],mc[36],mc[37],mc[38],mc[39],mc[40],mc[41],mc[42],mc[43],mc[44],mc[45],mc[46],mc[47],
            mc[48],mc[49],mc[50],mc[51],mc[52],mc[53],mc[54],mc[55],mc[56],mc[57],mc[58],mc[59],mc[60],mc[61],mc[62],mc[63],
            mc[64],mc[65],mc[66],mc[67],mc[68],mc[69],mc[70],mc[71],mc[72],mc[73],mc[74],mc[75],mc[76],mc[77],mc[78],mc[79],
            mc[80],mc[81],mc[82],mc[83],mc[84],mc[85],mc[86],mc[87],mc[88],mc[89],mc[90],mc[91],mc[92],mc[93],mc[94],mc[95],
            mc[96],mc[97],mc[98],mc[99],mc[100],mc[101],mc[102],mc[103],mc[104],mc[105],mc[106],mc[107],mc[108],mc[109],mc[110],mc[111],
            mc[112],mc[113],mc[114],mc[115],mc[116],mc[117],mc[118],mc[119],mc[120],mc[121],mc[122],mc[123],mc[124],mc[125],mc[126],mc[127]);
      }
      warpgroup_commit_batch();
    };

    int k = 0;
    if (K_TILES > 0) { full[0].wait(full[0].arrive()); issue_c1(0); k = 1; }
    for (; k < K_TILES; ++k) {
      int stage = k % 2, prev = (k - 1) % 2;
      full[stage].wait(full[stage].arrive());
      warpgroup_wait<0>();
      (void)empty[prev].arrive();
      issue_c1(k);
    }
    warpgroup_wait<0>();
    (void)empty[(k - 1) % 2].arrive();

    // Store accumulator to SMEM, then TMA to global output
    auto shape1  = cute::make_shape(cute::Int<64>{}, cute::Int<256>{});
    auto stride1 = cute::make_stride(cute::Int<256>{}, cute::Int<1>{});
    auto layout1 = cute::make_layout(shape1, stride1);
    auto tensor1 = cute::make_tensor(cute::make_smem_ptr<float>(output_s_1), layout1);
    store_fragment_d<CUTE_WGMMA_M64K16, 256>(tensor1, reinterpret_cast<float*>(mc));
    future fut0("", 58, 9);
    fut0.is_tma = true;
    fut0.set_atom(&choreo_copy_atom_t_0);
    cde::fence_proxy_async_shared_cta();
    if (__CHOREO_GROUPX4_SINGLE__) {
      // Consumer1: M row range = [blockIdx.x*128, blockIdx.x*128+64)
      cde::cp_async_bulk_tensor_2d_shared_to_global(
          &__choreo_tma_2_tensor_map, (blockIdx.y * 256), (blockIdx.x * 128), output_s_1);
      cde::cp_async_bulk_commit_group();
    }
  }

  // ----- CONSUMER 2 (warpgroup 2): rows 64..127 of A -----
  if (__choreo_vg4id_x == 2) {
    float mc[128];
    for (int i = 0; i < 128; ++i) mc[i] = 0.0f;

    // Prime: signal all stages empty
    for (int s = 0; s < 2; ++s) (void)empty[s].arrive();

    const int K_TILES = (K + 63) / 64;
    auto issue_c2 = [&](int tile_k) {
      int s = tile_k % 2;
      warpgroup_arrive();
      for (int w = 0; w < 4; ++w) {
        // +4096 swizzle offset selects rows 64..127 within the 128-row LHS tile
        f16* ma = (f16*)((w * 16 + s * 8192 + (lhs_load_s + 8192)));
        f16* mb = (f16*)((w * 16 + s * 32768 + rhs_load_s));
        uint64_t dma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma);
        uint64_t dmb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb);
        cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<
            static_cast<cute::SM90::GMMA::Major>(0),
            static_cast<cute::SM90::GMMA::Major>(0)>::fma(
            dma, dmb,
            mc[0],mc[1],mc[2],mc[3],mc[4],mc[5],mc[6],mc[7],mc[8],mc[9],mc[10],mc[11],mc[12],mc[13],mc[14],mc[15],
            mc[16],mc[17],mc[18],mc[19],mc[20],mc[21],mc[22],mc[23],mc[24],mc[25],mc[26],mc[27],mc[28],mc[29],mc[30],mc[31],
            mc[32],mc[33],mc[34],mc[35],mc[36],mc[37],mc[38],mc[39],mc[40],mc[41],mc[42],mc[43],mc[44],mc[45],mc[46],mc[47],
            mc[48],mc[49],mc[50],mc[51],mc[52],mc[53],mc[54],mc[55],mc[56],mc[57],mc[58],mc[59],mc[60],mc[61],mc[62],mc[63],
            mc[64],mc[65],mc[66],mc[67],mc[68],mc[69],mc[70],mc[71],mc[72],mc[73],mc[74],mc[75],mc[76],mc[77],mc[78],mc[79],
            mc[80],mc[81],mc[82],mc[83],mc[84],mc[85],mc[86],mc[87],mc[88],mc[89],mc[90],mc[91],mc[92],mc[93],mc[94],mc[95],
            mc[96],mc[97],mc[98],mc[99],mc[100],mc[101],mc[102],mc[103],mc[104],mc[105],mc[106],mc[107],mc[108],mc[109],mc[110],mc[111],
            mc[112],mc[113],mc[114],mc[115],mc[116],mc[117],mc[118],mc[119],mc[120],mc[121],mc[122],mc[123],mc[124],mc[125],mc[126],mc[127]);
      }
      warpgroup_commit_batch();
    };

    int k = 0;
    if (K_TILES > 0) { full[0].wait(full[0].arrive()); issue_c2(0); k = 1; }
    for (; k < K_TILES; ++k) {
      int stage = k % 2, prev = (k - 1) % 2;
      full[stage].wait(full[stage].arrive());
      warpgroup_wait<0>();
      (void)empty[prev].arrive();
      issue_c2(k);
    }
    warpgroup_wait<0>();
    (void)empty[(k - 1) % 2].arrive();

    // Store accumulator to SMEM, then TMA to global output
    auto shape2  = cute::make_shape(cute::Int<64>{}, cute::Int<256>{});
    auto stride2 = cute::make_stride(cute::Int<256>{}, cute::Int<1>{});
    auto layout2 = cute::make_layout(shape2, stride2);
    auto tensor2 = cute::make_tensor(cute::make_smem_ptr<float>(output_s_2), layout2);
    store_fragment_d<CUTE_WGMMA_M64K16, 256>(tensor2, reinterpret_cast<float*>(mc));
    future fut1("", 75, 9);
    fut1.is_tma = true;
    fut1.set_atom(&choreo_copy_atom_t_1);
    cde::fence_proxy_async_shared_cta();
    if (__CHOREO_GROUPX4_SINGLE__) {
      // Consumer2: M row range = [blockIdx.x*128+64, blockIdx.x*128+128)
      cde::cp_async_bulk_tensor_2d_shared_to_global(
          &__choreo_tma_3_tensor_map, (blockIdx.y * 256), (blockIdx.x * 128 + 64), output_s_2);
      cde::cp_async_bulk_commit_group();
    }
  }
}

void matmul(const choreo::spanned_view<choreo::f16, 2> & lhs,
            const choreo::spanned_view<choreo::f16, 2> & rhs,
            const choreo::spanned_view<choreo::f32, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &K = lhs.shape()[1];
  auto &M = lhs.shape()[0];
  auto &N = rhs.shape()[0];
  choreo::runtime_check(lhs.shape()[1] == rhs.shape()[1], "K mismatch");
  choreo::runtime_check(lhs.shape()[0] == output.shape()[0], "M mismatch");
  choreo::runtime_check(rhs.shape()[0] == output.shape()[1], "N mismatch");

  // TMA0: LHS A matrix — box 128x64, SWIZZLE_128B
  uint64_t tma0_shape[] = {K, M};
  uint64_t tma0_strides[] = {K * 2};
  uint32_t tma0_box[] = {64, 128};   // {K-dim, M-dim}
  uint32_t tma0_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap tma0_map{};
  choreo::abend_true(cuTensorMapEncodeTiled(
      &tma0_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, lhs.data(),
      tma0_shape, tma0_strides, tma0_box, tma0_elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS);

  // TMA1: RHS B matrix — box 256x64, SWIZZLE_128B
  uint64_t tma1_shape[] = {K, N};
  uint64_t tma1_strides[] = {K * 2};
  uint32_t tma1_box[] = {64, 256};   // {K-dim, N-dim}
  uint32_t tma1_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap tma1_map{};
  choreo::abend_true(cuTensorMapEncodeTiled(
      &tma1_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, rhs.data(),
      tma1_shape, tma1_strides, tma1_box, tma1_elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS);

  // TMA2: Output C1 (consumer1) — box 256x64, SWIZZLE_NONE
  uint64_t tma2_shape[] = {N, M};
  uint64_t tma2_strides[] = {N * 4};
  uint32_t tma2_box[] = {256, 64};   // {N-dim, M-dim}
  uint32_t tma2_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap tma2_map{};
  choreo::abend_true(cuTensorMapEncodeTiled(
      &tma2_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, output.data(),
      tma2_shape, tma2_strides, tma2_box, tma2_elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS);

  // TMA3: Output C2 (consumer2) — same map as C1 (same global tensor, different offset per CTA)
  uint64_t tma3_shape[] = {N, M};
  uint64_t tma3_strides[] = {N * 4};
  uint32_t tma3_box[] = {256, 64};
  uint32_t tma3_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap tma3_map{};
  choreo::abend_true(cuTensorMapEncodeTiled(
      &tma3_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, output.data(),
      tma3_shape, tma3_strides, tma3_box, tma3_elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS);

  // Grid: M/128 x N/256 CTAs, each CTA has 384 threads (3 warpgroups)
  dim3 gdims(((M + 127) / 128), ((N + 255) / 256), 1);
  dim3 bdims(384, 1, 1);
  cudaFuncSetAttribute(__choreo_device_matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, 229376 + (1024 - 1));
  __choreo_device_matmul<<<gdims, bdims, 229376 + (1024 - 1)>>>(
      lhs.data(), rhs.data(), output.data(), K, M, N,
      tma0_map, tma1_map, tma2_map, tma3_map);
  choreo::abend_true(cudaDeviceSynchronize());
}


int main(int argc, char** argv) {
  bool enable_timing = true;
  bool skip_verify = false;
  double user_flops = -1.0;
  auto is_disable_timing_arg = [](const char* s) { const char* t = "--disable-timing"; int i = 0; while (t[i] != '\0' && s[i] == t[i]) ++i; return t[i] == '\0' && s[i] == '\0'; };
  auto is_skip_verify_arg = [](const char* s) { const char* t = "--skip-verify"; int i = 0; while (t[i] != '\0' && s[i] == t[i]) ++i; return t[i] == '\0' && s[i] == '\0'; };
  auto is_verify_arg = [](const char* s) { const char* t = "--verify"; int i = 0; while (t[i] != '\0' && s[i] == t[i]) ++i; return t[i] == '\0' && s[i] == '\0'; };
  for (int i = 1; i < argc; ++i) {
    if (is_disable_timing_arg(argv[i])) { enable_timing = false; continue; }
    if (is_skip_verify_arg(argv[i])) { skip_verify = true; continue; }
    if (is_verify_arg(argv[i])) { /* --verify implies skip timing on some harnesses */ continue; }
    if (std::strncmp(argv[i], "--flops=", 8) == 0) { user_flops = std::atof(argv[i] + 8); continue; }
  }
  const char* timing_env = std::getenv("CHOREO_DISABLE_TIMING");
  if (timing_env && timing_env[0] == '1') enable_timing = false;
  const char* skip_verify_env = std::getenv("CHOREO_SKIP_VERIFY");
  if (skip_verify_env && skip_verify_env[0] == '1') skip_verify = true;

  size_t M = MATMUL_DEFAULT_M, N = MATMUL_DEFAULT_N, K = MATMUL_DEFAULT_K;
  auto lhs_h = choreo::make_spandata<choreo::f16>(M, K);
  auto rhs_h = choreo::make_spandata<choreo::f16>(N, K);
  auto res_h = choreo::make_spandata<choreo::f32>(M, N);
  lhs_h.fill_random(0, 2); rhs_h.fill_random(0, 2); res_h.fill(0.0f);

  half *a_d = nullptr, *b_d = nullptr;
  float *c_d = nullptr;
  choreo::abend_true(cudaMalloc(&a_d, M * K * sizeof(half)));
  choreo::abend_true(cudaMalloc(&b_d, N * K * sizeof(half)));
  choreo::abend_true(cudaMalloc(&c_d, M * N * sizeof(float)));
  choreo::abend_true(cudaMemcpy(a_d, lhs_h.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(b_d, rhs_h.data(), N * K * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(c_d, res_h.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());

  auto lhs_d = choreo::make_spanview<choreo::f16, 2>(a_d, {M, K});
  auto rhs_d = choreo::make_spanview<choreo::f16, 2>(b_d, {N, K});
  auto res_d = choreo::make_spanview<choreo::f32, 2>(c_d, {M, N});

  if (enable_timing) {
    int warmup = 10, repeat = 50;
    const char* warmup_env = std::getenv("CHOREO_TIMING_WARMUP");
    const char* repeat_env = std::getenv("CHOREO_TIMING_REPEAT");
    if (warmup_env) { int v = std::atoi(warmup_env); if (v >= 0) warmup = v; }
    if (repeat_env) { int v = std::atoi(repeat_env); if (v > 0) repeat = v; }
    choreo::TimerOption topt; topt.warmup = warmup; topt.repeat = repeat;
    auto avg_ms = choreo::timing([&]() { matmul(lhs_d, rhs_d, res_d); cudaDeviceSynchronize(); }, topt);
    std::cout << "Timing avg ms: " << avg_ms << "\n";
    double flops = (user_flops > 0.0) ? user_flops : (2.0 * double(M) * double(N) * double(K));
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    std::cout << "TFLOPS: " << tflops << "\n";
    double eff = (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0;
    std::cout << "HW efficiency: " << eff << "%\n";
  } else { matmul(lhs_d, rhs_d, res_d); }

  choreo::abend_true(cudaMemcpy(res_h.data(), c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaDeviceSynchronize());

  if (skip_verify) { std::cout << "Test Passed (verify skipped)\n" << std::endl; return 0; }

  auto lhs_view = lhs_h.view();
  auto rhs_view = rhs_h.view();
  auto res_view = res_h.view();
  float tolerance = 0.01f;
  auto rel_error = [](float ref, float got) {
    float abs_ref = std::abs(ref);
    float denom = abs_ref > 1e-6f ? abs_ref : 1.0f;
    return std::abs(ref - got) / denom;
  };
  for (size_t i = 0; i < 128; ++i) {
    for (size_t j = 0; j < 128; ++j) {
      float ref = 0.0f;
      for (size_t k = 0; k < lhs_view.shape()[1]; ++k)
        ref += __half2float(lhs_view[i][k]) * __half2float(rhs_view[j][k]);
      float got = res_view[i][j];
      auto delta = rel_error(ref, got);
      if (delta >= tolerance)
        std::cout << "[" << i << ", " << j << "] " << ref << " <-> " << got << ", delta: " << delta * 100 << "%\n";
      choreo::choreo_assert((delta < tolerance), "values are not equal.");
    }
  }
  std::cout << "Test Passed\n" << std::endl;
  return 0;
}
