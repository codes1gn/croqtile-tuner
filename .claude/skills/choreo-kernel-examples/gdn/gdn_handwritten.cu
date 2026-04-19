
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

template <typename T>
__device__ __forceinline__ T SHFL_XOR(T var, int lane_mask, int width) {
  return __shfl_xor_sync(uint32_t(-1), var, lane_mask, width);
}

template <int BK, int BV>
__global__ __launch_bounds__(32, 1) void gdn_kernel(
    const float * __restrict__ A_log,
    const bf16 * __restrict__ a,
    const bf16 * __restrict__ dt_bias,
    const bf16 * __restrict__ q,
    const bf16 * __restrict__ k,
    const bf16 * __restrict__ v,
    const bf16 * __restrict__ b,
    bf16 * __restrict__ o,
    float * __restrict__ initial_state_source,
    const int * __restrict__ initial_state_indices,
    float scale,
    float softplus_beta,
    float softplus_threshold,
    unsigned B, unsigned H, unsigned HV, unsigned K, unsigned V, unsigned T
) {
  constexpr int ROWS = BK / 32;
  int tid = threadIdx.x;

  int i_v = blockIdx.x;
  int i_n = blockIdx.y;
  int i_hv = blockIdx.z;
  int i_h = i_hv / (HV / H);

  int bos = i_n * T;

  float a_log_l = A_log[i_hv];
  float dt_bias_l = __bfloat162float(dt_bias[i_hv]);

  float h_reg[ROWS * BV];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
#pragma unroll
    for (int j = 0; j < BV; ++j) {
      h_reg[r * BV + j] = 0.0f;
    }
  }

  int idx = initial_state_indices[i_n];
  if (idx >= 0) {
    const float *h0 = initial_state_source + (size_t)idx * HV * K * V
                     + (size_t)i_hv * K * V
                     + (size_t)i_v * BV;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      int row = tid + r * 32;
#pragma unroll
      for (int j = 0; j < BV; ++j) {
        h_reg[r * BV + j] = h0[row * V + j];
      }
    }
  }

  const bf16 *p_a = a + (size_t)bos * HV + i_hv;
  const bf16 *p_b = b + (size_t)bos * HV + i_hv;
  const bf16 *p_q = q + ((size_t)bos * H + i_h) * K;
  const bf16 *p_k = k + ((size_t)bos * H + i_h) * K;
  const bf16 *p_v = v + ((size_t)bos * HV + i_hv) * V + i_v * BV;
  bf16 *p_o = o + ((size_t)bos * HV + i_hv) * V + i_v * BV;

  for (unsigned t = 0; t < T; ++t) {
    float b_a = __bfloat162float(*p_a);
    float b_b = __bfloat162float(*p_b);

    float q_local[ROWS], k_local[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      q_local[r] = __bfloat162float(p_q[tid + r * 32]);
      k_local[r] = __bfloat162float(p_k[tid + r * 32]);
    }
    float v_local = __bfloat162float(p_v[tid]);

    float x = b_a + dt_bias_l;
    float beta_x = softplus_beta * x;
    float softplus_x = beta_x < softplus_threshold
                           ? (1.0f / softplus_beta) * logf(1.0f + expf(beta_x))
                           : x;
    float g = -expf(a_log_l) * softplus_x;
    float beta_val = 1.0f / (1.0f + expf(-b_b));

    {
      float q_sum = 0.0f, k_sum = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        q_sum = fmaf(q_local[r], q_local[r], q_sum);
        k_sum = fmaf(k_local[r], k_local[r], k_sum);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += SHFL_XOR(q_sum, offset, 32);
        k_sum += SHFL_XOR(k_sum, offset, 32);
      }
      float q_div = rsqrtf(q_sum + 1e-6f);
      float k_div = rsqrtf(k_sum + 1e-6f);
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        q_local[r] *= q_div * scale;
        k_local[r] *= k_div;
      }
    }

    float h_scale = expf(g);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
      for (int j = 0; j < BV; ++j) {
        h_reg[r * BV + j] *= h_scale;
      }
    }

#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float v_sum = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        v_sum = fmaf(h_reg[r * BV + j], k_local[r], v_sum);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        v_sum += SHFL_XOR(v_sum, offset, 32);
      }
      if (j == tid) v_local -= v_sum;
    }

    v_local *= beta_val;

#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float vj = __shfl_sync(uint32_t(-1), v_local, j, 32);
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        h_reg[r * BV + j] += k_local[r] * vj;
      }
    }

#pragma unroll
    for (int j = 0; j < BV; ++j) {
      float o_sum = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        o_sum = fmaf(h_reg[r * BV + j], q_local[r], o_sum);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        o_sum += SHFL_XOR(o_sum, offset, 32);
      }
      if (tid == j) p_o[j] = __float2bfloat16(o_sum);
    }

    p_a += HV;
    p_b += HV;
    p_q += H * K;
    p_k += H * K;
    p_v += HV * V;
    p_o += HV * V;
  }

  if (idx >= 0) {
    float *h0 = initial_state_source + (size_t)idx * HV * K * V
               + (size_t)i_hv * K * V
               + (size_t)i_v * BV;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      int row = tid + r * 32;
#pragma unroll
      for (int j = 0; j < BV; ++j) {
        h0[row * V + j] = h_reg[r * BV + j];
      }
    }
  }
}

template <typename T>
void read_bin(const std::string &path, std::vector<T> &data) {
  std::ifstream file(path, std::ios::binary);
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  size_t count = size / sizeof(T);
  data.resize(count);
  file.read(reinterpret_cast<char *>(data.data()), size);
  file.close();
}

int main() {
  const unsigned B = 2;
  const unsigned T = 128;
  const unsigned H = 8;
  const unsigned K = 128;
  const unsigned V = 128;
  const unsigned HV = H * K / 64;
  const unsigned N = B;

  const size_t o_count = B * T * HV * V;

  const float softplus_beta = 1.0f;
  const float softplus_threshold = 20.0f;
  const float scale = 1.0f / std::sqrt(static_cast<float>(K));

  std::string ref_dir = "/home/baldlee/workspace/choreo-attn/gdn/reference/";

  std::vector<uint32_t> A_log_bits(HV);
  std::vector<uint16_t> a_bits(B * T * HV);
  std::vector<uint16_t> dt_bias_bits(HV);
  std::vector<uint16_t> q_bits(B * T * H * K);
  std::vector<uint16_t> k_bits(B * T * H * K);
  std::vector<uint16_t> v_bits(B * T * HV * V);
  std::vector<uint16_t> b_bits(B * T * HV);
  std::vector<int32_t> indices_bits(B);
  std::vector<uint16_t> o_expected(o_count);
  std::vector<uint32_t> initial_state_bits(B * HV * K * V);

  read_bin(ref_dir + "A_log.bin", A_log_bits);
  read_bin(ref_dir + "a.bin", a_bits);
  read_bin(ref_dir + "dt_bias.bin", dt_bias_bits);
  read_bin(ref_dir + "q.bin", q_bits);
  read_bin(ref_dir + "k.bin", k_bits);
  read_bin(ref_dir + "v.bin", v_bits);
  read_bin(ref_dir + "b.bin", b_bits);
  read_bin(ref_dir + "initial_state_indices.bin", indices_bits);
  read_bin(ref_dir + "o_expected.bin", o_expected);
  read_bin(ref_dir + "initial_state_source_in.bin", initial_state_bits);

  float *A_log_d, *iss_d;
  bf16 *a_d, *dt_bias_d, *q_d, *k_d, *v_d, *b_d, *o_d;
  int *indices_d;

  cudaMalloc(&A_log_d, HV * 4);
  cudaMalloc(&a_d, B * T * HV * 2);
  cudaMalloc(&dt_bias_d, HV * 2);
  cudaMalloc(&q_d, (size_t)B * T * H * K * 2);
  cudaMalloc(&k_d, (size_t)B * T * H * K * 2);
  cudaMalloc(&v_d, (size_t)B * T * HV * V * 2);
  cudaMalloc(&b_d, B * T * HV * 2);
  cudaMalloc(&o_d, (size_t)B * T * HV * V * 2);
  cudaMalloc(&iss_d, (size_t)B * HV * K * V * 4);
  cudaMalloc(&indices_d, B * 4);

  cudaMemcpy(A_log_d, A_log_bits.data(), HV * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(a_d, a_bits.data(), B * T * HV * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dt_bias_d, dt_bias_bits.data(), HV * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(q_d, q_bits.data(), (size_t)B * T * H * K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(k_d, k_bits.data(), (size_t)B * T * H * K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(v_d, v_bits.data(), (size_t)B * T * HV * V * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_bits.data(), B * T * HV * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(iss_d, initial_state_bits.data(), (size_t)B * HV * K * V * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(indices_d, indices_bits.data(), B * 4, cudaMemcpyHostToDevice);

  float *iss_backup;
  cudaMalloc(&iss_backup, (size_t)B * HV * K * V * 4);
  cudaMemcpy(iss_backup, iss_d, (size_t)B * HV * K * V * 4, cudaMemcpyDeviceToDevice);

  constexpr int BK = 128;
  constexpr int BV = 32;
  int NV = (V + BV - 1) / BV;
  dim3 grid(NV, N, HV);
  dim3 block(32);

  // Warmup
  for (int w = 0; w < 3; w++) {
    cudaMemcpy(iss_d, iss_backup, (size_t)B * HV * K * V * 4, cudaMemcpyDeviceToDevice);
    gdn_kernel<BK, BV><<<grid, block>>>(
        A_log_d, a_d, dt_bias_d, q_d, k_d, v_d, b_d, o_d, iss_d,
        indices_d, scale, softplus_beta, softplus_threshold,
        B, H, HV, K, V, T);
    cudaDeviceSynchronize();
  }

  cudaEvent_t start_ev, stop_ev;
  cudaEventCreate(&start_ev);
  cudaEventCreate(&stop_ev);

  std::cout << "=== Pure CUDA kernel (T=" << T << ", grid=(" << NV << "," << N << "," << HV << ")) ===" << std::endl;
  for (int run = 0; run < 10; run++) {
    cudaMemcpy(iss_d, iss_backup, (size_t)B * HV * K * V * 4, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(start_ev);
    gdn_kernel<BK, BV><<<grid, block>>>(
        A_log_d, a_d, dt_bias_d, q_d, k_d, v_d, b_d, o_d, iss_d,
        indices_d, scale, softplus_beta, softplus_threshold,
        B, H, HV, K, V, T);
    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);
    float ms = 0;
    cudaEventElapsedTime(&ms, start_ev, stop_ev);
    std::cout << "  Run " << run << ": " << ms << " ms" << std::endl;
  }
  cudaEventDestroy(start_ev);
  cudaEventDestroy(stop_ev);

  // Correctness check
  cudaMemcpy(iss_d, iss_backup, (size_t)B * HV * K * V * 4, cudaMemcpyDeviceToDevice);
  gdn_kernel<BK, BV><<<grid, block>>>(
      A_log_d, a_d, dt_bias_d, q_d, k_d, v_d, b_d, o_d, iss_d,
      indices_d, scale, softplus_beta, softplus_threshold,
      B, H, HV, K, V, T);
  cudaDeviceSynchronize();

  std::vector<uint16_t> o_host(o_count);
  cudaMemcpy(o_host.data(), o_d, o_count * 2, cudaMemcpyDeviceToHost);

  const int TOLERANCE = 32;
  size_t o_mismatch = 0, o_small_diff = 0;
  int max_diff = 0;
  for (size_t i = 0; i < o_count; ++i) {
    if (o_host[i] != o_expected[i]) {
      int diff = std::abs((int)o_host[i] - (int)o_expected[i]);
      if (diff > max_diff) max_diff = diff;
      if (diff <= TOLERANCE) {
        o_small_diff++;
      } else {
        o_mismatch++;
        if (o_mismatch <= 10)
          std::cout << "o[" << i << "]: 0x" << std::hex << o_host[i]
                    << " vs 0x" << o_expected[i] << std::dec
                    << " (diff=" << diff << ")" << std::endl;
      }
    }
  }

  std::cout << "o mismatches (>" << TOLERANCE << " ULP): " << o_mismatch << " / " << o_count << std::endl;
  std::cout << "o small diffs (1-" << TOLERANCE << " ULP): " << o_small_diff << " / " << o_count << std::endl;
  std::cout << "o max ULP diff: " << max_diff << std::endl;
  std::cout << (o_mismatch == 0 ? "TEST PASSED" : "TEST FAILED") << std::endl;

  cudaFree(A_log_d); cudaFree(a_d); cudaFree(dt_bias_d);
  cudaFree(q_d); cudaFree(k_d); cudaFree(v_d); cudaFree(b_d);
  cudaFree(o_d); cudaFree(iss_d); cudaFree(indices_d); cudaFree(iss_backup);
  return 0;
}
