#!/usr/bin/env python3
"""
Test and benchmark choreo-generated CUDA kernels against Triton reference.

Usage:
  python test_choreo_kernel.py <cu_file> [--fix-stride] [--bench] [-T T1,T2,...] [-B B]

Arguments:
  cu_file          Path to the choreo-generated .cu file
  --fix-stride     Apply known span_as stride fix before compiling (Deprecated)
  --bench          Run performance benchmark (default: correctness only)
  -T               Comma-separated T values (default: 4,128,512)
  -B               Batch size (default: 2)
  --nvcc-flags     Extra nvcc flags (quoted string)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference"))
from fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHOREO_RUNTIME = os.path.join(SCRIPT_DIR, "..", "choreo", "runtime")
CUTLASS_INCLUDE = "/home/baldlee/source/cutlass/include"
RMSNORM_INCLUDE = os.path.join(SCRIPT_DIR, "..", "rmsnorm")
CUDA_HOME = "/usr/local/cuda-13.0"
NVCC = os.path.join(CUDA_HOME, "bin", "nvcc")

H, K, V = 8, 128, 128
HV = H * K // 64
BK, BV = K, min(32, V)
SCALE = 1.0 / (K**0.5)
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0


def apply_stride_fix(cu_source: str) -> str:
    """Fix the known span_as stride bug for initial_state_source DMA.
    Replace cute::Int<32>{} with V in the stride of initial_state_source tensors.
    ATTENTION: Choreo has been updated and this function is deprecated.
    """
    lines = cu_source.split("\n")
    out = []
    for i, line in enumerate(lines):
        if "initial_state_source" in line and "make_stride" in line and "Int<32>" in line:
            line = line.replace("cute::Int<32>{}", "V")
        out.append(line)
    return "\n".join(out)


def fix_cu_seqlens(cu_source: str) -> str:
    """Fix cu_seqlens_dummy size from 1 to N."""
    cu_source = re.sub(
        r'cu_seqlens_dummy\(1,\s*0\)',
        'cu_seqlens_dummy(N, 0)',
        cu_source,
    )
    cu_source = re.sub(
        r'static_cast<size_t>\(1\)\}',
        'static_cast<size_t>(N)}',
        cu_source,
    )
    return cu_source


def replace_main_with_test(cu_source: str, B: int, T: int,
                           ref_dir: str) -> str:
    """Replace or append main() for testing with given B/T."""
    main_start = cu_source.find("int main(")
    if main_start == -1:
        main_start = cu_source.find("int main (")
    if main_start >= 0:
        prefix = cu_source[:main_start]
    else:
        prefix = cu_source + "\n\n"

    new_main = f"""
int main(int argc, char** argv) {{
  const int B = {B};
  const int T = {T};
  const int H = {H};
  const int K = {K};
  const int V = {V};
  const int HV = H * K / 64;
  const int N = B;
  const int BK = K;
  const int BV = std::min(32, V);

  const choreo::f32 softplus_beta = {SOFTPLUS_BETA}f;
  const choreo::f32 softplus_threshold = {SOFTPLUS_THRESHOLD}f;
  const choreo::f32 scale = 1.0f / std::sqrt(static_cast<choreo::f32>(K));

  // Read inputs from binary files
  auto read_bin = [](const std::string& path, void* dst, size_t bytes) {{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {{ std::cerr << "Cannot open " << path << std::endl; exit(1); }}
    f.read(reinterpret_cast<char*>(dst), bytes);
    f.close();
  }};

  std::string dir = "{ref_dir}/";

  size_t sz_A_log = HV * 4;
  size_t sz_a = (size_t)B * T * HV * 2;
  size_t sz_dtb = HV * 2;
  size_t sz_q = (size_t)B * T * H * K * 2;
  size_t sz_k = (size_t)B * T * H * K * 2;
  size_t sz_v = (size_t)B * T * HV * V * 2;
  size_t sz_b = (size_t)B * T * HV * 2;
  size_t sz_o = (size_t)B * T * HV * V * 2;
  size_t sz_init = (size_t)B * HV * K * V * 4;

  std::vector<char> A_log_buf(sz_A_log), a_buf(sz_a), dtb_buf(sz_dtb);
  std::vector<char> q_buf(sz_q), k_buf(sz_k), v_buf(sz_v), b_buf(sz_b);
  std::vector<char> init_buf(sz_init);
  std::vector<int32_t> idx_buf(B);

  read_bin(dir + "A_log.bin", A_log_buf.data(), sz_A_log);
  read_bin(dir + "a.bin", a_buf.data(), sz_a);
  read_bin(dir + "dt_bias.bin", dtb_buf.data(), sz_dtb);
  read_bin(dir + "q.bin", q_buf.data(), sz_q);
  read_bin(dir + "k.bin", k_buf.data(), sz_k);
  read_bin(dir + "v.bin", v_buf.data(), sz_v);
  read_bin(dir + "b.bin", b_buf.data(), sz_b);
  read_bin(dir + "initial_state_source_in.bin", init_buf.data(), sz_init);
  read_bin(dir + "initial_state_indices.bin", idx_buf.data(), B * 4);

  auto A_log = choreo::make_spanview<1, choreo::f32>(
      reinterpret_cast<choreo::f32*>(A_log_buf.data()), {{(size_t)HV}});
  auto a = choreo::make_spanview<3, choreo::bf16>(
      reinterpret_cast<choreo::bf16*>(a_buf.data()),
      {{(size_t)B, (size_t)T, (size_t)HV}});
  auto dt_bias = choreo::make_spanview<1, choreo::bf16>(
      reinterpret_cast<choreo::bf16*>(dtb_buf.data()), {{(size_t)HV}});
  auto q = choreo::make_spanview<4, choreo::bf16>(
      reinterpret_cast<choreo::bf16*>(q_buf.data()),
      {{(size_t)B, (size_t)T, (size_t)H, (size_t)K}});
  auto k = choreo::make_spanview<4, choreo::bf16>(
      reinterpret_cast<choreo::bf16*>(k_buf.data()),
      {{(size_t)B, (size_t)T, (size_t)H, (size_t)K}});
  auto v = choreo::make_spanview<4, choreo::bf16>(
      reinterpret_cast<choreo::bf16*>(v_buf.data()),
      {{(size_t)B, (size_t)T, (size_t)HV, (size_t)V}});
  auto b_in = choreo::make_spanview<3, choreo::bf16>(
      reinterpret_cast<choreo::bf16*>(b_buf.data()),
      {{(size_t)B, (size_t)T, (size_t)HV}});
  auto indices = choreo::make_spanview<1, choreo::s32>(
      idx_buf.data(), {{(size_t)B}});
  std::vector<choreo::s32> cu_seqlens_dummy(N, 0);
  auto cu_seqlens = choreo::make_spanview<1, choreo::s32>(
      cu_seqlens_dummy.data(), {{(size_t)N}});

  auto initial_state_source = choreo::make_spandata<choreo::f32>(B, HV, K, V);
  std::memcpy(initial_state_source.data(), init_buf.data(), sz_init);

  auto o = choreo::make_spandata<choreo::bf16>(B, T, HV, V);

  // Run kernel
  fused_sigmoid_gating_delta_rule_update_kernel_bf16_BK128_BV32_UseInitalState_UseQkL2normInKernel(
      A_log, a, dt_bias, q, k, v, b_in, o.view(), initial_state_source.view(),
      indices, cu_seqlens, scale, softplus_beta, softplus_threshold);

  // Write output
  std::ofstream out("choreo_o.bin", std::ios::binary);
  out.write(reinterpret_cast<const char*>(o.data()), sz_o);
  out.close();

  std::ofstream out2("choreo_final_state.bin", std::ios::binary);
  out2.write(reinterpret_cast<const char*>(initial_state_source.data()), sz_init);
  out2.close();

  return 0;
}}
"""
    return prefix + new_main


def _extract_heap_simulator_block(cu_source: str) -> str:
    """Extract the JIT memory reuse block from generated .cu source."""
    begin = cu_source.find("// JIT memory reuse begin")
    end = cu_source.find("// JIT memory reuse end")
    if begin < 0 or end < 0:
        return ""
    return cu_source[begin:end + len("// JIT memory reuse end")]


def _extract_kernel_name(cu_source: str) -> str:
    """Extract the __global__ kernel function name."""
    m = re.search(r'__global__\s+void\s+(__choreo_device_\w+)\s*\(', cu_source)
    return m.group(1) if m else ""


def _count_kernel_mr_offsets(cu_source: str) -> int:
    """Count mr_offset parameters in the __global__ kernel signature."""
    m = re.search(r'__global__\s+void\s+__choreo_device_\w+\s*\([^)]+\)',
                  cu_source, re.DOTALL)
    if not m:
        return 0
    sig = m.group(0)
    return len(re.findall(r'unsigned long mr_offset_', sig))


def replace_main_with_bench(cu_source: str,
                            B: int,
                            T: int,
                            ref_dir: str,
                            n_warmup: int = 3,
                            n_runs: int = 10) -> str:
    """Replace or append main() for benchmarking with CUDA event timing.
    Directly launches the __global__ kernel to avoid host wrapper overhead."""
    main_start = cu_source.find("int main(")
    if main_start == -1:
        main_start = cu_source.find("int main (")
    if main_start >= 0:
        prefix = cu_source[:main_start]
    else:
        prefix = cu_source + "\n\n"

    heap_block = _extract_heap_simulator_block(cu_source)
    kernel_name = _extract_kernel_name(cu_source)
    if not kernel_name:
        raise ValueError("Cannot find __global__ kernel name in .cu")

    # Count shared memory offset parameters from kernel signature
    n_chunks = _count_kernel_mr_offsets(cu_source)
    if n_chunks > 0:
        offsets_args = ", ".join(f"__co__shared_chunk_offsets[{i}]"
                                 for i in range(n_chunks))
        trailing_args = f"{offsets_args}, __co__shared_spm_size"
    else:
        trailing_args = ""

    # Detect extern __shared__ with explicit __smem_size in host wrapper
    smem_expr = "0"
    if n_chunks > 0:
        smem_expr = "(__co__shared_spm_size + 8) + (128 - 1)"
    else:
        m_smem = re.search(r'unsigned\s+__smem_size\s*=\s*([^;]+);', cu_source)
        if m_smem and 'extern __shared__' in cu_source:
            smem_expr = m_smem.group(1).strip()

    bench_launch = (
        f"{kernel_name}<<<gdims, bdims, smem>>>("
        f"d_A_log, d_a, d_dtb, d_q, d_k, d_v, d_b, d_o, d_init, d_idx, d_cu, "
        f"scale, softplus_beta, softplus_threshold, "
        f"false, false, true, true, "
        f"B, H, HV, K, N, T, V" +
        (f", {trailing_args}" if trailing_args else "") + ")")

    new_main = f"""
int main(int argc, char** argv) {{
  const unsigned B = {B};
  const unsigned T = {T};
  const unsigned H = {H};
  const unsigned K = {K};
  const unsigned V = {V};
  const unsigned HV = H * K / 64;
  const unsigned N = B;

  const float softplus_beta = {SOFTPLUS_BETA}f;
  const float softplus_threshold = {SOFTPLUS_THRESHOLD}f;
  const float scale = 1.0f / std::sqrt(static_cast<float>(K));

  auto read_bin = [](const std::string& path, void* dst, size_t bytes) {{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {{ std::cerr << "Cannot open " << path << std::endl; exit(1); }}
    f.read(reinterpret_cast<char*>(dst), bytes);
    f.close();
  }};

  std::string dir = "{ref_dir}/";

  size_t sz_A_log = HV * 4;
  size_t sz_a = (size_t)B * T * HV * 2;
  size_t sz_dtb = HV * 2;
  size_t sz_q = (size_t)B * T * H * K * 2;
  size_t sz_k = (size_t)B * T * H * K * 2;
  size_t sz_v = (size_t)B * T * HV * V * 2;
  size_t sz_b = (size_t)B * T * HV * 2;
  size_t sz_o = (size_t)B * T * HV * V * 2;
  size_t sz_init = (size_t)B * HV * K * V * 4;

  std::vector<char> h_A_log(sz_A_log), h_a(sz_a), h_dtb(sz_dtb);
  std::vector<char> h_q(sz_q), h_k(sz_k), h_v(sz_v), h_b(sz_b);
  std::vector<char> h_init(sz_init);
  std::vector<int32_t> h_idx(B);

  read_bin(dir + "A_log.bin", h_A_log.data(), sz_A_log);
  read_bin(dir + "a.bin", h_a.data(), sz_a);
  read_bin(dir + "dt_bias.bin", h_dtb.data(), sz_dtb);
  read_bin(dir + "q.bin", h_q.data(), sz_q);
  read_bin(dir + "k.bin", h_k.data(), sz_k);
  read_bin(dir + "v.bin", h_v.data(), sz_v);
  read_bin(dir + "b.bin", h_b.data(), sz_b);
  read_bin(dir + "initial_state_source_in.bin", h_init.data(), sz_init);
  read_bin(dir + "initial_state_indices.bin", h_idx.data(), B * 4);

  float *d_A_log; bf16 *d_a, *d_dtb, *d_q, *d_k, *d_v, *d_b, *d_o;
  float *d_init; int *d_idx, *d_cu;
  cudaMalloc(&d_A_log, sz_A_log);
  cudaMalloc(&d_a, sz_a);
  cudaMalloc(&d_dtb, sz_dtb);
  cudaMalloc(&d_q, sz_q);
  cudaMalloc(&d_k, sz_k);
  cudaMalloc(&d_v, sz_v);
  cudaMalloc(&d_b, sz_b);
  cudaMalloc(&d_o, sz_o);
  cudaMalloc(&d_init, sz_init);
  cudaMalloc(&d_idx, B * 4);
  cudaMalloc(&d_cu, N * 4);

  cudaMemcpy(d_A_log, h_A_log.data(), sz_A_log, cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, h_a.data(), sz_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dtb, h_dtb.data(), sz_dtb, cudaMemcpyHostToDevice);
  cudaMemcpy(d_q, h_q.data(), sz_q, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), sz_k, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), sz_v, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), sz_b, cudaMemcpyHostToDevice);
  cudaMemcpy(d_idx, h_idx.data(), B * 4, cudaMemcpyHostToDevice);
  cudaMemset(d_cu, 0, N * 4);

  {heap_block}

  dim3 gdims(((V + 31) / 32), N, HV);
  dim3 bdims(32, 1, 1);
  unsigned smem = {smem_expr};
  cudaFuncSetAttribute(
      {kernel_name},
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

  #define LAUNCH_KERNEL() {bench_launch}

  cudaEvent_t start_ev, stop_ev;
  cudaEventCreate(&start_ev);
  cudaEventCreate(&stop_ev);

  for (int i = 0; i < {n_warmup}; ++i) {{
    cudaMemcpy(d_init, h_init.data(), sz_init, cudaMemcpyHostToDevice);
    LAUNCH_KERNEL();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {{
      fprintf(stderr, "Kernel launch error: %s\\n", cudaGetErrorString(err));
      return 1;
    }}
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {{
      fprintf(stderr, "Kernel exec error: %s\\n", cudaGetErrorString(err));
      return 1;
    }}
  }}

  for (int i = 0; i < {n_runs}; ++i) {{
    cudaMemcpy(d_init, h_init.data(), sz_init, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(start_ev);
    LAUNCH_KERNEL();
    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);
    float ms = 0;
    cudaEventElapsedTime(&ms, start_ev, stop_ev);
    printf("TIME_MS %.6f\\n", (double)ms);
  }}

  #undef LAUNCH_KERNEL
  cudaEventDestroy(start_ev);
  cudaEventDestroy(stop_ev);
  cudaFree(d_A_log); cudaFree(d_a); cudaFree(d_dtb);
  cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_b);
  cudaFree(d_o); cudaFree(d_init); cudaFree(d_idx); cudaFree(d_cu);
  return 0;
}}
"""
    return prefix + new_main


def bench_choreo(cu_source: str,
                 B: int,
                 T: int,
                 ref_dir: str,
                 extra_flags: str = ""):
    """Compile and benchmark the choreo kernel, return sorted times in ms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        patched_cu = replace_main_with_bench(cu_source, B, T, ref_dir)
        patched_path = os.path.join(tmpdir, "kernel_bench.cu")
        with open(patched_path, "w") as f:
            f.write(patched_cu)

        exe_path = os.path.join(tmpdir, "kernel_bench.out")
        if not compile_cu(patched_path, exe_path, extra_flags):
            return None

        result = subprocess.run(exe_path,
                                capture_output=True,
                                text=True,
                                cwd=tmpdir)
        if result.returncode != 0:
            print(f"[bench choreo FAILED] {result.stdout}\n{result.stderr}")
            return None

        times = []
        for line in result.stdout.split("\n"):
            if line.startswith("TIME_MS "):
                times.append(float(line.split()[1]))
        times.sort()
        return times


def generate_ref_data(B: int, T: int, ref_dir: str):
    """Generate reference inputs and outputs using Triton kernel."""
    torch.manual_seed(42)

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    initial_state_source = torch.randn(B,
                                       HV,
                                       K,
                                       V,
                                       dtype=torch.float32,
                                       device="cuda")
    initial_state_in = initial_state_source.clone()
    initial_state_indices = torch.arange(B, dtype=torch.int64, device="cuda")

    o, final_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=SOFTPLUS_BETA,
        softplus_threshold=SOFTPLUS_THRESHOLD,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        scale=SCALE,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
        is_kda=False,
    )

    os.makedirs(ref_dir, exist_ok=True)

    def save(name, tensor):
        if tensor.dtype == torch.bfloat16:
            arr = tensor.view(torch.uint16)
        elif tensor.dtype == torch.float32:
            arr = tensor.view(torch.uint32)
        elif tensor.dtype in [torch.int64, torch.int32]:
            arr = tensor.to(torch.int32)
        else:
            raise ValueError(f"Unsupported: {tensor.dtype}")
        arr.cpu().numpy().tofile(os.path.join(ref_dir, f"{name}.bin"))

    save("A_log", A_log)
    save("a", a)
    save("dt_bias", dt_bias)
    save("q", q)
    save("k", k)
    save("v", v)
    save("b", b)
    save("initial_state_source_in", initial_state_in)
    save("initial_state_indices", initial_state_indices.int())
    save("o_expected", o)
    save("initial_state_source_out", final_state)

    return o, final_state


def compile_cu(cu_path: str, out_path: str, extra_flags: str = ""):
    """Compile .cu file with nvcc."""
    cmd = (f"{NVCC} -arch sm_90a -std=c++17 "
           f"-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ "
           f"-Xcompiler -static-libstdc++ -lcuda -O3 "
           f"-D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ "
           f"-I{RMSNORM_INCLUDE} -I{CUTLASS_INCLUDE} "
           f"-L{CUDA_HOME}/lib64 -lcuda "
           f"-I{CHOREO_RUNTIME} "
           f"--resource-usage "
           f"{extra_flags} "
           f"{cu_path} -o {out_path}")
    print(f"[compile] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[compile FAILED]\n{result.stderr}")
        return False
    for line in result.stderr.split("\n"):
        if any(k in line for k in ["registers", "spill", "stack", "smem"]):
            print(f"  {line.strip()}")
    return True


def run_choreo_kernel(exe_path: str, work_dir: str):
    """Run the compiled choreo kernel."""
    result = subprocess.run(exe_path,
                            capture_output=True,
                            text=True,
                            cwd=work_dir)
    if result.returncode != 0:
        print(
            f"[run FAILED] exit={result.returncode}\n{result.stdout}\n{result.stderr}"
        )
        return False
    return True


def compare_outputs(ref_o, choreo_o_path: str, ulp_tol=16):
    """Compare choreo output against Triton reference.
    Allows up to ulp_tol ULP difference in bf16 representation."""
    choreo_raw = np.fromfile(choreo_o_path, dtype=np.uint16)
    raw_ref = ref_o.view(torch.uint16).cpu().reshape(-1).numpy()

    n = min(len(choreo_raw), len(raw_ref))
    fail_count = 0
    small_diff_count = 0
    for i in range(n):
        if choreo_raw[i] != raw_ref[i]:
            diff = abs(int(choreo_raw[i]) - int(raw_ref[i]))
            if diff <= ulp_tol:
                small_diff_count += 1
            else:
                fail_count += 1
                if fail_count <= 5:
                    print(
                        f"  o[{i}]: choreo=0x{choreo_raw[i]:04x} ref=0x{raw_ref[i]:04x} diff={diff}"
                    )

    print(
        f"  Failures (>{ulp_tol} ULP): {fail_count}/{n}, small diffs (1-{ulp_tol} ULP): {small_diff_count}/{n}"
    )
    return fail_count == 0


def bench_triton(B: int, T: int, n_warmup=3, n_runs=10):
    """Benchmark Triton kernel."""
    torch.manual_seed(42)
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    iss = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    indices = torch.arange(B, dtype=torch.int64, device="cuda")

    for _ in range(n_warmup):
        s = iss.clone()
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=SOFTPLUS_BETA,
            softplus_threshold=SOFTPLUS_THRESHOLD,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=s,
            initial_state_indices=indices,
            scale=SCALE,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
            is_kda=False)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_runs):
        s = iss.clone()
        torch.cuda.synchronize()
        start_ev.record()
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=SOFTPLUS_BETA,
            softplus_threshold=SOFTPLUS_THRESHOLD,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=s,
            initial_state_indices=indices,
            scale=SCALE,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
            is_kda=False)
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    times.sort()
    return times


def main():
    parser = argparse.ArgumentParser(
        description="Test choreo kernel vs Triton")
    parser.add_argument("cu_file", help="Path to choreo-generated .cu file")
    parser.add_argument(
        "--fix-stride",
        action="store_true",
        help="Apply span_as stride fix for initial_state_source")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("-T",
                        type=str,
                        default="4,128",
                        help="Comma-separated T values")
    parser.add_argument("-B", type=int, default=2, help="Batch size")
    parser.add_argument("--nvcc-flags",
                        type=str,
                        default="",
                        help="Extra nvcc flags")
    args = parser.parse_args()

    cu_file = os.path.abspath(args.cu_file)
    if not os.path.exists(cu_file):
        print(f"Error: {cu_file} not found")
        sys.exit(1)

    T_values = [int(t) for t in args.T.split(",")]
    B = args.B

    with open(cu_file) as f:
        cu_source = f.read()

    if args.fix_stride:
        cu_source = apply_stride_fix(cu_source)
        print("[fix] Applied stride fix for initial_state_source")

    cu_source = fix_cu_seqlens(cu_source)

    all_passed = True
    for T_val in T_values:
        print(f"\n{'='*60}")
        print(f"Testing B={B} T={T_val}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_dir = os.path.join(tmpdir, "ref")
            print(f"[gen] Generating reference data (Triton) ...")
            ref_o, ref_final = generate_ref_data(B, T_val, ref_dir)

            patched_cu = replace_main_with_test(cu_source, B, T_val, ref_dir)
            patched_path = os.path.join(tmpdir, "kernel.cu")
            with open(patched_path, "w") as f:
                f.write(patched_cu)

            exe_path = os.path.join(tmpdir, "kernel.out")
            print(f"[compile] Compiling ...")
            if not compile_cu(patched_path, exe_path, args.nvcc_flags):
                all_passed = False
                continue

            print(f"[run] Running choreo kernel ...")
            if not run_choreo_kernel(exe_path, tmpdir):
                all_passed = False
                continue

            choreo_o_path = os.path.join(tmpdir, "choreo_o.bin")
            if not os.path.exists(choreo_o_path):
                print("[FAIL] No output file generated")
                all_passed = False
                continue

            print(f"[compare] Checking correctness ...")
            passed = compare_outputs(ref_o, choreo_o_path)
            if passed:
                print(f"  [PASS] B={B} T={T_val}")
            else:
                print(f"  [FAIL] B={B} T={T_val}")
                all_passed = False

        if args.bench:
            ref_dir_bench = os.path.join(tempfile.mkdtemp(), "ref")
            generate_ref_data(B, T_val, ref_dir_bench)
            print(f"[bench] Choreo B={B} T={T_val} ...")
            choreo_times = bench_choreo(cu_source, B, T_val, ref_dir_bench,
                                        args.nvcc_flags)
            if choreo_times:
                mid = len(choreo_times) // 2
                print(
                    f"  Choreo: min={choreo_times[0]:.3f} med={choreo_times[mid]:.3f} max={choreo_times[-1]:.3f} ms"
                )
            print(f"[bench] Triton B={B} T={T_val} ...")
            triton_times = bench_triton(B, T_val)
            mid = len(triton_times) // 2
            print(
                f"  Triton: min={triton_times[0]:.3f} med={triton_times[mid]:.3f} max={triton_times[-1]:.3f} ms"
            )
            if choreo_times:
                ratio = choreo_times[0] / triton_times[0] if triton_times[
                    0] > 0 else float('inf')
                print(f"  Ratio (Choreo/Triton min): {ratio:.2f}x")

    if all_passed:
        print(f"\n{'='*60}")
        print("ALL TESTS PASSED")
    else:
        print(f"\n{'='*60}")
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
