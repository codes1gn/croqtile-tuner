#!/usr/bin/env python3
"""Compare Choreo vs vLLM fused-MoE for Qwen3.5-35B-A3B dimensions.

Choreo: single grouped FP8 GEMM + data prep (quant, sort, scatter)
vLLM:   full MoE MLP (gate_up + SiLU + down projections, BF16)

The comparison is NOT apples-to-apples on FLOPs: vLLM's fused_experts does
2 GEMMs + activation (3x FLOPs), while Choreo does 1 GEMM. We report both
raw latency and normalized per-GEMM TFLOPS for context.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

M = 128
N = 512
K = 2048
NUM_EXPERTS = 256
TOPK = 8
EXPANDED_M = M * TOPK  # 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1,
                        help="CUDA device index")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--choreo-bin", default="",
                        help="Path to Choreo .cute.result binary")
    parser.add_argument("--skip-choreo", action="store_true")
    parser.add_argument("--skip-vllm", action="store_true")
    return parser.parse_args()


def benchmark_choreo(args: argparse.Namespace) -> dict:
    bin_path = args.choreo_bin
    if not bin_path:
        bin_path = os.path.join(
            REPO_ROOT, "build", "ai_tune", "fused_moe_op",
            "fused_moe_qwen35_35b_a3b_fp8_end2end_sm90_iter099_fusememset.cute.result")

    if not os.path.exists(bin_path):
        print(f"[choreo] Binary not found: {bin_path}")
        return {}

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["CHOREO_ENABLE_TIMING"] = "1"
    env["CHOREO_TIMING_WARMUP"] = str(args.warmup)
    env["CHOREO_TIMING_REPEAT"] = str(args.repeat)
    env["CHOREO_SKIP_VERIFY"] = "1"

    proc = subprocess.run(
        ["bash", bin_path, "--execute"],
        capture_output=True, text=True, env=env)
    output = proc.stdout + proc.stderr

    serving_ms_match = re.search(r"Serving-path avg ms:\s*([0-9.]+)", output)
    serving_tflops_match = re.search(r"Serving-path TFLOPS:\s*([0-9.]+)", output)
    e2e_ms_match = re.search(r"End-to-end avg ms:\s*([0-9.]+)", output)
    e2e_tflops_match = re.search(r"End-to-end TFLOPS:\s*([0-9.]+)", output)

    result = {"binary": bin_path}
    if serving_ms_match:
        result["serving_ms"] = float(serving_ms_match.group(1))
    if serving_tflops_match:
        result["serving_tflops"] = float(serving_tflops_match.group(1))
    if e2e_ms_match:
        result["e2e_ms"] = float(e2e_ms_match.group(1))
    if e2e_tflops_match:
        result["e2e_tflops"] = float(e2e_tflops_match.group(1))

    if not serving_ms_match:
        print(f"[choreo] Failed to parse output:\n{output}")
    return result


def benchmark_vllm(args: argparse.Namespace) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from vllm.model_executor.layers.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    hidden_states = torch.randn((M, K), device=device, dtype=dtype) / 10
    gating_output = torch.randn(
        (M, NUM_EXPERTS), device=device, dtype=torch.float32) / 10

    # vLLM MLP weights: gate_up is [experts, 2*N, K], down is [experts, K, N]
    w1 = torch.randn(
        (NUM_EXPERTS, 2 * N, K), device=device, dtype=dtype) / 10
    w2 = torch.randn(
        (NUM_EXPERTS, K, N), device=device, dtype=dtype) / 10

    def cuda_bench(fn, warmup, repeat):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / repeat

    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, gating_output, TOPK, True)

    topk_ms = cuda_bench(
        lambda: fused_topk(hidden_states, gating_output, TOPK, True),
        args.warmup, args.repeat)

    experts_ms = cuda_bench(
        lambda: fused_experts(hidden_states, w1, w2, topk_weights, topk_ids),
        args.warmup, args.repeat)

    e2e_ms = cuda_bench(
        lambda: (
            fused_topk(hidden_states, gating_output, TOPK, True),
            fused_experts(hidden_states, w1, w2,
                          *fused_topk(hidden_states, gating_output, TOPK, True)[:2]),
        )[-1],
        args.warmup, args.repeat)

    # vLLM FLOPs: gate_up GEMM (2*expanded_m*2N*K) + down GEMM (2*expanded_m*K*N)
    # = 2*1024*1024*2048 + 2*1024*2048*512 = 6.44e9
    vllm_gemm_flops = (
        2.0 * EXPANDED_M * 2 * N * K +  # gate_up
        2.0 * EXPANDED_M * K * N         # down
    )
    experts_tflops = (vllm_gemm_flops / (experts_ms / 1000.0)) / 1e12

    return {
        "topk_ms": topk_ms,
        "experts_ms": experts_ms,
        "e2e_ms": e2e_ms,
        "experts_tflops": experts_tflops,
        "vllm_total_flops": vllm_gemm_flops,
        "device": torch.cuda.get_device_name(0),
    }


def main():
    args = parse_args()

    choreo_single_gemm_flops = 2.0 * EXPANDED_M * N * K  # 2.147e9

    print("=" * 72)
    print("Choreo vs vLLM — Fused MoE — Qwen3.5-35B-A3B Dimensions")
    print("=" * 72)
    print(f"  M={M}, N={N}, K={K}, experts={NUM_EXPERTS}, topk={TOPK}")
    print(f"  expanded_m={EXPANDED_M}")
    print(f"  warmup={args.warmup}, repeat={args.repeat}, GPU={args.gpu}")
    print()

    choreo_result = {}
    vllm_result = {}

    if not args.skip_choreo:
        print("[Choreo] Running benchmark...")
        choreo_result = benchmark_choreo(args)
        if choreo_result:
            print(f"  Binary: {choreo_result.get('binary', '?')}")
            if "serving_ms" in choreo_result:
                print(f"  Serving-path:  {choreo_result['serving_ms']:.4f} ms"
                      f"  ({choreo_result.get('serving_tflops', 0):.2f} TFLOPS)")
            if "e2e_ms" in choreo_result:
                print(f"  End-to-end:    {choreo_result['e2e_ms']:.4f} ms"
                      f"  ({choreo_result.get('e2e_tflops', 0):.2f} TFLOPS)")
            print(f"  Workload: 1 grouped FP8 GEMM + data prep (quant, sort, scatter)")
            print(f"  FLOPs:    {choreo_single_gemm_flops:.3e}")
        print()

    if not args.skip_vllm:
        print("[vLLM] Running benchmark...")
        vllm_result = benchmark_vllm(args)
        if vllm_result:
            print(f"  Device: {vllm_result.get('device', '?')}")
            print(f"  fused_topk:    {vllm_result['topk_ms']:.4f} ms")
            print(f"  fused_experts: {vllm_result['experts_ms']:.4f} ms"
                  f"  ({vllm_result['experts_tflops']:.2f} TFLOPS)")
            print(f"  End-to-end:    {vllm_result['e2e_ms']:.4f} ms")
            print(f"  Workload: 2 BF16 GEMMs (gate_up[{2*N}x{K}] + down[{K}x{N}]) + SiLU")
            print(f"  FLOPs:    {vllm_result['vllm_total_flops']:.3e} (3x Choreo)")
        print()

    if choreo_result and vllm_result and "serving_ms" in choreo_result:
        print("-" * 72)
        print("Comparison Summary")
        print("-" * 72)

        choreo_ms = choreo_result["serving_ms"]
        vllm_experts_ms = vllm_result["experts_ms"]

        print(f"  Raw latency (serving/experts):")
        print(f"    Choreo:  {choreo_ms:.4f} ms  (1 FP8 GEMM + prep)")
        print(f"    vLLM:    {vllm_experts_ms:.4f} ms  (2 BF16 GEMMs + SiLU)")
        print(f"    Ratio:   {vllm_experts_ms / choreo_ms:.2f}x (vLLM / Choreo)")
        print()

        # Per-GEMM comparison (normalize by GEMM count)
        choreo_per_gemm_tflops = choreo_result.get("serving_tflops", 0)
        # vLLM does 3x the FLOPs in those 2 GEMMs, so per-GEMM TFLOPS = total / 3
        vllm_per_gemm_tflops = vllm_result["experts_tflops"] / 3.0

        print(f"  Per-GEMM TFLOPS (normalized):")
        print(f"    Choreo:  {choreo_per_gemm_tflops:.2f} TFLOPS"
              f"  (FP8×FP8, blockscaled)")
        print(f"    vLLM:    {vllm_per_gemm_tflops:.2f} TFLOPS"
              f"  (BF16, Triton grouped)")
        if vllm_per_gemm_tflops > 0:
            print(f"    Ratio:   {choreo_per_gemm_tflops / vllm_per_gemm_tflops:.2f}x"
                  f"  (Choreo / vLLM)")
        print()

        # End-to-end comparison
        if "e2e_ms" in choreo_result:
            choreo_e2e = choreo_result["e2e_ms"]
            vllm_e2e = vllm_result["e2e_ms"]
            print(f"  E2E latency (including routing):")
            print(f"    Choreo:  {choreo_e2e:.4f} ms")
            print(f"    vLLM:    {vllm_e2e:.4f} ms")
            print(f"    Ratio:   {vllm_e2e / choreo_e2e:.2f}x (vLLM / Choreo)")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
