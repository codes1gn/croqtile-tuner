#!/usr/bin/env bash
# cublas_baseline.sh — Measure cuBLAS baseline TFLOPS for a given shape.
#
# USAGE:
#   bash .claude/skills/croq-tune/tools/cublas_baseline.sh \
#       --dtype bf16fp32 --m 512 --n 512 --k 512 \
#       [--warmup 10] [--iters 50]
#
# OUTPUT (stdout): one-line JSON
#   {"tflops": 12.34, "dtype": "bf16fp32", "m": 512, "n": 512, "k": 512,
#    "warmup": 10, "iters": 50, "status": "ok"}
#
# EXIT CODES:
#   0  — success
#   1  — argument error or missing dependencies
#   2  — runtime error (benchmark failed)

set -euo pipefail

DTYPE=""
M=""
N=""
K=""
WARMUP=10
ITERS=50

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dtype)  DTYPE="$2";  shift 2 ;;
        --m)      M="$2";     shift 2 ;;
        --n)      N="$2";     shift 2 ;;
        --k)      K="$2";     shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --iters)  ITERS="$2"; shift 2 ;;
        *) echo "[cublas_baseline] ERROR: unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$DTYPE" || -z "$M" || -z "$N" || -z "$K" ]]; then
    echo "[cublas_baseline] ERROR: --dtype, --m, --n, --k required" >&2
    exit 1
fi

python3 - "$DTYPE" "$M" "$N" "$K" "$WARMUP" "$ITERS" <<'PYEOF'
import sys
import json
import time

dtype_arg, m_str, n_str, k_str, warmup_str, iters_str = sys.argv[1:7]
M, N, K = int(m_str), int(n_str), int(k_str)
WARMUP, ITERS = int(warmup_str), int(iters_str)

try:
    import torch
except ImportError:
    print(json.dumps({"status": "error", "error": "torch not available"}))
    sys.exit(2)

if not torch.cuda.is_available():
    print(json.dumps({"status": "error", "error": "CUDA not available"}))
    sys.exit(2)

# torch.mm dispatches to cuBLAS (cublasGemmEx / cublasLtMatmul) for fp16/bf16.
# Disable TF32 for fair precision-matched comparison.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

dtype_map = {
    "f16": (torch.float16, torch.float16),
    "bf16": (torch.bfloat16, torch.bfloat16),
    "bf16fp32": (torch.bfloat16, torch.float32),
    "f16fp32": (torch.float16, torch.float32),
    "f32": (torch.float32, torch.float32),
    "e4m3": (torch.float8_e4m3fn, torch.float16) if hasattr(torch, "float8_e4m3fn") else (torch.float16, torch.float16),
}

input_dtype, out_dtype = dtype_map.get(dtype_arg, (torch.float16, torch.float16))

try:
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).to(input_dtype)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).to(input_dtype)

    for _ in range(WARMUP):
        torch.mm(A, B)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(ITERS):
        torch.mm(A, B)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / ITERS
    flops = 2.0 * M * N * K
    tflops = (flops / (avg_ms * 1e-3)) / 1e12

    result = {
        "tflops": round(tflops, 4),
        "dtype": dtype_arg,
        "m": M,
        "n": N,
        "k": K,
        "warmup": WARMUP,
        "iters": ITERS,
        "avg_ms": round(avg_ms, 4),
        "status": "ok",
    }
    print(json.dumps(result))

except Exception as e:
    print(json.dumps({"status": "error", "error": str(e)}))
    sys.exit(2)
PYEOF
