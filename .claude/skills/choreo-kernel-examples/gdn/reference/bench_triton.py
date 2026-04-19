import torch
import time

torch.manual_seed(42)

B, H, K, V = 2, 8, 128, 128
HV = H * K // 64

scale = 1.0 / (K**0.5)
softplus_beta = 1.0
softplus_threshold = 20.0
use_qk_l2norm_in_kernel = True
is_kda = False

import sys
sys.path.insert(0, '/home/baldlee/workspace/choreo-attn/gdn/reference')
from fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update

for T in [4, 128, 512]:
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
    initial_state_source = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    initial_state_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    # warmup
    for _ in range(3):
        iss = initial_state_source.clone()
        o, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt_bias,
            softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
            q=q, k=k, v=v, b=b,
            initial_state_source=iss,
            initial_state_indices=initial_state_indices,
            scale=scale, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=None, is_kda=is_kda,
        )
    torch.cuda.synchronize()

    # benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(10):
        iss = initial_state_source.clone()
        torch.cuda.synchronize()
        start.record()
        o, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt_bias,
            softplus_beta=softplus_beta, softplus_threshold=softplus_threshold,
            q=q, k=k, v=v, b=b,
            initial_state_source=iss,
            initial_state_indices=initial_state_indices,
            scale=scale, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=None, is_kda=is_kda,
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median = times[len(times)//2]
    print(f"T={T:4d}: median={median:.3f} ms, min={times[0]:.3f} ms, all={[f'{t:.3f}' for t in times]}")
