import torch
import triton
import sys
sys.path.insert(0, '/home/baldlee/workspace/choreo-attn/gdn/reference')
from fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update_kernel

B, T, H, K, V = 2, 128, 8, 128, 128
HV = H * K // 64
BK, BV = 128, 32
NK, NV = 1, V // BV

A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
o = torch.empty(NK, B, T, HV, V, dtype=torch.bfloat16, device="cuda")
iss = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
h0_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
cu_seqlens = torch.tensor([0, T, 2*T], dtype=torch.int32, device="cuda")

grid = (NK, NV, B * HV)
kernel = fused_sigmoid_gating_delta_rule_update_kernel[grid](
    A_log, a, dt_bias, 1.0, 20.0,
    q, k, v, b, o, iss, h0_indices, cu_seqlens,
    1.0/(K**0.5), T,
    B=B, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
    USE_INITIAL_STATE=True, USE_QK_L2NORM_IN_KERNEL=True,
    IS_VARLEN=True, IS_KDA=False,
    num_warps=1, num_stages=3,
)

print(f"n_regs: {kernel.n_regs}")
print(f"n_spills: {kernel.n_spills}")
print(f"shared: {kernel.metadata.shared}")
