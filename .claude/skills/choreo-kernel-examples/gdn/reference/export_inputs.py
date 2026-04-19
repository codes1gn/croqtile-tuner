import torch
import numpy as np

torch.manual_seed(42)

B, T, H, K, V = 2, 128, 8, 128, 128
HV = H * K // 64

A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
initial_state_source = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
initial_state_source_in = initial_state_source.clone()
initial_state_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
cu_seqlens = None

scale = 1.0 / (K**0.5)
softplus_beta = 1.0
softplus_threshold = 20.0
use_qk_l2norm_in_kernel = True
is_kda = False

from fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update

o, initial_state_source_out = fused_sigmoid_gating_delta_rule_update(
    A_log=A_log,
    a=a,
    dt_bias=dt_bias,
    softplus_beta=softplus_beta,
    softplus_threshold=softplus_threshold,
    q=q,
    k=k,
    v=v,
    b=b,
    initial_state_source=initial_state_source,
    initial_state_indices=initial_state_indices,
    scale=scale,
    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    cu_seqlens=cu_seqlens,
    is_kda=is_kda,
)


def save_bin(name, tensor):
    if tensor.dtype == torch.bfloat16:
        arr = tensor.view(torch.uint16)
    elif tensor.dtype == torch.float32:
        arr = tensor.view(torch.uint32)
    elif tensor.dtype in [torch.int64, torch.int32]:
        arr = tensor.to(torch.int32)
    else:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    arr.cpu().numpy().tofile(
        f"/home/baldlee/workspace/choreo-attn/gdn/reference/{name}.bin"
    )


save_bin("A_log", A_log)
save_bin("a", a)
save_bin("dt_bias", dt_bias)
save_bin("q", q)
save_bin("k", k)
save_bin("v", v)
save_bin("b", b)
save_bin("initial_state_source_in", initial_state_source_in)
save_bin("initial_state_indices", initial_state_indices.int())

save_bin("o_expected", o)
save_bin("initial_state_source_out", initial_state_source_out)

print("All inputs and outputs saved to .bin files")
print(f"A_log shape: {A_log.shape}, dtype: {A_log.dtype}")
print(f"a shape: {a.shape}, dtype: {a.dtype}")
print(f"dt_bias shape: {dt_bias.shape}, dtype: {dt_bias.dtype}")
print(f"q shape: {q.shape}, dtype: {q.dtype}")
print(f"k shape: {k.shape}, dtype: {k.dtype}")
print(f"v shape: {v.shape}, dtype: {v.dtype}")
print(f"b shape: {b.shape}, dtype: {b.dtype}")
print(
    f"initial_state_source_in shape: {initial_state_source.shape}, dtype: {initial_state_source.dtype}"
)
print(f"o_expected shape: {o.shape}, dtype: {o.dtype}")
print(
    f"initial_state_source_out shape: {initial_state_source_out.shape}, dtype: {initial_state_source_out.dtype}"
)
