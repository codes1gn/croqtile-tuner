import torch

from fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update


def test_fused_sigmoid_gating_delta_rule_update_bf16():
    torch.manual_seed(42)

    B, T, H, K, V = 2, 4, 8, 128, 128
    HV = H * K // 64

    A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
    a = (
        torch.randn(B, T, HV, dtype=torch.float32, device="cuda")
        if True
        else torch.randn(HV, dtype=torch.float32, device="cuda")
    )
    dt_bias = (
        torch.randn(HV, dtype=torch.float32, device="cuda")
        if True
        else torch.randn(HV, K, dtype=torch.float32, device="cuda")
    )
    softplus_beta = 1.0
    softplus_threshold = 20.0

    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")

    initial_state_source = torch.randn(B, HV, K, V, dtype=torch.float32, device="cuda")
    initial_state_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")

    scale = None
    use_qk_l2norm_in_kernel = False
    cu_seqlens = None
    is_kda = True

    o = fused_sigmoid_gating_delta_rule_update(
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

    print(f"Output shape: {o.shape}")
    print(f"Output dtype: {o.dtype}")
    print(f"Output bf16 device: {o.device}")
    print(f"Output bf16 min: {o.min().item():.6f}")
    print(f"Output bf16 max: {o.max().item():.6f}")
    print(f"Output bf16 mean: {o.mean().item():.6f}")

    o_cpu = o.detach().cpu()
    o_fp32 = o.float()

    output_path = (
        "/home/baldlee/workspace/choreo-attn/gdn/reference/test_output_bf16.pt"
    )
    torch.save(
        {
            "o_bf16": o_cpu,
            "o_fp32": o_fp32.cpu(),
            "shape": o.shape,
            "dtype": str(o.dtype),
        },
        output_path,
    )
    print(f"Saved bf16 output to {output_path}")

    return o


if __name__ == "__main__":
    test_fused_sigmoid_gating_delta_rule_update_bf16()
