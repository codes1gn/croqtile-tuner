import torch
import helion
import helion.language as hl

M = N = K = 128
G = 32

torch.manual_seed(0)
a = (torch.randn(M, K, device="cuda") * 0.02).to(torch.float8_e4m3fn)
b = (torch.randn(K, N, device="cuda") * 0.02).to(torch.float8_e4m3fn)
a_s = torch.randint(124, 128, (M, K // G), device="cuda", dtype=torch.uint8)
b_s = torch.randint(124, 128, (N, K // G), device="cuda", dtype=torch.uint8)

cfg = helion.Config(
    block_sizes=[32, 32, 32],
    num_warps=4,
    num_stages=2,
    indexing="block_ptr",
)


@helion.kernel(config=cfg, settings=helion.Settings(autotune_effort="none", static_shapes=True))
def k(a, a_s, b, b_s):
    m, kd = a.shape
    _k2, n = b.shape
    out = torch.empty((m, n), dtype=torch.float16, device=a.device)
    for tm in hl.tile(m):
        for tn in hl.tile(n):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(kd):
                s1 = a_s[tm, tk // G]
                s2 = b_s[tn, tk // G]
                sa = s1[:, None]
                sb = s2[:, None]
                acc = hl.dot_scaled(
                    a[tm, tk], sa, "e4m3", b[tk, tn], sb, "e4m3", acc=acc, out_dtype=torch.float32
                )
            out[tm, tn] = acc.to(torch.float16)
    return out


print(k(a, a_s, b, b_s).shape)
