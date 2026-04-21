import torch, triton, triton.language as tl
M, N, K = 8192, 8192, 8192
BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 256, 128, 128, 4
num_stages, num_warps = 3, 16
@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
           stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    a_scale = tl.load(a_scale_ptr + offs_am, mask=offs_am < M, other=1.0)
    b_scale = tl.load(b_scale_ptr + offs_bn, mask=offs_bn < N, other=1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    acc = acc * a_scale[:, None] * b_scale[None, :]
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

torch.manual_seed(0)
a = torch.randn((M, K), device='cuda').to(torch.float8_e4m3fn)
b = torch.randn((K, N), device='cuda').to(torch.float8_e4m3fn)
a_s = torch.rand((M,), device='cuda') * 2 + 0.5
b_s = torch.rand((N,), device='cuda') * 2 + 0.5
c = torch.empty((M, N), device='cuda', dtype=torch.float16)
grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
for _ in range(5): kernel[grid](a, b, c, a_s, b_s, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M, num_stages=num_stages, num_warps=num_warps)
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(30): kernel[grid](a, b, c, a_s, b_s, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M, num_stages=num_stages, num_warps=num_warps)
end.record()
torch.cuda.synchronize()
print(f"GROUP_M={GROUP_M}: {2*M*N*K/(start.elapsed_time(end)/30)/1e9:.2f} TFLOPS")
