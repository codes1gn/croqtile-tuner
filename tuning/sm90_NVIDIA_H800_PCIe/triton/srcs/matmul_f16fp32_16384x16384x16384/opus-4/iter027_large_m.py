import torch, triton, triton.language as tl
M, N, K = 16384, 16384, 16384
BLOCK_M, BLOCK_N, BLOCK_K = 256, 128, 64
GROUP_SIZE_M = 8
num_stages, num_warps = 3, 8

@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M); num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n; group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M; group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc); a_ptrs += BLOCK_K * stride_ak; b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)

def matmul_triton(A, B):
    M_dim, K_dim = A.shape; K_dim, N_dim = B.shape
    C = torch.empty((M_dim, N_dim), device=A.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M_dim, META['BLOCK_M']) * triton.cdiv(N_dim, META['BLOCK_N']),)
    matmul_kernel[grid](A, B, C, M_dim, N_dim, K_dim, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=GROUP_SIZE_M, num_stages=num_stages, num_warps=num_warps)
    return C

if __name__ == "__main__":
    torch.manual_seed(42)
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    for _ in range(10): _ = matmul_triton(A, B)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(50): _ = matmul_triton(A, B)
    end.record(); torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / 50
    tflops = 2 * M * N * K / elapsed_ms / 1e9
    print(f"iter027_large_m: {BLOCK_M}x{BLOCK_N}x{BLOCK_K} s{num_stages}w{num_warps} => {tflops:.2f} TFLOPS")
