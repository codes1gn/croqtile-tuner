#!/usr/bin/env python3
"""
CuTe DSL F16 GEMM kernel using SM90 WGMMA instructions.
iter009_wgmma_f16: WGMMA based on hopper_helpers patterns.

Uses the higher-level helper functions for SMEM layout creation.
"""

import os
os.environ["CUTE_DSL_ARCH"] = "sm_90a"

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu.warpgroup import (
    OperandMajorMode, OperandSource,
    fence, commit_group, wait_group, Field
)

# Problem dimensions
M, N, K = 16384, 16384, 16384


class WgmmaF16Gemm:
    """F16 GEMM using WGMMA on Hopper."""
    
    def __init__(
        self,
        tile_m: int = 128,
        tile_n: int = 128,
        tile_k: int = 64,
        num_stages: int = 3,
    ):
        self._bM = tile_m
        self._bN = tile_n
        self._bK = tile_k
        self._num_stages = num_stages
        
        self.a_dtype = cutlass.Float16
        self.b_dtype = cutlass.Float16
        self.acc_dtype = cutlass.Float16
        
        self.warpgroup_size = 128
    
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        a_layout = utils.LayoutEnum.from_tensor(mA)
        b_layout = utils.LayoutEnum.from_tensor(mB)
        
        # Create WGMMA tiled MMA using helper
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            a_layout.sm90_mma_major_mode(),
            b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, self._bN),  # MMA tiler M=64, N=tile_n
            a_source=OperandSource.SMEM,
        )
        
        # MMA tiler MNK
        mma_tiler_mnk = (self._bM, self._bN, self._bK)
        
        # Create SMEM layouts using hopper helpers
        sA_layout = sm90_utils.make_smem_layout_a(
            a_layout, mma_tiler_mnk, self.a_dtype, self._num_stages
        )
        sB_layout = sm90_utils.make_smem_layout_b(
            b_layout, mma_tiler_mnk, self.b_dtype, self._num_stages
        )
        
        # Synchronous copy atoms
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.a_dtype)
        
        # Thread layout for copy
        num_threads = self.warpgroup_size
        tA = cute.make_layout(
            (num_threads // self._bK, self._bK), stride=(self._bK, 1)
        )
        tB = cute.make_layout(
            (num_threads // self._bK, self._bK), stride=(self._bK, 1)
        )
        vA = cute.make_layout((1, 1))
        vB = cute.make_layout((1, 1))
        
        tiled_copy_A = cute.make_tiled_copy_tv(copy_atom, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(copy_atom, tB, vB)
        
        # Grid
        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1
        
        self.kernel(
            mA, mB, mC,
            sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B,
            self.tiled_mma,
        ).launch(
            grid=grid_dim,
            block=[num_threads, 1, 1],
            stream=stream,
        )
    
    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        
        cta_tiler = (self._bM, self._bN, self._bK)
        tiler_coord = (bidx, bidy, None)
        
        # Global tiles
        gA = cute.local_tile(mA, tiler=cta_tiler, coord=tiler_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=cta_tiler, coord=tiler_coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, tiler=cta_tiler, coord=tiler_coord, proj=(1, 1, None))
        
        # Allocate SMEM with swizzle separated
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout.outer, 128, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(mB.element_type, sB_layout.outer, 128, swizzle=sB_layout.inner)
        
        # Thread copy slices
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)
        
        # MMA thread slice
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        
        # Fragments
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)
        
        k_tile_count = cute.size(gA, mode=[2])
        k_block_max = cute.size(tCsA, mode=[2])
        
        # Initialize WGMMA
        tiled_mma.set(Field.ACCUMULATE, False)
        
        # K-loop
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            # Load A, B to SMEM - use first stage
            cute.copy(tiled_copy_A, tAgA[None, None, k_tile], tAsA[None, None, None, 0])
            cute.copy(tiled_copy_B, tBgB[None, None, k_tile], tBsB[None, None, None, 0])
            cute.arch.syncthreads()
            
            # Compute MMA
            for k_block in cutlass.range(k_block_max, unroll_full=True):
                fence()
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCsA[None, None, k_block, 0],
                    tCsB[None, None, k_block, 0],
                    tCrC,
                )
                commit_group()
                wait_group(0)
                tiled_mma.set(Field.ACCUMULATE, True)
            
            cute.arch.syncthreads()
        
        # Store result
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC)


def verify():
    """Verify correctness."""
    torch.manual_seed(42)
    
    # Column-major (K-major) tensors
    A_storage = torch.randn(K, M, dtype=torch.float16, device="cuda")
    B_storage = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C_storage = torch.zeros(N, M, dtype=torch.float16, device="cuda")
    A = A_storage.t()
    B = B_storage.t()
    C = C_storage.t()
    
    a_tensor = from_dlpack(A, assumed_align=16)
    b_tensor = from_dlpack(B, assumed_align=16)
    c_tensor = from_dlpack(C, assumed_align=16)
    
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    
    gemm = WgmmaF16Gemm(tile_m=128, tile_n=128, tile_k=64, num_stages=3)
    
    print("Compiling...")
    compiled_fn = cute.compile(gemm, a_tensor, b_tensor, c_tensor, stream=current_stream)
    
    print("Running...")
    compiled_fn(a_tensor, b_tensor, c_tensor, current_stream)
    torch.cuda.synchronize()
    
    # Reference
    C_ref = torch.mm(A.float(), B.t().float()).half()
    
    max_abs_err = (C - C_ref).abs().max().item()
    
    if max_abs_err < 1e-1:
        print(f"VERIFY: PASS max_abs_err={max_abs_err:.6f}")
        return True
    else:
        print(f"VERIFY: FAIL max_abs_err={max_abs_err:.6f}")
        return False


def bench(warmup: int = 10, iters: int = 50):
    """Benchmark."""
    torch.manual_seed(42)
    
    A_storage = torch.randn(K, M, dtype=torch.float16, device="cuda")
    B_storage = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C_storage = torch.zeros(N, M, dtype=torch.float16, device="cuda")
    A = A_storage.t()
    B = B_storage.t()
    C = C_storage.t()
    
    a_tensor = from_dlpack(A, assumed_align=16)
    b_tensor = from_dlpack(B, assumed_align=16)
    c_tensor = from_dlpack(C, assumed_align=16)
    
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    
    gemm = WgmmaF16Gemm(tile_m=128, tile_n=128, tile_k=64, num_stages=3)
    compiled_fn = cute.compile(gemm, a_tensor, b_tensor, c_tensor, stream=current_stream)
    
    for _ in range(warmup):
        compiled_fn(a_tensor, b_tensor, c_tensor, current_stream)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        compiled_fn(a_tensor, b_tensor, c_tensor, current_stream)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iters
    tflops = 2 * M * N * K / elapsed_ms / 1e9
    
    print(f"TFLOPS: {tflops:.2f}   time_ms: {elapsed_ms:.3f}")
    return tflops, elapsed_ms


if __name__ == "__main__":
    print(f"CuTe DSL WGMMA GEMM: {M}x{N}x{K} F16")
    print("WGMMA with hopper_helpers layouts")
    
    if verify():
        bench()
