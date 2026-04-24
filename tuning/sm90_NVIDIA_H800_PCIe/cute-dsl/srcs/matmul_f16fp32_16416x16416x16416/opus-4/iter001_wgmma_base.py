#!/usr/bin/env python3
"""
CuTe DSL F16->FP32 GEMM kernel using SM90 SIMT FMA.
iter001_wgmma_base: Synchronous copy for non-aligned 16416 shape.

C = A @ B^T where A is (M,K), B is (N,K), C is (M,N)
"""

import os
from typing import Tuple

import torch
import cuda.bindings.driver as cuda

os.environ["CUTE_DSL_ARCH"] = "sm_90a"

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

# Problem dimensions
M, N, K = 16416, 16416, 16416


class F16Gemm:
    """F16->FP32 GEMM using CuTe DSL with synchronous copies."""
    
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 128, 32),
        num_stages: int = 2,
        num_threads: int = 256,
    ):
        self._cta_tiler = cta_tiler
        self._num_stages = num_stages
        self._num_threads = num_threads
        
        self._bM, self._bN, self._bK = self._cta_tiler
        
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )
    
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)
        
        # SMEM layouts with padding
        padding = 8
        
        sA_layout = cute.make_layout(
            (self._bM, self._bK, self._num_stages),
            stride=(1, (self._bM + padding), self._bK * (self._bM + padding)),
        )
        sB_layout = cute.make_layout(
            (self._bN, self._bK, self._num_stages),
            stride=(1, (self._bN + padding), self._bK * (self._bN + padding)),
        )
        
        # Use synchronous universal copy (no alignment requirements)
        atom_copy_A = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mA.element_type,
        )
        atom_copy_B = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mB.element_type,
        )
        
        # Thread layout for copy
        tA = cute.make_layout(
            (self._num_threads // self._bK, self._bK), stride=(self._bK, 1)
        )
        tB = cute.make_layout(
            (self._num_threads // self._bK, self._bK), stride=(self._bK, 1)
        )
        vA = cute.make_layout((1, 1))
        vB = cute.make_layout((1, 1))
        
        tiled_copy_A = cute.make_tiled_copy_tv(atom_copy_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_copy_B, tB, vB)
        
        # MMA layout - SIMT FMA
        atoms_layout = cute.make_layout(
            (self._num_threads // 16, 16, 1), stride=(16, 1, 0)
        )
        if cutlass.const_expr(self.c_major_mode == utils.LayoutEnum.COL_MAJOR):
            atoms_layout = cute.make_layout(
                (16, self._num_threads // 16, 1), stride=(1, 16, 0)
            )
        
        op = cute.nvgpu.MmaUniversalOp(cutlass.Float16)
        permutation_tiler_M = cute.make_layout(
            (atoms_layout.shape[0], 4), stride=(4, 1)
        )
        permutation_tiler_N = cute.make_layout(
            (atoms_layout.shape[1], 4), stride=(4, 1)
        )
        tiled_mma = cute.make_tiled_mma(
            op,
            atoms_layout,
            permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
        )
        
        # Grid dimensions
        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1
        
        self.kernel(
            mA, mB, mC,
            sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B,
            tiled_mma,
            epilogue_op,
        ).launch(
            grid=grid_dim,
            block=[cute.size(atoms_layout), 1, 1],
            stream=stream,
        )
    
    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        tiler_coord = (bidx, bidy, None)
        thr_mma = tiled_mma.get_slice(tidx)
        
        # Get tiles for this CTA
        gA = cute.local_tile(
            mA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        gB = cute.local_tile(
            mB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
        )
        gC = cute.local_tile(
            mC, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, 1, None)
        )
        
        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
        
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)
        
        # MMA partitions
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)
        
        k_tile_count = cute.size(tAgA, mode=[3])
        k_block_max = cute.size(tCrA, mode=[2])
        
        # Simple mainloop without complex predication
        smem_pipe = cutlass.Int32(0)
        
        # Load first tile
        cute.copy(tiled_copy_A, tAgA[None, None, None, 0], tAsA[None, None, None, 0])
        cute.copy(tiled_copy_B, tBgB[None, None, None, 0], tBsB[None, None, None, 0])
        self.cta_sync_barrier.arrive_and_wait()
        
        # Main loop
        for k_tile in range(k_tile_count):
            # Load from SMEM to registers
            tCsA_p = tCsA[None, None, None, smem_pipe]
            tCsB_p = tCsB[None, None, None, smem_pipe]
            
            # Prefetch next tile to alternate buffer
            next_smem_pipe = (smem_pipe + 1) % self._num_stages
            if k_tile + 1 < k_tile_count:
                cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile + 1], tAsA[None, None, None, next_smem_pipe])
                cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile + 1], tBsB[None, None, None, next_smem_pipe])
            
            # Compute
            for k_block in range(k_block_max, unroll_full=True):
                cute.autovec_copy(tCsA_p[None, None, k_block], tCrA[None, None, k_block])
                cute.autovec_copy(tCsB_p[None, None, k_block], tCrB[None, None, k_block])
                
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )
            
            # Sync before next iteration
            self.cta_sync_barrier.arrive_and_wait()
            smem_pipe = next_smem_pipe
        
        # Epilogue
        tCrC.store(epilogue_op(tCrC.load()))
        
        # Store result (unpredicated - assumes interior tiles)
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC)


def verify():
    """Verify correctness with small test first."""
    torch.manual_seed(42)
    
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    
    a_tensor = from_dlpack(A, assumed_align=16)
    b_tensor = from_dlpack(B, assumed_align=16)
    c_tensor = from_dlpack(C, assumed_align=16)
    
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    
    gemm = F16Gemm(cta_tiler=(96, 96, 32), num_stages=2, num_threads=256)
    
    compiled_fn = cute.compile(gemm, a_tensor, b_tensor, c_tensor, stream=current_stream)
    
    compiled_fn(a_tensor, b_tensor, c_tensor)
    torch.cuda.synchronize()
    
    # Reference: C = A @ B^T
    C_ref = torch.mm(A.float(), B.t().float()).half()
    
    max_abs_err = (C - C_ref).abs().max().item()
    
    if max_abs_err < 1e-2:
        print(f"VERIFY: PASS max_abs_err={max_abs_err:.6f}")
        return True
    else:
        # Debug: check a subset
        err = (C - C_ref).abs()
        print(f"VERIFY: FAIL max_abs_err={max_abs_err:.6f}")
        print(f"  Error at: {(err == max_abs_err).nonzero()[0].tolist()}")
        print(f"  C[0,0]={C[0,0].item():.6f} vs C_ref[0,0]={C_ref[0,0].item():.6f}")
        print(f"  C[127,127]={C[127,127].item():.6f} vs C_ref[127,127]={C_ref[127,127].item():.6f}")
        return False


def bench(warmup: int = 10, iters: int = 50):
    """Benchmark."""
    torch.manual_seed(42)
    
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    
    a_tensor = from_dlpack(A, assumed_align=16)
    b_tensor = from_dlpack(B, assumed_align=16)
    c_tensor = from_dlpack(C, assumed_align=16)
    
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    
    gemm = F16Gemm(cta_tiler=(96, 96, 32), num_stages=2, num_threads=256)
    compiled_fn = cute.compile(gemm, a_tensor, b_tensor, c_tensor, stream=current_stream)
    
    # Warmup
    for _ in range(warmup):
        compiled_fn(a_tensor, b_tensor, c_tensor)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        compiled_fn(a_tensor, b_tensor, c_tensor)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iters
    tflops = 2 * M * N * K / elapsed_ms / 1e9
    
    print(f"TFLOPS: {tflops:.2f}   time_ms: {elapsed_ms:.3f}")
    return tflops, elapsed_ms


if __name__ == "__main__":
    print(f"CuTe DSL GEMM: {M}x{N}x{K} F16->FP32")
    print("Using synchronous copy + SIMT FMA")
    
    if verify():
        bench()
