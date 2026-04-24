#!/usr/bin/env python3
"""
CuTe DSL FP32 GEMM kernel - Coalesced epilogue stores.
iter003_coalesced_store: Larger K tile (16) and more pipeline stages (4) for better latency hiding.

Target: 16384 x 16384 x 16384 matmul
C = A @ B^T where A is (M,K), B is (N,K), C is (M,N)
"""

import os
import argparse
import time
from typing import Tuple

import torch
import cuda.bindings.driver as cuda

os.environ["CUTE_DSL_ARCH"] = "sm_90a"

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

# Problem dimensions
M, N, K = 16384, 16384, 16384


class SgemmSimple:
    """FP32 GEMM using CuTe DSL - SIMT with async copies."""
    
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 128, 8),
        num_stages: int = 3,
        num_threads: int = 256,
    ):
        self._cta_tiler = cta_tiler
        self._num_stages = num_stages
        self._num_threads = num_threads
        
        self._bM, self._bN, self._bK = self._cta_tiler
        assert self._bM % 16 == 0
        assert self._bN % 16 == 0
        assert self._num_stages >= 3
        
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
        padding_a = 4 if self.a_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
        padding_b = 4 if self.b_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
        
        sA_layout = cute.make_layout(
            (self._bM, self._bK, self._num_stages),
            stride=(1, (self._bM + padding_a), self._bK * (self._bM + padding_a)),
        )
        sB_layout = cute.make_layout(
            (self._bN, self._bK, self._num_stages),
            stride=(1, (self._bN + padding_b), self._bK * (self._bN + padding_b)),
        )
        
        # Thread and value layouts for copy
        tA = cute.make_layout(
            (self._num_threads // self._bK, self._bK), stride=(self._bK, 1)
        )
        tB = cute.make_layout(
            (self._num_threads // self._bK, self._bK), stride=(self._bK, 1)
        )
        vA = cute.make_layout((1, 1))
        vB = cute.make_layout((1, 1))
        
        # FP32 = 32 bits, minimum for async copy
        atom_async_copy_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width,
        )
        atom_async_copy_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mB.element_type,
            num_bits_per_copy=mB.element_type.width,
        )
        
        # Vectorize for col-major layouts
        if cutlass.const_expr(self.a_major_mode == utils.LayoutEnum.COL_MAJOR):
            num_vectorized = 4 if (mA.layout[0].max_alignment % 16 == 0) else 1
            atom_async_copy_A = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mA.element_type,
                num_bits_per_copy=mA.element_type.width * num_vectorized,
            )
            major_mode_size = self._bM // num_vectorized
            tA = cute.make_layout(
                (major_mode_size, self._num_threads // major_mode_size),
                stride=(1, major_mode_size),
            )
            vA = cute.make_layout((num_vectorized, 1))
        
        if cutlass.const_expr(self.b_major_mode == utils.LayoutEnum.COL_MAJOR):
            num_vectorized = 4 if (mB.layout[0].max_alignment % 16 == 0) else 1
            atom_async_copy_B = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mB.element_type,
                num_bits_per_copy=mB.element_type.width * num_vectorized,
            )
            major_mode_size = self._bN // num_vectorized
            tB = cute.make_layout(
                (major_mode_size, self._num_threads // major_mode_size),
                stride=(1, major_mode_size),
            )
            vB = cute.make_layout((num_vectorized, 1))
        
        tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy_B, tB, vB)
        
        # MMA layout for FP32 SIMT
        atoms_layout = cute.make_layout(
            (self._num_threads // 16, 16, 1), stride=(16, 1, 0)
        )
        if cutlass.const_expr(self.c_major_mode == utils.LayoutEnum.COL_MAJOR):
            atoms_layout = cute.make_layout(
                (16, self._num_threads // 16, 1), stride=(1, 16, 0)
            )
        
        op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
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
        
        # Handle k-residue
        residue_k = mA.shape[1] - self._bK * gA.shape[2]
        gA = cute.domain_offset((0, residue_k, 0), gA)
        gB = cute.domain_offset((0, residue_k, 0), gB)
        
        # Shared memory
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
        
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)
        
        # Predicates
        mcA = cute.make_identity_tensor(mA.shape)
        mcB = cute.make_identity_tensor(mB.shape)
        cA = cute.local_tile(
            mcA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        cB = cute.local_tile(
            mcB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
        )
        cA = cute.domain_offset((0, residue_k, 0), cA)
        cB = cute.domain_offset((0, residue_k, 0), cB)
        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)
        
        # Predicate tensors
        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
                stride=(cute.size(tAsA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_rmem_tensor(
            cute.make_layout(
                (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        
        tApA_residue_k = cute.make_rmem_tensor(
            cute.make_layout(
                (tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
                stride=(cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),
                       cute.size(tAsA, mode=[2]), 1),
            ),
            cutlass.Boolean,
        )
        tBpB_residue_k = cute.make_rmem_tensor(
            cute.make_layout(
                (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                stride=(cute.size(tBsB, mode=[1]) * cute.size(tBsB, mode=[2]),
                       cute.size(tBsB, mode=[2]), 1),
            ),
            cutlass.Boolean,
        )
        
        # Set predicates
        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(
                    tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                )
        
        for rest_v in range(tApA_residue_k.shape[0]):
            for m in range(tApA_residue_k.shape[1]):
                for k in range(tApA_residue_k.shape[2]):
                    coord_A = tAcA[(0, rest_v), m, k, 0]
                    tApA_residue_k[rest_v, m, k] = cute.elem_less(
                        (coord_A[0], cutlass.Int32(-1)), (mA.shape[0], coord_A[1])
                    )
        for rest_v in range(tBpB_residue_k.shape[0]):
            for n in range(tBpB_residue_k.shape[1]):
                for k in range(tBpB_residue_k.shape[2]):
                    coord_B = tBcB[(0, rest_v), n, k, 0]
                    tBpB_residue_k[rest_v, n, k] = cute.elem_less(
                        (coord_B[0], cutlass.Int32(-1)), (mB.shape[0], coord_B[1])
                    )
        
        # Prologue
        k_pipe_max = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)
        
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA_residue_k,
        )
        cute.copy(
            tiled_copy_B,
            tBgB[None, None, None, gmem_pipe_read],
            tBsB[None, None, None, 0],
            pred=tBpB_residue_k,
        )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = gmem_pipe_read + 1 if gmem_pipe_read + 1 < k_tile_count else cutlass.Int32(0)
        
        for k_tile in range(1, k_pipe_max - 1):
            if k_tile < k_tile_count:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, gmem_pipe_read],
                    tAsA[None, None, None, k_tile],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, k_tile],
                    pred=tBpB,
                )
                gmem_pipe_read = gmem_pipe_read + 1 if gmem_pipe_read + 1 < k_tile_count else cutlass.Int32(0)
            cute.arch.cp_async_commit_group()
        
        if k_tile_count < k_pipe_max:
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cutlass.Boolean(0)
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cutlass.Boolean(0)
        
        # MMA partitions
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)
        
        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(k_pipe_max - 1)
        
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_p = tCsB[None, None, None, smem_pipe_read]
        
        k_block_max = cute.size(tCrA, mode=[2])
        
        if k_block_max > 1:
            cute.arch.cp_async_wait_group(k_pipe_max - 2)
            self.cta_sync_barrier.arrive_and_wait()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])
        
        # Mainloop
        for _ in range(k_tile_count):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(k_pipe_max - 2)
                    self.cta_sync_barrier.arrive_and_wait()
                
                k_block_next = (k_block + 1) % k_block_max
                cute.autovec_copy(tCsA_p[None, None, k_block_next], tCrA[None, None, k_block_next])
                cute.autovec_copy(tCsB_p[None, None, k_block_next], tCrB[None, None, k_block_next])
                
                if k_block == 0:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )
                
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )
                
                if k_block == 0:
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, gmem_pipe_read],
                        tBsB[None, None, None, smem_pipe_write],
                        pred=tBpB,
                    )
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == k_pipe_max:
                        smem_pipe_read = cutlass.Int32(0)
                    gmem_pipe_read = gmem_pipe_read + 1 if gmem_pipe_read + 1 < k_tile_count else cutlass.Int32(1)
        
        # Epilogue
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        tCrC.store(epilogue_op(tCrC.load()))
        
        cC = cute.make_identity_tensor(gC.shape)
        tCpC = thr_mma.partition_C(cC)
        predC = cute.make_rmem_tensor(tCrC.layout, cutlass.Boolean)
        residue_m = mC.shape[0] - cutlass.Int32(self._bM) * bidx
        residue_n = mC.shape[1] - cutlass.Int32(self._bN) * bidy
        for i in range(cute.size(tCrC.shape)):
            predC[i] = cute.elem_less(tCpC[i], (residue_m, residue_n))
        
        # Epilogue store - use default copy atom
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC, pred=predC)


def verify():
    """Verify correctness."""
    torch.manual_seed(42)
    
    # FP32 for SIMT baseline
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(N, K, dtype=torch.float32, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    
    a_tensor = from_dlpack(A, assumed_align=16)
    b_tensor = from_dlpack(B, assumed_align=16)
    c_tensor = from_dlpack(C, assumed_align=16)
    
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    
    # Increase K tile from 8 to 16 and stages from 3 to 4 for better latency hiding
    gemm = SgemmSimple(cta_tiler=(128, 128, 16), num_stages=4, num_threads=256)
    
    # Compile
    print("Compiling...")
    compiled_fn = cute.compile(gemm, a_tensor, b_tensor, c_tensor, stream=current_stream)
    
    # Run
    print("Running...")
    compiled_fn(a_tensor, b_tensor, c_tensor)
    torch.cuda.synchronize()
    
    # Reference
    C_ref = torch.mm(A, B.t())
    
    max_abs_err = (C - C_ref).abs().max().item()
    
    if max_abs_err < 1e-3:
        print(f"VERIFY: PASS max_abs_err={max_abs_err:.6f}")
        return True
    else:
        print(f"VERIFY: FAIL max_abs_err={max_abs_err:.6f}")
        return False


def bench(warmup: int = 10, iters: int = 50):
    """Benchmark."""
    torch.manual_seed(42)
    
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(N, K, dtype=torch.float32, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    
    a_tensor = from_dlpack(A, assumed_align=16)
    b_tensor = from_dlpack(B, assumed_align=16)
    c_tensor = from_dlpack(C, assumed_align=16)
    
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    
    # Increase K tile from 8 to 16 and stages from 3 to 4 for better latency hiding
    gemm = SgemmSimple(cta_tiler=(128, 128, 16), num_stages=4, num_threads=256)
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
    print(f"CuTe DSL SGEMM (FP32): {M}x{N}x{K}")
    print("SIMT baseline - async copy + FMA")
    
    if verify():
        bench()
