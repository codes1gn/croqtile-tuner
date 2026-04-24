#!/usr/bin/env python3
"""
CuTe DSL F16->FP32 GEMM kernel using SM90 WGMMA instructions.
iter002_wgmma_tma: WGMMA with synchronous copy for H100.

Target: 16384 x 16384 x 16384 matmul with F16 inputs, FP32 accumulator.
C = A @ B^T where A is (M,K) col-major, B is (N,K) col-major, C is (M,N) col-major.
"""

import os
os.environ["CUTE_DSL_ARCH"] = "sm_90a"

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu.warpgroup import (
    OperandMajorMode, OperandSource, make_smem_layout_atom, SmemLayoutAtomKind,
    fence, commit_group, wait_group, Field, MmaF16BF16Op
)

# Problem dimensions
M, N, K = 16384, 16384, 16384


class HopperF16GemmWgmma:
    """
    F16->FP32 GEMM using WGMMA on Hopper with synchronous copy.
    Uses WGMMA for MMA, SIMT for memory.
    """
    
    def __init__(
        self,
        cta_tiler: tuple = (128, 64, 64),
        num_threads: int = 128,
    ):
        self._cta_tiler = cta_tiler
        self._num_threads = num_threads
        
        self._bM, self._bN, self._bK = self._cta_tiler
        
        self.acc_dtype = cutlass.Float32
        self.a_dtype = cutlass.Float16
        self.b_dtype = cutlass.Float16
        
    
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)
        
        # For WGMMA: need to know if A is K-major or M-major
        # Column-major (M,K) means M-stride=1, so K is the major mode
        # Column-major for us means: shape=(M,K), but stored as (K,M).t() in PyTorch
        
        # Determine major modes
        a_is_k_major = self.a_major_mode.is_k_major_a()
        b_is_k_major = self.b_major_mode.is_k_major_b()
        
        # Get MMA major modes
        a_mma_major = self.a_major_mode.sm90_mma_major_mode()
        b_mma_major = self.b_major_mode.sm90_mma_major_mode()
        
        # Create WGMMA tiled MMA
        # MmaF16BF16Op requires tiler_mnk where K=16 for F16
        # Standard Hopper WGMMA shape is 64x64x16 or 64x128x16 etc.
        mma_op = MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            (64, self._bN, 16),  # M=64, N=bN, K=16
            OperandSource.SMEM,
            a_mma_major,
            b_mma_major,
        )
        self.tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(mma_op), (1, 1, 1))
        
        # SMEM layouts - match the MMA expectations with swizzle for bank conflict avoidance
        # For K-major (column-major) operands, use K_SW128 swizzle
        # Shape is (M/N, K) for one tile
        
        a_smem_atom_kind = sm90_utils.get_smem_layout_atom(
            self.a_major_mode, self.a_dtype, 
            self._bK if a_is_k_major else self._bM
        )
        b_smem_atom_kind = sm90_utils.get_smem_layout_atom(
            self.b_major_mode, self.b_dtype,
            self._bK if b_is_k_major else self._bN
        )
        
        a_smem_atom = make_smem_layout_atom(a_smem_atom_kind, self.a_dtype)
        b_smem_atom = make_smem_layout_atom(b_smem_atom_kind, self.b_dtype)
        
        # Full SMEM layout tiled to block size
        # For A: shape (bM, bK), for B: shape (bN, bK)
        sA_shape = (self._bM, self._bK)
        sB_shape = (self._bN, self._bK)
        
        sA_layout = cute.tile_to_shape(
            a_smem_atom, sA_shape,
            order=(0, 1) if a_is_k_major else (1, 0)
        )
        sB_layout = cute.tile_to_shape(
            b_smem_atom, sB_shape,
            order=(0, 1) if b_is_k_major else (1, 0)
        )
        
        # Copy atoms - use synchronous vector copy 
        # For column-major F16, the major dimension (K) is contiguous
        # Can vectorize along K direction
        
        # Thread layout for copy: threads distributed across M/N dimension
        # Value layout: each thread copies a vector along K
        
        # For col-major, K is contiguous, so vectorize K
        vec_elements = 8  # 8 F16 = 128 bits
        
        # A copy: bM threads, each copies bK/bM * vec_elements elements
        threads_M = min(self._bM, self._num_threads)
        
        tA = cute.make_layout(
            (threads_M, self._num_threads // threads_M),
            stride=(1, threads_M)
        )
        vA = cute.make_layout(
            (self._bM // threads_M, self._bK // (self._num_threads // threads_M))
        )
        
        threads_N = min(self._bN, self._num_threads)
        tB = cute.make_layout(
            (threads_N, self._num_threads // threads_N),
            stride=(1, threads_N)
        )
        vB = cute.make_layout(
            (self._bN // threads_N, self._bK // (self._num_threads // threads_N))
        )
        
        # Sync copy - no alignment issues
        copy_atom_A = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mA.element_type)
        copy_atom_B = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mB.element_type)
        
        tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(copy_atom_B, tB, vB)
        
        # Grid dimensions
        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1
        
        self.kernel(
            mA, mB, mC,
            sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B,
            self.tiled_mma,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
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
        
        # Shared memory - for WGMMA, need to separate swizzle from layout
        smem = cutlass.utils.SmemAllocator()
        
        # ComposedLayout has .outer and .inner attributes
        # Use .outer for layout shape and .inner for swizzle
        sA = smem.allocate_tensor(mA.element_type, sA_layout.outer, 128, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(mB.element_type, sB_layout.outer, 128, swizzle=sB_layout.inner)
        
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        
        # Copy partitions
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)
        
        # MMA partitions
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)
        
        k_tile_count = cute.size(tAgA, mode=[2])
        k_block_max = cute.size(tCrA, mode=[2])
        
        # Initialize WGMMA
        tiled_mma.set(Field.ACCUMULATE, False)
        
        # K-loop
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            # Copy G -> S
            cute.copy(tiled_copy_A, tAgA[None, None, k_tile], tAsA)
            cute.copy(tiled_copy_B, tBgB[None, None, k_tile], tBsB)
            cute.arch.syncthreads()
            
            # MMA
            for k_block in cutlass.range(k_block_max, unroll_full=True):
                fence()
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCsA[None, None, k_block],
                    tCsB[None, None, k_block],
                    tCrC,
                )
                commit_group()
                wait_group(0)
                tiled_mma.set(Field.ACCUMULATE, True)
            
            cute.arch.syncthreads()
        
        # Epilogue - store C
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC)


def verify():
    """Verify correctness."""
    torch.manual_seed(42)
    
    # Column-major tensors for proper alignment
    A_storage = torch.randn(K, M, dtype=torch.float16, device="cuda")
    B_storage = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C_storage = torch.zeros(N, M, dtype=torch.float32, device="cuda")
    A = A_storage.t()  # (M,K) column-major
    B = B_storage.t()  # (N,K) column-major  
    C = C_storage.t()  # (M,N) column-major
    
    a_tensor = from_dlpack(A, assumed_align=16)
    b_tensor = from_dlpack(B, assumed_align=16)
    c_tensor = from_dlpack(C, assumed_align=16)
    
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    
    gemm = HopperF16GemmWgmma(cta_tiler=(128, 64, 64), num_threads=128)
    
    print("Compiling...")
    compiled_fn = cute.compile(gemm, a_tensor, b_tensor, c_tensor, stream=current_stream)
    
    print("Running...")
    compiled_fn(a_tensor, b_tensor, c_tensor, current_stream)
    torch.cuda.synchronize()
    
    # Reference: C = A @ B^T with FP32 accumulation
    C_ref = torch.mm(A.float(), B.t().float())
    
    max_abs_err = (C - C_ref).abs().max().item()
    
    if max_abs_err < 1e-2:
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
    C_storage = torch.zeros(N, M, dtype=torch.float32, device="cuda")
    A = A_storage.t()
    B = B_storage.t()
    C = C_storage.t()
    
    a_tensor = from_dlpack(A, assumed_align=16)
    b_tensor = from_dlpack(B, assumed_align=16)
    c_tensor = from_dlpack(C, assumed_align=16)
    
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    
    gemm = HopperF16GemmWgmma(cta_tiler=(128, 64, 64), num_threads=128)
    compiled_fn = cute.compile(gemm, a_tensor, b_tensor, c_tensor, stream=current_stream)
    
    # Warmup
    for _ in range(warmup):
        compiled_fn(a_tensor, b_tensor, c_tensor, current_stream)
    torch.cuda.synchronize()
    
    # Benchmark
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
    print(f"CuTe DSL WGMMA GEMM: {M}x{N}x{K} F16->FP32")
    print("WGMMA + sync copy (col-major tensors)")
    
    if verify():
        bench()
