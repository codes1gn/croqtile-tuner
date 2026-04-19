# Investigation: Closing the Gap with cuBLAS

## Target Shape
- **M×N×K**: 16384×2048×2048
- **Datatype**: f16 → f32
- **GPU**: NVIDIA H800 PCIe (sm90a)

## Performance Baseline

| Kernel | TFLOPS | Efficiency | Gap vs cuBLAS |
|--------|--------|------------|---------------|
| cuBLAS (`nvjet_sm90_hsh_128x256_64x4_2x1_v_bz_coopA_TNN`) | 465.0 | 30.7% | baseline |
| iter025_maxreg128 | 383.4 | 25.3% | -17.5% |
| iter057_wn256_fixed (WARP_N=256, STAGES=2) | 214.0 | 14.1% | -54% |
| iter055_cublas_imitate (cluster+multicast) | 206.5 | 13.6% | -56% |

## Root Cause Analysis

### Instruction Count Comparison
- **cuBLAS**: 17.5M instructions
- **iter025**: 53.4M instructions (3× more)

This 3× instruction overhead is the primary cause of the performance gap.

### cuBLAS Architecture (decoded from kernel name)
```
nvjet_sm90_hsh_128x256_64x4_2x1_v_bz_coopA_TNN
         │   │       │   │   │       │
         │   │       │   │   │       └── Transposition: TNN (A^T, B not transposed)
         │   │       │   │   └── coopA: Cooperative loading on A operand (multicast)
         │   │       │   └── Cluster 2×1 (2 blocks along M)
         │   │       └── K=64, STAGES=4
         │   └── TILE_M=128, TILE_N=256
         └── hsh: likely a variant identifier
```

### cuBLAS Key Parameters
- **TILE_M**: 128
- **TILE_N**: 256
- **TILE_K**: 64
- **STAGES**: 4
- **CLUSTER_M×CLUSTER_N**: 2×1
- **Registers per thread**: 168
- **Shared memory**: 213KB dynamic
- **Grid**: (2, 57) = 114 blocks
- **WGMMA**: 64×256×16 (inferred from TILE_N=256)
- **Multicast**: A operand (LHS) shared within cluster

---

## Paths to Surpass cuBLAS

### Path 1: Non-Cluster Optimizations

#### 1A. Increase WARP_N from 128 to 256
- **Status**: Attempted (iter057)
- **Result**: 214 TFLOPS (worse than iter025's 383 TFLOPS)
- **Issue**: STAGES must be reduced from 4 to 2 to fit SMEM, limiting pipelining

**SMEM calculation for WARP_N=256:**
```
LHS: 128 × 64 × STAGES × 2 bytes
RHS: 256 × 64 × STAGES × 2 bytes
Output: 64 × 256 × 4 × 2 consumers = 128KB

STAGES=4: 64KB + 128KB + 128KB = 320KB > 233KB ✗
STAGES=3: 48KB + 96KB + 128KB = 272KB > 233KB ✗
STAGES=2: 32KB + 64KB + 128KB = 224KB < 233KB ✓
```

**Potential fix**: Reduce output double-buffering or use register-based staging.

#### 1B. Increase WARP_N to 192 (middle ground)
- **Status**: Not attempted
- **Rationale**: May fit STAGES=3

**SMEM calculation for WARP_N=192:**
```
LHS: 128 × 64 × 3 × 2 = 48KB
RHS: 192 × 64 × 3 × 2 = 72KB  
Output: 64 × 192 × 4 × 2 = 96KB
Total: 48KB + 72KB + 96KB = 216KB < 233KB ✓
```

**Required changes:**
- Change WGMMA from 64×128 to 64×192
- Adjust TMA tensor maps for 192 N tile
- Update output store paths

#### 1C. Optimize Loop Structure
- **Status**: Not attempted
- **Potential**: Reduce loop overhead, unroll more aggressively
- **Target**: Reduce instruction count from 53M towards 17M

#### 1D. Remove maxrregcount=128 with Better Occupancy
- **Status**: Tested in iter053, no improvement
- **Observation**: 128 registers is optimal for this kernel structure

---

### Path 2: Cluster + Multicast Optimizations

#### 2A. Fix Choreo-Generated Cluster Kernel
- **Status**: Attempted (iter055), failed with WGMMA serialization
- **Issue**: Choreo cluster codegen has bugs
  - `warpgroup_arrive()` called inside WGMMA loop
  - Possible barrier/sync issues

#### 2B. Manual CUDA Cluster Implementation
- **Status**: Attempted (iter058), GPU hung
- **Cause**: Block-scoped barriers don't work with cluster mode
- **Required**: CUTLASS-style pipeline implementation

#### Key CUTLASS Cluster Patterns (from sm90_pipeline.hpp)

```cpp
// 1. Barrier initialization for multicast
uint32_t multicast_consumer_arrival_count = 
    (cluster_x + cluster_y - 1) * num_consumer_warpgroups;
    
initialize_barrier_array_pair_aligned(
    full_barrier, empty_barrier,
    producer_arv_cnt, multicast_consumer_arrival_count);

// 2. TMA multicast load (only leader issues)
if (block_rank_in_cluster == 0) {
    cp.async.bulk.tensor.2d.shared::cluster.global
        .mbarrier::complete_tx::bytes.multicast::cluster
        [dst], [desc, {coord}], [mbar], multicast_mask;
}

// 3. Consumer mask calculation
block_id_mask_ = calculate_multicast_mask<McastDirection>(
    cluster_shape, AtomThrShape, block_id_in_cluster);
```

#### 2C. Use CUTLASS Pipeline Components Directly
- **Status**: Not attempted
- **Approach**: Integrate CUTLASS's `PipelineTmaAsync` or `PipelineTransactionAsync`
- **Complexity**: High - requires understanding CUTLASS collective/pipeline architecture

---

## Detailed cuBLAS Feature Analysis

### Hyperparameters
| Parameter | cuBLAS | iter025 | Notes |
|-----------|--------|---------|-------|
| TILE_M | 128 | 128 | Same |
| TILE_N | 256 | 128 | cuBLAS 2× larger |
| TILE_K | 64 | 64 | Same |
| STAGES | 4 | 4 | Same |
| WARP_M | 64 | 64 | Fixed for WGMMA |
| WARP_N | 256 | 128 | cuBLAS 2× larger |
| WARP_K | 16 | 16 | Fixed for WGMMA |
| Registers | 168 | 128 | cuBLAS uses more |
| Cluster | 2×1 | none | Key difference |
| Multicast | coopA (LHS) | none | Key difference |

### WGMMA Comparison
| Metric | cuBLAS (64×256) | iter025 (64×128) |
|--------|-----------------|------------------|
| FLOPs per instruction | 2×64×256×16 = 524K | 2×64×128×16 = 262K |
| Output registers | 128 floats | 64 floats |
| K iterations per tile | 64/16 = 4 | 64/16 = 4 |
| Total WGMMA per block-tile | 4 | 4 |

### Cluster Benefits
1. **Reduced TMA bandwidth**: LHS loaded once per cluster (2 blocks), not per block
2. **Better L2 utilization**: Shared data stays in L2 longer
3. **Reduced instruction count**: ~2× fewer LHS loads

### Instruction Breakdown (estimated)
| Component | cuBLAS | iter025 | Ratio |
|-----------|--------|---------|-------|
| WGMMA | ~2M | ~4M | 2× (WARP_N difference) |
| TMA load | ~1M | ~3M | 3× (no multicast) |
| Control/sync | ~2M | ~6M | 3× (more iterations) |
| Store | ~1M | ~2M | 2× |
| **Total** | ~6M* | ~15M* | 2.5× |

*Note: Actual measurements show 17.5M vs 53.4M, suggesting additional overhead sources.

---

## Recommendations

### Short Term (Non-Cluster)
1. Try WARP_N=192 with STAGES=3
2. Profile iter025 vs iter057 to understand STAGES impact
3. Investigate instruction count breakdown with SASS analysis

### Medium Term (Cluster)  
1. Study CUTLASS's `sm90_mma_tma_gmma_ss_warpspecialized.hpp` in detail
2. Implement using CUTLASS pipeline components
3. Or fix Choreo's cluster codegen bugs

### Long Term
1. Contribute cluster fixes back to Choreo
2. Build proper CUTLASS-like pipeline abstraction

---

## File References

### Tuning Artifacts
- `tuning/sm90_NVIDIA_H800_PCIe/croqtile/srcs/matmul_f16fp32_16384x2048x2048/gpt-5-4/`
  - `iter025_maxreg128.cu` - Current best kernel
  - `iter055_cublas_imitate.cu` - Cluster attempt (broken)
  - `iter057_wn256_fixed.cu` - WARP_N=256 attempt
  - `iter058_cluster2.cu` - Manual cluster attempt (caused GPU hang)

### CUTLASS References
- `extern/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp` - Cluster pipeline
- `extern/cutlass/include/cute/arch/copy_sm90_tma.hpp` - Multicast TMA primitives
- `extern/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` - GEMM collective

### Choreo References  
- `/home/albert/workspace/croqtile/benchmark/performance/matmul/matmul_f16_dyn_sm90_tbc.co` - Cluster kernel example
- `/home/albert/workspace/croqtile/runtime/choreo_cute.h` - TMA multicast function
- `/home/albert/workspace/croqtile/lib/Target/GPU/cute_codegen.cpp` - Cluster codegen

---

## Appendix: Profile Data

### cuBLAS Profile (ncu_baseline_round51.ncu-rep)
```
Grid: (2, 57, 1)
Block: (384, 1, 1)
Cluster: 2
Registers: 168
SMEM Dynamic: 213KB
Duration: ~38μs (at full clocks)
Tensor utilization: 89.6%
```

### iter025 Profile (ncu_iter025_gpu0.ncu-rep)
```
Grid: (114, 1, 1)  
Block: (384, 1, 1)
Cluster: none
Registers: 128
SMEM Dynamic: ~200KB
Duration: ~45μs (at full clocks)
Tensor utilization: 68.5%
```
