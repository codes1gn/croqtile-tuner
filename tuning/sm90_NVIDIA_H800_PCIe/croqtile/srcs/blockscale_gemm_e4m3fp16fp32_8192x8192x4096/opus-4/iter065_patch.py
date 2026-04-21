import sys
src = open(sys.argv[1]).read()

# Add cp.async for scale factors - this uses the async copy engine
# Insert prefetch at loop start for next iteration's scales

# First, add a shared memory buffer for scale factors
old_shm = """__shared__ half __dyn_smem_base__;"""
new_shm = """__shared__ half __dyn_smem_base__;
  __shared__ float scale_a_smem[8];  // 8 scale values for LHS block
  __shared__ float scale_b_smem[8];  // 8 scale values for RHS block"""
src = src.replace(old_shm, new_shm, 1)

# Actually, let's just try using cooperative fetch + wait pattern
# Change the scale loading to use cooperative loads - all threads in warp load together

old_scale = """mc_scale_a_col_idx = (__iv_iv_k * 128) / 128;
        float mc_scale_a_val = *((float*)scale_lhs + (((mc_scale_a_row_idx * (K / 128)) + mc_scale_a_col_idx)));
        mc_scale_b_col_idx = (threadIdx.x >> 7) * 64;
        mc_scale_b_row_idx = (__iv_iv_k * 128) / 128;
        float mc_scale_b_val = *((float*)scale_rhs + (((mc_scale_b_row_idx * (N / 128)) + (mc_scale_b_col_idx / 128))));"""

# Simplify with cached loads and bitmask operations
new_scale = """mc_scale_a_col_idx = __iv_iv_k;
        const float* __restrict__ scale_a_ptr = (const float*)scale_lhs + (mc_scale_a_row_idx * (K / 128) + mc_scale_a_col_idx);
        float mc_scale_a_val = *scale_a_ptr;
        mc_scale_b_col_idx = (threadIdx.x >> 7) << 6;
        mc_scale_b_row_idx = __iv_iv_k;
        const float* __restrict__ scale_b_ptr = (const float*)scale_rhs + (mc_scale_b_row_idx * (N / 128) + (mc_scale_b_col_idx >> 7));
        float mc_scale_b_val = *scale_b_ptr;"""

src = src.replace(old_scale, new_scale, 1)

open(sys.argv[1], 'w').write(src)
