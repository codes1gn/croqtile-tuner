import sys
src = open(sys.argv[1]).read()

# Add shared memory for scale factors after __dyn_smem_base__
old_smem = """__shared__ half __dyn_smem_base__;"""
new_smem = """__shared__ half __dyn_smem_base__;
  // Scale factor shared memory - preload for current and next iteration
  __shared__ float scale_a_smem[4][2];  // [stage][row] for 2 M-tiles * 64 elements per tile
  __shared__ float scale_b_smem[4][2];  // [stage][col] for 2 N-tiles * 64 elements per tile"""
src = src.replace(old_smem, new_smem, 1)

# For now, let's try a simpler change - use __ldcs (streaming load) for scales
old_scale_a = """float mc_scale_a_val = *((float*)scale_lhs + (((mc_scale_a_row_idx * (K / 128)) + mc_scale_a_col_idx)));"""
new_scale_a = """float mc_scale_a_val;
        asm("ld.global.cs.f32 %0, [%1];" : "=f"(mc_scale_a_val) : "l"((float*)scale_lhs + (((mc_scale_a_row_idx * (K / 128)) + mc_scale_a_col_idx))));"""
src = src.replace(old_scale_a, new_scale_a, 1)

old_scale_b = """float mc_scale_b_val = *((float*)scale_rhs + (((mc_scale_b_row_idx * (N / 128)) + (mc_scale_b_col_idx / 128))));"""
new_scale_b = """float mc_scale_b_val;
        asm("ld.global.cs.f32 %0, [%1];" : "=f"(mc_scale_b_val) : "l"((float*)scale_rhs + (((mc_scale_b_row_idx * (N / 128)) + (mc_scale_b_col_idx / 128)))));"""
src = src.replace(old_scale_b, new_scale_b, 1)

open(sys.argv[1], 'w').write(src)
