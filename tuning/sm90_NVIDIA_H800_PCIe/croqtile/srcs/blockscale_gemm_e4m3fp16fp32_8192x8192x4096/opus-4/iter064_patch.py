import sys
src = open(sys.argv[1]).read()

# Add async copy for scales - load scale factors for next iteration while computing current
# We can use __pipeline_memcpy_async for small data

# Find the scale loading section and add async pattern
old_pattern = """mc_scale_a_col_idx = (__iv_iv_k * 128) / 128;
        float mc_scale_a_val = *((float*)scale_lhs + (((mc_scale_a_row_idx * (K / 128)) + mc_scale_a_col_idx)));
        mc_scale_b_col_idx = (threadIdx.x >> 7) * 64;
        mc_scale_b_row_idx = (__iv_iv_k * 128) / 128;
        float mc_scale_b_val = *((float*)scale_rhs + (((mc_scale_b_row_idx * (N / 128)) + (mc_scale_b_col_idx / 128))));"""

# Change to use vectorized load
new_pattern = """mc_scale_a_col_idx = __iv_iv_k;
        float mc_scale_a_val = __ldg((const float*)scale_lhs + (((mc_scale_a_row_idx * (K / 128)) + mc_scale_a_col_idx)));
        mc_scale_b_col_idx = (threadIdx.x >> 7) << 6;  // * 64
        mc_scale_b_row_idx = __iv_iv_k;
        float mc_scale_b_val = __ldg((const float*)scale_rhs + (((mc_scale_b_row_idx * (N / 128)) + (mc_scale_b_col_idx >> 7))));"""

src = src.replace(old_pattern, new_pattern, 1)

open(sys.argv[1], 'w').write(src)
