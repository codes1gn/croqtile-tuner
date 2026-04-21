import sys
src = open(sys.argv[1]).read()

# Find the scale computation area and add precomputed offsets
# The scales are accessed with repeated offset calculations - let's hoist them
old = """    int mc_scale_a_row_idx;
    int mc_scale_a_col_idx;"""

new = """    int mc_scale_a_row_idx;
    int mc_scale_a_col_idx;
    // Precompute scale strides
    const int scale_a_row_stride = (K / 128);
    const int scale_b_col_stride = (N / 128);"""

src = src.replace(old, new, 1)

# Update scale_lhs access
old_scale_a = """mc_scale_a_col_idx = (__iv_iv_k * 128) / 128;
        float mc_scale_a_val = *((float*)scale_lhs + (((mc_scale_a_row_idx * (K / 128)) + mc_scale_a_col_idx)));"""

new_scale_a = """mc_scale_a_col_idx = __iv_iv_k;
        float mc_scale_a_val = *((float*)scale_lhs + ((mc_scale_a_row_idx * scale_a_row_stride) + mc_scale_a_col_idx));"""

src = src.replace(old_scale_a, new_scale_a, 1)

# Update scale_rhs access
old_scale_b = """mc_scale_b_col_idx = (threadIdx.x >> 7) * 64;
        mc_scale_b_row_idx = (__iv_iv_k * 128) / 128;
        float mc_scale_b_val = *((float*)scale_rhs + (((mc_scale_b_row_idx * (N / 128)) + (mc_scale_b_col_idx / 128))));"""

new_scale_b = """mc_scale_b_col_idx = (threadIdx.x >> 7) * 64;
        mc_scale_b_row_idx = __iv_iv_k;
        float mc_scale_b_val = *((float*)scale_rhs + ((mc_scale_b_row_idx * scale_b_col_stride) + (mc_scale_b_col_idx >> 7)));"""

src = src.replace(old_scale_b, new_scale_b, 1)

open(sys.argv[1], 'w').write(src)
