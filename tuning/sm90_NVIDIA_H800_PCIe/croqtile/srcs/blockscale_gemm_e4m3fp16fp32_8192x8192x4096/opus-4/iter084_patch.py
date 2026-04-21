import sys
src = open(sys.argv[1]).read()

# Replace the scale multiplication with fused multiply-add
# Current: ((mc[...] * mc_scale_a_val) * mc_scale_b_val)
# Can be: fma(mc[...], scale_a * scale_b)
old = """float* mc_scale_a_ptr = (float*)(((__iv_iv_k + DIV_BLK_K * mc_scale_a_row_idx) + scale_lhs));"""
new = """float* mc_scale_a_ptr = (float*)(((__iv_iv_k + DIV_BLK_K * mc_scale_a_row_idx) + scale_lhs));
        // Prefetch next iteration's scale
        if (__iv_iv_k + 1 < (int)(K / 128)) {
          asm volatile("prefetch.global.L2 [%0];" :: "l"(mc_scale_a_ptr + DIV_BLK_K));
        }"""

src = src.replace(old, new, 1)

open(sys.argv[1], 'w').write(src)
