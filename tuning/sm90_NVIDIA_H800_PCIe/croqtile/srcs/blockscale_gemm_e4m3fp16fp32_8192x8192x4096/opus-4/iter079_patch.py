import sys
src = open(sys.argv[1]).read()

# Add prefetch storage for scale factors
# Find the start of the consumer section and add scale register prefetch
old_mc_init = """__CHOREO_IF_WARPGROUPX4_ID_IN_V__(1, 3) {
        mc[0] = 0;"""

new_mc_init = """__CHOREO_IF_WARPGROUPX4_ID_IN_V__(1, 3) {
        // Prefetch first iteration's scale factors to registers
        float prefetch_scale_a = 0.0f, prefetch_scale_b = 0.0f;
        mc[0] = 0;"""

src = src.replace(old_mc_init, new_mc_init, 1)

open(sys.argv[1], 'w').write(src)
