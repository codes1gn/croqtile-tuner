import sys
src = open(sys.argv[1]).read()

# Add double-buffering hint to the consumer loop
# Change the MMA loop to use prefetching
old_warp_loop = """for (__iv_iv_warp = 0; __iv_iv_warp < 4; ++__iv_iv_warp) {"""
new_warp_loop = """#pragma unroll 1
        for (__iv_iv_warp = 0; __iv_iv_warp < 4; ++__iv_iv_warp) {"""
src = src.replace(old_warp_loop, new_warp_loop, 1)

open(sys.argv[1], 'w').write(src)
