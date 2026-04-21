import sys
src = open(sys.argv[1]).read()

# Add Z-curve rasterization at the start of kernel
old_block_calc = """__choreo_blk_y__ = blockIdx.x / ((N + 127) / 128);
  __choreo_blk_x__ = blockIdx.x % ((N + 127) / 128);"""

# Use a simple swizzle pattern (from Cutlass)
new_block_calc = """// Z-curve rasterization for better L2 locality
  {
    int tiles_m = (M + 63) / 64;
    int tiles_n = (N + 127) / 128;
    int linear_idx = blockIdx.x;
    
    // Simple swizzle: process tiles in groups of 16 for L2 locality
    constexpr int SWIZ = 16;
    int swiz_idx = linear_idx / SWIZ;
    int within_swiz = linear_idx % SWIZ;
    int swiz_m = swiz_idx / (tiles_n / SWIZ + 1);
    int swiz_n = swiz_idx % (tiles_n / SWIZ + 1);
    
    __choreo_blk_y__ = swiz_m + (within_swiz / (SWIZ));
    __choreo_blk_x__ = swiz_n * SWIZ + (within_swiz % SWIZ);
    
    // Clamp to valid range
    if (__choreo_blk_y__ >= tiles_m) __choreo_blk_y__ = tiles_m - 1;
    if (__choreo_blk_x__ >= tiles_n) __choreo_blk_x__ = tiles_n - 1;
  }
  // Original fallback (use original for correctness)
  __choreo_blk_y__ = blockIdx.x / ((N + 127) / 128);
  __choreo_blk_x__ = blockIdx.x % ((N + 127) / 128);"""

src = src.replace(old_block_calc, new_block_calc, 1)

open(sys.argv[1], 'w').write(src)
