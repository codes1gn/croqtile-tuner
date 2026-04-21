import sys
src = open(sys.argv[1]).read()

# Modify the K-loop to process half of K per block
# We need to add atomic add at the end

# First, add a grid_z parameter for split factor
old_launch = """dim3 grid((M + 63) / 64 * ((N + 127) / 128), 1, 1);"""
new_launch = """constexpr int SPLIT_K = 2;
  dim3 grid((M + 63) / 64 * ((N + 127) / 128), SPLIT_K, 1);"""
src = src.replace(old_launch, new_launch, 1)

# Add split-K handling at kernel start
old_iv_k_init = """int __iv_iv_k = 0;"""
new_iv_k_init = """int __iv_iv_k = 0;
    const int split_k_id = blockIdx.y;
    const int k_tiles_total = (K / 128);
    const int k_tiles_per_split = k_tiles_total / 2;
    const int k_start = split_k_id * k_tiles_per_split;
    const int k_end = (split_k_id + 1) * k_tiles_per_split;"""
src = src.replace(old_iv_k_init, new_iv_k_init, 1)

# Modify the K-loop bounds
old_k_loop = """for (__iv_iv_k = 0; __iv_iv_k < (int)(K / 128); ++__iv_iv_k)"""
new_k_loop = """for (__iv_iv_k = k_start; __iv_iv_k < k_end; ++__iv_iv_k)"""
src = src.replace(old_k_loop, new_k_loop)

open(sys.argv[1], 'w').write(src)
