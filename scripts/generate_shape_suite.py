#!/usr/bin/env python3
"""Generate the complete deduplicated shape suite and print statistics.

Enumerates all shapes across 7 scenarios (10 sweep lines), deduplicates,
classifies near/far, and outputs:
  - JSON for manifest.json scenarios section
  - JSON for state.json shape entries
  - List of directory keys to create
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

MIN_M, MIN_N, MIN_K = 128, 256, 128

# Reference shape for region classification
REF = (4096, 8192, 8192)

# --- Size lists per scenario ---
SQUARE_SIZES = [256, 512, 768, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384]

SWEEP_M_SIZES = [128, 256, 384, 512, 768, 1024, 2048, 3072, 4096, 6144, 8192, 16384]

SWEEP_N_SIZES = [256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768]

SWEEP_K_SIZES = [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 32768]

TIED_SWEEP_SIZES = [256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
TIED_VARY_K_SIZES = [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384]
TIED_VARY_N_SIZES = [256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 32768]
TIED_VARY_M_SIZES = [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384]

FIXED_ANCHORS = [4096, 8192]
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "scripts" / "shape_suite_data.json"

def valid(m, n, k):
    return m >= MIN_M and n >= MIN_N and k >= MIN_K

def is_near(m, n, k):
    return (2048 <= m <= 8192) and (4096 <= n <= 16384) and (4096 <= k <= 16384)

def gen_shapes():
    """Returns dict: (M,N,K) -> set of scenario tags."""
    shapes = defaultdict(set)

    # Scenario 1: square M=N=K=S
    for s in SQUARE_SIZES:
        if valid(s, s, s):
            shapes[(s, s, s)].add("square")

    # Scenario 2: sweep_m (N=K=8192)
    for m in SWEEP_M_SIZES:
        if valid(m, 8192, 8192):
            shapes[(m, 8192, 8192)].add("sweep_m")

    # Scenario 3: sweep_n (M=4096, K=8192)
    for n in SWEEP_N_SIZES:
        if valid(4096, n, 8192):
            shapes[(4096, n, 8192)].add("sweep_n")

    # Scenario 4: sweep_k (M=4096, N=8192)
    for k in SWEEP_K_SIZES:
        if valid(4096, 8192, k):
            shapes[(4096, 8192, k)].add("sweep_k")

    # Scenario 5: sweep_mn (M=N tied)
    for anchor_k in FIXED_ANCHORS:
        tag = f"sweep_mn_fixK{anchor_k}"
        for t in TIED_SWEEP_SIZES:
            if valid(t, t, anchor_k):
                shapes[(t, t, anchor_k)].add(tag)

    for anchor_mn in FIXED_ANCHORS:
        tag = f"sweep_mn_fixMN{anchor_mn}"
        for k in TIED_VARY_K_SIZES:
            if valid(anchor_mn, anchor_mn, k):
                shapes[(anchor_mn, anchor_mn, k)].add(tag)

    # Scenario 6: sweep_mk (M=K tied)
    for anchor_n in FIXED_ANCHORS:
        tag = f"sweep_mk_fixN{anchor_n}"
        for t in TIED_SWEEP_SIZES:
            if valid(t, anchor_n, t):
                shapes[(t, anchor_n, t)].add(tag)

    for anchor_mk in FIXED_ANCHORS:
        tag = f"sweep_mk_fixMK{anchor_mk}"
        for n in TIED_VARY_N_SIZES:
            if valid(anchor_mk, n, anchor_mk):
                shapes[(anchor_mk, n, anchor_mk)].add(tag)

    # Scenario 7: sweep_nk (N=K tied)
    for anchor_m in FIXED_ANCHORS:
        tag = f"sweep_nk_fixM{anchor_m}"
        for t in TIED_SWEEP_SIZES:
            if valid(anchor_m, t, t):
                shapes[(anchor_m, t, t)].add(tag)

    for anchor_nk in FIXED_ANCHORS:
        tag = f"sweep_nk_fixNK{anchor_nk}"
        for m in TIED_VARY_M_SIZES:
            if valid(m, anchor_nk, anchor_nk):
                shapes[(m, anchor_nk, anchor_nk)].add(tag)

    return shapes

def main():
    shapes = gen_shapes()
    
    # Sort by (M, N, K) for deterministic output
    sorted_shapes = sorted(shapes.keys())
    
    near_shapes = [(m,n,k) for m,n,k in sorted_shapes if is_near(m,n,k)]
    far_shapes = [(m,n,k) for m,n,k in sorted_shapes if not is_near(m,n,k)]
    
    print(f"=== Shape Suite Statistics ===")
    print(f"Total unique shapes: {len(sorted_shapes)}")
    print(f"Near (from-current-best, 30 iter): {len(near_shapes)}")
    print(f"Far (also from-scratch, 150 iter): {len(far_shapes)}")
    print(f"x2 dtypes (f16+e4m3) = {len(sorted_shapes)*2} total shape-dtype entries")
    print()
    
    # Per-scenario counts
    scenario_counts = defaultdict(int)
    for mnk, tags in shapes.items():
        for t in tags:
            scenario_counts[t] += 1
    print("=== Per-scenario shape counts (before dedup) ===")
    for sc in sorted(scenario_counts.keys()):
        print(f"  {sc}: {scenario_counts[sc]}")
    print()
    
    # Print all shapes with their scenarios and region
    print("=== All unique shapes ===")
    for m, n, k in sorted_shapes:
        region = "NEAR" if is_near(m, n, k) else "FAR"
        tags = sorted(shapes[(m, n, k)])
        print(f"  {m:>5}x{n:>5}x{k:>5}  [{region:4s}]  scenarios: {', '.join(tags)}")
    
    # Generate directory keys for both dtypes
    print(f"\n=== Directory keys ({len(sorted_shapes)*2} total) ===")
    keys = []
    for dtype in ["f16", "e4m3"]:
        for m, n, k in sorted_shapes:
            key = f"{dtype}_{m}x{n}x{k}"
            keys.append(key)
    
    # Output JSON data
    output = {
        "unique_shapes": [[m,n,k] for m,n,k in sorted_shapes],
        "shape_scenarios": {f"{m}x{n}x{k}": sorted(list(tags)) for (m,n,k), tags in shapes.items()},
        "near_shapes": [[m,n,k] for m,n,k in near_shapes],
        "far_shapes": [[m,n,k] for m,n,k in far_shapes],
        "keys": keys,
        "counts": {
            "unique_shapes": len(sorted_shapes),
            "near": len(near_shapes),
            "far": len(far_shapes),
            "total_with_dtypes": len(sorted_shapes) * 2
        }
    }
    
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nData written to scripts/shape_suite_data.json")

if __name__ == "__main__":
    main()
