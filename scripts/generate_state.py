#!/usr/bin/env python3
"""Generate the expanded tuning/state.json with all 268 shape entries.

Preserves existing 8 done f16 shapes. Adds all new shapes as pending.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAPE_SUITE_PATH = ROOT / "scripts" / "shape_suite_data.json"
STATE_PATH = ROOT / "tuning" / "state.json"

with SHAPE_SUITE_PATH.open(encoding="utf-8") as f:
    data = json.load(f)

with STATE_PATH.open(encoding="utf-8") as f:
    existing = json.load(f)

existing_shapes = existing["shapes"]

new_state = {
    "version": 3,
    "description": "Global tuning state — tracks per-shape kernel status across all scenarios and dtypes. All work on main branch. v3: 134 unique shapes x 2 dtypes = 268 entries.",
    "shapes": {}
}

# Preserve existing done shapes
for key, info in existing_shapes.items():
    new_state["shapes"][key] = info

shape_scenarios = data["shape_scenarios"]

for dtype in ["f16", "e4m3"]:
    for shape in data["unique_shapes"]:
        m, n, k = shape
        key = f"{dtype}_{m}x{n}x{k}"
        mnk_key = f"{m}x{n}x{k}"
        scenarios = shape_scenarios.get(mnk_key, [])
        
        if key in new_state["shapes"]:
            # Already exists (one of the 8 done shapes) — add scenarios list
            if "scenarios" not in new_state["shapes"][key]:
                new_state["shapes"][key]["scenarios"] = scenarios
            continue
        
        is_near = (2048 <= m <= 8192) and (4096 <= n <= 16384) and (4096 <= k <= 16384)
        
        new_state["shapes"][key] = {
            "status": "pending",
            "mode": "from_current_best",
            "current_iter": 0,
            "best_tflops": 0,
            "baseline_tflops": 0,
            "best_kernel": "",
            "scenarios": scenarios,
            "region": "near" if is_near else "far"
        }

# Count stats
total = len(new_state["shapes"])
done = sum(1 for v in new_state["shapes"].values() if v["status"] == "done")
pending = sum(1 for v in new_state["shapes"].values() if v["status"] == "pending")
near = sum(1 for v in new_state["shapes"].values() if v.get("region") == "near")
far = sum(1 for v in new_state["shapes"].values() if v.get("region") == "far")

print(f"Total entries: {total}")
print(f"Done (preserved): {done}")
print(f"Pending (new): {pending}")
print(f"Near: {near}, Far: {far}")

with STATE_PATH.open("w", encoding="utf-8") as f:
    json.dump(new_state, f, indent=2)
    f.write("\n")

print("Written to tuning/state.json")
