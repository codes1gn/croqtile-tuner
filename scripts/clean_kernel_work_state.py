#!/usr/bin/env python3
"""Clean invalid or active tuning work state so the next session starts from a safe INIT."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


ALL_DSLS = ["croqtile", "cuda", "cute", "triton", "tilelang", "helion", "cutile"]
INCLUDE_PATTERN = re.compile(r'^\s*#include\s+"([^"]+)"')


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_in_repo(root: Path, candidate: str) -> Path:
    path = Path(candidate)
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def escapes_repo(root: Path, candidate: str) -> bool:
    resolved = resolve_in_repo(root, candidate)
    try:
        resolved.relative_to(root)
        return False
    except ValueError:
        return True


def has_external_include(root: Path, source_file: Path) -> bool:
    if not source_file.exists():
        return False
    for line in source_file.read_text(encoding="utf-8").splitlines():
        match = INCLUDE_PATTERN.match(line)
        if not match:
            continue
        include_target = match.group(1)
        if Path(include_target).is_absolute() and escapes_repo(root, include_target):
            return True
    return False


def should_clean_shape(root: Path, dsl: str, shape_key: str, entry: dict, invalid_only: bool) -> bool:
    if entry.get("status") != "in_progress":
        return False
    if not invalid_only:
        return True
    best_kernel = str(entry.get("best_kernel", ""))
    if best_kernel.startswith("/") or "-paper/" in best_kernel or "_paper/" in best_kernel:
        return True
    if best_kernel and escapes_repo(root, best_kernel):
        return True
    if best_kernel and has_external_include(root, resolve_in_repo(root, best_kernel)):
        return True

    active_state = root / ".claude" / "skills" / "fsm-engine" / "state" / dsl / "loop-state.json"
    if not active_state.exists():
        checkpoint = root / "tuning" / "aitune" / dsl / "checkpoints" / f"{shape_key}.json"
        if checkpoint.exists():
            return True
    return False


def clean_dsl(root: Path, dsl: str, invalid_only: bool) -> list[str]:
    actions: list[str] = []
    state_dir = root / ".claude" / "skills" / "fsm-engine" / "state" / dsl
    for transient in [state_dir / "loop-state.json", state_dir / "compaction-summary.md"]:
        if transient.exists():
            transient.unlink()
            actions.append(f"removed {transient.relative_to(root)}")

    tuning_root = root / "tuning" / "aitune" / dsl
    state_path = tuning_root / "state.json"
    state = load_json(state_path)
    if not state:
        return actions

    changed = False
    for shape_key, entry in state.items():
        if not should_clean_shape(root, dsl, shape_key, entry, invalid_only):
            continue

        entry["status"] = "pending"
        entry["current_iter"] = 0
        entry["best_tflops"] = 0
        entry["baseline_tflops"] = 0
        entry["best_kernel"] = ""
        changed = True

        for subdir in ["logs", "srcs", "perf", "cmd", "memory"]:
            target = tuning_root / subdir / shape_key
            if target.exists():
                shutil.rmtree(target)
                actions.append(f"removed {target.relative_to(root)}")

        checkpoint = tuning_root / "checkpoints" / f"{shape_key}.json"
        if checkpoint.exists():
            checkpoint.unlink()
            actions.append(f"removed {checkpoint.relative_to(root)}")

    if changed:
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        actions.append(f"updated {state_path.relative_to(root)}")

    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean active or invalid kernel work state.")
    parser.add_argument("--dsl", default="all", help="Single DSL or 'all'.")
    parser.add_argument("--invalid-only", action="store_true", help="Only clean in-progress shapes with clearly invalid external resume sources.")
    args = parser.parse_args()

    root = repo_root()
    dsls = ALL_DSLS if args.dsl == "all" else [args.dsl]
    invalid = [dsl for dsl in dsls if dsl not in ALL_DSLS]
    if invalid:
        raise SystemExit(f"Unsupported DSLs: {', '.join(invalid)}")

    actions: list[str] = []
    for dsl in dsls:
        actions.extend(clean_dsl(root, dsl, args.invalid_only))

    if actions:
        for action in actions:
            print(action)
    else:
        print("No matching work state needed cleanup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())