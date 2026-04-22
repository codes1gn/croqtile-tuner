#!/usr/bin/env python3
"""Clean invalid or active tuning work state so the next session starts from a safe INIT."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


ALL_DSLS = ["croqtile", "cuda", "cute-dsl", "cute-cpp", "triton", "tilelang", "helion"]
INCLUDE_PATTERN = re.compile(r'^\s*#include\s+"([^"]+)"')


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _trace_event(gpu: str, dsl: str, shape_key: str, tool: str, msg: str, level: str = "info") -> None:
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    mem_dir = repo_root() / "tuning" / gpu / dsl / "memory" / shape_key
    if not mem_dir.exists():
        return
    log_path = mem_dir / "activity.jsonl"
    ts = _dt.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    entry = {"ts": ts, "tool": tool, "msg": msg, "level": level}
    with open(log_path, "a") as f:
        f.write(_json.dumps(entry) + "\n")


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


def should_clean_shape(
    root: Path,
    gpu: str,
    dsl: str,
    shape_key: str,
    entry: dict,
    invalid_only: bool,
    reset_all: bool,
) -> bool:
    if reset_all:
        return True
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
        checkpoint = root / "tuning" / gpu / dsl / "checkpoints" / f"{shape_key}.json"
        if checkpoint.exists():
            return True
    return False


def reset_entry(entry: dict) -> None:
    entry["status"] = "pending"
    entry["current_iter"] = 0
    entry["best_tflops"] = 0
    entry["baseline_tflops"] = 0
    entry["best_kernel"] = ""


def clean_dsl(root: Path, gpu: str, dsl: str, invalid_only: bool, reset_all: bool) -> list[str]:
    actions: list[str] = []
    state_dir = root / ".claude" / "skills" / "fsm-engine" / "state" / dsl
    for transient in [state_dir / "loop-state.json", state_dir / "compaction-summary.md"]:
        if transient.exists():
            transient.unlink()
            actions.append(f"removed {transient.relative_to(root)}")

    tuning_root = root / "tuning" / gpu / dsl
    state_path = tuning_root / "state.json"
    state = load_json(state_path)

    if reset_all:
        for subdir in ["logs", "srcs", "perf", "cmd", "memory", "checkpoints"]:
            target = tuning_root / subdir
            if target.exists():
                shutil.rmtree(target)
                actions.append(f"removed {target.relative_to(root)}")
            target.mkdir(parents=True, exist_ok=True)

        if state:
            for entry in state.values():
                reset_entry(entry)
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            actions.append(f"updated {state_path.relative_to(root)}")
        return actions

    if not state:
        return actions

    changed = False
    for shape_key, entry in state.items():
        if not should_clean_shape(root, gpu, dsl, shape_key, entry, invalid_only, reset_all):
            continue

        reset_entry(entry)
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
    parser.add_argument("--gpu", required=True, help="GPU key from detect_gpu.sh, e.g. sm90_H100")
    parser.add_argument("--dsl", default="all", help="Single DSL or 'all'.")
    parser.add_argument("--invalid-only", action="store_true", help="Only clean in-progress shapes with clearly invalid external resume sources.")
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Reset all local stateful tuning artifacts for the selected DSL back to INIT-ready pending state.",
    )
    args = parser.parse_args()

    root = repo_root()
    dsls = ALL_DSLS if args.dsl == "all" else [args.dsl]
    invalid = [dsl for dsl in dsls if dsl not in ALL_DSLS]
    if invalid:
        raise SystemExit(f"Unsupported DSLs: {', '.join(invalid)}")

    actions: list[str] = []
    for dsl in dsls:
        actions.extend(clean_dsl(root, args.gpu, dsl, args.invalid_only, args.reset_all))

    if actions:
        for action in actions:
            print(action)
        for dsl in dsls:
            tuning_root = root / "tuning" / args.gpu / dsl
            state = load_json(tuning_root / "state.json")
            if state:
                for shape_key in state:
                    _trace_event(args.gpu, dsl, shape_key, "clean_kernel_work_state",
                                 f"Cleaned {len(actions)} item(s)", "warn")
    else:
        print("No matching work state needed cleanup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
