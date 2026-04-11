#!/usr/bin/env python3
"""Validate that tuning resume state only depends on local repo artifacts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


ALL_DSLS = ["croqtile", "cuda", "cute", "triton", "tilelang", "helion", "cutile"]
INCLUDE_PATTERN = re.compile(r'^\s*#include\s+"([^"]+)"')


@dataclass
class Problem:
    dsl: str
    kind: str
    path: str
    detail: str


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_in_repo(root: Path, candidate: str) -> Path:
    path = Path(candidate)
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def is_within_repo(root: Path, candidate: str) -> bool:
    resolved = resolve_in_repo(root, candidate)
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def scan_external_includes(root: Path, source_file: Path, dsl: str) -> list[Problem]:
    if not source_file.exists():
        return []
    problems: list[Problem] = []
    for index, line in enumerate(source_file.read_text(encoding="utf-8").splitlines(), start=1):
        match = INCLUDE_PATTERN.match(line)
        if not match:
            continue
        include_target = match.group(1)
        if Path(include_target).is_absolute() and not is_within_repo(root, include_target):
            problems.append(
                Problem(
                    dsl=dsl,
                    kind="external-include",
                    path=str(source_file.relative_to(root)),
                    detail=f"line {index} includes external path {include_target}",
                )
            )
    return problems


def validate_dsl(root: Path, dsl: str) -> list[Problem]:
    problems: list[Problem] = []
    active_state = root / ".claude" / "skills" / "fsm-engine" / "state" / dsl / "loop-state.json"
    tuning_state_path = root / "tuning" / "aitune" / dsl / "state.json"
    tuning_state = load_json(tuning_state_path)
    in_progress_keys: list[str] = []

    if tuning_state:
        for shape_key, entry in tuning_state.items():
            if entry.get("status") != "in_progress":
                continue
            in_progress_keys.append(shape_key)
            best_kernel = entry.get("best_kernel", "")
            if best_kernel and not is_within_repo(root, best_kernel):
                problems.append(
                    Problem(
                        dsl=dsl,
                        kind="external-best-kernel",
                        path=str(tuning_state_path.relative_to(root)),
                        detail=f"shape {shape_key} best_kernel escapes repo: {best_kernel}",
                    )
                )
            if best_kernel:
                problems.extend(scan_external_includes(root, resolve_in_repo(root, best_kernel), dsl))

            checkpoint = root / "tuning" / "aitune" / dsl / "checkpoints" / f"{shape_key}.json"
            checkpoint_data = load_json(checkpoint)
            if checkpoint_data:
                checkpoint_best = checkpoint_data.get("best_kernel", "")
                if checkpoint_best and not is_within_repo(root, checkpoint_best):
                    problems.append(
                        Problem(
                            dsl=dsl,
                            kind="external-checkpoint-kernel",
                            path=str(checkpoint.relative_to(root)),
                            detail=f"shape {shape_key} checkpoint best_kernel escapes repo: {checkpoint_best}",
                        )
                    )
                if checkpoint_best:
                    problems.extend(scan_external_includes(root, resolve_in_repo(root, checkpoint_best), dsl))

                if entry.get("current_iter", 0) > 0:
                    memory_dir = root / "tuning" / "aitune" / dsl / "memory" / shape_key
                    raw_log = memory_dir / "rounds.raw.jsonl"
                    md_log = memory_dir / "rounds.md"
                    if not raw_log.exists():
                        problems.append(
                            Problem(dsl=dsl, kind="missing-round-memory-raw", path=str(raw_log.relative_to(root)), detail=f"shape {shape_key} is in progress but raw round history is missing")
                        )
                    if not md_log.exists():
                        problems.append(
                            Problem(dsl=dsl, kind="missing-round-memory-md", path=str(md_log.relative_to(root)), detail=f"shape {shape_key} is in progress but markdown round history is missing")
                        )

    if in_progress_keys and not active_state.exists():
        problems.append(
            Problem(
                dsl=dsl,
                kind="missing-active-loop-state",
                path=str(active_state.relative_to(root)),
                detail=f"in-progress shapes exist without active loop-state: {', '.join(sorted(in_progress_keys))}",
            )
        )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local-only tuning resume state.")
    parser.add_argument("--dsl", default="all", help="Single DSL or 'all'.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    root = repo_root()
    dsls = ALL_DSLS if args.dsl == "all" else [args.dsl]
    invalid = [dsl for dsl in dsls if dsl not in ALL_DSLS]
    if invalid:
        raise SystemExit(f"Unsupported DSLs: {', '.join(invalid)}")

    problems: list[Problem] = []
    for dsl in dsls:
        problems.extend(validate_dsl(root, dsl))

    if args.json:
        print(json.dumps([asdict(problem) for problem in problems], indent=2, sort_keys=True))
    elif problems:
        for problem in problems:
            print(f"[{problem.dsl}] {problem.kind}: {problem.path} :: {problem.detail}")
    else:
        print("Resume validation passed: all checked sources stay within the current repo.")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())