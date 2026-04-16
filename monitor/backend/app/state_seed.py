from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .models import IterationLog, Task
from .runtime_settings import get_default_model

logger = logging.getLogger("croqtuner.state_seed")

_SHAPE_KEY_PATTERN = re.compile(
    r"^(?:matmul_)?([A-Za-z0-9_]+?)_(\d+)x(\d+)x(\d+)$"
)


def _find_state_files(tuning_dir: Path) -> list[Path]:
    """Discover state.json files in tuning/<gpu>/<dsl>/state.json and legacy tuning/state.json."""
    found: list[Path] = []
    root_state = tuning_dir / "state.json"
    if root_state.exists():
        found.append(root_state)

    if tuning_dir.exists():
        for gpu_dir in tuning_dir.iterdir():
            if not gpu_dir.is_dir():
                continue
            for dsl_dir in gpu_dir.iterdir():
                if not dsl_dir.is_dir():
                    continue
                nested = dsl_dir / "state.json"
                if nested.exists():
                    found.append(nested)
    return found


async def seed_tasks_from_state_if_empty(session: AsyncSession) -> bool:
    result = await session.execute(select(func.count(Task.id)))
    task_count = result.scalar_one()
    if task_count:
        return False

    state_paths = _find_state_files(settings.tuning_dir)
    if not state_paths:
        logger.info("Skipping state seed: no state.json found under %s", settings.tuning_dir)
        return False

    shapes: dict[str, dict] = {}
    for state_path in state_paths:
        try:
            raw = json.loads(state_path.read_text())
            shapes.update(raw.get("shapes", {}))
            logger.info("Loaded shapes from %s", state_path)
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to read %s for initial seed", state_path)

    await session.execute(delete(IterationLog))
    await session.execute(delete(Task))

    default_model = await get_default_model(session)

    completed_rows: list[dict] = []
    waiting_rows: list[dict] = []
    active_row: dict | None = None

    base_completed = datetime(2020, 1, 1, tzinfo=timezone.utc)
    base_waiting = datetime(2020, 6, 1, tzinfo=timezone.utc)
    active_ts = datetime(1999, 1, 1, tzinfo=timezone.utc)

    for key, info in shapes.items():
        match = _SHAPE_KEY_PATTERN.match(key)
        if not match:
            logger.warning("Skipping malformed shape key during seed: %s", key)
            continue

        dtype, m_raw, n_raw, k_raw = match.groups()
        mode = "cursor_ide"
        max_iter = 30
        row = {
            "shape_key": key,
            "dtype": dtype,
            "m": int(m_raw),
            "n": int(n_raw),
            "k": int(k_raw),
            "mode": info.get("mode", mode),
            "max_iterations": max_iter,
            "current_iteration": int(info.get("current_iter", 0) or 0),
            "best_tflops": _to_float_or_none(info.get("best_tflops")),
            "baseline_tflops": _to_float_or_none(info.get("baseline_tflops")),
            "best_kernel": info.get("best_kernel") or None,
            "model": default_model,
        }

        state_status = info.get("status", "pending")
        if state_status == "done":
            completed_rows.append({
                **row,
                "status": "completed",
                "created_at": base_completed + timedelta(seconds=len(completed_rows)),
            })
        elif state_status == "active":
            active_row = {
                **row,
                "status": "pending",
                "created_at": active_ts,
            }
        else:
            waiting_rows.append({
                **row,
                "status": "waiting",
                "created_at": base_waiting + timedelta(seconds=len(waiting_rows)),
            })

    for row in completed_rows:
        session.add(_task_from_seed_row(row))
    for row in waiting_rows:
        session.add(_task_from_seed_row(row))
    if active_row:
        session.add(_task_from_seed_row(active_row))

    await session.flush()
    await _import_iteration_logs(session)
    await session.commit()

    logger.info(
        "Seeded tasks from state.json: %d completed, %d waiting, %d active-as-pending",
        len(completed_rows),
        len(waiting_rows),
        1 if active_row else 0,
    )
    return True


def _task_from_seed_row(row: dict) -> Task:
    created_at = row.pop("created_at")
    task = Task(**row)
    task.created_at = created_at
    task.updated_at = created_at
    return task


def _find_results_tsv_for_seed(shape_key: str) -> Path | None:
    """Search nested tuning tree for results.tsv matching a shape key."""
    tuning_dir = settings.tuning_dir
    if not tuning_dir.exists():
        return None

    parts = shape_key.split("/")
    if len(parts) == 4:
        candidate = tuning_dir / parts[0] / parts[1] / "logs" / parts[2] / parts[3] / "results.tsv"
        if candidate.exists():
            return candidate
        return None

    if len(parts) == 3:
        logs_dir = tuning_dir / parts[0] / parts[1] / "logs" / parts[2]
        if logs_dir.is_dir():
            for model_dir in logs_dir.iterdir():
                if model_dir.is_dir():
                    candidate = model_dir / "results.tsv"
                    if candidate.exists():
                        return candidate
        return None

    # Nested layout: tuning/<gpu>/<dsl>/logs/<shape_key>/<model>/results.tsv
    for gpu_dir in tuning_dir.iterdir():
        if not gpu_dir.is_dir():
            continue
        for dsl_dir in gpu_dir.iterdir():
            if not dsl_dir.is_dir():
                continue
            shape_dir = dsl_dir / "logs" / shape_key
            if shape_dir.is_dir():
                for model_dir in shape_dir.iterdir():
                    if model_dir.is_dir():
                        candidate = model_dir / "results.tsv"
                        if candidate.exists():
                            return candidate

    # Legacy flat: tuning/logs/<shape_key>/results.tsv
    flat = tuning_dir / "logs" / shape_key / "results.tsv"
    if flat.exists():
        return flat

    return None


async def _import_iteration_logs(session: AsyncSession) -> None:
    result = await session.execute(select(Task.id, Task.shape_key))
    task_by_key = {shape_key: task_id for task_id, shape_key in result.all()}

    for shape_key, task_id in task_by_key.items():
        results_path = _find_results_tsv_for_seed(shape_key)
        if results_path is None:
            continue

        try:
            lines = results_path.read_text(errors="replace").splitlines()
        except OSError:
            logger.warning("Failed to read results file during seed: %s", results_path)
            continue

        iteration = 0
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split("\t")
            if not parts:
                continue

            first = parts[0].lower()
            if first in ("iter", "round"):
                continue

            # Detect format like artifact_scanner does:
            # croqtile: round(int), iter, kernel, tflops, decision, ...
            # cuda:     iter(str),  kernel, tflops, decision, bottleneck, ...
            try:
                int(parts[0])
                tflops_idx, decision_idx, kernel_idx = 3, 4, 2
            except ValueError:
                tflops_idx, decision_idx, kernel_idx = 2, 3, 1

            if len(parts) <= tflops_idx:
                continue

            iteration += 1

            tflops_raw = parts[tflops_idx].strip() if len(parts) > tflops_idx else ""
            tflops = _to_float_or_none(tflops_raw)
            decision = parts[decision_idx].strip() if len(parts) > decision_idx else None
            if decision == "BASELINE":
                decision = None
            kernel = parts[kernel_idx].strip() if len(parts) > kernel_idx else None
            bottleneck = parts[decision_idx + 1].strip() if len(parts) > decision_idx + 1 else None
            idea = parts[decision_idx + 2].strip() if len(parts) > decision_idx + 2 else None

            session.add(
                IterationLog(
                    task_id=task_id,
                    iteration=iteration,
                    kernel_path=kernel or None,
                    tflops=tflops,
                    decision=decision or None,
                    bottleneck=bottleneck or None,
                    idea_summary=idea or None,
                )
            )


def _to_float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None