"""Background scanner that auto-discovers tuning runs from disk artifacts.

Walks tuning/<gpu>/<dsl>/logs/<shape_key>/results.tsv and
tuning/<gpu>/<dsl>/checkpoints/<shape_key>.json to find runs that
agents started outside the web UI.  Creates DB tasks for any new
shape keys and refreshes metrics from checkpoint files.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .events import event_bus
from .models import Task
from .runtime_settings import get_default_model

logger = logging.getLogger("croqtuner.artifact_scanner")

_SHAPE_KEY_RE = re.compile(
    r"^(?:matmul_)?([A-Za-z0-9_]+?)_(\d+)x(\d+)x(\d+)(_fs)?$"
)


def discover_shape_keys() -> dict[str, dict]:
    """Walk the nested tuning tree and return discovered shape keys with metadata."""
    tuning_dir = settings.tuning_dir
    if not tuning_dir.exists():
        return {}

    found: dict[str, dict] = {}

    for gpu_dir in tuning_dir.iterdir():
        if not gpu_dir.is_dir():
            continue
        for dsl_dir in gpu_dir.iterdir():
            if not dsl_dir.is_dir():
                continue

            logs_dir = dsl_dir / "logs"
            if logs_dir.exists():
                for key_dir in logs_dir.iterdir():
                    if key_dir.is_dir() and (key_dir / "results.tsv").exists():
                        found.setdefault(key_dir.name, {})["results_tsv"] = key_dir / "results.tsv"

            cp_dir = dsl_dir / "checkpoints"
            if cp_dir.exists():
                for cp_file in cp_dir.iterdir():
                    if cp_file.is_file() and cp_file.suffix == ".json":
                        key = cp_file.stem
                        found.setdefault(key, {})["checkpoint"] = cp_file

            srcs_dir = dsl_dir / "srcs"
            if srcs_dir.exists():
                for key_dir in srcs_dir.iterdir():
                    if key_dir.is_dir():
                        found.setdefault(key_dir.name, {})["has_srcs"] = True

    return found


def _parse_shape_key(key: str) -> dict | None:
    m = _SHAPE_KEY_RE.match(key)
    if not m:
        return None
    dtype, m_raw, n_raw, k_raw, fs = m.groups()
    op_type_part = key.split(f"_{dtype}_")[0] if f"_{dtype}_" in key else "matmul"
    return {
        "op_type": op_type_part,
        "dtype": dtype,
        "m": int(m_raw),
        "n": int(n_raw),
        "k": int(k_raw),
        "mode": "from_scratch" if fs else "from_current_best",
    }


def _read_checkpoint_metrics(cp_path: Path) -> dict:
    metrics: dict = {}
    try:
        cp = json.loads(cp_path.read_text())
        it = cp.get("iteration") or cp.get("current_iter")
        if it is not None:
            metrics["current_iteration"] = int(it)
        best = cp.get("current_best_tflops") or cp.get("best_tflops")
        if best not in (None, ""):
            metrics["best_tflops"] = float(best)
        baseline = cp.get("baseline_tflops")
        if baseline not in (None, ""):
            metrics["baseline_tflops"] = float(baseline)
        best_kernel = cp.get("current_best_kernel") or cp.get("best_kernel")
        if best_kernel:
            metrics["best_kernel"] = best_kernel
    except (json.JSONDecodeError, OSError, ValueError):
        pass
    return metrics


def _count_results_iterations(results_path: Path) -> int:
    try:
        lines = results_path.read_text(errors="replace").splitlines()
        count = 0
        for line in lines:
            if line.strip() and not line.startswith("#") and not line.lower().startswith("iter\t"):
                count += 1
        return count
    except OSError:
        return 0


async def scan_and_create_tasks(session: AsyncSession) -> int:
    """Discover tuning runs on disk, create tasks for new ones, and refresh metrics for existing ones.

    Returns the number of new tasks created.
    """
    discovered = discover_shape_keys()
    if not discovered:
        return 0

    result = await session.execute(select(Task))
    existing: dict[str, Task] = {t.shape_key: t for t in result.scalars().all()}

    default_model = await get_default_model(session)
    created = 0
    changed = False

    for shape_key, info in discovered.items():
        metrics = _collect_metrics(info)

        if shape_key in existing:
            task = existing[shape_key]
            updated = False
            if metrics.get("current_iteration", 0) > (task.current_iteration or 0):
                task.current_iteration = metrics["current_iteration"]
                updated = True
            if metrics.get("best_tflops") is not None and (task.best_tflops is None or metrics["best_tflops"] > task.best_tflops):
                task.best_tflops = metrics["best_tflops"]
                updated = True
            if metrics.get("baseline_tflops") is not None and task.baseline_tflops is None:
                task.baseline_tflops = metrics["baseline_tflops"]
                updated = True
            if metrics.get("best_kernel") and not task.best_kernel:
                task.best_kernel = metrics["best_kernel"]
                updated = True
            if updated:
                task.updated_at = datetime.now(timezone.utc)
                changed = True
            continue

        parsed = _parse_shape_key(shape_key)
        if parsed is None:
            logger.debug("Skipping unparseable shape key from disk: %s", shape_key)
            continue

        max_iter = 150 if parsed["mode"] == "from_scratch" else 30

        task = Task(
            shape_key=shape_key,
            op_type=parsed["op_type"],
            dtype=parsed["dtype"],
            m=parsed["m"],
            n=parsed["n"],
            k=parsed["k"],
            mode=parsed["mode"],
            max_iterations=max_iter,
            status="pending",
            current_iteration=metrics.get("current_iteration", 0),
            best_tflops=metrics.get("best_tflops"),
            baseline_tflops=metrics.get("baseline_tflops"),
            best_kernel=metrics.get("best_kernel"),
            model=default_model,
        )
        session.add(task)
        created += 1
        logger.info(
            "Auto-discovered task: %s (iter=%d, best=%.2f TFLOPS)",
            shape_key, metrics.get("current_iteration", 0), metrics.get("best_tflops") or 0,
        )

    if created or changed:
        await session.commit()

    return created


def _collect_metrics(info: dict) -> dict:
    """Extract metrics from checkpoint and results.tsv."""
    metrics: dict = {}

    if "checkpoint" in info:
        metrics.update(_read_checkpoint_metrics(info["checkpoint"]))

    if "results_tsv" in info:
        iter_count = _count_results_iterations(info["results_tsv"])
        metrics["current_iteration"] = max(metrics.get("current_iteration", 0), iter_count)

    return metrics
