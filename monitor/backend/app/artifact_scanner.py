"""Background scanner that auto-discovers tuning runs from disk artifacts.

Walks tuning/<gpu>/<dsl>/logs/<shape_key>/<model>/results.tsv and
tuning/<gpu>/<dsl>/checkpoints/<shape_key>/<model>/current_idea.json to find runs
that agents started outside the web UI.  Creates DB tasks for any new shape keys
and refreshes metrics from checkpoint files.

The unique task identity is the compound key  "{gpu}/{dsl}/{shape_key}/{model}"
stored in the Task.shape_key column, so that the same kernel shape tuned by
different models on different GPU/DSL combinations is tracked as separate tasks.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .agent_detector import DetectedAgent, detect_running_agents
from .config import settings
from .events import event_bus
from .models import IterationLog, Task

logger = logging.getLogger("croqtuner.artifact_scanner")

_SHAPE_KEY_RE = re.compile(
    r"^(.+?)_([A-Za-z0-9]+)_(\d+)x(\d+)x(\d+)$"
)

_GPU_NAME_RE = re.compile(r"^sm\d+_(.+)$")


def _gpu_dir_to_device(gpu_dir_name: str) -> str:
    """Extract human-readable device name from gpu dir, e.g. 'sm86_NVIDIA_GeForce_RTX_3070' → 'NVIDIA GeForce RTX 3070'."""
    m = _GPU_NAME_RE.match(gpu_dir_name)
    if m:
        return m.group(1).replace("_", " ")
    return gpu_dir_name


def _normalize_model(dir_name: str) -> str | None:
    """Treat the 'default' model placeholder as None (no model specified)."""
    if dir_name == "default":
        return None
    return dir_name


def _compound_key(gpu: str, dsl: str, shape_key: str, model: str | None) -> str:
    """Build the compound key used to uniquely identify a task.

    When model is None (the 'default' placeholder), the key is 3-part:
        {gpu}/{dsl}/{shape_key}
    Otherwise it includes the model:
        {gpu}/{dsl}/{shape_key}/{model}
    """
    if model:
        return f"{gpu}/{dsl}/{shape_key}/{model}"
    return f"{gpu}/{dsl}/{shape_key}"


def discover_shape_keys() -> dict[str, dict]:
    """Walk the nested tuning tree and return compound keys with metadata.

    Returns a dict keyed by  "{gpu_dir}/{dsl_dir}/{shape_key}/{model}"  so that the
    same kernel shape tuned by different models on different GPU/DSL combinations
    are kept as distinct entries.
    Each value dict contains: results_tsv, checkpoint, has_srcs, dsl, device, model.
    """
    tuning_dir = settings.tuning_dir
    if not tuning_dir.exists():
        return {}

    found: dict[str, dict] = {}

    for gpu_dir in tuning_dir.iterdir():
        if not gpu_dir.is_dir():
            continue
        device = _gpu_dir_to_device(gpu_dir.name)

        for dsl_dir in gpu_dir.iterdir():
            if not dsl_dir.is_dir():
                continue
            dsl = dsl_dir.name  # e.g. "croqtile", "cuda"

            logs_dir = dsl_dir / "logs"
            if logs_dir.exists():
                for key_dir in logs_dir.iterdir():
                    if not key_dir.is_dir():
                        continue
                    for model_dir in key_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                        model = _normalize_model(model_dir.name)
                        compound = _compound_key(gpu_dir.name, dsl, key_dir.name, model)
                        entry = found.setdefault(compound, {"dsl": dsl, "device": device, "shape_key": key_dir.name, "model": model})
                        results_tsv = model_dir / "results.tsv"
                        if results_tsv.exists():
                            entry["results_tsv"] = results_tsv
                        idea_log = model_dir / "idea-log.jsonl"
                        if idea_log.exists():
                            entry["idea_log"] = idea_log
                        attempt_log = model_dir / "attempt-log.jsonl"
                        if attempt_log.exists():
                            entry["attempt_log"] = attempt_log

            cp_dir = dsl_dir / "checkpoints"
            if cp_dir.exists():
                for key_item in cp_dir.iterdir():
                    if not key_item.is_dir():
                        continue
                    for model_dir in key_item.iterdir():
                        if model_dir.is_dir():
                            idea_file = model_dir / "current_idea.json"
                            if idea_file.exists():
                                model = _normalize_model(model_dir.name)
                                compound = _compound_key(gpu_dir.name, dsl, key_item.name, model)
                                entry = found.setdefault(compound, {"dsl": dsl, "device": device, "shape_key": key_item.name, "model": model})
                                entry["checkpoint"] = idea_file

            srcs_dir = dsl_dir / "srcs"
            if srcs_dir.exists():
                for key_dir in srcs_dir.iterdir():
                    if not key_dir.is_dir():
                        continue
                    for model_dir in key_dir.iterdir():
                        if model_dir.is_dir():
                            model = _normalize_model(model_dir.name)
                            compound = _compound_key(gpu_dir.name, dsl, key_dir.name, model)
                            entry = found.setdefault(compound, {"dsl": dsl, "device": device, "shape_key": key_dir.name, "model": model})
                            entry["has_srcs"] = True

    return found


def _parse_shape_key(shape_key: str) -> dict | None:
    """Parse just the bare shape key (without gpu/dsl prefix).

    Handles formats like:
      gemm_sp_fp16fp32_16384x16384x16384  → op_type=gemm_sp, dtype=fp16fp32
      matmul_fp16_4096x4096x4096          → op_type=matmul,  dtype=fp16
      gemm_e4m3f32_8192x8192x8192         → op_type=gemm,    dtype=e4m3f32
    """
    m = _SHAPE_KEY_RE.match(shape_key)
    if not m:
        return None
    op_type, dtype, m_raw, n_raw, k_raw = m.groups()
    return {
        "op_type": op_type,
        "dtype": dtype,
        "m": int(m_raw),
        "n": int(n_raw),
        "k": int(k_raw),
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


def _parse_results_tsv(results_path: Path) -> dict:
    """Parse results.tsv and return baseline_tflops, best_tflops (excl. baseline), iteration_count."""
    try:
        lines = results_path.read_text(errors="replace").splitlines()
    except OSError:
        return {}

    baseline_tflops: float | None = None
    best_tflops: float | None = None
    iter_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if not parts:
            continue

        first = parts[0].lower()
        # Skip header lines
        if first in ("iter", "round"):
            continue

        # Detect format: croqtile has numeric round as col0; cuda has iter-name as col0
        try:
            int(parts[0])
            # croqtile format: round, iter, kernel, tflops, decision, ...
            tflops_idx, decision_idx = 3, 4
        except ValueError:
            # cuda format: iter, kernel, tflops, decision, bottleneck, ...
            tflops_idx, decision_idx = 2, 3

        if len(parts) <= max(tflops_idx, decision_idx):
            continue

        try:
            tflops = float(parts[tflops_idx].strip())
        except ValueError:
            tflops = 0.0

        decision = parts[decision_idx].strip().upper() if len(parts) > decision_idx else ""
        bottleneck = parts[decision_idx + 1].strip().lower() if len(parts) > decision_idx + 1 else ""

        # Only treat as cuBLAS/external baseline if the kernel name contains "baseline"
        # or the bottleneck is literally "baseline". The BASELINE decision in Croqtile
        # format means "initial draft" — it's NOT an external reference benchmark.
        kernel_col = parts[tflops_idx - 1].strip() if tflops_idx > 0 else ""
        is_baseline = bottleneck == "baseline" or "baseline" in kernel_col.lower()

        if is_baseline:
            if tflops > 0:
                baseline_tflops = tflops
        else:
            iter_count += 1
            if tflops > 0 and (best_tflops is None or tflops > best_tflops):
                best_tflops = tflops

    result: dict = {"current_iteration": iter_count}
    if baseline_tflops is not None:
        result["baseline_tflops"] = baseline_tflops
    if best_tflops is not None:
        result["best_tflops"] = best_tflops
    return result


async def prune_stale_tasks(session: AsyncSession) -> list[int]:
    """Remove DB tasks whose tuning artifacts no longer exist on disk.

    Called at startup and periodically by the scheduler.
    Only prunes scanner-created tasks (compound keys with '/').
    Never prunes running tasks to avoid race conditions.
    Returns the list of pruned task IDs.
    """
    discovered = discover_shape_keys()

    result = await session.execute(select(Task))
    existing: dict[str, Task] = {t.shape_key: t for t in result.scalars().all()}

    pruned_ids: list[int] = []
    for shape_key, task in list(existing.items()):
        if "/" not in shape_key:
            continue
        if shape_key in discovered:
            continue
        if task.status == "running":
            continue
        logger.info("Pruning stale task %d (%s) — disk artifacts removed", task.id, shape_key)
        pruned_ids.append(task.id)
        await session.delete(task)

    if pruned_ids:
        await session.commit()
        for tid in pruned_ids:
            await event_bus.publish("task_deleted", {"id": tid})
        logger.info("Pruned %d stale task(s) from DB", len(pruned_ids))

    return pruned_ids


async def scan_and_create_tasks(session: AsyncSession) -> int:
    """Discover tuning runs on disk, create tasks for new ones, and refresh metrics for existing ones.

    Uses compound key "{gpu}/{dsl}/{shape_key}/{model}" for uniqueness so that the
    same kernel shape tuned by different models on different GPU/DSL combinations
    are tracked separately.

    Tasks whose disk artifacts have been deleted are automatically removed from the
    database (only scanner-created tasks with compound shape_keys containing '/').

    Returns the number of new tasks created.
    """
    discovered = discover_shape_keys()

    result = await session.execute(select(Task))
    existing: dict[str, Task] = {t.shape_key: t for t in result.scalars().all()}

    created = 0
    changed = False

    pruned_in_scan: list[int] = []
    for shape_key, task in list(existing.items()):
        if "/" not in shape_key:
            continue
        if shape_key in discovered:
            continue
        if task.status == "running":
            continue
        logger.info("Pruning stale task %d (%s) — disk artifacts removed", task.id, shape_key)
        pruned_in_scan.append(task.id)
        await session.delete(task)
        changed = True

    if not discovered:
        if changed:
            await session.commit()
        return 0

    bare_keys_with_tasks: set[str] = set()
    for sk in existing:
        bare = sk.split("/")[-1] if "/" in sk else sk
        bare_keys_with_tasks.add(bare)

    for compound_key, info in discovered.items():
        metrics = _collect_metrics(info)
        bare_shape_key = info.get("shape_key", compound_key.split("/")[-1])
        dsl = info.get("dsl", "unknown")
        device = info.get("device")
        model = info.get("model")

        if compound_key in existing:
            task = existing[compound_key]
            updated = False
            new_iter = metrics.get("current_iteration", 0)
            old_iter = task.current_iteration or 0
            if new_iter != old_iter:
                task.current_iteration = new_iter
                updated = True
                if "results_tsv" in info and new_iter > old_iter:
                    await _import_new_iteration_logs(session, task.id, info["results_tsv"], old_iter)
            if metrics.get("best_tflops") is not None and metrics["best_tflops"] != task.best_tflops:
                task.best_tflops = metrics["best_tflops"]
                updated = True
            if metrics.get("baseline_tflops") is not None and metrics["baseline_tflops"] != task.baseline_tflops:
                task.baseline_tflops = metrics["baseline_tflops"]
                updated = True
            if metrics.get("best_kernel") and not task.best_kernel:
                task.best_kernel = metrics["best_kernel"]
                updated = True
            if device and not task.device:
                task.device = device
                updated = True
            if model and task.model != model:
                task.model = model
                updated = True
            if dsl and not task.dsl:
                task.dsl = dsl
                updated = True
            if "idea_log" in info:
                await _enrich_from_idea_log(session, task.id, info["idea_log"])
            if "attempt_log" in info:
                await _import_attempt_log(session, task.id, info["attempt_log"])
            if updated:
                task.updated_at = datetime.now(timezone.utc)
                changed = True
            continue

        if bare_shape_key in bare_keys_with_tasks:
            logger.debug(
                "Skipping %s — a task for bare key %s already exists",
                compound_key, bare_shape_key,
            )
            continue

        parsed = _parse_shape_key(bare_shape_key)
        if parsed is None:
            logger.debug("Skipping unparseable shape key from disk: %s", compound_key)
            continue

        initial_status = "pending"

        task = Task(
            shape_key=compound_key,
            op_type=parsed["op_type"],
            dtype=parsed["dtype"],
            m=parsed["m"],
            n=parsed["n"],
            k=parsed["k"],
            dsl=dsl,
            mode="opencode",
            device=device,
            max_iterations=30,
            status=initial_status,
            current_iteration=metrics.get("current_iteration", 0),
            best_tflops=metrics.get("best_tflops"),
            baseline_tflops=metrics.get("baseline_tflops"),
            best_kernel=metrics.get("best_kernel"),
            model=model,
        )
        session.add(task)
        await session.flush()
        if "results_tsv" in info:
            await _import_iteration_logs_from_tsv(session, task.id, info["results_tsv"])
        if "idea_log" in info:
            await _enrich_from_idea_log(session, task.id, info["idea_log"])
        if "attempt_log" in info:
            await _import_attempt_log(session, task.id, info["attempt_log"])
        created += 1
        logger.info(
            "Auto-discovered task: %s [%s/%s/%s] (iter=%d, best=%.3f TFLOPS)",
            bare_shape_key, device or "?", dsl, model, metrics.get("current_iteration", 0), metrics.get("best_tflops") or 0,
        )

    if created or changed:
        await session.commit()
        for tid in pruned_in_scan:
            await event_bus.publish("task_deleted", {"id": tid})

    # Detect running agents and mark their tasks as "running"
    await _sync_agent_status(session)

    return created


async def _sync_agent_status(session: AsyncSession) -> None:
    """Match detected agents to tasks and update status/agent_type/model."""
    try:
        agents = detect_running_agents()
    except Exception:
        return

    if not agents:
        return

    active_agents: dict[str, "DetectedAgent"] = {}
    for agent in agents:
        if agent.kernel_path:
            active_agents[agent.kernel_path] = agent

    if not active_agents:
        return

    result = await session.execute(select(Task))
    tasks = result.scalars().all()

    for task in tasks:
        # Compound key format: "{gpu}/{dsl}/{shape_key}/{model}" — the kernel shape is at index [-2]
        parts = task.shape_key.split("/")
        bare_key = parts[-2] if len(parts) >= 2 else task.shape_key
        agent = active_agents.get(bare_key) or active_agents.get(task.shape_key)
        if not agent:
            for kp, ag in active_agents.items():
                if kp in task.shape_key:
                    agent = ag
                    break
        if not agent:
            continue

        updated = False
        if agent.model_id and agent.model_id != task.model:
            logger.info("Agent model update: %s model %s → %s", task.shape_key, task.model, agent.model_id)
            task.model = agent.model_id
            updated = True
        if agent.agent_type and agent.agent_type != task.agent_type:
            task.agent_type = agent.agent_type
            updated = True
        mode = agent.agent_type if agent.agent_type != "unknown" else None
        if mode and mode != task.mode:
            task.mode = mode
            updated = True

        # If a live agent is detected for this task, it is running — regardless of
        # request_budget (which is a scheduler dispatch counter, not a liveness flag).
        if task.status in ("pending", "cancelled", "completed"):
            task.status = "running"
            task.started_at = task.started_at or datetime.now(timezone.utc)
            updated = True
            logger.info("Agent sync: %s → running (agent=%s, model=%s)", task.shape_key, agent.agent_type, agent.model_id)

        if updated:
            task.updated_at = datetime.now(timezone.utc)
            await event_bus.publish("task_update", task.to_dict())

    await session.commit()


async def _import_new_iteration_logs(session: AsyncSession, task_id: int, results_path: Path, skip: int) -> None:
    """Import only the iteration log rows that come after `skip` existing rows."""
    await _import_iteration_logs_from_tsv(session, task_id, results_path, start_after=skip)


async def _import_iteration_logs_from_tsv(session: AsyncSession, task_id: int, results_path: Path, start_after: int = 0) -> None:
    """Populate IterationLog rows from results.tsv for a freshly-created task.
    
    Baseline rows (iter000/cuBLAS/torch) get iteration=0 and decision='BASELINE'.
    Tuning iterations start at 1.
    """
    try:
        lines = results_path.read_text(errors="replace").splitlines()
    except OSError:
        return

    tuning_iter = 0
    data_row_count = 0
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

        try:
            int(parts[0])
            tflops_idx, decision_idx, kernel_idx = 3, 4, 2
        except ValueError:
            tflops_idx, decision_idx, kernel_idx = 2, 3, 1

        if len(parts) <= tflops_idx:
            continue

        try:
            tflops_raw = parts[tflops_idx].strip()
            tflops = float(tflops_raw) if tflops_raw not in ("", "INVALID", "FAIL") else None
        except ValueError:
            tflops = None

        decision = parts[decision_idx].strip() if len(parts) > decision_idx else None
        bottleneck_raw = parts[decision_idx + 1].strip() if len(parts) > decision_idx + 1 else ""
        kernel = parts[kernel_idx].strip() if len(parts) > kernel_idx else None
        idea = parts[decision_idx + 2].strip() if len(parts) > decision_idx + 2 else None

        is_baseline = (
            bottleneck_raw.lower() in ("baseline", "baseline_profile")
            or "baseline" in (kernel or "").lower()
            or (kernel or "").lower().startswith("framework/")
        )

        data_row_count += 1
        if data_row_count <= start_after:
            continue

        if is_baseline:
            iter_num = 0
            decision = "BASELINE"
        else:
            tuning_iter += 1
            iter_num = start_after + tuning_iter

        session.add(IterationLog(
            task_id=task_id,
            iteration=iter_num,
            kernel_path=kernel,
            tflops=tflops,
            decision=decision,
            bottleneck=bottleneck_raw or None,
            idea_summary=idea,
        ))


async def _enrich_from_idea_log(session: AsyncSession, task_id: int, idea_log_path: Path) -> None:
    """Merge bottleneck/category/idea_summary from idea-log.jsonl into existing IterationLog rows."""
    try:
        lines = idea_log_path.read_text(errors="replace").splitlines()
    except OSError:
        return

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        iteration = entry.get("iteration") or entry.get("iter")
        if iteration is None:
            continue
        try:
            iteration = int(iteration)
        except (TypeError, ValueError):
            continue

        result = await session.execute(
            select(IterationLog).where(
                IterationLog.task_id == task_id,
                IterationLog.iteration == iteration,
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            if not existing.bottleneck and entry.get("bottleneck"):
                existing.bottleneck = str(entry["bottleneck"])
            if not existing.idea_summary and entry.get("idea_summary"):
                existing.idea_summary = str(entry["idea_summary"])
        else:
            session.add(IterationLog(
                task_id=task_id,
                iteration=iteration,
                bottleneck=str(entry.get("bottleneck", "")) or None,
                idea_summary=str(entry.get("idea_summary", "")) or None,
            ))


async def _import_attempt_log(session: AsyncSession, task_id: int, attempt_log_path: Path) -> None:
    """Create IterationLog entries with decision=COMPILE_FAIL from attempt-log.jsonl."""
    try:
        lines = attempt_log_path.read_text(errors="replace").splitlines()
    except OSError:
        return

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        iteration = entry.get("iteration") or entry.get("iter")
        if iteration is None:
            continue
        try:
            iteration = int(iteration)
        except (TypeError, ValueError):
            continue

        result = await session.execute(
            select(IterationLog).where(
                IterationLog.task_id == task_id,
                IterationLog.iteration == iteration,
            )
        )
        if result.scalar_one_or_none():
            continue

        session.add(IterationLog(
            task_id=task_id,
            iteration=iteration,
            kernel_path=entry.get("kernel") or entry.get("kernel_path"),
            decision="COMPILE_FAIL",
            idea_summary=str(entry.get("error", entry.get("idea_summary", ""))) or None,
        ))


def _collect_metrics(info: dict) -> dict:
    """Extract metrics from checkpoint and results.tsv.

    results.tsv is the authoritative source for iteration count, best_tflops (excl. baseline),
    and baseline_tflops.  checkpoint files supplement with the current best kernel name.
    """
    metrics: dict = {}

    if "checkpoint" in info:
        metrics.update(_read_checkpoint_metrics(info["checkpoint"]))

    if "results_tsv" in info:
        tsv_metrics = _parse_results_tsv(info["results_tsv"])
        # results.tsv wins for iteration count, baseline, and best_tflops
        metrics.update(tsv_metrics)

    return metrics
