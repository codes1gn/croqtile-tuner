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

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .agent_detector import DetectedAgent, detect_running_agents
from .config import settings
from .events import event_bus
from .models import IterationLog, Task

try:
    from .agent import _detect_session_from_db
except ImportError:
    _detect_session_from_db = None

logger = logging.getLogger("croqtuner.artifact_scanner")


def _is_baseline_row(kernel: str | None, bottleneck: str | None) -> bool:
    """Detect whether a results.tsv row is the external baseline (cuBLAS/torch)."""
    bn = (bottleneck or "").lower()
    k = (kernel or "").lower()
    return (
        bn in ("baseline", "baseline_profile")
        or "baseline" in k
        or k.startswith("framework/")
    )


_SHAPE_KEY_RE = re.compile(
    r"^(.+?)_([A-Za-z0-9]+)_(\d+)x(\d+)x(\d+)$"
)

_CANONICAL_DTYPE = {
    "f16": "fp16", "f32": "fp32", "f64": "fp64",
    "fp16": "fp16", "fp32": "fp32", "fp64": "fp64",
    "bf16": "bf16",
    "e4m3": "e4m3", "e5m2": "e5m2",
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
}


def _normalize_dtype(raw: str) -> str:
    """Normalize a compound dtype to canonical form.

    e.g. 'f16f32' → 'fp16fp32', 'f16fp32' → 'fp16fp32', 'fp16' → 'fp16'
    """
    low = raw.lower()
    tokens = sorted(_CANONICAL_DTYPE.keys(), key=len, reverse=True)
    result = []
    pos = 0
    while pos < len(low):
        matched = False
        for tok in tokens:
            if low[pos:].startswith(tok):
                result.append(_CANONICAL_DTYPE[tok])
                pos += len(tok)
                matched = True
                break
        if not matched:
            result.append(low[pos])
            pos += 1
    return "".join(result)


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
                        config_file = model_dir / "task_config.json"
                        if config_file.exists():
                            try:
                                cfg = json.loads(config_file.read_text())
                                if cfg.get("task_uid"):
                                    entry["task_uid"] = cfg["task_uid"]
                                if cfg.get("mode"):
                                    entry["mode"] = cfg["mode"]
                                if cfg.get("status"):
                                    entry["status"] = cfg["status"]
                            except (json.JSONDecodeError, OSError):
                                pass

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
        "dtype": _normalize_dtype(dtype),
        "m": int(m_raw),
        "n": int(n_raw),
        "k": int(k_raw),
    }


def _parse_shape_identity(shape_key: str) -> tuple[str, str]:
    """Extract (op_type, 'MxNxK') from a bare shape key for fuzzy matching.

    Returns ("", "") if unparseable.
    """
    parsed = _parse_shape_key(shape_key)
    if not parsed:
        return ("", "")
    return (parsed["op_type"], f"{parsed['m']}x{parsed['n']}x{parsed['k']}")


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

        kernel_col = parts[tflops_idx - 1].strip() if tflops_idx > 0 else ""
        is_baseline = _is_baseline_row(kernel_col, bottleneck)

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
    Respects task_uid: a task whose uid appears in any discovered artifact is kept
    even if its shape_key has drifted (agent changed path at runtime).
    Returns the list of pruned task IDs.
    """
    discovered = discover_shape_keys()
    discovered_uids: set[str] = set()
    for info in discovered.values():
        uid = info.get("task_uid")
        if uid:
            discovered_uids.add(uid)

    result = await session.execute(select(Task))
    all_tasks = list(result.scalars().all())

    pruned_ids: list[int] = []
    for task in all_tasks:
        if "/" not in task.shape_key:
            continue
        if task.shape_key in discovered:
            continue
        if task.task_uid and task.task_uid in discovered_uids:
            continue
        if task.status == "running":
            continue
        logger.info("Pruning stale task %d (%s) — disk artifacts removed", task.id, task.shape_key)
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

    # Deduplicate entries that share the same task_uid — keep the one with
    # more progress (results_tsv present, higher iteration count).
    uid_best: dict[str, str] = {}  # task_uid → best compound_key
    for compound_key, info in discovered.items():
        uid = info.get("task_uid")
        if not uid:
            continue
        if uid not in uid_best:
            uid_best[uid] = compound_key
            continue
        prev_key = uid_best[uid]
        prev_info = discovered[prev_key]
        prev_has_tsv = "results_tsv" in prev_info
        curr_has_tsv = "results_tsv" in info
        if curr_has_tsv and not prev_has_tsv:
            uid_best[uid] = compound_key
        elif curr_has_tsv and prev_has_tsv:
            prev_metrics = _collect_metrics(prev_info)
            curr_metrics = _collect_metrics(info)
            if (curr_metrics.get("current_iteration", 0) >
                    prev_metrics.get("current_iteration", 0)):
                uid_best[uid] = compound_key

    # Remove duplicate entries (same uid, less progress)
    keys_to_remove: set[str] = set()
    for compound_key, info in discovered.items():
        uid = info.get("task_uid")
        if uid and uid in uid_best and uid_best[uid] != compound_key:
            keys_to_remove.add(compound_key)
    for k in keys_to_remove:
        logger.info("Dedup: dropping %s (same task_uid %s as %s)",
                     k, discovered[k].get("task_uid"), uid_best[discovered[k]["task_uid"]])
        del discovered[k]

    result = await session.execute(select(Task))
    all_tasks = list(result.scalars().all())
    existing_by_key: dict[str, Task] = {t.shape_key: t for t in all_tasks}
    existing_by_uid: dict[str, Task] = {t.task_uid: t for t in all_tasks if t.task_uid}

    # Build set of discovered task_uids for stale-pruning
    discovered_uids: set[str] = set()
    for info in discovered.values():
        uid = info.get("task_uid")
        if uid:
            discovered_uids.add(uid)

    created = 0
    changed = False

    pruned_in_scan: list[int] = []
    for task in list(all_tasks):
        shape_key = task.shape_key
        if "/" not in shape_key:
            continue
        # Task is still on disk (by compound key) — keep
        if shape_key in discovered:
            continue
        # Task is still on disk (by task_uid, but agent moved artifacts to a new path) — keep
        if task.task_uid and task.task_uid in discovered_uids:
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

    for compound_key, info in discovered.items():
        metrics = _collect_metrics(info)
        bare_shape_key = info.get("shape_key", compound_key.split("/")[-1])
        dsl = info.get("dsl", "unknown")
        device = info.get("device")
        model = info.get("model")
        disk_uid = info.get("task_uid")
        disk_mode = info.get("mode")
        disk_status = info.get("status")

        # Skip cancelled artifacts — don't create new tasks from dead entries
        if disk_status == "cancelled":
            logger.debug("Skipping cancelled artifact: %s", compound_key)
            continue

        # Match: first by task_uid, then by compound key, then by bare shape+dsl
        task: Task | None = None
        if disk_uid and disk_uid in existing_by_uid:
            task = existing_by_uid[disk_uid]
            if task.shape_key != compound_key:
                logger.info(
                    "Task %d (uid=%s) path changed: %s → %s",
                    task.id, disk_uid, task.shape_key, compound_key,
                )
                task.shape_key = compound_key
                changed = True
        elif compound_key in existing_by_key:
            task = existing_by_key[compound_key]
        else:
            # Fall back: match by structural identity (op_type + dsl + dimensions)
            # Handles shape key variations like f16f32 vs f16fp32
            disk_op, disk_dims = _parse_shape_identity(bare_shape_key)
            for t in all_tasks:
                if t.id in [x.id for x in [task] if task]:
                    continue
                if t.dsl != dsl:
                    continue
                t_op = t.op_type or ""
                t_dims = f"{t.m}x{t.n}x{t.k}" if t.m and t.n and t.k else ""
                if disk_op == t_op and disk_dims == t_dims:
                    task = t
                    logger.info(
                        "Matched task %d (uid=%s, %s) to disk artifact %s by op+dims",
                        t.id, t.task_uid, t.shape_key, compound_key,
                    )
                    task.shape_key = compound_key
                    changed = True
                    break

        if task is not None:
            from .artifact_config import sync_artifact_to_task
            if await sync_artifact_to_task(session, task):
                await session.flush()
            updated = False
            new_iter = metrics.get("current_iteration", 0)
            old_iter = task.current_iteration or 0

            log_count_result = await session.execute(
                select(func.count()).where(IterationLog.task_id == task.id)
            )
            log_count = log_count_result.scalar() or 0

            if log_count == 0 and "results_tsv" in info:
                await _import_iteration_logs_from_tsv(session, task.id, info["results_tsv"])
                updated = True
            elif new_iter != old_iter:
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
            if device and task.device != device:
                task.device = device
                updated = True
            if model and not task.model:
                task.model = model
                updated = True
            if dsl and not task.dsl:
                task.dsl = dsl
                updated = True
            if disk_mode and task.mode != disk_mode:
                task.mode = disk_mode
                updated = True
            # Re-parse shape_key fields if they drifted
            parsed = _parse_shape_key(bare_shape_key)
            if parsed:
                for field in ("op_type", "dtype", "m", "n", "k"):
                    disk_val = parsed[field]
                    if getattr(task, field) != disk_val:
                        setattr(task, field, disk_val)
                        updated = True
            if "idea_log" in info:
                await _enrich_from_idea_log(session, task.id, info["idea_log"])
            if "attempt_log" in info:
                await _import_attempt_log(session, task.id, info["attempt_log"])
            if updated:
                task.updated_at = datetime.now(timezone.utc)
                changed = True
                from .artifact_config import sync_task_to_artifact
                await sync_task_to_artifact(session, task)
            continue

        # New task — parse and create
        parsed = _parse_shape_key(bare_shape_key)
        if parsed is None:
            logger.debug("Skipping unparseable shape key from disk: %s", compound_key)
            continue

        initial_status = "pending"

        new_task = Task(
            shape_key=compound_key,
            op_type=parsed["op_type"],
            dtype=parsed["dtype"],
            m=parsed["m"],
            n=parsed["n"],
            k=parsed["k"],
            dsl=dsl,
            mode=disk_mode or "opencode",
            device=device,
            max_iterations=30,
            status=initial_status,
            current_iteration=metrics.get("current_iteration", 0),
            best_tflops=metrics.get("best_tflops"),
            baseline_tflops=metrics.get("baseline_tflops"),
            best_kernel=metrics.get("best_kernel"),
            model=model,
        )
        if disk_uid:
            new_task.task_uid = disk_uid
        session.add(new_task)
        await session.flush()
        if "results_tsv" in info:
            await _import_iteration_logs_from_tsv(session, new_task.id, info["results_tsv"])
        if "idea_log" in info:
            await _enrich_from_idea_log(session, new_task.id, info["idea_log"])
        if "attempt_log" in info:
            await _import_attempt_log(session, new_task.id, info["attempt_log"])
        from .artifact_config import sync_artifact_to_task, sync_task_to_artifact
        await sync_artifact_to_task(session, new_task)
        await session.flush()
        await sync_task_to_artifact(session, new_task)

        created += 1
        logger.info(
            "Auto-discovered task: %s (uid=%s) [%s/%s/%s] (iter=%d, best=%.3f TFLOPS)",
            bare_shape_key, disk_uid or "new", device or "?", dsl, model,
            metrics.get("current_iteration", 0), metrics.get("best_tflops") or 0,
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
        session_id = agent.session_id
        if not session_id and agent.agent_type == "opencode" and _detect_session_from_db:
            session_id = _detect_session_from_db(0)
        if session_id and session_id != task.opencode_session_id:
            logger.info("Session link: %s → %s", task.shape_key, session_id)
            task.opencode_session_id = session_id
            updated = True
        if agent.agent_type and agent.agent_type != task.agent_type:
            task.agent_type = agent.agent_type
            updated = True
        mode = agent.agent_type if agent.agent_type != "unknown" else None
        if mode and mode != task.mode:
            task.mode = mode
            updated = True

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

        is_baseline = _is_baseline_row(kernel, bottleneck_raw)

        if is_baseline:
            iter_num = 0
            decision = "BASELINE"
        else:
            tuning_iter += 1
            if tuning_iter <= start_after:
                continue
            iter_num = tuning_iter

        existing = await session.execute(
            select(IterationLog).where(
                IterationLog.task_id == task_id,
                IterationLog.iteration == iter_num,
            )
        )
        existing_log = existing.scalar_one_or_none()
        if existing_log is not None:
            # TSV is authoritative — overwrite values that may have been
            # captured incorrectly from agent chat text via regex
            if tflops is not None and existing_log.tflops != tflops:
                existing_log.tflops = tflops
            if decision and existing_log.decision != decision:
                existing_log.decision = decision
            if kernel and not existing_log.kernel_path:
                existing_log.kernel_path = kernel
            if (bottleneck_raw or None) and not existing_log.bottleneck:
                existing_log.bottleneck = bottleneck_raw or None
            if idea and not existing_log.idea_summary:
                existing_log.idea_summary = idea
            continue

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
