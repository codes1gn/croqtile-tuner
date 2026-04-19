"""CroqTuner Agent Bot — FastAPI backend."""

import asyncio
import logging
import shutil
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from .agent_detector import detect_running_agents, get_all_agent_sessions
from .database import async_session, get_session, init_db
from .events import event_bus
from .iteration_history import read_iteration_history
from .models import AgentLog, IterationLog, Task, TaskSession
from .opencode_sessions import read_session_history
from .runtime_settings import (
    available_models,
    available_variants,
    get_default_model,
    get_default_variant,
    set_default_model,
    get_auto_wake_enabled,
    set_auto_wake_enabled,
    get_use_proxy,
    set_use_proxy,
)
from .scheduler import scheduler
from .schemas import (
    AgentLogResponse,
    AutoWakeSettingsResponse,
    AutoWakeSettingsUpdate,
    HealthResponse,
    IterationLogResponse,
    ModelSettingsResponse,
    ModelSettingsUpdate,
    ProxySettingsResponse,
    ProxySettingsUpdate,
    ResumeRequest,
    SessionHistoryResponse,
    TaskCreate,
    TaskResponse,
    TaskSessionResponse,
    TaskUpdate,
)
from .artifact_scanner import prune_stale_tasks, scan_and_create_tasks
from .config import invalidate_model_cache, settings
from .file_watcher import TuningDirWatcher
from .state_seed import seed_tasks_from_state_if_empty
from .task_runtime import apply_live_runtime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("croqtuner")


def _ensure_writable() -> None:
    if settings.read_only_mode:
        raise HTTPException(403, "Monitor is in read-only mode")


def _detect_gpu_name() -> str | None:
    """Return the first GPU name from nvidia-smi, e.g. 'NVIDIA GeForce RTX 3070'."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0].strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
        pass
    return None


_watcher = TuningDirWatcher(settings.tuning_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    async with async_session() as session:
        await seed_tasks_from_state_if_empty(session)
    async with async_session() as session:
        pruned_ids = await prune_stale_tasks(session)
        if pruned_ids:
            logger.info("Startup: cleaned %d stale task(s) from DB", len(pruned_ids))
    async with async_session() as session:
        await scan_and_create_tasks(session)
    # Initialize default settings (use_proxy defaults to true for proxychains4)
    async with async_session() as session:
        use_proxy = await get_use_proxy(session)
        await session.commit()
        logger.info("Proxy setting initialized: %s", use_proxy)
    _watcher.start(asyncio.get_running_loop())
    if settings.scheduler_enabled:
        await scheduler.start()
    else:
        logger.warning("Scheduler disabled by CROQTUNER_SCHEDULER_ENABLED=false")
    if settings.read_only_mode:
        logger.warning("Monitor write APIs disabled by CROQTUNER_READ_ONLY_MODE=true")
    yield
    if settings.scheduler_enabled:
        await scheduler.stop()
    _watcher.stop()


app = FastAPI(title="CroqTuner Agent Bot", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/tasks", response_model=list[TaskResponse])
async def list_tasks(
    status: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Task).order_by(Task.created_at.desc())
    if status:
        stmt = stmt.where(Task.status == status)
    result = await session.execute(stmt)
    return [
        TaskResponse(**apply_live_runtime(t, t.to_dict()))
        for t in result.scalars().all()
    ]


@app.post("/api/tasks", response_model=TaskResponse, status_code=201)
async def create_task(
    body: TaskCreate,
    session: AsyncSession = Depends(get_session),
):
    _ensure_writable()
    from .artifact_scanner import _normalize_dtype
    max_iter = 30
    canonical_dtype = _normalize_dtype(body.dtype)
    shape_key = f"{body.op_type}_{canonical_dtype}_{body.m}x{body.n}x{body.k}"

    task = Task(
        shape_key=shape_key,
        op_type=body.op_type,
        dtype=canonical_dtype,
        m=body.m,
        n=body.n,
        k=body.k,
        dsl=body.dsl,
        mode=body.mode,
        model=body.model,
        variant=body.variant,
        request_budget=body.request_budget,
        max_iterations=max_iter,
        status="pending",
        current_iteration=0,
        agent_type=body.mode,
        device=_detect_gpu_name(),
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    await event_bus.publish("task_update", task.to_dict())
    return TaskResponse(**task.to_dict())


@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int, session: AsyncSession = Depends(get_session)):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return TaskResponse(**apply_live_runtime(task, task.to_dict()))


@app.patch("/api/tasks/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: int,
    body: TaskUpdate,
    session: AsyncSession = Depends(get_session),
):
    _ensure_writable()
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    changed = False

    if body.model is not None or body.variant is not None:
        if task.status in ("running", "completed"):
            raise HTTPException(
                400, "Cannot change model for running or completed tasks"
            )
        if body.model is not None:
            task.model = body.model
            changed = True
        if body.variant is not None:
            task.variant = body.variant
            changed = True

    if body.request_budget is not None and body.request_budget != task.request_budget:
        task.request_budget = body.request_budget
        changed = True

    if body.status == "cancelled":
        if task.status not in ("pending", "running", "waiting"):
            raise HTTPException(
                400, "Can only cancel pending, running, or waiting tasks"
            )
        if task.status != "cancelled":
            task.status = "cancelled"
            changed = True
    elif body.status == "pending" and task.status == "waiting":
        task.status = "pending"
        changed = True
    elif body.status == "waiting" and task.status == "pending":
        if task.id in scheduler.active_task_ids:
            raise HTTPException(400, "Cannot demote a task that is running")
        task.status = "waiting"
        changed = True
    elif body.status is not None and body.status != task.status:
        raise HTTPException(
            400, f"Cannot change task from {task.status} to {body.status}"
        )

    if changed:
        task.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await event_bus.publish("task_update", task.to_dict())
        from .artifact_config import sync_task_to_artifact
        await sync_task_to_artifact(session, task)

    return TaskResponse(**task.to_dict())


def _delete_task_artifacts(shape_key: str) -> None:
    """Delete ALL disk artifacts for a task so the scanner never re-discovers it.

    Artifacts are the single source of truth.  The scanner creates DB records
    from artifacts; it deletes DB records when artifacts disappear.  Therefore:
    - Deleting a task means deleting its artifacts — the DB entry then vanishes
      automatically on the next prune_stale_tasks() pass.
    - We must delete EVERY directory that discover_shape_keys() inspects so that
      no leftover file can trigger task re-creation.

    Compound key format: "{gpu}/{dsl}/{bare_shape_key}/{model}"  (scanner-created)
    Bare key format:     "{bare_shape_key}"                       (UI-created tasks)
    """
    from .config import settings

    tuning_dir = settings.tuning_dir.resolve()
    # ALL subdirs that discover_shape_keys() inspects — must stay in sync with that function.
    subdirs = ["logs", "srcs", "checkpoints", "cmd", "perf", "bin", "memory", "baseline"]

    parts = shape_key.split("/")

    def _rmdir_if_empty(d: Path) -> None:
        try:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass

    if len(parts) >= 3:
        # Compound key: delete the specific model-scoped leaf directory
        gpu, dsl = parts[0], parts[1]
        bare_key = parts[2]
        model = parts[3] if len(parts) >= 4 else None
        dsl_root = tuning_dir / gpu / dsl
        if not dsl_root.exists():
            return
        for subdir in subdirs:
            target = (dsl_root / subdir / bare_key / model) if model else (dsl_root / subdir / bare_key)
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                logger.info("Deleted artifacts: %s", target)
                _rmdir_if_empty(target.parent)
    else:
        # Bare key: search all gpu/dsl combos on disk for matching bare_shape_key dirs
        bare_key = parts[0]
        if not tuning_dir.exists():
            return
        for gpu_dir in tuning_dir.iterdir():
            if not gpu_dir.is_dir():
                continue
            for dsl_dir in gpu_dir.iterdir():
                if not dsl_dir.is_dir():
                    continue
                for subdir in subdirs:
                    target = dsl_dir / subdir / bare_key
                    if target.exists():
                        shutil.rmtree(target, ignore_errors=True)
                        logger.info("Deleted bare-key artifacts: %s", target)
                        _rmdir_if_empty(target.parent)


@app.delete("/api/tasks/{task_id}", status_code=204)
async def delete_task(
    task_id: int,
    clean_artifacts: bool = Query(True),
    session: AsyncSession = Depends(get_session),
):
    _ensure_writable()
    """Delete a task.

    Design: artifacts are the single source of truth.

    With clean_artifacts=True (default):
      1. Stop any running agent worker for this task.
      2. Delete all artifacts from disk — every directory that the scanner inspects.
      3. Immediately delete the DB record and emit task_deleted so the UI updates
         without waiting for the 2-second file-watcher debounce.
      4. The file watcher will fire a prune scan, but prune_stale_tasks will find
         nothing to do (record already gone from DB, artifacts already gone from disk).

    With clean_artifacts=False:
      Artifacts remain on disk.  The next file-watcher scan will re-create the DB
      record from the surviving artifacts.  Use this only if you want to preserve
      the tuning history while temporarily removing the UI entry.
    """
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    # Stop any active scheduler worker first so it doesn't restart the process.
    await scheduler.cancel_active_task(task_id)

    if clean_artifacts:
        _delete_task_artifacts(task.shape_key)

    # Remove the DB record immediately so the UI doesn't see a zombie task.
    # If clean_artifacts=True, the file watcher's subsequent prune scan will find
    # the record already gone and do nothing. If clean_artifacts=False, the scanner
    # will re-create the record from the surviving artifacts — which is intentional.
    await session.delete(task)
    await session.commit()
    await event_bus.publish("task_deleted", {"id": task_id})


@app.post("/api/tasks/{task_id}/retry", response_model=TaskResponse, status_code=200)
async def retry_task(task_id: int, session: AsyncSession = Depends(get_session)):
    _ensure_writable()
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status not in ("pending", "completed", "cancelled"):
        raise HTTPException(
            400, "Can only retry pending, completed, or cancelled tasks"
        )

    # Update existing task status instead of creating a new task
    task.status = "pending"
    task.current_iteration = 0
    task.request_budget = max(task.request_budget or 1, 1)
    task.error_message = None  # Clear any previous error
    await session.commit()
    await session.refresh(task)

    # Persist the new status to disk artifact so it survives rescans
    from .artifact_config import sync_task_to_artifact
    await sync_task_to_artifact(session, task)

    await event_bus.publish("task_update", task.to_dict())
    return TaskResponse(**task.to_dict())


@app.post("/api/tasks/{task_id}/resume", response_model=TaskResponse, status_code=200)
async def resume_task(
    task_id: int,
    body: ResumeRequest,
    session: AsyncSession = Depends(get_session),
):
    _ensure_writable()
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status in ("running", "pending"):
        raise HTTPException(400, f"Cannot resume a {task.status} task")

    from_iter = min(body.from_iteration, task.current_iteration, task.max_iterations)

    # Update existing task status instead of creating a new task
    task.status = "pending"
    task.current_iteration = from_iter
    task.request_budget = max(task.request_budget or 1, 1)
    task.error_message = None  # Clear any previous error
    await session.commit()
    await session.refresh(task)

    # Persist the new status to disk artifact so it survives rescans
    from .artifact_config import sync_task_to_artifact
    await sync_task_to_artifact(session, task)

    await event_bus.publish("task_update", task.to_dict())
    return TaskResponse(**task.to_dict())


@app.get("/api/tasks/{task_id}/logs", response_model=list[IterationLogResponse])
async def get_iteration_logs(
    task_id: int, session: AsyncSession = Depends(get_session)
):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    # Prefer DB logs (imported from results.tsv during scan) for accuracy
    result = await session.execute(
        select(IterationLog)
        .where(IterationLog.task_id == task_id)
        .order_by(IterationLog.iteration.desc())
    )
    db_logs = result.scalars().all()
    if db_logs:
        return [IterationLogResponse(**log.to_dict()) for log in db_logs]

    # Fallback: read directly from disk (for tasks not yet scanned)
    artifact_logs = read_iteration_history(task.shape_key, task.id)
    return [IterationLogResponse(**log) for log in artifact_logs]


@app.get("/api/tasks/{task_id}/agent-logs", response_model=list[AgentLogResponse])
async def get_agent_logs(
    task_id: int,
    limit: int = Query(100, ge=1, le=1000),
    session: AsyncSession = Depends(get_session),
):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    result = await session.execute(
        select(AgentLog)
        .where(AgentLog.task_id == task_id)
        .order_by(AgentLog.timestamp.desc())
        .limit(limit)
    )
    return [AgentLogResponse(**log.to_dict()) for log in result.scalars().all()]


@app.get("/api/tasks/{task_id}/session-history", response_model=SessionHistoryResponse)
async def get_session_history(
    task_id: int,
    limit: int = Query(200, ge=1, le=1000),
    sid: str | None = Query(None, alias="session_id"),
    session: AsyncSession = Depends(get_session),
):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    target_sid = sid or task.opencode_session_id
    if not target_sid:
        return SessionHistoryResponse(
            session_id=None,
            session_title=None,
            session_directory=None,
            entries=[],
        )
    history = await read_session_history(target_sid, limit=limit)
    return SessionHistoryResponse(**history)


@app.get("/api/tasks/{task_id}/sessions", response_model=list[TaskSessionResponse])
async def get_task_sessions(
    task_id: int,
    session: AsyncSession = Depends(get_session),
):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    result = await session.execute(
        select(TaskSession).where(TaskSession.task_id == task_id).order_by(TaskSession.started_at)
    )
    return [
        TaskSessionResponse(**ts.to_dict())
        for ts in result.scalars().all()
    ]


@app.get("/api/tasks/{task_id}/session-history/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history_by_id(
    task_id: int,
    session_id: str,
    limit: int = Query(200, ge=1, le=1000),
    session: AsyncSession = Depends(get_session),
):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    history = await read_session_history(session_id, limit=limit)
    return SessionHistoryResponse(**history)


@app.get("/api/tasks/{task_id}/activity-log")
async def get_activity_log(
    task_id: int,
    limit: int = Query(200, ge=1, le=2000),
    session: AsyncSession = Depends(get_session),
):
    """Serve the harness activity log (activity.jsonl) from disk for a task."""
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    entries = _read_activity_log(task.shape_key, limit)
    return entries


def _read_activity_log(shape_key: str, limit: int = 200) -> list[dict]:
    """Read activity.jsonl from the task's memory/ directory."""
    import json as _json

    parts = shape_key.split("/")
    if len(parts) < 3:
        return []

    gpu, dsl, bare_key = parts[0], parts[1], parts[2]
    model = parts[3] if len(parts) >= 4 else None

    if model:
        log_path = settings.tuning_dir / gpu / dsl / "memory" / bare_key / model / "activity.jsonl"
    else:
        log_path = settings.tuning_dir / gpu / dsl / "memory" / bare_key / "activity.jsonl"

    if not log_path.exists():
        return []

    entries: list[dict] = []
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(_json.loads(line))
            except _json.JSONDecodeError:
                continue
    except OSError:
        return []

    return entries[-limit:]


@app.get("/api/settings/model", response_model=ModelSettingsResponse)
async def get_model_settings(session: AsyncSession = Depends(get_session)):
    return ModelSettingsResponse(
        default_model=await get_default_model(session),
        default_variant=await get_default_variant(session),
        available_models=available_models(),
        available_variants=available_variants(),
    )


@app.patch("/api/settings/model", response_model=ModelSettingsResponse)
async def update_model_settings(
    body: ModelSettingsUpdate,
    session: AsyncSession = Depends(get_session),
):
    _ensure_writable()
    model, variant = await set_default_model(
        session, body.default_model, body.default_variant
    )
    await session.commit()
    return ModelSettingsResponse(
        default_model=model,
        default_variant=variant,
        available_models=available_models(),
        available_variants=available_variants(),
    )


@app.post("/api/settings/model/refresh", response_model=ModelSettingsResponse)
async def refresh_models(session: AsyncSession = Depends(get_session)):
    """Force-refresh the cached model list from opencode and cursor-agent CLIs."""
    _ensure_writable()
    invalidate_model_cache()
    return ModelSettingsResponse(
        default_model=await get_default_model(session),
        default_variant=await get_default_variant(session),
        available_models=available_models(),
        available_variants=available_variants(),
    )


@app.get("/api/settings/auto-wake", response_model=AutoWakeSettingsResponse)
async def get_auto_wake_settings(session: AsyncSession = Depends(get_session)):
    """Get the auto-wake toggle state.

    When enabled, the scheduler will automatically start opencode for pending tasks.
    When disabled, the monitor only observes and polls artifacts (pure monitor mode).
    """
    return AutoWakeSettingsResponse(
        auto_wake_enabled=await get_auto_wake_enabled(session),
    )


@app.patch("/api/settings/auto-wake", response_model=AutoWakeSettingsResponse)
async def update_auto_wake_settings(
    body: AutoWakeSettingsUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Toggle the auto-wake setting.

    When enabled: scheduler will auto-start opencode for pending tasks.
    When disabled: pure monitor mode, no automatic task execution.
    """
    _ensure_writable()
    enabled = await set_auto_wake_enabled(session, body.auto_wake_enabled)
    await session.commit()
    logger.info("Auto-wake setting changed to: %s", enabled)
    await event_bus.publish("auto_wake_changed", {"enabled": enabled})
    return AutoWakeSettingsResponse(auto_wake_enabled=enabled)


@app.get("/api/settings/proxy", response_model=ProxySettingsResponse)
async def get_proxy_settings(session: AsyncSession = Depends(get_session)):
    use_proxy = await get_use_proxy(session)
    await session.commit()  # Commit to persist the default setting if newly created
    return ProxySettingsResponse(use_proxy=use_proxy)


@app.patch("/api/settings/proxy", response_model=ProxySettingsResponse)
async def update_proxy_settings(
    body: ProxySettingsUpdate,
    session: AsyncSession = Depends(get_session),
):
    _ensure_writable()
    enabled = await set_use_proxy(session, body.use_proxy)
    await session.commit()
    logger.info("Proxy setting changed to: %s", enabled)
    return ProxySettingsResponse(use_proxy=enabled)


@app.get("/api/events")
async def sse_events():
    return EventSourceResponse(event_bus.subscribe(), ping=15)


@app.get("/api/health", response_model=HealthResponse)
async def health(session: AsyncSession = Depends(get_session)):
    gpu_info = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        gpu_info = "nvidia-smi not available"

    counts_result = await session.execute(
        select(Task.status, func.count(Task.id)).group_by(Task.status)
    )
    task_counts = {status: count for status, count in counts_result.all()}
    for status in (
        "waiting",
        "pending",
        "running",
        "completed",
        "cancelled",
    ):
        task_counts.setdefault(status, 0)

    return HealthResponse(
        status="ok",
        scheduler_running=scheduler.running,
        read_only_mode=settings.read_only_mode,
        active_task_id=scheduler.active_task_id,
        active_task_ids=scheduler.active_task_ids,
        auto_wake_enabled=await get_auto_wake_enabled(session),
        use_proxy=await get_use_proxy(session),
        gpu_info=gpu_info,
        default_model=await get_default_model(session),
        default_variant=await get_default_variant(session),
        available_models=available_models(),
        available_variants=available_variants(),
        task_counts=task_counts,
    )


@app.post("/api/scan", status_code=200)
async def trigger_scan(session: AsyncSession = Depends(get_session)):
    """Manually trigger an artifact scan to discover/refresh tasks from disk."""
    _ensure_writable()
    created = await scan_and_create_tasks(session)
    return {"created": created}


@app.get("/api/agents")
async def list_agents():
    """List all detected AI agents currently running.

    Returns agents grouped by type: cursor_cli, opencode.
    Each agent includes PID, session ID (if detected), working directory, and active kernel.
    """
    return get_all_agent_sessions()


@app.get("/api/agents/running")
async def list_running_agents():
    """List all currently running agents with their details."""
    agents = detect_running_agents()
    return [
        {
            "agent_type": a.agent_type,
            "pid": a.pid,
            "session_id": a.session_id,
            "working_dir": a.working_dir,
            "kernel": a.kernel_path,
            "command": a.command[:300],
            "model_id": a.model_id,
            "model_display": a.model_display,
        }
        for a in agents
    ]
