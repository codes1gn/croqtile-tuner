"""CroqTuner Agent Bot — FastAPI backend."""

import asyncio
import logging
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from .agent_detector import detect_running_agents, get_all_agent_sessions
from .database import async_session, get_session, init_db
from .events import event_bus
from .iteration_history import read_iteration_history
from .models import AgentLog, IterationLog, Task
from .opencode_sessions import read_session_history
from .runtime_settings import (
    available_models,
    available_variants,
    get_default_model,
    get_default_variant,
    set_default_model,
    get_auto_wake_enabled,
    set_auto_wake_enabled,
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
    ResumeRequest,
    SessionHistoryResponse,
    TaskCreate,
    TaskResponse,
    TaskUpdate,
)
from .artifact_scanner import prune_stale_tasks, scan_and_create_tasks
from .config import invalidate_model_cache
from .state_seed import seed_tasks_from_state_if_empty
from .task_runtime import apply_live_runtime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("croqtuner")


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
    await scheduler.start()
    yield
    await scheduler.stop()


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
    max_iter = 30
    shape_key = f"{body.op_type}_{body.dtype}_{body.m}x{body.n}x{body.k}"

    task = Task(
        shape_key=shape_key,
        op_type=body.op_type,
        dtype=body.dtype,
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
        if scheduler.active_task_id == task.id:
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

    return TaskResponse(**task.to_dict())


@app.delete("/api/tasks/{task_id}", status_code=204)
async def delete_task(task_id: int, session: AsyncSession = Depends(get_session)):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status == "running":
        raise HTTPException(400, "Cannot delete a running task; cancel it first")
    if task.status not in ("pending", "waiting", "completed", "cancelled"):
        raise HTTPException(400, "Cannot delete this task in its current state")
    await session.delete(task)
    await session.commit()
    await event_bus.publish("task_deleted", {"id": task_id})


@app.post("/api/tasks/{task_id}/retry", response_model=TaskResponse, status_code=201)
async def retry_task(task_id: int, session: AsyncSession = Depends(get_session)):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status not in ("pending", "completed", "cancelled"):
        raise HTTPException(
            400, "Can only retry pending, completed, or cancelled tasks"
        )

    retry = Task(
        shape_key=task.shape_key,
        dtype=task.dtype,
        m=task.m,
        n=task.n,
        k=task.k,
        dsl=task.dsl,
        mode=task.mode,
        model=task.model,
        variant=task.variant,
        max_iterations=task.max_iterations,
        status="pending",
        current_iteration=0,
        baseline_tflops=task.baseline_tflops,
        best_kernel=task.best_kernel,
    )
    session.add(retry)
    await session.commit()
    await session.refresh(retry)

    await event_bus.publish("task_update", retry.to_dict())
    return TaskResponse(**retry.to_dict())


@app.post("/api/tasks/{task_id}/resume", response_model=TaskResponse, status_code=201)
async def resume_task(
    task_id: int,
    body: ResumeRequest,
    session: AsyncSession = Depends(get_session),
):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status in ("running", "pending"):
        raise HTTPException(400, f"Cannot resume a {task.status} task")

    from_iter = min(body.from_iteration, task.current_iteration, task.max_iterations)

    resumed = Task(
        shape_key=task.shape_key,
        dtype=task.dtype,
        m=task.m,
        n=task.n,
        k=task.k,
        dsl=task.dsl,
        mode=task.mode,
        model=task.model,
        variant=task.variant,
        max_iterations=task.max_iterations,
        status="pending",
        current_iteration=from_iter,
        best_tflops=task.best_tflops,
        baseline_tflops=task.baseline_tflops,
        best_kernel=task.best_kernel,
    )
    session.add(resumed)
    await session.commit()
    await session.refresh(resumed)

    await event_bus.publish("task_update", resumed.to_dict())
    return TaskResponse(**resumed.to_dict())


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
    session: AsyncSession = Depends(get_session),
):
    task = await session.get(Task, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if not task.opencode_session_id:
        return SessionHistoryResponse(
            session_id=None,
            session_title=None,
            session_directory=None,
            entries=[],
        )
    history = await read_session_history(task.opencode_session_id, limit=limit)
    return SessionHistoryResponse(**history)


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
    enabled = await set_auto_wake_enabled(session, body.auto_wake_enabled)
    await session.commit()
    logger.info("Auto-wake setting changed to: %s", enabled)
    await event_bus.publish("auto_wake_changed", {"enabled": enabled})
    return AutoWakeSettingsResponse(auto_wake_enabled=enabled)


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
        active_task_id=scheduler.active_task_id,
        auto_wake_enabled=await get_auto_wake_enabled(session),
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
        }
        for a in agents
    ]
