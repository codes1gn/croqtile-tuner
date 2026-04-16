"""Heartbeat scheduler: picks pending tasks and dispatches them one at a time.

Auto-sweep: promotes one waiting task to pending when no pending/running tasks exist.
Respawn: any stopped/failed task with progress is automatically re-queued as pending.
Auto-wake: when enabled via web toggle, auto-starts opencode for pending tasks.
"""

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select, func

from .agent import poll_artifacts, run_task, terminate_stray_opencode_processes
from .artifact_scanner import scan_and_create_tasks
from .config import settings
from .database import async_session
from .events import event_bus
from .models import Task
from .runtime_settings import get_default_model, get_default_variant, get_auto_wake_enabled

logger = logging.getLogger("croqtuner.scheduler")


class Scheduler:
    def __init__(self) -> None:
        self.running = False
        self.active_task_id: int | None = None
        self._task: asyncio.Task | None = None
        self._worker: asyncio.Task | None = None
        self._scan_counter: int = 0

    async def start(self) -> None:
        self.running = True
        await self._recover_stale_tasks()
        self._task = asyncio.create_task(self._loop())
        logger.info("Scheduler started")

    async def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    async def _recover_stale_tasks(self) -> None:
        live_pids = _find_live_opencode_pids()

        async with async_session() as session:
            result = await session.execute(
                select(Task).where(Task.status == "running")
            )
            running_tasks = result.scalars().all()

            adopted_task: Task | None = None
            for task in running_tasks:
                if live_pids and adopted_task is None:
                    adopted_task = task
                    logger.info(
                        "Adopting live task %d (%s) — opencode still running (PIDs %s)",
                        task.id, task.shape_key, live_pids,
                    )
                else:
                    task.status = "pending"
                    task.updated_at = datetime.now(timezone.utc)
                    logger.info("Recovered stale task %d (%s) -> pending", task.id, task.shape_key)

            await session.commit()

        if adopted_task is not None:
            self.active_task_id = adopted_task.id
            task_snapshot = _snapshot(adopted_task)
            self._worker = asyncio.create_task(self._adopt_worker(task_snapshot))

    async def _loop(self) -> None:
        while self.running:
            try:
                self._scan_counter += 1
                if self._scan_counter % 6 == 0:
                    await self._scan_disk()

                if self._worker is None or self._worker.done():
                    self._worker = None
                    self.active_task_id = None
                    await self._try_dispatch()
            except Exception:
                logger.exception("Scheduler loop error")
            await asyncio.sleep(5)

    async def _scan_disk(self) -> None:
        """Periodically scan tuning dir for runs started outside the UI."""
        try:
            async with async_session() as session:
                created = await scan_and_create_tasks(session)
                if created:
                    logger.info("Artifact scan: created %d new task(s) from disk", created)
        except Exception:
            logger.exception("Artifact scan error")

    async def _try_dispatch(self) -> None:
        async with async_session() as session:
            # Check if auto-wake is enabled
            auto_wake = await get_auto_wake_enabled(session)
            
            result = await session.execute(
                select(Task)
                .where(Task.status == "pending")
                .order_by(Task.created_at.asc())
                .limit(1)
            )
            task = result.scalar_one_or_none()

            if task is None:
                await self._auto_sweep(session)
                result = await session.execute(
                    select(Task)
                    .where(Task.status == "pending")
                    .order_by(Task.created_at.asc())
                    .limit(1)
                )
                task = result.scalar_one_or_none()

            if task is None:
                return
            
            # If auto-wake is disabled, only poll artifacts (monitor mode)
            if not auto_wake:
                logger.debug("Auto-wake disabled, skipping task dispatch")
                return

            task.status = "running"
            if not task.model:
                task.model = await get_default_model(session)
            if not task.variant and task.variant != "":
                task.variant = await get_default_variant(session)
            task.started_at = datetime.now(timezone.utc)
            task.updated_at = datetime.now(timezone.utc)
            await session.commit()

            self.active_task_id = task.id
            logger.info("Dispatching task %d (%s) model=%s variant=%s", task.id, task.shape_key, task.model, task.variant)
            await event_bus.publish("task_update", task.to_dict())

            task_snapshot = _snapshot(task)

        self._worker = asyncio.create_task(self._run_worker(task_snapshot))

    async def _auto_sweep(self, session) -> None:
        """Promote one waiting task to pending when the queue is empty."""
        pending_count = (await session.execute(
            select(func.count(Task.id)).where(Task.status == "pending")
        )).scalar() or 0
        running_count = (await session.execute(
            select(func.count(Task.id)).where(Task.status == "running")
        )).scalar() or 0

        if pending_count > 0 or running_count > 0:
            return

        result = await session.execute(
            select(Task)
            .where(Task.status == "waiting")
            .order_by(Task.created_at.asc())
            .limit(1)
        )
        waiting_task = result.scalar_one_or_none()
        if waiting_task is None:
            return

        waiting_task.status = "pending"
        waiting_task.updated_at = datetime.now(timezone.utc)
        await session.commit()
        logger.info(
            "Auto-sweep: promoted waiting task %d (%s) to pending",
            waiting_task.id, waiting_task.shape_key,
        )
        await event_bus.publish("task_update", waiting_task.to_dict())

    async def _run_worker(self, task: Task) -> None:
        try:
            exit_code = await run_task(task, async_session)
        except asyncio.CancelledError:
            logger.info("Worker for task %d cancelled", task.id)
            return
        except Exception:
            logger.exception("Worker for task %d crashed", task.id)
            exit_code = -1

        await self._finalize_task(task.id, exit_code, source="run_worker")

    async def _finalize_task(self, task_id: int, exit_code: int | None, source: str = "") -> None:
        """Common finalization for both _run_worker and _adopt_worker.
        
        Always respawns: if the task has progress but isn't done, re-queue it.
        """
        should_autocontinue = False
        async with async_session() as session:
            db_task = await session.get(Task, task_id)
            if not db_task:
                self.active_task_id = None
                return

            if db_task.status == "cancelled":
                pass
            elif db_task.current_iteration >= db_task.max_iterations:
                db_task.status = "completed"
                db_task.completed_at = datetime.now(timezone.utc)
                logger.info("Task %d completed (%d/%d)", task_id, db_task.current_iteration, db_task.max_iterations)
            elif db_task.current_iteration > 0:
                db_task.status = "stopped"
                reason = f"exit={exit_code}" if exit_code else "normally"
                db_task.error_message = (
                    f"opencode ended ({reason}) at iter {db_task.current_iteration}"
                    f"/{db_task.max_iterations} [{source}] — will respawn"
                )
                should_autocontinue = True
                logger.info(
                    "Task %d stopped at iter %d/%d — will respawn",
                    task_id, db_task.current_iteration, db_task.max_iterations,
                )
            else:
                db_task.status = "failed"
                if not db_task.error_message:
                    db_task.error_message = f"opencode exited with code {exit_code} (no progress)"
                logger.warning("Task %d failed with no progress (exit %s)", task_id, exit_code)

            db_task.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await event_bus.publish("task_update", db_task.to_dict())

        self.active_task_id = None
        logger.info("Task %d finished [%s] exit_code=%s", task_id, source, exit_code)

        if should_autocontinue:
            await self._respawn(task_id)

    MAX_RESPAWNS = 5

    async def _respawn(self, task_id: int) -> None:
        """Re-queue a stopped task as pending with exponential backoff.

        Caps at MAX_RESPAWNS retries, then marks the task as failed.
        """
        async with async_session() as session:
            db_task = await session.get(Task, task_id)
            if not db_task or db_task.status != "stopped":
                return

            if db_task.current_iteration >= db_task.max_iterations:
                db_task.status = "completed"
                db_task.completed_at = datetime.now(timezone.utc)
                db_task.updated_at = datetime.now(timezone.utc)
                await session.commit()
                await event_bus.publish("task_update", db_task.to_dict())
                return

            count = db_task.respawn_count + 1
            if count > self.MAX_RESPAWNS:
                db_task.status = "failed"
                db_task.error_message = (
                    f"Max respawns exceeded ({self.MAX_RESPAWNS}) at iter "
                    f"{db_task.current_iteration}/{db_task.max_iterations}"
                )
                db_task.updated_at = datetime.now(timezone.utc)
                await session.commit()
                await event_bus.publish("task_update", db_task.to_dict())
                logger.warning("Task %d failed: max respawns exceeded", task_id)
                return

            delay = min(3 * (2 ** db_task.respawn_count), 120)
            db_task.respawn_count = count
            await session.commit()

        logger.info("Respawn: waiting %ds before re-queuing task %d (attempt %d/%d)",
                     delay, task_id, count, self.MAX_RESPAWNS)
        await asyncio.sleep(delay)

        async with async_session() as session:
            db_task = await session.get(Task, task_id)
            if not db_task or db_task.status != "stopped":
                return
            db_task.status = "pending"
            db_task.error_message = None
            db_task.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await event_bus.publish("task_update", db_task.to_dict())
            logger.info(
                "Respawn: re-queued task %d (%s) at iter %d/%d (attempt %d/%d)",
                db_task.id, db_task.shape_key,
                db_task.current_iteration, db_task.max_iterations,
                count, self.MAX_RESPAWNS,
            )

    async def _adopt_worker(self, task: Task) -> None:
        """Poll artifacts for a task whose opencode subprocess is still running
        from before the backend restarted."""
        from .config import settings

        logger.info("Adopt-worker: polling artifacts for task %d (%s)", task.id, task.shape_key)
        try:
            while True:
                await asyncio.sleep(settings.heartbeat_sec)

                async with async_session() as session:
                    db_task = await session.get(Task, task.id)
                    if db_task and db_task.status == "cancelled":
                        terminated = terminate_stray_opencode_processes()
                        logger.info("Adopt-worker: task %d cancelled, terminated %s", task.id, terminated)
                        self.active_task_id = None
                        return

                await poll_artifacts(task, async_session)

                live = _find_live_opencode_pids()
                if not live:
                    logger.info("Adopt-worker: opencode exited for task %d", task.id)
                    break
        except asyncio.CancelledError:
            logger.info("Adopt-worker for task %d cancelled", task.id)
            return

        await poll_artifacts(task, async_session)
        await self._finalize_task(task.id, exit_code=0, source="adopt_worker")


def _snapshot(task: Task) -> Task:
    """Create a detached copy of a Task for passing to worker coroutines."""
    return Task(
        id=task.id,
        shape_key=task.shape_key,
        dtype=task.dtype,
        m=task.m,
        n=task.n,
        k=task.k,
        dsl=task.dsl,
        mode=task.mode,
        max_iterations=task.max_iterations,
        status=task.status,
        current_iteration=task.current_iteration,
        best_tflops=task.best_tflops,
        baseline_tflops=task.baseline_tflops,
        best_kernel=task.best_kernel,
        model=task.model,
        variant=task.variant,
        opencode_session_id=task.opencode_session_id,
    )


def _find_live_opencode_pids() -> list[int]:
    import os
    import subprocess

    from .config import settings

    try:
        result = subprocess.run(
            ["pgrep", "-af", "opencode run --print-logs"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    pids: list[int] = []
    project_root = str(settings.project_dir.resolve())
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or project_root not in line:
            continue
        pid_text, _, cmdline = line.partition(" ")
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if "opencode run --print-logs" not in cmdline:
            continue
        try:
            os.kill(pid, 0)
            pids.append(pid)
        except ProcessLookupError:
            continue
    return pids


scheduler = Scheduler()
