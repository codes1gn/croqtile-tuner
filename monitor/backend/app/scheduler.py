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
from .artifact_scanner import prune_stale_tasks, scan_and_create_tasks
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
        from .agent_detector import detect_running_agents

        live_pids = _find_live_opencode_pids()
        # Also check for cursor-agent or other agent types actively targeting any task
        live_agents = detect_running_agents()
        live_agent_kernel_paths = {ag.kernel_path for ag in live_agents if ag.kernel_path}

        async with async_session() as session:
            result = await session.execute(
                select(Task).where(Task.status == "running")
            )
            running_tasks = result.scalars().all()

            adopted_task: Task | None = None
            for task in running_tasks:
                # Check if any live agent (opencode or cursor-agent) is targeting this task
                parts = task.shape_key.split("/")
                task_kernel = parts[-2] if len(parts) >= 2 else task.shape_key
                agent_is_live = (
                    any(kp in task.shape_key for kp in live_agent_kernel_paths)
                    or task_kernel in live_agent_kernel_paths
                )

                if (live_pids or agent_is_live) and adopted_task is None:
                    adopted_task = task
                    logger.info(
                        "Adopting live task %d (%s) — agent still running (opencode PIDs=%s, agent_kernel_paths=%s)",
                        task.id, task.shape_key, live_pids, live_agent_kernel_paths,
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
        """Periodically scan tuning dir for runs started outside the UI,
        and prune tasks whose disk artifacts have been deleted."""
        try:
            async with async_session() as session:
                pruned = await prune_stale_tasks(session)
                if pruned:
                    logger.info("Periodic cleanup: pruned %d stale task(s)", pruned)
                    for task_id in pruned:
                        await event_bus.publish("task_deleted", {"id": task_id})
                created = await scan_and_create_tasks(session)
                if created:
                    logger.info("Artifact scan: created %d new task(s) from disk", created)
        except Exception:
            logger.exception("Artifact scan error")

    async def _try_dispatch(self) -> None:
        async with async_session() as session:
            auto_wake = await get_auto_wake_enabled(session)

            result = await session.execute(
                select(Task)
                .where(Task.status == "pending")
                .order_by(Task.created_at.asc())
            )
            candidates = list(result.scalars().all())

            if not candidates:
                await self._auto_sweep(session)
                result = await session.execute(
                    select(Task)
                    .where(Task.status == "pending")
                    .order_by(Task.created_at.asc())
                )
                candidates = list(result.scalars().all())

            if not candidates:
                return

            if not auto_wake:
                logger.debug("Auto-wake disabled, skipping task dispatch")
                return

            task = None
            for candidate in candidates:
                budget = candidate.request_budget if candidate.request_budget is not None else 1
                if budget > 0:
                    task = candidate
                    break
                logger.debug(
                    "Task %d (%s) has no budget remaining, skipping",
                    candidate.id, candidate.shape_key,
                )

            if task is None:
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
            logger.info(
                "Dispatching task %d (%s) model=%s variant=%s budget_remaining=%d",
                task.id, task.shape_key, task.model, task.variant, task.request_budget,
            )
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
        async with async_session() as session:
            db_task = await session.get(Task, task.id)
            if db_task:
                budget = db_task.request_budget if db_task.request_budget is not None else 1
                db_task.request_budget = max(budget - 1, 0)
                db_task.request_number = (db_task.request_number or 0) + 1
                await session.commit()
                logger.info(
                    "Request #%d for task %d, budget_remaining=%d",
                    db_task.request_number, task.id, db_task.request_budget,
                )
                task.request_number = db_task.request_number

        import time as _time
        t0 = _time.monotonic()
        iter_before = task.current_iteration

        try:
            exit_code = await run_task(task, async_session)
        except asyncio.CancelledError:
            logger.info("Worker for task %d cancelled", task.id)
            return
        except Exception:
            logger.exception("Worker for task %d crashed", task.id)
            exit_code = -1

        elapsed = _time.monotonic() - t0
        await self._finalize_task(
            task.id, exit_code, source="run_worker",
            elapsed_s=elapsed, iter_before=iter_before,
        )

    _FAST_EXIT_THRESHOLD_S = 90
    _FAST_EXIT_MAX_CONSECUTIVE = 3

    async def _finalize_task(
        self,
        task_id: int,
        exit_code: int | None,
        source: str = "",
        elapsed_s: float | None = None,
        iter_before: int | None = None,
    ) -> None:
        """Common finalization for both _run_worker and _adopt_worker.

        Status transitions:
          - max iterations reached → completed
          - cancelled by user → stays cancelled
          - fast-exit with no progress (3 consecutive) → waiting
          - otherwise → pending (ready for re-dispatch if budget allows)
        """
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
            elif exit_code == -2:
                db_task.status = "waiting"
                db_task.error_message = (
                    f"Rate limited / fatal error at iter {db_task.current_iteration}"
                    f"/{db_task.max_iterations} [{source}]"
                )
                logger.warning(
                    "Task %d → waiting (rate limited) at iter %d/%d",
                    task_id, db_task.current_iteration, db_task.max_iterations,
                )
            else:
                is_fast_exit = (
                    elapsed_s is not None
                    and elapsed_s < self._FAST_EXIT_THRESHOLD_S
                    and iter_before is not None
                    and db_task.current_iteration == iter_before
                )
                count = (db_task.respawn_count or 0) + 1
                db_task.respawn_count = count

                if is_fast_exit and count >= self._FAST_EXIT_MAX_CONSECUTIVE:
                    db_task.status = "waiting"
                    db_task.error_message = (
                        f"Agent failed to start tuning ({count} consecutive fast exits "
                        f"in <{self._FAST_EXIT_THRESHOLD_S}s with no iteration progress). "
                        f"Model may not support skill triggers. [{source}]"
                    )
                    logger.warning(
                        "Task %d → waiting (fast-exit loop detected, %d consecutive in <%.0fs)",
                        task_id, count, self._FAST_EXIT_THRESHOLD_S,
                    )
                else:
                    db_task.status = "pending"
                    reason = f"exit={exit_code}" if exit_code else "normally"
                    extra = f" (fast exit {elapsed_s:.0f}s)" if is_fast_exit else ""
                    db_task.error_message = (
                        f"Agent ended ({reason}) at iter {db_task.current_iteration}"
                        f"/{db_task.max_iterations}{extra} [{source}]"
                    )
                    logger.info(
                        "Task %d → pending at iter %d/%d (respawn #%d%s)",
                        task_id, db_task.current_iteration, db_task.max_iterations,
                        count, extra,
                    )

            db_task.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await event_bus.publish("task_update", db_task.to_dict())

        self.active_task_id = None
        logger.info("Task %d finished [%s] exit_code=%s", task_id, source, exit_code)

    async def _adopt_worker(self, task: Task) -> None:
        """Poll artifacts for a task whose agent subprocess is still running
        from before the backend restarted."""
        from .agent_detector import detect_running_agents
        from .config import settings

        logger.info("Adopt-worker: polling artifacts for task %d (%s)", task.id, task.shape_key)
        parts = task.shape_key.split("/")
        task_kernel = parts[-2] if len(parts) >= 2 else task.shape_key

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

                # Liveness: check opencode AND cursor-agent
                live_opencode = _find_live_opencode_pids()
                if live_opencode:
                    continue

                live_agents = detect_running_agents()
                agent_live = any(
                    ag.kernel_path and (ag.kernel_path == task_kernel or ag.kernel_path in task.shape_key)
                    for ag in live_agents
                )
                if not agent_live:
                    logger.info("Adopt-worker: agent exited for task %d", task.id)
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
        request_number=task.request_number,
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
