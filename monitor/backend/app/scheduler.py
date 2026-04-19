"""Heartbeat scheduler: dispatches pending tasks in parallel.

Tuning agents use the GPU only in short bursts (compile, profile, bench)
via harness scripts that serialise GPU access.  The rest of the time is
spent thinking / coding / reading, so multiple agents can interleave.

Auto-sweep: promotes waiting tasks to pending when no pending/running tasks exist.
Respawn: any stopped/failed task with progress is automatically re-queued as pending.
Auto-wake: when enabled via web toggle, auto-starts agents for pending tasks.
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
        self._active_workers: dict[int, asyncio.Task] = {}
        self._task: asyncio.Task | None = None
        self._scan_counter: int = 0

    @property
    def active_task_id(self) -> int | None:
        """Backward-compatible: return the first active task ID (or None)."""
        return next(iter(self._active_workers), None)

    @property
    def active_task_ids(self) -> list[int]:
        return list(self._active_workers.keys())

    async def start(self) -> None:
        if not settings.scheduler_enabled:
            self.running = False
            logger.warning("Scheduler start skipped because CROQTUNER_SCHEDULER_ENABLED=false")
            return
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
        for tid, worker in list(self._active_workers.items()):
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        self._active_workers.clear()
        logger.info("Scheduler stopped")

    async def cancel_active_task(self, task_id: int) -> bool:
        """Cancel the running worker for task_id.

        Returns True if a worker was cancelled, False otherwise.
        """
        worker = self._active_workers.get(task_id)
        if worker is None or worker.done():
            self._active_workers.pop(task_id, None)
            return False
        logger.info("Cancelling worker for deleted task %d", task_id)
        worker.cancel()
        try:
            await asyncio.wait_for(worker, timeout=5)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        self._active_workers.pop(task_id, None)
        return True

    async def _recover_stale_tasks(self) -> None:
        from .agent_detector import detect_running_agents

        live_pids = _find_live_opencode_pids()
        live_agents = detect_running_agents()
        live_agent_kernel_paths = {ag.kernel_path for ag in live_agents if ag.kernel_path}

        async with async_session() as session:
            result = await session.execute(
                select(Task).where(Task.status == "running")
            )
            running_tasks = result.scalars().all()

            for task in running_tasks:
                parts = task.shape_key.split("/")
                task_kernel = parts[-2] if len(parts) >= 2 else task.shape_key
                agent_is_live = (
                    any(kp in task.shape_key for kp in live_agent_kernel_paths)
                    or task_kernel in live_agent_kernel_paths
                )

                if live_pids or agent_is_live:
                    logger.info(
                        "Adopting live task %d (%s) — agent still running",
                        task.id, task.shape_key,
                    )
                    snapshot = _snapshot(task)
                    worker = asyncio.create_task(self._adopt_worker(snapshot))
                    self._active_workers[task.id] = worker
                else:
                    task.status = "pending"
                    task.updated_at = datetime.now(timezone.utc)
                    logger.info("Recovered stale task %d (%s) -> pending", task.id, task.shape_key)

            await session.commit()

    async def _loop(self) -> None:
        while self.running:
            try:
                self._scan_counter += 1
                # Scan more frequently when tasks are running (hot scan)
                scan_interval = 2 if self._active_workers else 6
                if self._scan_counter % scan_interval == 0:
                    await self._scan_disk()

                # Clean up finished workers
                finished = [tid for tid, w in self._active_workers.items() if w.done()]
                for tid in finished:
                    self._active_workers.pop(tid, None)

                # Dispatch any pending tasks that aren't already running
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
        if not settings.scheduler_enabled:
            return
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

            to_dispatch: list[Task] = []
            exhausted: list[Task] = []
            for candidate in candidates:
                if candidate.id in self._active_workers:
                    continue
                budget = candidate.request_budget if candidate.request_budget is not None else 1
                if budget > 0:
                    to_dispatch.append(candidate)
                else:
                    exhausted.append(candidate)

            # Move budget-exhausted tasks to "waiting"
            for ex in exhausted:
                ex.status = "waiting"
                ex.error_message = (
                    "Request budget exhausted — use Resume or increase budget to continue"
                )
                ex.updated_at = datetime.now(timezone.utc)
                logger.info(
                    "Task %d (%s) has no budget remaining → waiting",
                    ex.id, ex.shape_key,
                )
                await event_bus.publish("task_update", ex.to_dict())
            if exhausted:
                await session.commit()
                from .artifact_config import sync_task_to_artifact
                for ex in exhausted:
                    await sync_task_to_artifact(session, ex)

            # Dispatch all eligible tasks in parallel
            for task in to_dispatch:
                task.status = "running"
                if not task.model:
                    task.model = await get_default_model(session)
                if not task.variant and task.variant != "":
                    task.variant = await get_default_variant(session)
                task.started_at = datetime.now(timezone.utc)
                task.updated_at = datetime.now(timezone.utc)
                await session.commit()

                logger.info(
                    "Dispatching task %d (uid=%s, %s) model=%s variant=%s budget_remaining=%d",
                    task.id, task.task_uid, task.shape_key, task.model, task.variant, task.request_budget,
                )
                await event_bus.publish("task_update", task.to_dict())

                from .artifact_config import sync_task_to_artifact
                await sync_task_to_artifact(session, task)

                snapshot = _snapshot(task)
                worker = asyncio.create_task(self._run_worker(snapshot))
                self._active_workers[task.id] = worker

    async def _auto_sweep(self, session) -> None:
        """Promote waiting tasks to pending when no pending tasks exist."""
        pending_count = (await session.execute(
            select(func.count(Task.id)).where(Task.status == "pending")
        )).scalar() or 0

        if pending_count > 0:
            return

        result = await session.execute(
            select(Task)
            .where(Task.status == "waiting")
            .order_by(Task.created_at.asc())
        )
        waiting_tasks = list(result.scalars().all())
        if not waiting_tasks:
            return

        for wt in waiting_tasks:
            wt.status = "pending"
            wt.updated_at = datetime.now(timezone.utc)
            logger.info(
                "Auto-sweep: promoted waiting task %d (%s) to pending",
                wt.id, wt.shape_key,
            )
            await event_bus.publish("task_update", wt.to_dict())
        await session.commit()

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

    # How many seconds a run must last before it's NOT considered a "fast exit".
    # Models with slow cold starts (tool resolution, workspace indexing) can take
    # 60-120s before emitting any output, so 90s is too aggressive; use 180s.
    # Override with env: FAST_EXIT_THRESHOLD_S=<int>
    _FAST_EXIT_THRESHOLD_S: int = int(__import__("os").environ.get("FAST_EXIT_THRESHOLD_S", "180"))
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
        # Scan disk to pick up artifact path changes the agent may have made
        try:
            async with async_session() as scan_session:
                await scan_and_create_tasks(scan_session)
        except Exception:
            logger.exception("Post-run scan failed for task %d", task_id)

        async with async_session() as session:
            db_task = await session.get(Task, task_id)
            if not db_task:
                self._active_workers.pop(task_id, None)
                return

            if db_task.status == "cancelled":
                pass
            # NOTE: Do NOT auto-complete based on max_iterations - tuning should continue indefinitely
            # until user explicitly stops. max_iterations is just a UI hint, not a hard limit.
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
                made_progress = (
                    iter_before is not None
                    and db_task.current_iteration > iter_before
                )
                if made_progress:
                    # Reset consecutive fast-exit counter when real progress was made
                    db_task.respawn_count = 0
                    count = 0
                else:
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
                    extra = ""
                    if is_fast_exit:
                        quality_hint = (
                            " — model may have produced unusable output; "
                            "check LiveLog or try a higher-capability model"
                            if not exit_code  # exit=0 with no progress usually means garbled/wrong output
                            else ""
                        )
                        extra = f" (fast exit {elapsed_s:.0f}s{quality_hint})"
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

            # Persist final status/budget to disk so it survives DB resets
            from .artifact_config import sync_task_to_artifact
            await sync_task_to_artifact(session, db_task)

        self._active_workers.pop(task_id, None)
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
                        self._active_workers.pop(task.id, None)
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
        task_uid=task.task_uid,
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
    """Return PIDs of live opencode and cursor-agent dispatched processes."""
    import os
    import subprocess

    from .config import settings

    project_root = str(settings.project_dir.resolve())
    pids: list[int] = []

    def _pgrep_collect(pattern: str, cmdline_must_contain: str) -> None:
        try:
            result = subprocess.run(
                ["pgrep", "-af", pattern],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or project_root not in line:
                continue
            pid_text, _, cmdline = line.partition(" ")
            if not pid_text.isdigit():
                continue
            pid = int(pid_text)
            if cmdline_must_contain not in cmdline:
                continue
            try:
                os.kill(pid, 0)
                if pid not in pids:
                    pids.append(pid)
            except ProcessLookupError:
                continue

    _pgrep_collect("opencode run --print-logs", "opencode run --print-logs")
    _pgrep_collect("cursor-agent --print", "cursor-agent")
    return pids


scheduler = Scheduler()
