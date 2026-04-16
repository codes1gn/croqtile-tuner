"""opencode invocation and output monitoring."""

import asyncio
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .events import event_bus
from .models import AgentLog, IterationLog, Task

ITER_PATTERN = re.compile(
    r"iter[_]?(\d+).*?(\d+(?:\.\d+)?)\s*TFLOPS.*?(KEEP|DISCARD)",
    re.IGNORECASE,
)
TFLOPS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*TFLOPS", re.IGNORECASE)
BASELINE_PATTERN = re.compile(r"[Bb]aseline.*?(\d+(?:\.\d+)?)\s*TFLOPS")
SESSION_ID_PATTERNS = (
    re.compile(r"sessionID=([A-Za-z0-9_]+)"),
    re.compile(r"/session/([A-Za-z0-9_]+)/message"),
)
FATAL_ERROR_PATTERN = re.compile(
    r"(ProviderModelNotFoundError|Model not found:|EACCES:|stream error|(?:^|\s)fatal error:)",
    re.IGNORECASE,
)


@dataclass
class RunState:
    fatal_error_message: str | None = None


def _bare_shape_key(compound_key: str) -> str:
    """Extract bare shape key from compound key: 'sm86_.../croqtile/matmul_...' → 'matmul_...'."""
    return compound_key.split("/")[-1] if "/" in compound_key else compound_key


def build_prompt(task: Task) -> str:
    dsl = task.dsl or "cuda"
    shape_key = _bare_shape_key(task.shape_key)
    model = task.model or "default"
    return (
        f"/croq-tune {dsl} {task.dtype} {shape_key} "
        f"--model {model}"
    )


def build_command(task: Task) -> list[str]:
    prompt = build_prompt(task)
    project_dir = str(settings.project_dir.resolve())
    command = [settings.opencode_bin, "run", "--print-logs"]
    model = task.model or settings.opencode_model
    if model:
        command.extend(["--model", model])
    variant = getattr(task, "variant", None) or ""
    if variant:
        command.extend(["--variant", variant])
    command.extend([prompt, project_dir])
    return command


def is_fatal_agent_error(line: str) -> bool:
    return FATAL_ERROR_PATTERN.search(line) is not None


def extract_session_id(line: str) -> str | None:
    for pattern in SESSION_ID_PATTERNS:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None


async def _iter_stream_lines(stream: asyncio.StreamReader):
    buffer = b""
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        buffer += chunk
        while True:
            newline_index = buffer.find(b"\n")
            if newline_index == -1:
                if len(buffer) > 65536:
                    yield buffer
                    buffer = b""
                break
            yield buffer[:newline_index]
            buffer = buffer[newline_index + 1 :]

    if buffer:
        yield buffer


async def _read_stream(
    stream: asyncio.StreamReader,
    task_id: int,
    level: str,
    session_factory,
    run_state: RunState,
) -> None:
    """Read subprocess output line-by-line, store as AgentLog, parse iterations."""
    async for raw in _iter_stream_lines(stream):
        line = raw.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue

        if run_state.fatal_error_message is None and is_fatal_agent_error(line):
            run_state.fatal_error_message = line

        async with session_factory() as session:
            log = AgentLog(task_id=task_id, level=level, message=line)
            session.add(log)
            task_updated = False
            task = None

            session_id = extract_session_id(line)
            if session_id:
                task = await session.get(Task, task_id)
                if task and task.opencode_session_id != session_id:
                    task.opencode_session_id = session_id
                    task.updated_at = datetime.now(timezone.utc)
                    task_updated = True

            m = ITER_PATTERN.search(line)
            if m:
                iteration = int(m.group(1))
                tflops = float(m.group(2))
                decision = m.group(3).upper()

                existing = await session.execute(
                    select(IterationLog).where(
                        IterationLog.task_id == task_id,
                        IterationLog.iteration == iteration,
                    )
                )
                if not existing.scalar_one_or_none():
                    iter_log = IterationLog(
                        task_id=task_id,
                        iteration=iteration,
                        tflops=tflops,
                        decision=decision,
                    )
                    session.add(iter_log)

                if task is None:
                    task = await session.get(Task, task_id)
                if task:
                    task.current_iteration = max(task.current_iteration, iteration)
                    if decision == "KEEP" and (task.best_tflops is None or tflops > task.best_tflops):
                        task.best_tflops = tflops
                    task.updated_at = datetime.now(timezone.utc)
                    task_updated = True

            await session.commit()

            if m:
                await event_bus.publish("iteration", {
                    "task_id": task_id,
                    "iteration": int(m.group(1)),
                    "tflops": float(m.group(2)),
                    "decision": m.group(3).upper(),
                })

            if task_updated and task is not None:
                await event_bus.publish("task_update", task.to_dict())

        await event_bus.publish("agent_log", {
            "task_id": task_id,
            "level": level,
            "message": line,
        })


def _find_artifacts(key: str) -> tuple[Path | None, Path | None]:
    """Locate checkpoint and results.tsv in the nested tuning tree.

    Accepts compound keys ({gpu}/{dsl}/{shape_key}/{model}) or bare keys.
    Layout: tuning/<gpu>/<dsl>/checkpoints/<shape_key>/<model>/current_idea.json
            tuning/<gpu>/<dsl>/logs/<shape_key>/<model>/results.tsv
    """
    tuning_dir = settings.tuning_dir.resolve()
    checkpoint_path: Path | None = None
    results_path: Path | None = None

    if not tuning_dir.exists():
        return checkpoint_path, results_path

    parts = key.split("/")
    if len(parts) == 4:
        gpu, dsl, shape, model = parts
        cp = tuning_dir / gpu / dsl / "checkpoints" / shape / model / "current_idea.json"
        if cp.exists():
            checkpoint_path = cp
        rp = tuning_dir / gpu / dsl / "logs" / shape / model / "results.tsv"
        if rp.exists():
            results_path = rp
        return checkpoint_path, results_path

    for gpu_dir in tuning_dir.iterdir():
        if not gpu_dir.is_dir():
            continue
        for dsl_dir in gpu_dir.iterdir():
            if not dsl_dir.is_dir():
                continue
            cp_dir = dsl_dir / "checkpoints" / key
            if cp_dir.is_dir():
                for model_dir in cp_dir.iterdir():
                    idea = model_dir / "current_idea.json"
                    if idea.exists() and checkpoint_path is None:
                        checkpoint_path = idea
                        break
            logs_dir = dsl_dir / "logs" / key
            if logs_dir.is_dir():
                for model_dir in logs_dir.iterdir():
                    rp = model_dir / "results.tsv"
                    if rp.exists() and results_path is None:
                        results_path = rp
                        break
            if checkpoint_path and results_path:
                return checkpoint_path, results_path

    return checkpoint_path, results_path


async def poll_artifacts(task: Task, session_factory) -> None:
    """Read checkpoint and results.tsv from filesystem to update task metrics.
    
    IMPORTANT: This function NEVER changes task.status. Status transitions are
    handled exclusively by the scheduler's _finalize_task. This prevents stale
    artifact files (e.g. state.json with status=done from a prior run) from
    prematurely completing a running task.
    """
    checkpoint_path, results_path = _find_artifacts(task.shape_key)

    update: dict = {}

    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            cp = json.loads(checkpoint_path.read_text())
            cp_iter = cp.get("iteration") or cp.get("current_iter")
            if cp_iter is not None:
                update["current_iteration"] = int(cp_iter)
            cp_best = cp.get("current_best_tflops") or cp.get("best_tflops")
            if cp_best not in (None, ""):
                update["best_tflops"] = float(cp_best)
            cp_baseline = cp.get("baseline_tflops")
            if cp_baseline not in (None, ""):
                update["baseline_tflops"] = float(cp_baseline)
            cp_best_kernel = cp.get("current_best_kernel") or cp.get("best_kernel")
            if cp_best_kernel:
                update["best_kernel"] = cp_best_kernel
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    if results_path is not None and results_path.exists():
        try:
            lines = results_path.read_text().strip().split("\n")
            data_lines = [l for l in lines if l and not l.startswith("#") and not l.startswith("iter\t")]
            if data_lines:
                last = data_lines[-1].split("\t")
                if len(last) >= 3:
                    try:
                        iter_num = int(last[0])
                        if iter_num > update.get("current_iteration", 0):
                            update["current_iteration"] = iter_num
                    except ValueError:
                        pass
        except OSError:
            pass

    if update:
        async with session_factory() as session:
            t = await session.get(Task, task.id)
            if t:
                changed = False
                for attr in ("current_iteration", "best_tflops", "baseline_tflops", "best_kernel"):
                    val = update.get(attr)
                    if val is None:
                        continue
                    old = getattr(t, attr, None)
                    if attr == "current_iteration":
                        val = max(val, old or 0)
                    if old != val:
                        setattr(t, attr, val)
                        changed = True
                if changed:
                    t.updated_at = datetime.now(timezone.utc)
                    await session.commit()
                    await event_bus.publish("task_update", t.to_dict())


async def run_task(task: Task, session_factory) -> int:
    """Launch opencode subprocess, monitor output, return exit code."""
    run_state = RunState()
    if settings.mock_mode:
        cmd = ["python3", str(Path(__file__).parent.parent.parent / "scripts" / "mock_opencode.py"),
               task.shape_key, str(task.max_iterations)]
    else:
        cmd = build_command(task)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(settings.project_dir.resolve()),
        start_new_session=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": settings.cuda_visible_devices},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout_task = asyncio.create_task(
        _read_stream(proc.stdout, task.id, "info", session_factory, run_state)
    )
    stderr_task = asyncio.create_task(
        _read_stream(proc.stderr, task.id, "error", session_factory, run_state)
    )

    poll_task = asyncio.create_task(_poll_loop(task, session_factory, proc))

    try:
        await asyncio.gather(stdout_task, stderr_task)
        exit_code = await proc.wait()
    except asyncio.CancelledError:
        await _terminate_process(proc)
        raise
    finally:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass

    await poll_artifacts(task, session_factory)
    if run_state.fatal_error_message:
        async with session_factory() as session:
            db_task = await session.get(Task, task.id)
            if db_task:
                db_task.error_message = run_state.fatal_error_message
                db_task.updated_at = datetime.now(timezone.utc)
                await session.commit()
                await event_bus.publish("task_update", db_task.to_dict())
        return exit_code if exit_code != 0 else 1
    return exit_code


async def _poll_loop(task: Task, session_factory, proc: asyncio.subprocess.Process) -> None:
    """Periodically poll filesystem artifacts while process is alive."""
    try:
        while proc.returncode is None:
            await asyncio.sleep(settings.heartbeat_sec)
            async with session_factory() as session:
                db_task = await session.get(Task, task.id)
                if db_task and db_task.status == "cancelled":
                    await _terminate_process(proc)
                    return
            await poll_artifacts(task, session_factory)
    except asyncio.CancelledError:
        return


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
        return
    except asyncio.TimeoutError:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    await proc.wait()


def terminate_stray_opencode_processes() -> list[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-af", "opencode run --print-logs"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    terminated: list[int] = []
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
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            terminated.append(pid)
        except ProcessLookupError:
            continue

    deadline = time.time() + 10
    alive = set(terminated)
    while alive and time.time() < deadline:
        remaining = set()
        for pid in alive:
            try:
                os.kill(pid, 0)
                remaining.add(pid)
            except ProcessLookupError:
                continue
        alive = remaining
        if alive:
            time.sleep(0.2)

    for pid in list(alive):
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            continue

    return terminated
