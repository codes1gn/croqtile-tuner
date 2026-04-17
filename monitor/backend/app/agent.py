"""opencode invocation and output monitoring."""

import asyncio
import json
import logging
import os
import re
import signal
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
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
    re.compile(r'"sessionID"\s*:\s*"([A-Za-z0-9_]+)"'),
    re.compile(r'"session_id"\s*:\s*"([A-Za-z0-9_]+)"'),
    re.compile(r"/session/([A-Za-z0-9_]+)/message"),
    re.compile(r"service=session\s+id=(ses_[A-Za-z0-9_]+)"),
    re.compile(r"session=(ses_[A-Za-z0-9_]+)"),
)
FATAL_ERROR_PATTERN = re.compile(
    r"(ProviderModelNotFoundError|Model not found:|EACCES:|FreeUsageLimitError|Rate limit exceeded|(?:^|\s)fatal error:)",
    re.IGNORECASE,
)
# "stream error" matches too aggressively (hits benign title-agent failures),
# so only flag it when it appears outside of a title/small-model error context.
_STREAM_ERROR_PATTERN = re.compile(r"stream error", re.IGNORECASE)
_TITLE_AGENT_PATTERN = re.compile(r"agent=title|small=true|modelID=gpt-5-mini")


logger = logging.getLogger("croqtuner.agent")


@dataclass
class RunState:
    fatal_error_message: str | None = None
    launch_epoch_ms: int = 0


def _build_env() -> dict[str, str]:
    """Build subprocess environment with CUDA on PATH."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": settings.cuda_visible_devices}
    cuda_home = env.get("CUDA_HOME", "/usr/local/cuda")
    cuda_bin = f"{cuda_home}/bin"
    path = env.get("PATH", "")
    if cuda_bin not in path:
        env["PATH"] = f"{cuda_bin}:{path}"
    if "CUDA_HOME" not in env:
        env["CUDA_HOME"] = cuda_home
    return env


def _bare_shape_key(compound_key: str) -> str:
    """Extract bare shape key from compound key: 'sm86_.../croqtile/matmul_...' → 'matmul_...'."""
    return compound_key.split("/")[-1] if "/" in compound_key else compound_key


def _sanitize_model_id(model: str) -> str:
    """Convert provider/model format to path-safe ID: 'github-copilot/gpt-5-mini' -> 'gpt-5-mini'."""
    return model.rsplit("/", 1)[-1] if "/" in model else model


def build_prompt(task: Task) -> str:
    dsl = task.dsl or "cuda"
    shape_key = _bare_shape_key(task.shape_key)
    model = _sanitize_model_id(task.model or "default")
    return (
        f"read .claude/skills/croq-tune/SKILL.md then kick off the tuning experiment, "
        f"tune {dsl} {task.dtype} {shape_key} --model {model}. "
        f"IMPORTANT: do NOT ask the user any questions — make all decisions autonomously."
    )


def build_command(task: Task) -> list[str]:
    prompt = build_prompt(task)
    project_dir = str(settings.project_dir.resolve())
    command = [settings.opencode_bin, "run", "--print-logs", "--format", "json"]
    model = task.model or settings.opencode_model
    if model:
        command.extend(["--model", model])
    variant = getattr(task, "variant", None) or ""
    if variant:
        command.extend(["--variant", variant])
    command.extend([prompt, project_dir])
    return command


def is_fatal_agent_error(line: str) -> bool:
    if FATAL_ERROR_PATTERN.search(line) is not None:
        return True
    if _STREAM_ERROR_PATTERN.search(line) and not _TITLE_AGENT_PATTERN.search(line):
        return True
    return False


def extract_session_id(line: str) -> str | None:
    for pattern in SESSION_ID_PATTERNS:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None



def _detect_session_from_db(after_epoch_ms: int) -> str | None:
    """Query opencode's SQLite DB for the newest session created after `after_epoch_ms`.

    This is the reliable fallback when the session ID doesn't appear in stdout/stderr.
    """
    db_path = settings.opencode_db_path
    if not db_path.exists():
        return None
    project_dir = str(settings.project_dir.resolve())
    try:
        conn = sqlite3.connect(str(db_path), timeout=2)
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM session "
            "WHERE directory = ? AND time_created > ? "
            "ORDER BY time_created DESC LIMIT 1",
            (project_dir, after_epoch_ms),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0]
    except (sqlite3.Error, OSError):
        pass
    return None


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


async def _tail_file(
    path: Path,
    task_id: int,
    level: str,
    session_factory,
    run_state: RunState,
    stop_event: asyncio.Event,
) -> None:
    """Tail a log file line-by-line, processing each line through the parser."""
    pos = 0
    buf = ""
    while not stop_event.is_set():
        try:
            raw = path.read_bytes()
        except OSError:
            await asyncio.sleep(0.5)
            continue

        if len(raw) > pos:
            new_data = raw[pos:].decode("utf-8", errors="replace")
            pos = len(raw)
            buf += new_data

            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.rstrip()
                if not line:
                    continue
                await _process_line(line, task_id, level, session_factory, run_state)
        else:
            await asyncio.sleep(0.5)

    try:
        remaining = path.read_bytes()[pos:].decode("utf-8", errors="replace")
        buf += remaining
        for line in buf.split("\n"):
            line = line.rstrip()
            if line:
                await _process_line(line, task_id, level, session_factory, run_state)
    except OSError:
        pass


async def _process_line(
    line: str,
    task_id: int,
    level: str,
    session_factory,
    run_state: RunState,
) -> None:
    """Process a single output line: store as AgentLog and parse iterations."""
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
        rn = None
        if m:
            iteration = int(m.group(1))
            tflops = float(m.group(2))
            decision = m.group(3).upper()

            t = await session.get(Task, task_id)
            if t:
                rn = t.request_number

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
                    request_number=rn,
                    tflops=tflops,
                    decision=decision,
                )
                session.add(iter_log)

            if task is None:
                task = t or await session.get(Task, task_id)
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
                "request_number": rn,
            })

        if task_updated and task is not None:
            await event_bus.publish("task_update", task.to_dict())

    await event_bus.publish("agent_log", {
        "task_id": task_id,
        "level": level,
        "message": line,
    })


async def run_task(task: Task, session_factory) -> int:
    """Launch opencode subprocess, monitor output, return exit code.

    Uses file-based I/O instead of pipes because opencode's default
    formatter hangs when stdout is connected to a pipe.
    """
    run_state = RunState(launch_epoch_ms=int(time.time() * 1000))
    if settings.mock_mode:
        cmd = ["python3", str(Path(__file__).parent.parent.parent / "scripts" / "mock_opencode.py"),
               task.shape_key, str(task.max_iterations)]
    else:
        cmd = build_command(task)

    stdout_fd, stdout_name = tempfile.mkstemp(
        prefix=f"croqtune_stdout_{task.id}_", suffix=".log"
    )
    stderr_fd, stderr_name = tempfile.mkstemp(
        prefix=f"croqtune_stderr_{task.id}_", suffix=".log"
    )
    stdout_path = Path(stdout_name)
    stderr_path = Path(stderr_name)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(settings.project_dir.resolve()),
            start_new_session=True,
            env=_build_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=stdout_fd,
            stderr=stderr_fd,
        )
        os.close(stdout_fd)
        os.close(stderr_fd)

        stop_event = asyncio.Event()

        stdout_task = asyncio.create_task(
            _tail_file(stdout_path, task.id, "info", session_factory, run_state, stop_event)
        )
        stderr_task = asyncio.create_task(
            _tail_file(stderr_path, task.id, "error", session_factory, run_state, stop_event)
        )
        poll_task = asyncio.create_task(_poll_loop(task, session_factory, proc, run_state))

        try:
            while True:
                try:
                    exit_code = await asyncio.wait_for(proc.wait(), timeout=10)
                    break
                except asyncio.TimeoutError:
                    if run_state.fatal_error_message:
                        logger.warning(
                            "Fatal error detected for task %d, terminating: %s",
                            task.id, run_state.fatal_error_message[:120],
                        )
                        await _terminate_process(proc)
                        exit_code = -2
                        break
            stop_event.set()
            await asyncio.gather(stdout_task, stderr_task)
        except asyncio.CancelledError:
            await _terminate_process(proc)
            stop_event.set()
            stdout_task.cancel()
            stderr_task.cancel()
            raise
        finally:
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass
    finally:
        try:
            stdout_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            stderr_path.unlink(missing_ok=True)
        except OSError:
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


async def _poll_loop(
    task: Task,
    session_factory,
    proc: asyncio.subprocess.Process,
    run_state: RunState | None = None,
) -> None:
    """Periodically poll filesystem artifacts and detect session ID while process is alive."""
    session_linked = False
    try:
        while proc.returncode is None:
            await asyncio.sleep(settings.heartbeat_sec)
            async with session_factory() as session:
                db_task = await session.get(Task, task.id)
                if db_task and db_task.status == "cancelled":
                    await _terminate_process(proc)
                    return

                if not session_linked and db_task and not db_task.opencode_session_id:
                    epoch_ms = run_state.launch_epoch_ms if run_state else 0
                    sid = _detect_session_from_db(epoch_ms)
                    if sid:
                        db_task.opencode_session_id = sid
                        db_task.updated_at = datetime.now(timezone.utc)
                        await session.commit()
                        await event_bus.publish("task_update", db_task.to_dict())
                        session_linked = True
                        logger.info("Linked session %s to task %d via DB", sid, task.id)
                elif db_task and db_task.opencode_session_id:
                    session_linked = True

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
