"""Bidirectional sync between DB Task fields and a task_config.json in the artifacts filesystem.

The config file lives at:
    tuning/<gpu>/<dsl>/logs/<shape_key>/<model>/task_config.json

It stores DB-only metadata that must survive DB resets/recreation:
    - model (AI model used)
    - sessions (list of session records)
    - agent_type
    - device
    - variant
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .models import Task, TaskSession

logger = logging.getLogger("croqtuner.artifact_config")

CONFIG_FILENAME = "task_config.json"


def _resolve_config_path(shape_key: str) -> Path | None:
    """Resolve the task_config.json path from a compound or bare shape_key."""
    parts = shape_key.split("/")
    tuning_dir = settings.tuning_dir

    if len(parts) == 4:
        return tuning_dir / parts[0] / parts[1] / "logs" / parts[2] / parts[3] / CONFIG_FILENAME
    if len(parts) == 3:
        logs_dir = tuning_dir / parts[0] / parts[1] / "logs" / parts[2]
        if logs_dir.is_dir():
            for model_dir in logs_dir.iterdir():
                if model_dir.is_dir():
                    return model_dir / CONFIG_FILENAME
    for gpu_dir in tuning_dir.iterdir():
        if not gpu_dir.is_dir():
            continue
        for dsl_dir in gpu_dir.iterdir():
            if not dsl_dir.is_dir():
                continue
            shape_dir = dsl_dir / "logs" / shape_key
            if shape_dir.is_dir():
                for model_dir in shape_dir.iterdir():
                    if model_dir.is_dir():
                        return model_dir / CONFIG_FILENAME
    return None


def read_artifact_config(shape_key: str) -> dict:
    """Read task_config.json from the artifacts filesystem. Returns {} if not found."""
    config_path = _resolve_config_path(shape_key)
    if config_path is None or not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read artifact config: %s", config_path)
        return {}


def write_artifact_config(shape_key: str, config: dict) -> bool:
    """Write task_config.json to the artifacts filesystem. Returns True on success."""
    config_path = _resolve_config_path(shape_key)
    if config_path is None:
        return False
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2, default=str) + "\n")
        return True
    except OSError:
        logger.warning("Failed to write artifact config: %s", config_path)
        return False


async def sync_task_to_artifact(session: AsyncSession, task: Task) -> bool:
    """Write current DB task state to artifact config file.

    Called after any DB update that changes model/session/agent metadata.
    """
    result = await session.execute(
        select(TaskSession).where(TaskSession.task_id == task.id).order_by(TaskSession.started_at)
    )
    db_sessions = result.scalars().all()

    config = read_artifact_config(task.shape_key)
    config["task_uid"] = task.task_uid
    config["model"] = task.model
    config["variant"] = task.variant or ""
    config["agent_type"] = task.agent_type
    config["device"] = task.device
    config["mode"] = task.mode
    config["status"] = task.status
    config["request_budget"] = task.request_budget
    config["request_number"] = task.request_number
    config["max_iterations"] = task.max_iterations
    config["respawn_count"] = task.respawn_count
    config["opencode_session_id"] = task.opencode_session_id
    config["sessions"] = [s.to_dict() for s in db_sessions]

    return write_artifact_config(task.shape_key, config)


async def sync_artifact_to_task(session: AsyncSession, task: Task) -> bool:
    """Read artifact config and update DB task + sessions if the artifact has data the DB lacks.

    Called during startup seed or scanner when a task exists in artifacts but DB is fresh.
    Returns True if any DB changes were made.
    """
    config = read_artifact_config(task.shape_key)
    if not config:
        return False

    changed = False

    if config.get("task_uid") and task.task_uid != config["task_uid"]:
        task.task_uid = config["task_uid"]
        changed = True
    if config.get("model") and task.model != config["model"]:
        task.model = config["model"]
        changed = True
    if config.get("variant") and not task.variant:
        task.variant = config["variant"]
        changed = True
    if config.get("agent_type") and not task.agent_type:
        task.agent_type = config["agent_type"]
        changed = True
    if config.get("device") and not task.device:
        task.device = config["device"]
        changed = True
    if config.get("mode") and task.mode != config["mode"]:
        task.mode = config["mode"]
        changed = True
    # Restore status/budget from artifact so tasks survive DB resets.
    # "running" is NOT restored -- the agent process died with the backend.
    # It reverts to "pending" so the scheduler can re-dispatch if budget > 0.
    cfg_status = config.get("status")
    if cfg_status and task.status == "pending" and cfg_status != "running":
        task.status = cfg_status
        changed = True
    if config.get("request_budget") is not None:
        task.request_budget = config["request_budget"]
        changed = True
    if config.get("request_number") is not None:
        task.request_number = config["request_number"]
        changed = True
    if config.get("max_iterations") is not None:
        task.max_iterations = config["max_iterations"]
        changed = True
    if config.get("respawn_count") is not None:
        task.respawn_count = config["respawn_count"]
        changed = True
    if config.get("opencode_session_id") and not task.opencode_session_id:
        task.opencode_session_id = config["opencode_session_id"]
        changed = True

    artifact_sessions = config.get("sessions", [])
    if artifact_sessions:
        existing_result = await session.execute(
            select(TaskSession.session_id).where(TaskSession.task_id == task.id)
        )
        existing_sids = {r[0] for r in existing_result.all()}

        for s in artifact_sessions:
            sid = s.get("session_id")
            if sid and sid not in existing_sids:
                session.add(TaskSession(
                    task_id=task.id,
                    session_id=sid,
                    agent_type=s.get("agent_type"),
                    model=s.get("model"),
                    request_number=s.get("request_number"),
                ))
                existing_sids.add(sid)
                changed = True

    return changed


async def register_session(
    session: AsyncSession,
    task: Task,
    session_id: str,
    agent_type: str | None = None,
    model: str | None = None,
    request_number: int | None = None,
) -> TaskSession | None:
    """Register a new session for a task. Deduplicates by session_id. Updates artifact config."""
    existing = await session.execute(
        select(TaskSession).where(
            TaskSession.task_id == task.id,
            TaskSession.session_id == session_id,
        )
    )
    if existing.scalar_one_or_none() is not None:
        return None

    ts = TaskSession(
        task_id=task.id,
        session_id=session_id,
        agent_type=agent_type or task.agent_type,
        model=model or task.model,
        request_number=request_number or task.request_number,
    )
    session.add(ts)

    if not task.opencode_session_id:
        task.opencode_session_id = session_id

    await session.flush()
    await sync_task_to_artifact(session, task)

    return ts
