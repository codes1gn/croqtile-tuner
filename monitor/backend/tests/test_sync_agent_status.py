from __future__ import annotations

import pytest
from sqlalchemy import select

from app import artifact_scanner
from app.agent_detector import DetectedAgent
from app.models import Task
from app.scheduler import scheduler


def _task(shape: str, status: str) -> Task:
    return Task(
        shape_key=f"sm90_NVIDIA_H800_PCIe/croqtile/{shape}/claude-4-5-opus-high",
        op_type="matmul",
        dtype="fp16fp32",
        m=512,
        n=16384,
        k=16384,
        mode="cursor_cli",
        dsl="croqtile",
        max_iterations=30,
        status=status,
        current_iteration=5,
        request_budget=1,
    )


@pytest.mark.asyncio
async def test_sync_agent_status_respects_state_machine(monkeypatch, db_session):
    pending = _task("pending_shape", "pending")
    waiting = _task("waiting_shape", "waiting")
    completed = _task("completed_shape", "completed")
    cancelled = _task("cancelled_shape", "cancelled")
    running = _task("running_shape", "running")

    db_session.add_all([pending, waiting, completed, cancelled, running])
    await db_session.commit()

    agents = [
        DetectedAgent(
            agent_type="cursor_cli",
            pid=111,
            command="cursor-agent --print",
            working_dir=None,
            session_id=None,
            kernel_path="pending_shape",
            model_id="claude-4-5-opus-high",
        ),
        DetectedAgent(
            agent_type="cursor_cli",
            pid=222,
            command="cursor-agent --print",
            working_dir=None,
            session_id=None,
            kernel_path="waiting_shape",
            model_id="claude-4-5-opus-high",
        ),
        DetectedAgent(
            agent_type="cursor_cli",
            pid=333,
            command="cursor-agent --print",
            working_dir=None,
            session_id=None,
            kernel_path="completed_shape",
            model_id="claude-4-5-opus-high",
        ),
    ]

    async def _no_publish(*_args, **_kwargs):
        return None

    async def _no_sync_task_to_artifact(*_args, **_kwargs):
        return True

    monkeypatch.setattr(artifact_scanner, "detect_running_agents", lambda: agents)
    monkeypatch.setattr(artifact_scanner.event_bus, "publish", _no_publish)
    monkeypatch.setattr("app.artifact_config.sync_task_to_artifact", _no_sync_task_to_artifact)
    scheduler._active_workers.clear()

    await artifact_scanner._sync_agent_status(db_session)
    await db_session.commit()

    result = await db_session.execute(select(Task))
    by_shape = {t.shape_key.split("/")[-2]: t for t in result.scalars().all()}

    assert by_shape["pending_shape"].status == "running"
    assert by_shape["waiting_shape"].status == "waiting"
    assert by_shape["completed_shape"].status == "completed"
    assert by_shape["cancelled_shape"].status == "cancelled"
    assert by_shape["running_shape"].status == "waiting"
