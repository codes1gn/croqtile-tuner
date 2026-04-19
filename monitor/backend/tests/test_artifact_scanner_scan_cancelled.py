from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app import artifact_scanner
from app.models import Task


@pytest.mark.asyncio
async def test_scan_creates_cancelled_tasks_from_artifacts(tmp_path, monkeypatch, db_session):
    tuning_dir = tmp_path / "tuning"
    model_dir = (
        tuning_dir
        / "sm90_NVIDIA_H800_PCIe"
        / "croqtile"
        / "logs"
        / "matmul_fp16fp32_512x16384x16384"
        / "claude-4-5-opus-high"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "task_config.json").write_text(
        json.dumps(
            {
                "task_uid": "abc123def456",
                "status": "cancelled",
                "mode": "cursor_cli",
                "request_budget": 0,
            }
        )
    )
    (model_dir / "results.tsv").write_text(
        "iter\tkernel\ttflops\tdecision\tbottleneck\n"
        "0\titer000_cublas\t100.0\tBASELINE\tbaseline\n"
        "1\titer001_candidate\t110.0\tKEEP\tmemory\n"
    )

    async def _no_publish(*_args, **_kwargs):
        return None

    async def _no_side_effect(*_args, **_kwargs):
        return None

    monkeypatch.setattr(artifact_scanner.settings, "tuning_dir", tuning_dir)
    monkeypatch.setattr(artifact_scanner.event_bus, "publish", _no_publish)
    monkeypatch.setattr(artifact_scanner, "_sync_agent_status", _no_side_effect)
    monkeypatch.setattr(artifact_scanner, "_discover_cursor_sessions", _no_side_effect)

    created = await artifact_scanner.scan_and_create_tasks(db_session)
    assert created == 1

    result = await db_session.execute(select(Task))
    tasks = result.scalars().all()
    assert len(tasks) == 1
    assert tasks[0].status == "cancelled"
    assert tasks[0].task_uid == "abc123def456"
