from __future__ import annotations

import asyncio

import pytest

from app.agent import run_task
from app.models import Task


class _DummySession:
    async def commit(self):
        return None

    async def get(self, *_args, **_kwargs):
        return None


class _DummySessionFactory:
    def __call__(self):
        return self

    async def __aenter__(self):
        return _DummySession()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeProc:
    def __init__(self):
        self.returncode = None
        self.pid = 12345

    async def wait(self):
        self.returncode = 0
        return 0


def _make_task() -> Task:
    return Task(
        id=42,
        shape_key="sm90_NVIDIA_H800_PCIe/croqtile/matmul_fp16fp32_512x16384x16384/claude-4-5-opus-high",
        op_type="matmul",
        dtype="fp16fp32",
        m=512,
        n=16384,
        k=16384,
        mode="cursor_cli",
        dsl="croqtile",
        max_iterations=30,
        status="pending",
        current_iteration=0,
        request_budget=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_proxy,expected_prefix", [(True, "proxychains4"), (False, "cursor-agent")])
async def test_run_task_proxy_prefix(monkeypatch, use_proxy: bool, expected_prefix: str):
    captured: dict[str, tuple[str, ...]] = {}

    async def _fake_create_subprocess_exec(*cmd, **_kwargs):
        captured["cmd"] = tuple(cmd)
        return _FakeProc()

    async def _no_tail_file(*_args, **_kwargs):
        return None

    async def _no_poll_loop(*_args, **_kwargs):
        await asyncio.sleep(0)

    async def _no_poll_artifacts(*_args, **_kwargs):
        return None

    async def _fake_get_use_proxy(_session):
        return use_proxy

    monkeypatch.setattr("app.agent.settings.mock_mode", False)
    monkeypatch.setattr("app.agent.build_command", lambda _task: ["cursor-agent", "--print"])
    monkeypatch.setattr("app.runtime_settings.get_use_proxy", _fake_get_use_proxy)
    monkeypatch.setattr("app.agent.asyncio.create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr("app.agent._tail_file", _no_tail_file)
    monkeypatch.setattr("app.agent._poll_loop", _no_poll_loop)
    monkeypatch.setattr("app.agent.poll_artifacts", _no_poll_artifacts)

    exit_code = await run_task(_make_task(), _DummySessionFactory())
    assert exit_code == 0
    assert captured["cmd"][0] == expected_prefix
