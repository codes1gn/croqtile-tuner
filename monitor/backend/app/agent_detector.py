"""Detect and monitor different AI agent types, including their active model."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .config import settings

logger = logging.getLogger("croqtuner.agent_detector")

AgentType = Literal["cursor_cli", "opencode", "unknown"]

_CURSOR_CLI_CONFIG = Path.home() / ".cursor" / "cli-config.json"


@dataclass
class DetectedAgent:
    """Information about a detected agent process."""
    agent_type: AgentType
    pid: int
    command: str
    working_dir: str | None
    session_id: str | None
    kernel_path: str | None
    model_id: str | None = None
    model_display: str | None = None
    extra: dict = field(default_factory=dict)


AGENT_PATTERNS: dict[AgentType, list[re.Pattern]] = {
    "cursor_cli": [
        re.compile(r"cursor-agent"),
        re.compile(r"cursor.*run.*--print-logs"),
    ],
    "opencode": [
        re.compile(r"opencode\s+run"),
        re.compile(r"opencode.*--print-logs"),
    ],
}

SESSION_PATTERNS = [
    re.compile(r"sessionID=([A-Za-z0-9_-]+)"),
    re.compile(r"/session/([A-Za-z0-9_-]+)/"),
    re.compile(r"session[_-]id[=:]([A-Za-z0-9_-]+)", re.IGNORECASE),
]


def _identify_agent_type(cmdline: str) -> AgentType:
    for agent_type, patterns in AGENT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(cmdline):
                return agent_type
    return "unknown"


def _extract_session_id(cmdline: str) -> str | None:
    for pattern in SESSION_PATTERNS:
        match = pattern.search(cmdline)
        if match:
            return match.group(1)
    return None


def _extract_working_dir(pid: int) -> str | None:
    try:
        cwd_link = f"/proc/{pid}/cwd"
        if os.path.islink(cwd_link):
            return os.readlink(cwd_link)
    except (OSError, PermissionError):
        pass
    return None


def _find_active_kernel(working_dir: str | None) -> str | None:
    tuning_dir = settings.tuning_dir
    if not tuning_dir.exists():
        return None

    checkpoints = list(tuning_dir.glob("**/checkpoints/**/current_idea.json"))
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0].parent.parent.name


def _is_own_user_process(pid: int) -> bool:
    try:
        stat_path = f"/proc/{pid}/status"
        with open(stat_path) as f:
            for line in f:
                if line.startswith("Uid:"):
                    real_uid = int(line.split()[1])
                    return real_uid == os.getuid()
    except (OSError, PermissionError, ValueError):
        pass
    return False


# ---------------------------------------------------------------------------
# Model detection — platform-specific
# ---------------------------------------------------------------------------

def detect_cursor_model() -> tuple[str | None, str | None]:
    """Read model from Cursor CLI config.

    Returns (model_id, display_name) e.g. ("claude-4.6-opus-max", "Opus 4.6 1M Max").
    The model_id is prefixed with "cursor/" for the monitor's naming convention.
    """
    try:
        data = json.loads(_CURSOR_CLI_CONFIG.read_text())
        model = data.get("model") or data.get("selectedModel") or {}
        model_id = model.get("modelId") or model.get("modelID")
        display = model.get("displayName") or model.get("displayNameShort")
        if model_id:
            qualified = f"cursor/{model_id}" if "/" not in model_id else model_id
            return qualified, display
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return None, None


def detect_opencode_model(session_id: str | None = None) -> tuple[str | None, str | None]:
    """Read model from the most recent opencode session message.

    If session_id is given, looks at that session; otherwise picks the latest.
    Returns (qualified_model_id, display) e.g. ("github-copilot/gpt-5-mini", "gpt-5-mini").
    """
    db_path = settings.opencode_db_path
    if not db_path.exists():
        return None, None
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        if session_id:
            cur.execute(
                "SELECT data FROM message WHERE session_id = ? ORDER BY time_created DESC LIMIT 10",
                (session_id,),
            )
        else:
            cur.execute(
                "SELECT data FROM message ORDER BY time_created DESC LIMIT 10",
            )
        for (raw,) in cur.fetchall():
            data = json.loads(raw)
            model_id = data.get("modelID")
            provider_id = data.get("providerID")
            if model_id:
                qualified = f"{provider_id}/{model_id}" if provider_id and "/" not in model_id else model_id
                conn.close()
                return qualified, model_id
            model_obj = data.get("model")
            if isinstance(model_obj, dict):
                mid = model_obj.get("modelID") or ""
                pid = model_obj.get("providerID", "")
                effective = mid or pid
                if effective:
                    if pid and mid and pid != mid and "/" not in mid:
                        qualified = f"{pid}/{mid}"
                    elif "/" in effective:
                        qualified = effective
                    else:
                        qualified = effective
                    conn.close()
                    return qualified, mid or pid
        conn.close()
    except (sqlite3.Error, json.JSONDecodeError, OSError):
        pass
    return None, None


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------

def detect_running_agents() -> list[DetectedAgent]:
    """Detect all running AI agents on the system, including their model."""
    agents: list[DetectedAgent] = []
    project_root = str(settings.project_dir.resolve())

    try:
        result = subprocess.run(
            ["ps", "auxww"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return agents

        lines = result.stdout.strip().split("\n")[1:]

        for line in lines:
            parts = line.split(None, 10)
            if len(parts) < 11:
                continue

            pid_str = parts[1]
            cmdline = parts[10] if len(parts) > 10 else ""

            try:
                pid = int(pid_str)
            except ValueError:
                continue

            agent_type = _identify_agent_type(cmdline)
            if agent_type == "unknown":
                continue

            if not _is_own_user_process(pid):
                continue

            working_dir = _extract_working_dir(pid)

            if project_root not in cmdline and project_root != working_dir:
                continue

            session_id = _extract_session_id(cmdline)
            kernel_path = _find_active_kernel(working_dir or project_root)

            model_id: str | None = None
            model_display: str | None = None
            if agent_type == "cursor_cli":
                model_id, model_display = detect_cursor_model()
            elif agent_type == "opencode":
                model_id, model_display = detect_opencode_model(session_id)

            agents.append(DetectedAgent(
                agent_type=agent_type,
                pid=pid,
                command=cmdline[:500],
                working_dir=working_dir,
                session_id=session_id,
                kernel_path=kernel_path,
                model_id=model_id,
                model_display=model_display,
            ))

    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        pass

    return agents


def detect_agent_for_task(shape_key: str) -> DetectedAgent | None:
    agents = detect_running_agents()
    for agent in agents:
        if agent.kernel_path and shape_key in agent.kernel_path:
            return agent
    return None


def get_all_agent_sessions() -> dict[AgentType, list[dict]]:
    agents = detect_running_agents()
    grouped: dict[AgentType, list[dict]] = {
        "cursor_cli": [],
        "opencode": [],
    }
    for agent in agents:
        grouped[agent.agent_type].append({
            "pid": agent.pid,
            "session_id": agent.session_id,
            "working_dir": agent.working_dir,
            "kernel": agent.kernel_path,
            "command": agent.command[:200],
            "model_id": agent.model_id,
            "model_display": agent.model_display,
        })
    return grouped
