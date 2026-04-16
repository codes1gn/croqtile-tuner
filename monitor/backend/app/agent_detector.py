"""Detect and monitor different AI agent types."""

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import settings

AgentType = Literal["cursor_ide", "cursor_cli", "opencode", "copilot_ide", "unknown"]


@dataclass
class DetectedAgent:
    """Information about a detected agent process."""
    agent_type: AgentType
    pid: int
    command: str
    working_dir: str | None
    session_id: str | None
    kernel_path: str | None


# Patterns to identify different agents from their command line
AGENT_PATTERNS = {
    "cursor_ide": [
        re.compile(r"cursor.*--type=extensionHost"),
        re.compile(r"Cursor Helper"),
    ],
    "cursor_cli": [
        re.compile(r"cursor-agent"),
        re.compile(r"cursor.*run.*--print-logs"),
    ],
    "opencode": [
        re.compile(r"opencode\s+run"),
        re.compile(r"opencode.*--print-logs"),
    ],
    "copilot_ide": [
        re.compile(r"copilot-agent"),
        re.compile(r"github\.copilot"),
        re.compile(r"vscode.*copilot"),
    ],
}

# Session ID patterns
SESSION_PATTERNS = [
    re.compile(r"sessionID=([A-Za-z0-9_-]+)"),
    re.compile(r"/session/([A-Za-z0-9_-]+)/"),
    re.compile(r"session[_-]id[=:]([A-Za-z0-9_-]+)", re.IGNORECASE),
]


def _identify_agent_type(cmdline: str) -> AgentType:
    """Identify agent type from command line."""
    for agent_type, patterns in AGENT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(cmdline):
                return agent_type
    return "unknown"


def _extract_session_id(cmdline: str) -> str | None:
    """Extract session ID from command line or environment."""
    for pattern in SESSION_PATTERNS:
        match = pattern.search(cmdline)
        if match:
            return match.group(1)
    return None


def _extract_working_dir(pid: int) -> str | None:
    """Get working directory of a process."""
    try:
        cwd_link = f"/proc/{pid}/cwd"
        if os.path.islink(cwd_link):
            return os.readlink(cwd_link)
    except (OSError, PermissionError):
        pass
    return None


def _find_active_kernel(working_dir: str | None) -> str | None:
    """Find the most recently modified kernel checkpoint in the tuning tree."""
    tuning_dir = settings.tuning_dir
    if not tuning_dir.exists():
        return None
    
    checkpoints = list(tuning_dir.glob("**/checkpoints/*.json"))
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0].stem


def _is_own_user_process(pid: int) -> bool:
    """Check if a process belongs to the current user."""
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


def detect_running_agents() -> list[DetectedAgent]:
    """Detect all running AI agents on the system."""
    agents = []
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

            # IDE agents (cursor_ide, copilot_ide) serve multiple workspaces
            # from a shared extensionHost, so we can't filter by working dir.
            # For CLI/opencode agents, check the command line for project path.
            if agent_type not in ("cursor_ide", "copilot_ide"):
                if project_root not in cmdline and "tuning" not in cmdline.lower():
                    continue
            
            session_id = _extract_session_id(cmdline)
            kernel_path = _find_active_kernel(working_dir or project_root)
            
            agents.append(DetectedAgent(
                agent_type=agent_type,
                pid=pid,
                command=cmdline[:500],
                working_dir=working_dir,
                session_id=session_id,
                kernel_path=kernel_path,
            ))
    
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        pass
    
    return agents


def detect_agent_for_task(shape_key: str) -> DetectedAgent | None:
    """Find an agent working on a specific task."""
    agents = detect_running_agents()
    
    for agent in agents:
        if agent.kernel_path and shape_key in agent.kernel_path:
            return agent
    
    return None


def get_all_agent_sessions() -> dict[AgentType, list[dict]]:
    """Get all agent sessions grouped by type."""
    agents = detect_running_agents()
    
    grouped: dict[AgentType, list[dict]] = {
        "cursor_ide": [],
        "cursor_cli": [],
        "opencode": [],
        "copilot_ide": [],
    }
    
    for agent in agents:
        grouped[agent.agent_type].append({
            "pid": agent.pid,
            "session_id": agent.session_id,
            "working_dir": agent.working_dir,
            "kernel": agent.kernel_path,
            "command": agent.command[:200],
        })
    
    return grouped
