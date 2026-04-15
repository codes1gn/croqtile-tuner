from __future__ import annotations

import json

from .config import settings
from .models import Task


def apply_live_runtime(task: Task, payload: dict) -> dict:
    live_state = read_live_fsm_state(task)
    if not live_state:
        return payload

    payload["current_iteration"] = max(payload.get("current_iteration") or 0, live_state["iteration"])
    return payload


def read_live_fsm_state(task: Task) -> dict | None:
    state_file = settings.skills_dir / "fsm-engine" / "state" / (
        "loop-state_from_scratch.json" if task.mode == "from_scratch" else "loop-state.json"
    )
    if not state_file.exists():
        return None

    try:
        raw = json.loads(state_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    fsm = raw.get("fsm") or {}
    if fsm.get("shape_key") != task.shape_key:
        return None

    iteration = fsm.get("iteration")
    if not isinstance(iteration, int):
        return None

    return {
        "state": fsm.get("current_state"),
        "iteration": iteration,
    }