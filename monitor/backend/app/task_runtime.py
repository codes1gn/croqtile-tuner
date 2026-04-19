from __future__ import annotations

import json

from .models import Task


def apply_live_runtime(task: Task, payload: dict) -> dict:
    live_state = read_live_fsm_state(task)
    if not live_state:
        return payload

    payload["current_iteration"] = max(payload.get("current_iteration") or 0, live_state["iteration"])
    return payload


def read_live_fsm_state(task: Task) -> dict | None:
    """Read live iteration state from croq-tune checkpoint file."""
    from .agent import _find_artifacts
    from .artifact_scanner import parse_checkpoint_iteration

    checkpoint_path, _ = _find_artifacts(task.shape_key)

    if checkpoint_path is None or not checkpoint_path.exists():
        return None

    try:
        cp = json.loads(checkpoint_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    iteration = parse_checkpoint_iteration(cp)
    if iteration is None:
        return None

    return {
        "state": "running",
        "iteration": iteration,
    }