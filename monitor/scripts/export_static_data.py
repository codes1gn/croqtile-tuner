"""Export monitor.db to static JSON files for static hosting (Vercel/GH Pages)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "monitor.db"
OUT_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public" / "static-data"


def rows_to_dicts(cursor: sqlite3.Cursor) -> list[dict]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def export():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Tasks
    c.execute("SELECT * FROM tasks ORDER BY id")
    tasks = rows_to_dicts(c)
    # Parse session_ids from task_sessions table
    for task in tasks:
        c.execute("SELECT session_id FROM task_sessions WHERE task_id = ?", (task["id"],))
        task["session_ids"] = [r[0] for r in c.fetchall()]

    (OUT_DIR / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=None))

    # Health (synthetic from current state)
    task_counts: dict[str, int] = {}
    for task in tasks:
        s = task.get("status", "unknown")
        task_counts[s] = task_counts.get(s, 0) + 1

    health = {
        "status": "static",
        "scheduler_running": False,
        "read_only_mode": True,
        "active_task_id": None,
        "active_task_ids": [],
        "auto_wake_enabled": False,
        "use_proxy": False,
        "gpu_info": "H800 PCIe (exported snapshot)",
        "default_model": "",
        "default_variant": "",
        "available_models": [],
        "available_variants": [],
        "task_counts": task_counts,
    }
    (OUT_DIR / "health.json").write_text(json.dumps(health, ensure_ascii=False))

    # Iteration logs per task
    logs_dir = OUT_DIR / "tasks"
    logs_dir.mkdir(exist_ok=True)

    for task in tasks:
        tid = task["id"]
        task_dir = logs_dir / str(tid)
        task_dir.mkdir(exist_ok=True)

        c.execute("SELECT * FROM iteration_logs WHERE task_id = ? ORDER BY iteration", (tid,))
        (task_dir / "logs.json").write_text(json.dumps(rows_to_dicts(c), ensure_ascii=False))

        c.execute("SELECT * FROM task_sessions WHERE task_id = ? ORDER BY id", (tid,))
        (task_dir / "sessions.json").write_text(json.dumps(rows_to_dicts(c), ensure_ascii=False))

    conn.close()
    print(f"Exported static data to {OUT_DIR.relative_to(OUT_DIR.parent.parent)}")
    print(f"  tasks: {len(tasks)}")
    total_logs = sum(1 for _ in (OUT_DIR / "tasks").rglob("logs.json"))
    print(f"  task log files: {total_logs}")


if __name__ == "__main__":
    export()
