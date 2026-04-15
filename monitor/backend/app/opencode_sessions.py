import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import settings


def _sync_iso_from_ms(epoch_ms: int | float | None) -> str | None:
    if epoch_ms is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()


def _compact_json(value: Any, limit: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=True) if not isinstance(value, str) else value
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def _tool_text(payload: dict[str, Any]) -> tuple[str, str | None, str | None]:
    state = payload.get("state") or {}
    status = state.get("status")
    tool = payload.get("tool")
    input_data = state.get("input") or {}
    output_data = state.get("output")
    title = state.get("title") or input_data.get("description") or input_data.get("filePath") or input_data.get("command")

    lines = []
    if title:
        lines.append(str(title))
    if tool:
        lines.append(f"tool={tool}")
    if status:
        lines.append(f"status={status}")
    if output_data:
        lines.append(f"output: {_compact_json(output_data, limit=400)}")
    return ("\n".join(lines), tool, status)


def _entry_from_row(row: sqlite3.Row) -> dict[str, Any] | None:
    part_data = json.loads(row["part_data"])
    message_data = json.loads(row["message_data"])
    part_type = part_data.get("type")
    role = message_data.get("role", "assistant")
    text: str | None = None
    tool: str | None = None
    status: str | None = None

    if part_type in ("text", "reasoning"):
        text = part_data.get("text")
    elif part_type == "tool":
        text, tool, status = _tool_text(part_data)
    elif part_type in ("step-start", "step-finish"):
        return None
    else:
        return None

    if not text:
        return None

    return {
        "id": row["part_id"],
        "message_id": row["message_id"],
        "role": role,
        "kind": part_type,
        "text": text,
        "tool": tool,
        "status": status,
        "timestamp": _sync_iso_from_ms(row["time_created"]),
    }


def _read_session_history_sync(session_id: str, limit: int, db_path: Path) -> dict[str, Any]:
    empty = {
        "session_id": session_id,
        "session_title": None,
        "session_directory": None,
        "entries": [],
    }
    if not db_path.exists():
        return empty

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return empty

    try:
        session_row = conn.execute(
            "select id, title, directory from session where id = ?",
            (session_id,),
        ).fetchone()

        rows = conn.execute(
            """
            select
                part.id as part_id,
                part.message_id as message_id,
                part.time_created as time_created,
                part.data as part_data,
                message.data as message_data
            from part
            join message on message.id = part.message_id
            where part.session_id = ?
            order by part.time_created desc
            limit ?
            """,
            (session_id, limit),
        ).fetchall()
    except sqlite3.Error:
        conn.close()
        return empty
    finally:
        conn.close()

    entries = []
    for row in reversed(rows):
        try:
            entry = _entry_from_row(row)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if entry is not None:
            entries.append(entry)

    return {
        "session_id": session_id,
        "session_title": session_row["title"] if session_row else None,
        "session_directory": session_row["directory"] if session_row else None,
        "entries": entries,
    }


async def read_session_history(session_id: str, limit: int = 200) -> dict[str, Any]:
    return await asyncio.to_thread(
        _read_session_history_sync,
        session_id,
        limit,
        Path(settings.opencode_db_path),
    )