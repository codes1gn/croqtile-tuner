import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from .config import settings

logger = logging.getLogger("croqtuner.sessions")


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


def _read_cursor_transcript_sync(session_id: str, limit: int) -> dict[str, Any] | None:
    """Read a cursor-agent transcript from its JSONL file.

    Returns None if the transcript doesn't exist (caller should fall back to opencode DB).
    """
    transcripts_dir = settings.cursor_transcripts_dir
    transcript_path = transcripts_dir / session_id / f"{session_id}.jsonl"
    if not transcript_path.exists():
        return None

    entries: list[dict[str, Any]] = []
    tool_count = 0
    try:
        with open(transcript_path, "r", errors="replace") as f:
            for line_num, raw in enumerate(f):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                role = obj.get("role", "assistant")
                msg = obj.get("message", {})
                content = msg.get("content", [])
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                if not isinstance(content, list):
                    continue

                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type", "")
                    if item_type == "text":
                        text = item.get("text", "").strip()
                        if text:
                            entries.append({
                                "id": f"cursor-{session_id[:8]}-{line_num}",
                                "message_id": f"msg-{line_num}",
                                "role": role,
                                "kind": "text",
                                "text": text[:4000],
                                "tool": None,
                                "status": None,
                                "timestamp": None,
                            })
                    elif item_type == "tool_use":
                        tool_name = item.get("name", "unknown")
                        tool_input = item.get("input", {})
                        text_parts: list[str] = []

                        if tool_name == "Shell":
                            cmd = tool_input.get("command", "")
                            desc = tool_input.get("description", "")
                            if desc:
                                text_parts.append(desc)
                            if cmd:
                                text_parts.append(f"$ {str(cmd)[:2000]}")
                        elif tool_name in ("Read", "Write", "Glob", "Grep"):
                            path = tool_input.get("path") or tool_input.get("glob_pattern") or tool_input.get("pattern") or ""
                            if path:
                                text_parts.append(f"{tool_name} {path}")
                            if tool_name == "Grep" and tool_input.get("pattern"):
                                text_parts.append(f"pattern: {tool_input['pattern']}")
                        elif tool_name == "StrReplace":
                            path = tool_input.get("path", "")
                            old = tool_input.get("old_string", "")[:200]
                            text_parts.append(f"Edit {path}")
                            if old:
                                text_parts.append(f"replace: {old}...")
                        elif tool_name == "TodoWrite":
                            todos = tool_input.get("todos", [])
                            for td in todos[:5]:
                                status = td.get("status", "?")
                                content = td.get("content", "")[:100]
                                text_parts.append(f"[{status}] {content}")
                        else:
                            desc = tool_input.get("description") or tool_input.get("command") or tool_input.get("path") or ""
                            if desc:
                                text_parts.append(str(desc)[:800])
                            elif tool_input:
                                text_parts.append(_compact_json(tool_input, limit=800))

                        if not text_parts:
                            text_parts.append(tool_name)

                        tool_count += 1
                        entries.append({
                            "id": f"cursor-{session_id[:8]}-{line_num}-t{tool_count}",
                            "message_id": f"msg-{line_num}",
                            "role": role,
                            "kind": "tool",
                            "text": "\n".join(text_parts),
                            "tool": tool_name,
                            "status": "completed",
                            "timestamp": None,
                        })
    except OSError:
        return None

    if limit and len(entries) > limit:
        entries = entries[-limit:]

    return {
        "session_id": session_id,
        "session_title": f"Cursor CLI session {session_id[:12]}",
        "session_directory": str(transcript_path.parent),
        "entries": entries,
    }


async def read_session_history(session_id: str, limit: int = 200) -> dict[str, Any]:
    # Try cursor transcript first (JSONL files)
    cursor_result = await asyncio.to_thread(
        _read_cursor_transcript_sync, session_id, limit
    )
    if cursor_result is not None:
        return cursor_result

    # Fall back to opencode SQLite DB
    return await asyncio.to_thread(
        _read_session_history_sync,
        session_id,
        limit,
        Path(settings.opencode_db_path),
    )