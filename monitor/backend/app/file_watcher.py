"""File watcher that triggers artifact scans when tuning directory changes.

Uses watchdog to monitor the tuning directory tree.  When a results.tsv,
current_idea.json, or .co/.cu source file is created/modified/deleted, it
debounces the events and triggers a scan_and_create_tasks() call, which in
turn emits SSE events so the frontend updates in real-time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("croqtuner.file_watcher")

WATCHED_PATTERNS = {
    "results.tsv",
    "current_idea.json",
    "idea-log.jsonl",
    "attempt-log.jsonl",
    "activity.jsonl",
}
WATCHED_SUFFIXES = {".co", ".cu", ".json", ".tsv"}


class _TuningDirHandler(FileSystemEventHandler):
    """Debounces file events and queues a scan."""

    def __init__(self, loop: asyncio.AbstractEventLoop, debounce_sec: float = 2.0) -> None:
        self._loop = loop
        self._debounce_sec = debounce_sec
        self._timer: threading.Timer | None = None
        self._activity_timer: threading.Timer | None = None
        self._scan_queued = False

    def _is_relevant(self, path: str) -> bool:
        name = Path(path).name
        if name in WATCHED_PATTERNS:
            return True
        suffix = Path(path).suffix
        return suffix in WATCHED_SUFFIXES

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = getattr(event, "src_path", "")
        if not self._is_relevant(src):
            return

        if Path(src).name == "activity.jsonl":
            self._schedule_activity_notify(src)
        else:
            self._schedule_scan()

    def _schedule_scan(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce_sec, self._fire_scan)
        self._timer.daemon = True
        self._timer.start()

    def _fire_scan(self) -> None:
        self._timer = None
        asyncio.run_coroutine_threadsafe(_run_scan(), self._loop)

    def _schedule_activity_notify(self, path: str) -> None:
        if self._activity_timer is not None:
            self._activity_timer.cancel()
        self._activity_timer = threading.Timer(0.5, self._fire_activity_notify, args=[path])
        self._activity_timer.daemon = True
        self._activity_timer.start()

    def _fire_activity_notify(self, path: str) -> None:
        self._activity_timer = None
        asyncio.run_coroutine_threadsafe(_emit_activity_event(path), self._loop)


async def _emit_activity_event(path: str) -> None:
    """Emit an SSE event when activity.jsonl changes so UI refreshes in real-time."""
    from .events import event_bus

    try:
        p = Path(path)
        # path looks like: tuning/<gpu>/<dsl>/memory/<bare_key>[/<model>]/activity.jsonl
        parts = p.parts
        try:
            mem_idx = parts.index("memory")
        except ValueError:
            return
        await event_bus.publish("activity_log_update", {"path": str(p)})
    except Exception:
        logger.exception("Activity log event error")


async def _run_scan() -> None:
    from .artifact_scanner import prune_stale_tasks, scan_and_create_tasks
    from .database import async_session

    try:
        async with async_session() as session:
            await prune_stale_tasks(session)
            created = await scan_and_create_tasks(session)
            if created:
                logger.info("File watcher scan: created %d new task(s)", created)
    except Exception:
        logger.exception("File watcher scan error")


class TuningDirWatcher:
    """Manages a watchdog Observer on the tuning directory.
    
    Also runs an internal periodic scanner that triggers every PERIODIC_SCAN_INTERVAL_SEC
    to catch changes that watchdog might miss (e.g., git checkout, bulk cp operations).
    This internal scanner is always enabled and cannot be disabled by the user.
    """

    PERIODIC_SCAN_INTERVAL_SEC: float = 30.0  # Full scan every 30 seconds

    def __init__(self, tuning_dir: Path) -> None:
        self._tuning_dir = tuning_dir
        self._observer: Observer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._periodic_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        
        if not self._tuning_dir.exists():
            logger.info("File watcher: tuning dir %s does not exist, skipping watchdog", self._tuning_dir)
        else:
            handler = _TuningDirHandler(loop, debounce_sec=2.0)
            self._observer = Observer()
            self._observer.schedule(handler, str(self._tuning_dir), recursive=True)
            self._observer.daemon = True
            self._observer.start()
            logger.info("File watcher started on %s", self._tuning_dir)

        # Start the internal periodic scanner (always enabled)
        self._stop_event = asyncio.Event()
        self._periodic_task = loop.create_task(self._periodic_scan_loop())
        logger.info("Internal periodic scanner started (interval=%ds)", int(self.PERIODIC_SCAN_INTERVAL_SEC))

    async def _periodic_scan_loop(self) -> None:
        """Internal periodic scanner that runs independently of scheduler settings."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.PERIODIC_SCAN_INTERVAL_SEC)
                if self._stop_event.is_set():
                    break
                await _run_scan()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Periodic scan error")

    def stop(self) -> None:
        # Stop periodic scanner
        if self._stop_event is not None:
            self._stop_event.set()
        if self._periodic_task is not None:
            self._periodic_task.cancel()
            self._periodic_task = None
            logger.info("Internal periodic scanner stopped")
        
        # Stop watchdog observer
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("File watcher stopped")
