"""SSE event bus for broadcasting task updates to connected clients."""

import asyncio
import json
from typing import AsyncGenerator


class EventBus:
    """Simple pub/sub event bus using asyncio.Queue per subscriber."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[dict]] = []

    async def publish(self, event_type: str, data: dict) -> None:
        payload = {"type": event_type, "data": data}
        dead: list[asyncio.Queue[dict]] = []
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

    async def subscribe(self) -> AsyncGenerator[str, None]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=256)
        self._subscribers.append(q)
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15)
                    yield json.dumps(payload)
                except asyncio.TimeoutError:
                    yield json.dumps({"type": "ping", "data": {}})
        finally:
            if q in self._subscribers:
                self._subscribers.remove(q)


event_bus = EventBus()
