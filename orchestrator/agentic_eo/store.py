"""In-memory pass state.

Phase 1: single-process, single-consumer per pass. Phase 2 likely needs a
fan-out (per-client queues) once eval harness + live frontend read the
same pass concurrently.
"""
from __future__ import annotations

import asyncio
from typing import Any, Coroutine
from uuid import uuid4

from pydantic import BaseModel

from .events import to_sse
from .schema import PassRequest, PassSummary


class PassState:
    def __init__(self, pass_id: str, request: PassRequest) -> None:
        self.pass_id = pass_id
        self.request = request
        self.queue: asyncio.Queue[BaseModel] = asyncio.Queue(maxsize=1024)
        self.closed = asyncio.Event()
        self.summary: PassSummary | None = None
        self._task: asyncio.Task[Any] | None = None
        # Tile imagery cache: tile_id -> (content_type, image_bytes). Lets the
        # frontend lazy-fetch /pass/{id}/tile/{tid}/image while events stream
        # in, instead of stuffing PNG bytes into the SSE payload.
        self.tile_images: dict[str, tuple[str, bytes]] = {}

    async def publish(self, event: BaseModel) -> None:
        await self.queue.put(event)

    async def close(self) -> None:
        self.closed.set()

    async def event_stream(self):
        """Async generator the SSE handler subscribes to."""
        while True:
            if self.closed.is_set() and self.queue.empty():
                return
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=15.0)
                yield to_sse(event)
            except asyncio.TimeoutError:
                if self.closed.is_set() and self.queue.empty():
                    return
                # keep-alive comment so intermediate proxies don't time out
                yield {"event": "ping", "data": "{}"}


class PassStore:
    def __init__(self) -> None:
        self._passes: dict[str, PassState] = {}

    def create(self, request: PassRequest) -> str:
        pass_id = uuid4().hex[:12]
        self._passes[pass_id] = PassState(pass_id, request)
        return pass_id

    def get(self, pass_id: str) -> PassState | None:
        return self._passes.get(pass_id)

    def spawn(self, pass_id: str, coro: Coroutine[Any, Any, Any]) -> None:
        state = self._passes[pass_id]
        state._task = asyncio.create_task(coro)


PASS_STORE = PassStore()
