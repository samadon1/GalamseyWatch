"""SSE event encoding helpers."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def to_sse(model: BaseModel | dict[str, Any]) -> dict[str, str]:
    """Format an event as the dict shape sse-starlette expects.

    Returns ``{"event": <type>, "data": <json>}`` so EventSource consumers
    can listen for typed events with ``addEventListener("vlm_done", ...)``.
    """
    payload = model.model_dump() if isinstance(model, BaseModel) else dict(model)
    name = payload.get("event", "message")
    return {"event": name, "data": json.dumps(payload)}
