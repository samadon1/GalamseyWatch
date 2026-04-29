"""Event schema for an orbital pass.

The same schema feeds the SSE stream (frontend visualization) and the eval
harness (offline analysis). One format, two consumers.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# --- Request ---------------------------------------------------------------

class AOI(BaseModel):
    name: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float


PassMode = Literal["threshold", "rules", "agent"]


class PassRequest(BaseModel):
    aoi: AOI
    tile_count: int = Field(default=20, ge=1, le=200)
    mode: PassMode = "threshold"
    bandwidth_kb: int = Field(default=512, description="Downlink budget for the pass in KB")
    timestamp: Optional[str] = None  # ISO; defaults to now


class PassStarted(BaseModel):
    pass_id: str


# --- Detection / action types ---------------------------------------------

class BoundingBox(BaseModel):
    label: str
    bbox: list[float]  # [x0, y0, x1, y1] normalized 0-1
    confidence: float


class Detection(BaseModel):
    tile_id: str
    boxes: list[BoundingBox] = Field(default_factory=list)
    description: str = ""
    overall_confidence: float = 0.0


Action = Literal["flag", "request_neighbor", "request_hires", "downlink", "discard"]


# --- Stream events --------------------------------------------------------

class _EventBase(BaseModel):
    pass_id: str
    seq: int


class PassStartedEvent(_EventBase):
    event: Literal["pass_started"] = "pass_started"
    aoi: AOI
    tile_count: int
    bandwidth_kb: int
    mode: PassMode


class TileArrivedEvent(_EventBase):
    event: Literal["tile_arrived"] = "tile_arrived"
    tile_id: str
    lon: float
    lat: float
    fetch_ms: int
    image_url: Optional[str] = None  # /pass/{id}/tile/{tid}/image when imagery is real
    image_available: bool = True
    cloud_cover: Optional[float] = None
    captured_at: Optional[str] = None
    source: Optional[str] = None
    size_km: Optional[float] = None


class VLMDoneEvent(_EventBase):
    event: Literal["vlm_done"] = "vlm_done"
    tile_id: str
    detection: Detection
    inference_ms: int


class AgentThinkingEvent(_EventBase):
    event: Literal["agent_thinking"] = "agent_thinking"
    tile_id: str
    scratchpad: Optional[str] = None


class AgentDecidedEvent(_EventBase):
    event: Literal["agent_decided"] = "agent_decided"
    tile_id: str
    action: Action
    reasoning: str
    decision_ms: int


class BudgetUpdateEvent(_EventBase):
    event: Literal["budget_update"] = "budget_update"
    bandwidth_used_kb: int
    bandwidth_remaining_kb: int


class PassSummary(BaseModel):
    pass_id: str
    aoi: AOI
    tiles_processed: int
    tiles_flagged: int
    tiles_downlinked: int
    tiles_discarded: int
    bandwidth_used_kb: int
    bandwidth_remaining_kb: int
    elapsed_ms: int
    flagged_summary: list[str] = Field(default_factory=list)


class PassCompleteEvent(_EventBase):
    event: Literal["pass_complete"] = "pass_complete"
    summary: PassSummary
