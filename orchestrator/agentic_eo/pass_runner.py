"""Async pass loop.

Imagery is real Sentinel-2 from SimSat (RGB + SWIR composites fetched in
parallel). VLM perception is the v9-e3 fine-tuned LFM2.5-VL-450M loaded
in-process. Agent policy is still mocked (threshold over a derived
confidence), that swap lands next.
"""
from __future__ import annotations

import asyncio
import logging
from time import perf_counter

import httpx

logger = logging.getLogger(__name__)

from .aoi import plan_grid
from .models.agent import build_tile_prompt, get_default_agent
from .models.vlm import VLM, VlmBox, VlmResult

# Resolved once at module load. Production default is LFM2-2.6B; alternatives
# (text-only LFM2.5-VL, unified SFT'd VLM) plug in via the POLICY_AGENT env var.
_AGENT = get_default_agent()
from .schema import (
    AgentDecidedEvent,
    AgentThinkingEvent,
    BoundingBox,
    BudgetUpdateEvent,
    Detection,
    PassCompleteEvent,
    PassRequest,
    PassStartedEvent,
    PassSummary,
    TileArrivedEvent,
    VLMDoneEvent,
)
from .simsat_client import SimSatClient
from .store import PASS_STORE

_DOWNLINK_KB_PER_TILE = 80
_RGB_BANDS = ["red", "green", "blue"]
_SWIR_BANDS = ["swir22", "swir16", "nir"]


def _confidence_from_boxes(boxes: list[VlmBox]) -> float:
    """Derive a 0-1 confidence the policy can threshold on.

    The VLM doesn't emit per-box scores, so we proxy: 0 boxes is no signal,
    1 small box is a hint, 1 large box or 2+ boxes is high confidence.
    Mirrors the spirit of the dashboard's bbox-area-as-confidence ranking.
    """
    if not boxes:
        return 0.0
    max_area = max(b.area for b in boxes)
    return min(1.0, 0.5 + 0.2 * len(boxes) + max_area * 1.5)


def _no_signal_detection(tile_id: str, reason: str) -> Detection:
    return Detection(
        tile_id=tile_id,
        boxes=[],
        description=reason,
        overall_confidence=0.0,
    )


def _detection_from_vlm(tile_id: str, result: VlmResult) -> Detection:
    return Detection(
        tile_id=tile_id,
        boxes=[
            BoundingBox(label=b.label, bbox=b.bbox, confidence=1.0)
            for b in result.boxes
        ],
        description=result.description or "(no description returned)",
        overall_confidence=_confidence_from_boxes(result.boxes),
    )


def _threshold_decision(detection: Detection) -> tuple[str, str, int]:
    """Baseline policy, threshold on derived confidence. Returns
    (action, reason, decision_ms). The agent ablation compares against this."""
    if detection.overall_confidence >= 0.85:
        return "downlink", "high-confidence detection within budget", 0
    if detection.overall_confidence >= 0.65:
        return "flag", "moderate-confidence detection; flagged for end-of-pass review", 0
    return "discard", "below threshold", 0


async def _agent_decision(
    *,
    tile_id: str,
    lon: float,
    lat: float,
    cloud_cover: float | None,
    captured_at: str | None,
    detection: Detection,
    bandwidth_remaining_kb: int,
    bandwidth_total_kb: int,
) -> tuple[str, str, int]:
    """LFM2-1.2B tool-calling decision. Falls back to discard if the agent
    fails to emit a parseable call (counted in eval as a parser failure)."""
    max_area = (
        max(
            (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1])
            for b in detection.boxes
        )
        if detection.boxes
        else 0.0
    )
    prompt = build_tile_prompt(
        tile_id=tile_id,
        lon=lon,
        lat=lat,
        cloud_cover=cloud_cover,
        captured_at=captured_at,
        boxes_count=len(detection.boxes),
        max_area=max_area,
        description=detection.description,
        overall_confidence=detection.overall_confidence,
        bandwidth_remaining_kb=bandwidth_remaining_kb,
        bandwidth_total_kb=bandwidth_total_kb,
    )
    decision = await _AGENT.decide(prompt)
    return decision.action, decision.reason, decision.decision_ms


async def run_pass(pass_id: str, req: PassRequest) -> None:
    state = PASS_STORE.get(pass_id)
    if state is None:
        return

    simsat = SimSatClient()
    started = perf_counter()
    seq = 0

    def next_seq() -> int:
        nonlocal seq
        seq += 1
        return seq

    await state.publish(
        PassStartedEvent(
            pass_id=pass_id,
            seq=next_seq(),
            aoi=req.aoi,
            tile_count=req.tile_count,
            bandwidth_kb=req.bandwidth_kb,
            mode=req.mode,
        )
    )

    tiles = plan_grid(req.aoi, req.tile_count)
    bandwidth_used = 0
    flagged_descriptions: list[str] = []
    downlinked = 0
    flagged_only = 0
    discarded = 0
    processed = 0

    try:
        for tile_id, lon, lat in tiles:
            # --- Tile fetch (RGB + SWIR in parallel) ---------------------
            image_available = True
            cloud_cover: float | None = None
            captured_at: str | None = None
            source: str | None = None
            size_km: float | None = None
            image_url: str | None = None
            fetch_ms = 0
            rgb_bytes: bytes | None = None
            swir_bytes: bytes | None = None

            try:
                rgb_task = simsat.fetch_tile(lon=lon, lat=lat, bands=_RGB_BANDS)
                swir_task = simsat.fetch_tile(lon=lon, lat=lat, bands=_SWIR_BANDS)
                rgb_fetch, swir_fetch = await asyncio.gather(rgb_task, swir_task)
                fetch_ms = max(rgb_fetch.fetch_ms, swir_fetch.fetch_ms)
                if rgb_fetch.image_available and swir_fetch.image_available:
                    rgb_bytes = rgb_fetch.image_bytes
                    swir_bytes = swir_fetch.image_bytes
                    state.tile_images[tile_id] = (
                        rgb_fetch.content_type,
                        rgb_fetch.image_bytes,
                    )
                    image_url = f"/pass/{pass_id}/tile/{tile_id}/image"
                    cloud_cover = rgb_fetch.cloud_cover
                    captured_at = rgb_fetch.datetime
                    source = rgb_fetch.source
                    size_km = rgb_fetch.size_km
                else:
                    image_available = False
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                # One bad tile shouldn't kill the pass, mark unavailable so the
                # policy discards and the loop continues.
                image_available = False
                logger.warning(
                    "pass %s: SimSat fetch failed for %s: %s", pass_id, tile_id, e
                )

            await state.publish(
                TileArrivedEvent(
                    pass_id=pass_id,
                    seq=next_seq(),
                    tile_id=tile_id,
                    lon=lon,
                    lat=lat,
                    fetch_ms=fetch_ms,
                    image_url=image_url,
                    image_available=image_available,
                    cloud_cover=cloud_cover,
                    captured_at=captured_at,
                    source=source,
                    size_km=size_km,
                )
            )

            # --- VLM perception (real LFM2.5-VL-450M v9-e3) -------------
            if image_available and rgb_bytes is not None and swir_bytes is not None:
                try:
                    result = await VLM.detect(rgb_bytes, swir_bytes)
                    detection = _detection_from_vlm(tile_id, result)
                    inference_ms = result.inference_ms
                except Exception as e:  # noqa: BLE001, surface any VLM error per-tile
                    logger.warning(
                        "pass %s: VLM error on %s: %s", pass_id, tile_id, e
                    )
                    detection = _no_signal_detection(
                        tile_id, f"VLM inference failed: {e}"
                    )
                    inference_ms = 0
            else:
                detection = _no_signal_detection(
                    tile_id,
                    "Imagery unavailable (e.g. ocean, no Sentinel-2 pass within window).",
                )
                inference_ms = 0

            await state.publish(
                VLMDoneEvent(
                    pass_id=pass_id,
                    seq=next_seq(),
                    tile_id=tile_id,
                    detection=detection,
                    inference_ms=inference_ms,
                )
            )

            await state.publish(
                AgentThinkingEvent(
                    pass_id=pass_id,
                    seq=next_seq(),
                    tile_id=tile_id,
                    scratchpad=(
                        f"confidence={detection.overall_confidence:.2f}; "
                        f"boxes={len(detection.boxes)}; "
                        f"cloud_cover={cloud_cover if cloud_cover is not None else 'n/a'}; "
                        f"budget_remaining_kb={req.bandwidth_kb - bandwidth_used}"
                    ),
                )
            )

            # --- Policy decision ----------------------------------------
            if req.mode == "agent" and image_available:
                action, reasoning, decision_ms = await _agent_decision(
                    tile_id=tile_id,
                    lon=lon,
                    lat=lat,
                    cloud_cover=cloud_cover,
                    captured_at=captured_at,
                    detection=detection,
                    bandwidth_remaining_kb=max(0, req.bandwidth_kb - bandwidth_used),
                    bandwidth_total_kb=req.bandwidth_kb,
                )
            else:
                action, reasoning, decision_ms = _threshold_decision(detection)
            await state.publish(
                AgentDecidedEvent(
                    pass_id=pass_id,
                    seq=next_seq(),
                    tile_id=tile_id,
                    action=action,  # type: ignore[arg-type]
                    reasoning=reasoning,
                    decision_ms=decision_ms,
                )
            )

            if action == "downlink":
                bandwidth_used += _DOWNLINK_KB_PER_TILE
                downlinked += 1
                flagged_descriptions.append(detection.description)
            elif action == "flag":
                flagged_only += 1
                flagged_descriptions.append(detection.description)
            else:
                discarded += 1
            processed += 1

            await state.publish(
                BudgetUpdateEvent(
                    pass_id=pass_id,
                    seq=next_seq(),
                    bandwidth_used_kb=bandwidth_used,
                    bandwidth_remaining_kb=max(0, req.bandwidth_kb - bandwidth_used),
                )
            )

            if bandwidth_used >= req.bandwidth_kb:
                break

        elapsed_ms = int((perf_counter() - started) * 1000)
        summary = PassSummary(
            pass_id=pass_id,
            aoi=req.aoi,
            tiles_processed=processed,
            tiles_flagged=flagged_only,
            tiles_downlinked=downlinked,
            tiles_discarded=discarded,
            bandwidth_used_kb=bandwidth_used,
            bandwidth_remaining_kb=max(0, req.bandwidth_kb - bandwidth_used),
            elapsed_ms=elapsed_ms,
            flagged_summary=flagged_descriptions[:10],
        )
        state.summary = summary
        await state.publish(
            PassCompleteEvent(pass_id=pass_id, seq=next_seq(), summary=summary)
        )
    finally:
        await simsat.close()
        await state.close()
