"""Baseline policy comparison for the agentic-EO orchestrator.

Runs a single Bibiani-cluster pass once, caches RGB+SWIR imagery and VLM
perception results to disk, then replays four budget-allocation policies
against the cached perception:

    send_everything  downlink every tile until the budget is exhausted
    random           uniform random {downlink, flag, discard}, seeded
    threshold        perception-only confidence threshold (no agent)
    agent            LFM2-2.6B tool-calling agent (real model)

Ground truth for "pits captured" is the set of tiles where the perception
model emits at least one bounding box. The same VLM runs once per tile
and is cached, so every policy sees the same perception output. The
benchmark therefore measures budget-allocation quality, not perception
quality.

Outputs a Markdown table on stdout plus a CSV at
orchestrator/scripts/baseline_results.csv that backs Slide 13.5 of the
talk.

Usage:
    cd orchestrator
    uv run python scripts/benchmark_policies.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from agentic_eo.aoi import plan_grid  # noqa: E402
from agentic_eo.models.agent import build_tile_prompt, get_default_agent  # noqa: E402

_AGENT = get_default_agent()
from agentic_eo.models.vlm import VLM  # noqa: E402
from agentic_eo.pass_runner import (  # noqa: E402
    _DOWNLINK_KB_PER_TILE,
    _RGB_BANDS,
    _SWIR_BANDS,
    _confidence_from_boxes,
    _threshold_decision,
)
from agentic_eo.schema import AOI, BoundingBox, Detection  # noqa: E402
from agentic_eo.simsat_client import SimSatClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- Fixed experiment knobs --------------------------------------------------

BIBIANI_AOI = AOI(
    name="Bibiani cluster",
    lon_min=-2.79,
    lat_min=5.62,
    lon_max=-2.71,
    lat_max=5.66,
)
TILE_COUNT = 6
BANDWIDTH_KB = 512
RANDOM_SEED = 42

CACHE_DIR = ROOT / ".benchmark_cache"
RESULTS_CSV = ROOT / "scripts" / "baseline_results.csv"


@dataclass
class CachedTile:
    tile_id: str
    lon: float
    lat: float
    cloud_cover: float | None
    captured_at: str | None
    image_available: bool
    detection: Detection
    has_pit: bool  # ground truth: VLM emitted at least one bbox


# --- Cache phase: fetch + VLM once, persist to disk --------------------------

async def cache_pass() -> list[CachedTile]:
    """Run one pass, fetch RGB+SWIR + perception per tile, cache to disk.

    Idempotent: if the cache exists for a tile, skip the SimSat fetch and
    the VLM call. Delete .benchmark_cache/ to force a fresh run.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    simsat = SimSatClient()
    tiles_plan = plan_grid(BIBIANI_AOI, TILE_COUNT)
    cached: list[CachedTile] = []

    try:
        for tile_id, lon, lat in tiles_plan:
            tile_dir = CACHE_DIR / tile_id
            meta_path = tile_dir / "meta.json"

            if meta_path.exists():
                logger.info("[cache] %s: hit", tile_id)
                meta = json.loads(meta_path.read_text())
                detection = Detection.model_validate(meta["detection"])
                cached.append(
                    CachedTile(
                        tile_id=tile_id,
                        lon=meta["lon"],
                        lat=meta["lat"],
                        cloud_cover=meta.get("cloud_cover"),
                        captured_at=meta.get("captured_at"),
                        image_available=meta.get("image_available", True),
                        detection=detection,
                        has_pit=len(detection.boxes) > 0,
                    )
                )
                continue

            logger.info("[fetch] %s: lon=%.4f lat=%.4f", tile_id, lon, lat)
            tile_dir.mkdir(parents=True, exist_ok=True)
            cloud_cover: float | None = None
            captured_at: str | None = None
            image_available = True
            rgb_bytes: bytes | None = None
            swir_bytes: bytes | None = None

            try:
                rgb_task = simsat.fetch_tile(lon=lon, lat=lat, bands=_RGB_BANDS)
                swir_task = simsat.fetch_tile(lon=lon, lat=lat, bands=_SWIR_BANDS)
                rgb_fetch, swir_fetch = await asyncio.gather(rgb_task, swir_task)
                if rgb_fetch.image_available and swir_fetch.image_available:
                    rgb_bytes = rgb_fetch.image_bytes
                    swir_bytes = swir_fetch.image_bytes
                    cloud_cover = rgb_fetch.cloud_cover
                    captured_at = rgb_fetch.datetime
                    (tile_dir / "rgb.png").write_bytes(rgb_bytes)
                    (tile_dir / "swir.png").write_bytes(swir_bytes)
                else:
                    image_available = False
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                logger.warning("[fetch] %s: error %s", tile_id, e)
                image_available = False

            if image_available and rgb_bytes is not None and swir_bytes is not None:
                logger.info("[vlm] %s: running perception", tile_id)
                vlm_result = await VLM.detect(rgb_bytes, swir_bytes)
                detection = Detection(
                    tile_id=tile_id,
                    boxes=[
                        BoundingBox(label=b.label, bbox=b.bbox, confidence=1.0)
                        for b in vlm_result.boxes
                    ],
                    description=vlm_result.description or "(no description)",
                    overall_confidence=_confidence_from_boxes(vlm_result.boxes),
                )
            else:
                detection = Detection(
                    tile_id=tile_id,
                    boxes=[],
                    description="Imagery unavailable",
                    overall_confidence=0.0,
                )

            meta = {
                "lon": lon,
                "lat": lat,
                "cloud_cover": cloud_cover,
                "captured_at": captured_at,
                "image_available": image_available,
                "detection": detection.model_dump(),
            }
            meta_path.write_text(json.dumps(meta, indent=2))

            cached.append(
                CachedTile(
                    tile_id=tile_id,
                    lon=lon,
                    lat=lat,
                    cloud_cover=cloud_cover,
                    captured_at=captured_at,
                    image_available=image_available,
                    detection=detection,
                    has_pit=len(detection.boxes) > 0,
                )
            )
    finally:
        await simsat.close()

    return cached


# --- Policies ----------------------------------------------------------------

PolicyFn = Callable[[CachedTile, int], Awaitable[tuple[str, str]]]

_RAND = random.Random(RANDOM_SEED)


async def policy_send_everything(tile: CachedTile, remaining_kb: int) -> tuple[str, str]:
    if remaining_kb >= _DOWNLINK_KB_PER_TILE:
        return "downlink", "always-downlink baseline"
    return "discard", "budget exhausted"


async def policy_random(tile: CachedTile, remaining_kb: int) -> tuple[str, str]:
    choice = _RAND.choice(["downlink", "flag", "discard"])
    if choice == "downlink" and remaining_kb < _DOWNLINK_KB_PER_TILE:
        return "discard", "random pick was downlink but budget exhausted"
    return choice, f"random ({choice})"


async def policy_threshold(tile: CachedTile, remaining_kb: int) -> tuple[str, str]:
    action, reason, _ = _threshold_decision(tile.detection)
    if action == "downlink" and remaining_kb < _DOWNLINK_KB_PER_TILE:
        return "discard", "threshold said downlink but budget exhausted"
    return action, reason


async def policy_agent(tile: CachedTile, remaining_kb: int) -> tuple[str, str]:
    if not tile.image_available:
        return "discard", "imagery unavailable"
    max_area = (
        max(
            (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1])
            for b in tile.detection.boxes
        )
        if tile.detection.boxes
        else 0.0
    )
    prompt = build_tile_prompt(
        tile_id=tile.tile_id,
        lon=tile.lon,
        lat=tile.lat,
        cloud_cover=tile.cloud_cover,
        captured_at=tile.captured_at,
        boxes_count=len(tile.detection.boxes),
        max_area=max_area,
        description=tile.detection.description,
        overall_confidence=tile.detection.overall_confidence,
        bandwidth_remaining_kb=remaining_kb,
        bandwidth_total_kb=BANDWIDTH_KB,
    )
    decision = await _AGENT.decide(prompt)
    if decision.action == "downlink" and remaining_kb < _DOWNLINK_KB_PER_TILE:
        return "discard", "agent said downlink but budget exhausted"
    return decision.action, decision.reason


# --- Replay ------------------------------------------------------------------

@dataclass
class PolicyResult:
    name: str
    tiles_downlinked: int
    pits_captured: int
    pits_total: int
    bytes_used: int
    pits_per_kb: float


async def replay(name: str, fn: PolicyFn, tiles: list[CachedTile]) -> PolicyResult:
    bandwidth_used = 0
    downlinked = 0
    pits_captured = 0
    pits_total = sum(1 for t in tiles if t.has_pit)
    per_tile_log: list[str] = []

    for t in tiles:
        remaining = max(0, BANDWIDTH_KB - bandwidth_used)
        action, reason = await fn(t, remaining)
        per_tile_log.append(
            f"  {t.tile_id} (pit={t.has_pit}, conf={t.detection.overall_confidence:.2f}) "
            f"-> {action}  [{reason}]"
        )
        if action == "downlink":
            bandwidth_used += _DOWNLINK_KB_PER_TILE
            downlinked += 1
            if t.has_pit:
                pits_captured += 1

    for line in per_tile_log:
        logger.info(line)

    pits_per_kb = pits_captured / bandwidth_used if bandwidth_used > 0 else 0.0
    return PolicyResult(
        name=name,
        tiles_downlinked=downlinked,
        pits_captured=pits_captured,
        pits_total=pits_total,
        bytes_used=bandwidth_used,
        pits_per_kb=pits_per_kb,
    )


# --- Reporting ---------------------------------------------------------------

def print_table(results: list[PolicyResult]) -> None:
    print()
    print("| Policy | Tiles downlinked | Pits captured | Bytes used | Pits / KB |")
    print("|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r.name} | {r.tiles_downlinked} | "
            f"{r.pits_captured} of {r.pits_total} | "
            f"{r.bytes_used} KB | {r.pits_per_kb:.4f} |"
        )
    print()


def write_csv(results: list[PolicyResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "policy",
                "tiles_downlinked",
                "pits_captured",
                "pits_total",
                "bytes_used_kb",
                "pits_per_kb",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.name,
                    r.tiles_downlinked,
                    r.pits_captured,
                    r.pits_total,
                    r.bytes_used,
                    f"{r.pits_per_kb:.6f}",
                ]
            )
    logger.info("wrote %s", path)


# --- Entrypoint --------------------------------------------------------------

async def main() -> None:
    logger.info("== caching pass: %d tiles over %s ==", TILE_COUNT, BIBIANI_AOI.name)
    tiles = await cache_pass()
    pits_total = sum(1 for t in tiles if t.has_pit)
    logger.info(
        "ground truth: %d/%d tiles contain >=1 perception bbox",
        pits_total,
        len(tiles),
    )

    policies: list[tuple[str, PolicyFn]] = [
        ("send-everything", policy_send_everything),
        ("random (seed=42)", policy_random),
        ("threshold (no agent)", policy_threshold),
        ("LFM2-2.6B agent", policy_agent),
    ]

    results: list[PolicyResult] = []
    for name, fn in policies:
        _RAND.seed(RANDOM_SEED)  # determinism for the random policy
        logger.info("== replaying: %s ==", name)
        results.append(await replay(name, fn, tiles))

    print_table(results)
    write_csv(results, RESULTS_CSV)


if __name__ == "__main__":
    asyncio.run(main())
