"""Fetch 10 diverse Ghana tiles for the Phase A.2 mini validation.

Pulls RGB + SWIR composites from SimSat across deliberately-mixed terrain
(mining hotspots, pristine forest, water, urban, coastal) so the labeling
prompt can be tested on edge cases the Bibiani 6-tile sample didn't cover.

Output layout (parallels orchestrator/.benchmark_cache/<tid>/):
    orchestrator/.benchmark_cache/diverse_v1/<tid>/
        rgb.png
        swir.png
        meta.json   # tile coords, cloud_cover, captured_at, label_hint

Run:
    cd orchestrator && uv run python scripts/fetch_diverse_validation_tiles.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Add agentic_eo to path so we can import the SimSat client.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentic_eo.simsat_client import SimSatClient  # noqa: E402

_CACHE_DIR = ROOT / ".benchmark_cache" / "diverse_v1"
_RGB_BANDS = ["red", "green", "blue"]
_SWIR_BANDS = ["swir22", "swir16", "nir"]

# Deliberately-diverse 10-tile sample across Ghana.
# label_hint is the rough scene category (NOT ground truth, just for human review).
TILES: list[dict] = [
    {"id": "d000", "lon": -2.10,  "lat": 5.55, "label_hint": "mining_pra_basin_bogoso"},
    {"id": "d001", "lon": -2.00,  "lat": 5.45, "label_hint": "mining_ankobra_basin_prestea"},
    {"id": "d002", "lon": -0.55,  "lat": 6.20, "label_hint": "forest_atewa_reserve_threatened"},
    {"id": "d003", "lon": -1.38,  "lat": 5.35, "label_hint": "forest_kakum_pristine"},
    {"id": "d004", "lon": -3.13,  "lat": 6.55, "label_hint": "forest_bia_pristine"},
    {"id": "d005", "lon": -1.42,  "lat": 6.50, "label_hint": "water_lake_bosumtwi"},
    {"id": "d006", "lon":  0.05,  "lat": 7.80, "label_hint": "water_lake_volta"},
    {"id": "d007", "lon": -0.20,  "lat": 5.55, "label_hint": "urban_accra"},
    {"id": "d008", "lon": -1.62,  "lat": 6.69, "label_hint": "urban_kumasi"},
    {"id": "d009", "lon": -2.20,  "lat": 4.85, "label_hint": "coastal_axim_cloudprone"},
    # Positive controls for the two tools never exercised on the first 16 tiles.
    # If these don't elicit request_higher_resolution / request_neighbor_tile,
    # the labeling prompt is broken in a way scaling to 1000 won't fix.
    {"id": "d010", "lon": -2.81,  "lat": 5.63, "label_hint": "edge_west_of_bibiani_cluster_neighbor_east"},
    {"id": "d011", "lon": -0.60,  "lat": 6.18, "label_hint": "atewa_periphery_small_candidate_hires"},
    {"id": "d012", "lon": -1.95,  "lat": 5.40, "label_hint": "pra_tributary_sediment_plume_neighbor"},
]


async def fetch_one(client: SimSatClient, tile: dict) -> None:
    out_dir = _CACHE_DIR / tile["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {tile['id']:5s} ({tile['lon']:>+7.3f}, {tile['lat']:>+6.3f}) [{tile['label_hint']}]")

    rgb_task = client.fetch_tile(lon=tile["lon"], lat=tile["lat"], bands=_RGB_BANDS)
    swir_task = client.fetch_tile(lon=tile["lon"], lat=tile["lat"], bands=_SWIR_BANDS)
    rgb, swir = await asyncio.gather(rgb_task, swir_task)

    if not (rgb.image_available and swir.image_available):
        print(f"    -> NO IMAGE (rgb={rgb.image_available}, swir={swir.image_available})")
        meta = {
            **tile,
            "image_available": False,
            "rgb_available": rgb.image_available,
            "swir_available": swir.image_available,
            "cloud_cover": rgb.cloud_cover,
            "captured_at": rgb.datetime,
            "source": rgb.source,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return

    (out_dir / "rgb.png").write_bytes(rgb.image_bytes)
    (out_dir / "swir.png").write_bytes(swir.image_bytes)

    meta = {
        **tile,
        "image_available": True,
        "cloud_cover": rgb.cloud_cover,
        "captured_at": rgb.datetime,
        "source": rgb.source,
        "size_km": rgb.size_km,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"    -> rgb={len(rgb.image_bytes):>6d}B  swir={len(swir.image_bytes):>6d}B  cloud={rgb.cloud_cover}  captured={rgb.datetime}")


def _tile_already_cached(tile: dict) -> bool:
    out_dir = _CACHE_DIR / tile["id"]
    return (
        (out_dir / "rgb.png").exists()
        and (out_dir / "swir.png").exists()
        and (out_dir / "meta.json").exists()
    )


async def main() -> None:
    print(f"Fetching {len(TILES)} diverse tiles into {_CACHE_DIR}/ (sequential)")
    client = SimSatClient()
    try:
        for tile in TILES:
            if _tile_already_cached(tile):
                print(f"  {tile['id']:5s} already cached, skip")
                continue
            try:
                await fetch_one(client, tile)
            except Exception as e:  # noqa: BLE001
                print(f"  {tile['id']:5s} FAILED: {type(e).__name__}: {e}")
    finally:
        await client.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
