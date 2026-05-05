"""Fetch + cache N tiles for inline Claude Code labeling.

Used by Option A of the Phase A.2 plan: fetch a deterministic sample of
diverse Ghana coordinates via SimSat, cache each tile's RGB + SWIR + meta,
emit a manifest. Claude Code (this conversation) then reads tiles from the
manifest and emits labels into a JSONL file.

No Anthropic API calls. Pro Max subscription covers the labeling labor.

Usage:
    cd training && uv run python scripts/fetch_unified_corpus.py [N=20]
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "orchestrator"))

from agentic_eo.simsat_client import SimSatClient  # noqa: E402

# Reuse the sampler + context synth from the Modal prep script.
sys.path.insert(0, str(ROOT))
from scripts.prepare_unified_decision_dataset_modal import (  # noqa: E402
    PerPassContext,
    TileCoord,
    build_user_message,
    sample_coordinates,
    synthesize_context,
)

_CACHE_DIR = ROOT / "data" / "unified_v1_cache"
_MANIFEST = _CACHE_DIR / "manifest.json"
_LABELS_JSONL = ROOT / "data" / "unified_v1" / "labels.jsonl"
_RGB_BANDS = ["red", "green", "blue"]
_SWIR_BANDS = ["swir22", "swir16", "nir"]


async def fetch_and_cache(n: int) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _LABELS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    coords = sample_coordinates(n=n, seed=42)
    print(f"Sampled {len(coords)} coords across strata:")
    by_stratum: dict[str, int] = {}
    for c in coords:
        by_stratum[c.stratum] = by_stratum.get(c.stratum, 0) + 1
    for s, cnt in sorted(by_stratum.items(), key=lambda kv: -kv[1]):
        print(f"  {s:8s}: {cnt}")

    print(f"\nFetching to {_CACHE_DIR}/ (sequential, polite)...")
    client = SimSatClient()

    import random
    rng = random.Random(42)  # deterministic context synthesis
    manifest: list[dict] = []

    try:
        for i, coord in enumerate(coords):
            tile_dir = _CACHE_DIR / coord.coord_id
            tile_dir.mkdir(parents=True, exist_ok=True)

            rgb_path = tile_dir / "rgb.png"
            swir_path = tile_dir / "swir.png"
            meta_path = tile_dir / "meta.json"
            ctx_path = tile_dir / "context.json"

            if rgb_path.exists() and swir_path.exists() and meta_path.exists() and ctx_path.exists():
                print(f"  [{i+1:>3}/{len(coords)}] {coord.coord_id} already cached, skip")
                # Still need to add to manifest:
                manifest.append({
                    "coord_id": coord.coord_id,
                    "lon": coord.lon,
                    "lat": coord.lat,
                    "stratum": coord.stratum,
                    "rgb_path": str(rgb_path.relative_to(ROOT)),
                    "swir_path": str(swir_path.relative_to(ROOT)),
                    "meta_path": str(meta_path.relative_to(ROOT)),
                    "context_path": str(ctx_path.relative_to(ROOT)),
                    "user_message_preview": build_user_message(
                        coord, PerPassContext(**json.loads(ctx_path.read_text()))
                    )[:200],
                })
                # Advance the rng even when cached so context generation stays
                # deterministic across runs:
                synthesize_context(coord, cloud_cover=0, captured_at="", tile_imagery_issue=None, rng=rng)
                continue

            try:
                rgb_task = client.fetch_tile(lon=coord.lon, lat=coord.lat, bands=_RGB_BANDS)
                swir_task = client.fetch_tile(lon=coord.lon, lat=coord.lat, bands=_SWIR_BANDS)
                rgb, swir = await asyncio.gather(rgb_task, swir_task)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i+1:>3}/{len(coords)}] {coord.coord_id} FETCH FAILED: {e}")
                continue

            if not (rgb.image_available and swir.image_available):
                print(f"  [{i+1:>3}/{len(coords)}] {coord.coord_id} no imagery available, skip")
                continue

            rgb_path.write_bytes(rgb.image_bytes)
            swir_path.write_bytes(swir.image_bytes)

            meta = {
                "coord_id": coord.coord_id,
                "lon": coord.lon,
                "lat": coord.lat,
                "stratum": coord.stratum,
                "mission_priors": coord.mission_priors,
                "cloud_cover": rgb.cloud_cover,
                "captured_at": rgb.datetime,
                "source": rgb.source,
                "size_km": rgb.size_km,
            }
            meta_path.write_text(json.dumps(meta, indent=2))

            ctx = synthesize_context(
                coord,
                cloud_cover=rgb.cloud_cover,
                captured_at=rgb.datetime,
                tile_imagery_issue=None,
                rng=rng,
            )
            from dataclasses import asdict
            ctx_path.write_text(json.dumps(asdict(ctx), indent=2))

            user_msg = build_user_message(coord, ctx)

            manifest.append({
                "coord_id": coord.coord_id,
                "lon": coord.lon,
                "lat": coord.lat,
                "stratum": coord.stratum,
                "rgb_path": str(rgb_path.relative_to(ROOT)),
                "swir_path": str(swir_path.relative_to(ROOT)),
                "meta_path": str(meta_path.relative_to(ROOT)),
                "context_path": str(ctx_path.relative_to(ROOT)),
                "user_message_preview": user_msg[:200],
            })

            print(
                f"  [{i+1:>3}/{len(coords)}] {coord.coord_id} ({coord.lon:>+7.3f}, {coord.lat:>+6.3f}) "
                f"stratum={coord.stratum:8s}  rgb={len(rgb.image_bytes)//1024}KB"
            )
    finally:
        await client.close()

    _MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {_MANIFEST.relative_to(ROOT)}")
    print(f"Total cached: {len(manifest)} tiles")
    print(f"\nNext: Claude Code reads tiles from manifest, appends labels to {_LABELS_JSONL.relative_to(ROOT)}")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    asyncio.run(fetch_and_cache(n))


if __name__ == "__main__":
    main()
