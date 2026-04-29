"""AOI tile planning.

Lays out a regular lon/lat grid of tile centers covering an AOI bounding
box. Phase 1: pick grid dims that approximate the AOI aspect ratio and
fit ``n`` tiles. Real orbital tile planning eventually accounts for
swath geometry, satellite ground track, and overlap; this is the simple
version that's good enough for a single-pass demo over a small AOI.
"""
from __future__ import annotations

import math

from .schema import AOI

KM_PER_DEG_LAT = 111.0


def plan_grid(aoi: AOI, n: int) -> list[tuple[str, float, float]]:
    """Return ``n`` (tile_id, lon, lat) centers covering ``aoi``."""
    if n <= 0:
        return []

    lat_extent = aoi.lat_max - aoi.lat_min
    lon_extent = aoi.lon_max - aoi.lon_min
    if lat_extent <= 0 or lon_extent <= 0:
        # Degenerate AOI: collapse to its center and emit n copies.
        cx = (aoi.lon_min + aoi.lon_max) / 2
        cy = (aoi.lat_min + aoi.lat_max) / 2
        return [(f"t{i:03d}", cx, cy) for i in range(n)]

    # Approximate the AOI aspect ratio using metric km, not raw degrees,
    # so a 1°×1° AOI near the equator gets a balanced grid.
    mean_lat = (aoi.lat_min + aoi.lat_max) / 2
    km_per_deg_lon = KM_PER_DEG_LAT * math.cos(math.radians(mean_lat))
    aoi_w_km = lon_extent * km_per_deg_lon
    aoi_h_km = lat_extent * KM_PER_DEG_LAT
    aspect = aoi_w_km / aoi_h_km if aoi_h_km > 0 else 1.0

    rows = max(1, int(round(math.sqrt(n / aspect))))
    cols = max(1, math.ceil(n / rows))
    # If we now have more cells than tiles, keep the first n in row-major order.

    tiles: list[tuple[str, float, float]] = []
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= n:
                return tiles
            lat = aoi.lat_min + (r + 0.5) * (lat_extent / rows)
            lon = aoi.lon_min + (c + 0.5) * (lon_extent / cols)
            tiles.append((f"t{i:03d}", lon, lat))
            i += 1
    return tiles
