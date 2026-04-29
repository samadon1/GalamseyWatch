"""Async SimSat client. Fetches Sentinel-2 tiles from a SimSat instance.

SimSat is the DPhi Space simulator that exposes orbit-tied Sentinel-2
imagery via a REST API. Defaults match the existing GalamseyWatch
dashboard (`app/src/app/api/simsat/sentinel/route.ts`):

- ``size_km=1.28`` matches SmallMinesDS training patch scale (128 px ×
  10 m/px); larger tiles shrink pits below the model's perception.
- ``timestamp=2024-01-15T00:00:00Z`` pins to Ghana's peak dry season so
  cloud cover is minimal.
- ``window_seconds=730 days`` gives SimSat a deep pool to pick the
  least-cloudy tile from.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_SIMSAT_URL = os.environ.get(
    "SIMSAT_BASE_URL",
    "https://simsat-sim-943572188770.us-central1.run.app",
)
DEFAULT_SIZE_KM = 1.28
DEFAULT_TIMESTAMP = "2024-01-15T00:00:00Z"
DEFAULT_WINDOW_SECONDS = 730 * 24 * 60 * 60  # 730 days
DEFAULT_BANDS = ["red", "green", "blue"]


@dataclass
class TileFetch:
    image_bytes: bytes
    content_type: str
    image_available: bool
    cloud_cover: float | None
    datetime: str | None
    source: str | None
    size_km: float
    footprint: Any | None
    fetch_ms: int


class SimSatClient:
    def __init__(
        self,
        base_url: str = DEFAULT_SIMSAT_URL,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_tile(
        self,
        lon: float,
        lat: float,
        *,
        timestamp: str = DEFAULT_TIMESTAMP,
        bands: list[str] | None = None,
        size_km: float = DEFAULT_SIZE_KM,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> TileFetch:
        from time import perf_counter

        bands = bands or DEFAULT_BANDS
        params: list[tuple[str, Any]] = [
            ("lon", lon),
            ("lat", lat),
            ("timestamp", timestamp),
            ("size_km", size_km),
            ("window_seconds", window_seconds),
            ("return_type", "png"),
        ]
        for b in bands:
            params.append(("spectral_bands", b))

        started = perf_counter()
        resp = await self.client.get(
            f"{self.base_url}/data/image/sentinel",
            params=params,
            headers={"accept": "image/png"},
        )
        elapsed_ms = int((perf_counter() - started) * 1000)
        resp.raise_for_status()

        meta_raw = resp.headers.get("sentinel_metadata") or resp.headers.get(
            "sentinel-metadata"
        )
        meta: dict[str, Any] = {}
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
            except json.JSONDecodeError:
                meta = {}

        return TileFetch(
            image_bytes=resp.content,
            content_type=resp.headers.get("content-type", "image/png"),
            image_available=bool(meta.get("image_available", True)),
            cloud_cover=_as_float(meta.get("cloud_cover")),
            datetime=meta.get("datetime"),
            source=meta.get("source"),
            size_km=float(meta.get("size_km", size_km)),
            footprint=meta.get("footprint"),
            fetch_ms=elapsed_ms,
        )


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
