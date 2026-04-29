"""FastAPI app for the agentic-EO orchestrator.

Endpoints:
    POST /pass/start            kick off a pass over an AOI
    GET  /pass/{id}/events      SSE stream of pass events
    GET  /pass/{id}/summary     final summary
    GET  /                      health
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .pass_runner import run_pass
from .schema import PassRequest, PassStarted, PassSummary
from .store import PASS_STORE

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s · %(name)s · %(levelname)s · %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="agentic-eo orchestrator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # narrow in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str | int]:
    return {"service": "agentic-eo", "version": "0.1.0", "phase": 1}


@app.post("/pass/start", response_model=PassStarted)
async def start_pass(req: PassRequest) -> PassStarted:
    pass_id = PASS_STORE.create(req)
    PASS_STORE.spawn(pass_id, run_pass(pass_id, req))
    return PassStarted(pass_id=pass_id)


@app.get("/pass/{pass_id}/events")
async def pass_events(pass_id: str) -> EventSourceResponse:
    state = PASS_STORE.get(pass_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"unknown pass: {pass_id}")
    return EventSourceResponse(state.event_stream())


@app.get("/pass/{pass_id}/summary", response_model=PassSummary)
async def pass_summary(pass_id: str) -> PassSummary:
    state = PASS_STORE.get(pass_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"unknown pass: {pass_id}")
    if state.summary is None:
        raise HTTPException(status_code=409, detail="pass not complete")
    return state.summary


@app.get("/pass/{pass_id}/tile/{tile_id}/image")
async def pass_tile_image(pass_id: str, tile_id: str) -> Response:
    """Serve a cached tile PNG fetched from SimSat during the pass run."""
    state = PASS_STORE.get(pass_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"unknown pass: {pass_id}")
    cached = state.tile_images.get(tile_id)
    if cached is None:
        raise HTTPException(status_code=404, detail=f"unknown tile: {tile_id}")
    content_type, image_bytes = cached
    return Response(
        content=image_bytes,
        media_type=content_type,
        headers={"cache-control": "public, max-age=3600, immutable"},
    )
