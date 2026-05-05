"""Generate the unified-VLM training corpus on Modal.

Phase A.2 (full) of the agentic-EO research plan (see
``docs/UNIFIED_VLM_PLAN.md``). Status: SCAFFOLD ONLY, not runnable end-to-end
yet. Components marked TODO need implementation + verification against current
Anthropic SDK and SimSat API surfaces before execution.

Pipeline:
    1. Sample N coordinates across Ghana with stratified geographic mix.
    2. For each, fetch RGB + SWIR composites from SimSat (sequential, polite).
    3. Synthesize per-pass scalar context (budget, neighbors, mission priors).
    4. Call Claude as labeling adjudicator (prompt caching on system + tools).
    5. Write JSONL training examples in v9 VLM-SFT format extended with the
       action label.
    6. Stratified manual audit on ~10% of the corpus (separate script run).

Usage (when complete):
    cd training && uv run modal run scripts/prepare_unified_decision_dataset_modal.py \\
        --n 1000 --output unified_v1_train

Cost estimate (per the plan):
    - Claude image input: 2 images × 1000 tiles × ~1500 input tokens × $3/MTok
      input ~= $9 (with caching, much less).
    - Output: ~100 tokens × 1000 = $1.50.
    - Cache writes (system + tools, ~2000 tokens, $3.75/MTok): ~$0.01/call × 1000.
    - Total: ~$15-20 for the full 1000-example corpus.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import modal

logger = logging.getLogger(__name__)

# --- Modal infra ---------------------------------------------------------

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "anthropic>=0.40.0",  # TODO verify current major before run
        "httpx>=0.27.0",
        "pydantic>=2.0",
    )
)

app = modal.App("galamsey-unified-decision-dataset")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)


# --- Data classes --------------------------------------------------------

@dataclass(frozen=True)
class TileCoord:
    """A single (lon, lat) sample with stratification metadata."""

    coord_id: str
    lon: float
    lat: float
    stratum: str  # "mining" | "forest" | "water" | "urban" | "mixed" | "edge"
    mission_priors: str  # AOI-specific operational note


@dataclass(frozen=True)
class PerPassContext:
    """Synthesized scalar context for one labeling call."""

    budget_remaining_kb: int
    budget_total_kb: int
    prior_tiles_downlinked: int
    cloud_cover: float | None
    captured_at: str | None
    tile_imagery_issue: str | None  # null | "partial_swath" | "high_visual_cloud"
    mission_priors: str
    neighbor_summary: dict[str, Any]


@dataclass(frozen=True)
class LabeledExample:
    """One (tile, context, action) training example."""

    coord_id: str
    lon: float
    lat: float
    stratum: str
    rgb_path: str
    swir_path: str
    context: PerPassContext
    action: str       # one of the 5 Action literals
    direction: str | None  # only for request_neighbor_tile
    reason: str
    raw_response: str
    adjudicator_model: str


# --- Stratified coordinate sampler ---------------------------------------

# Hand-curated AOIs across Ghana with mission priors. The sampler picks within
# these regions and adds Gaussian jitter to produce N diverse coordinates.
_AOIs: list[dict] = [
    # Mining hotspots ---------------------------------------------------
    {"name": "bibiani_cluster",        "lon": -2.75, "lat": 5.64, "radius_deg": 0.05,
     "stratum": "mining",  "prior": "Bibiani: known active galamsey cluster, known-distribution training site"},
    {"name": "pra_basin_bogoso",       "lon": -2.10, "lat": 5.55, "radius_deg": 0.08,
     "stratum": "mining",  "prior": "Pra basin near Bogoso: heavy galamsey, river sediment plumes common"},
    {"name": "ankobra_basin_prestea",  "lon": -2.00, "lat": 5.45, "radius_deg": 0.07,
     "stratum": "mining",  "prior": "Ankobra basin near Prestea: heavy galamsey, mixed legal/illegal"},
    {"name": "obuasi_concession",      "lon": -1.65, "lat": 6.30, "radius_deg": 0.05,
     "stratum": "mining",  "prior": "Obuasi: legal AngloGold concession plus illegal periphery activity"},
    {"name": "asutifi_kenyasi",        "lon": -2.45, "lat": 6.95, "radius_deg": 0.06,
     "stratum": "mining",  "prior": "Asutifi: newer galamsey frontier, expanding"},
    # Forest (mostly negative; some encroachment) -----------------------
    {"name": "atewa_reserve",          "lon": -0.55, "lat": 6.20, "radius_deg": 0.10,
     "stratum": "forest",  "prior": "Atewa: known galamsey encroachment frontier, threatened reserve"},
    {"name": "kakum_park",             "lon": -1.38, "lat": 5.35, "radius_deg": 0.08,
     "stratum": "forest",  "prior": "Kakum: protected park, low expected disturbance"},
    {"name": "bia_park",               "lon": -3.13, "lat": 6.55, "radius_deg": 0.10,
     "stratum": "forest",  "prior": "Bia: protected park, buffer-zone farming nearby"},
    # Water -------------------------------------------------------------
    {"name": "lake_bosumtwi",          "lon": -1.42, "lat": 6.50, "radius_deg": 0.04,
     "stratum": "water",   "prior": "Lake Bosumtwi: crater lake, no mining possible"},
    {"name": "lake_volta_central",     "lon":  0.05, "lat": 7.50, "radius_deg": 0.20,
     "stratum": "water",   "prior": "Lake Volta: large reservoir, agriculture on shores"},
    # Urban -------------------------------------------------------------
    {"name": "accra_metro",            "lon": -0.20, "lat": 5.55, "radius_deg": 0.08,
     "stratum": "urban",   "prior": "Accra: urban, SWIR brightness is built environment"},
    {"name": "kumasi_metro",           "lon": -1.62, "lat": 6.69, "radius_deg": 0.06,
     "stratum": "urban",   "prior": "Kumasi: urban, SWIR brightness is built environment"},
    # Mixed / agricultural ---------------------------------------------
    {"name": "northern_savanna",       "lon": -1.00, "lat": 9.50, "radius_deg": 0.30,
     "stratum": "mixed",   "prior": "Northern savanna: dry-season agriculture, no mining"},
    {"name": "central_agriculture",    "lon": -1.20, "lat": 7.30, "radius_deg": 0.25,
     "stratum": "mixed",   "prior": "Central region farmland: cocoa and food crops, no mining"},
    # Edge cases (cloud-prone, swath edges) ----------------------------
    {"name": "coastal_axim",           "lon": -2.20, "lat": 4.85, "radius_deg": 0.08,
     "stratum": "edge",    "prior": "Coastal Axim: cloud-prone, dense forest"},
]

# Stratum sampling weights (must sum to ~1.0).
_STRATUM_WEIGHTS = {
    "mining": 0.30,
    "forest": 0.25,
    "water":  0.10,
    "urban":  0.10,
    "mixed":  0.15,
    "edge":   0.10,
}


def sample_coordinates(n: int, seed: int = 42) -> list[TileCoord]:
    """Stratified random sample of N coordinates over Ghana.

    Picks a stratum per ``_STRATUM_WEIGHTS``, picks an AOI within that stratum,
    adds Gaussian jitter within the AOI's ``radius_deg``. Deterministic given
    ``seed`` so the train/val/test split reproduces.
    """
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict]] = {}
    for aoi in _AOIs:
        by_stratum.setdefault(aoi["stratum"], []).append(aoi)

    coords: list[TileCoord] = []
    for i in range(n):
        stratum = rng.choices(
            list(_STRATUM_WEIGHTS.keys()), weights=list(_STRATUM_WEIGHTS.values()), k=1
        )[0]
        aoi = rng.choice(by_stratum[stratum])
        # Gaussian jitter, sigma = radius_deg / 2 so most samples land within radius.
        sigma = aoi["radius_deg"] / 2
        lon = aoi["lon"] + rng.gauss(0, sigma)
        lat = aoi["lat"] + rng.gauss(0, sigma)
        coords.append(
            TileCoord(
                coord_id=f"u{i:04d}",
                lon=lon,
                lat=lat,
                stratum=stratum,
                mission_priors=aoi["prior"],
            )
        )
    return coords


# --- Per-pass context synthesizer ----------------------------------------

def synthesize_context(
    coord: TileCoord,
    *,
    cloud_cover: float | None,
    captured_at: str | None,
    tile_imagery_issue: str | None,
    rng: random.Random,
) -> PerPassContext:
    """Generate plausible scalar context for one labeling call.

    Uses skewed distributions (not uniform) to reflect real pass state:
      - Budget: roughly uniform over [0, 512] (any pass moment is equally likely).
      - Prior tiles downlinked: log-normal-ish, anchored to budget consumption.
      - Neighbor summary: structured, with each direction either null or a
        synthesized {action, boxes, scene_hint} based on the AOI's prior.

    TODO: tighten the neighbor synthesis to actually look at adjacent
    coordinates in the sampler (so neighbor signals are spatially coherent
    rather than fully random).
    """
    budget_total = 512
    budget_remaining = rng.randint(0, budget_total)
    spent = budget_total - budget_remaining
    prior_downlinked = spent // 80  # 80 KB per downlink

    # Synthesize neighbor signals (placeholder; should be informed by
    # actual adjacent samples once we have spatial proximity tracking).
    neighbor_summary: dict[str, Any] = {}
    for direction in ("north", "south", "east", "west"):
        if rng.random() < 0.3:
            neighbor_summary[direction] = None  # not visited / out of AOI
        else:
            scene_hint = _scene_hint_for_stratum(coord.stratum, rng)
            action_for_hint = _action_for_scene(scene_hint, rng)
            neighbor_summary[direction] = {
                "action": action_for_hint,
                "boxes": rng.randint(0, 8) if action_for_hint == "downlink" else 0,
                "scene_hint": scene_hint,
            }

    return PerPassContext(
        budget_remaining_kb=budget_remaining,
        budget_total_kb=budget_total,
        prior_tiles_downlinked=prior_downlinked,
        cloud_cover=cloud_cover,
        captured_at=captured_at,
        tile_imagery_issue=tile_imagery_issue,
        mission_priors=coord.mission_priors,
        neighbor_summary=neighbor_summary,
    )


def _scene_hint_for_stratum(stratum: str, rng: random.Random) -> str:
    """Pick a plausible neighbor scene hint conditioned on the stratum."""
    options: dict[str, list[str]] = {
        "mining": ["cluster", "cluster_continuation", "forest", "mixed"],
        "forest": ["forest", "mixed", "cluster_continuation"],
        "water":  ["water", "forest", "urban"],
        "urban":  ["urban", "mixed", "forest"],
        "mixed":  ["mixed", "forest", "urban"],
        "edge":   ["forest", "cloud", "water"],
    }
    return rng.choice(options.get(stratum, ["mixed"]))


def _action_for_scene(hint: str, rng: random.Random) -> str:
    """Map a scene hint to a plausible neighbor action."""
    if hint in ("cluster", "cluster_continuation"):
        return rng.choices(["downlink", "flag_for_review"], weights=[0.7, 0.3])[0]
    if hint in ("forest", "water", "urban", "mixed", "cloud"):
        return rng.choices(["discard", "flag_for_review"], weights=[0.85, 0.15])[0]
    return "discard"


# --- Labeling prompt template --------------------------------------------

# Source of truth: docs/UNIFIED_VLM_PLAN.md Section 2 (validated 2026-05-04
# on 19 tiles). Keep this in sync with the plan.
LABELING_SYSTEM_PROMPT = """\
You are an on-orbit Earth-observation policy adjudicator. Given two views of \
a Sentinel-2 patch (natural-color RGB + SWIR false-color composite) and the \
per-pass operational context, decide which ONE of the five tools to call.

Tools:
- discard: Skip this tile. Default for forest, water, cloud, or undisturbed land.
- flag_for_review: Log as text only. Use for moderate-confidence detections \
(1-2 small candidates, ambiguous descriptions) worth recording but not bandwidth.
- request_higher_resolution: Ask for a higher-res recapture next pass. Use \
when you see a small candidate (1 tiny box, or dispersed sub-resolution \
candidates) that needs more pixels to confirm.
- request_neighbor_tile: Fetch an adjacent tile when a feature visibly \
continues off-frame. Requires a "direction" field, one of: \
"north" | "south" | "east" | "west".
- downlink_now: Use the precious downlink budget to send THIS tile's image \
to ground. Reserve for high-confidence detections: 2+ clear pits, OR a \
single large pit, AND active galamsey indicators (sediment plumes, exposed \
lateritic soil, turbid water). If in doubt, prefer flag_for_review.

Disambiguation rules:
- SWIR brightness in the absence of exposed-soil patterns or sediment plumes \
is more likely infrastructure (rooftops, asphalt, cleared paths), not mining. \
Urban tiles have very bright SWIR but no pits.
- Rectilinear field patterns are agriculture, not mining. Galamsey pits are \
amorphous and cluster near water.
- If imagery is partial (large no-data regions, swath-edge artifacts) or the \
tile is heavily cloud-occluded regardless of metadata cloud_cover value, \
prefer flag_for_review over downlink. Don't burn bandwidth on unreliable input.

Reply with EXACTLY ONE tool call in this JSON format and nothing else:
{
  "action": "discard|flag_for_review|request_higher_resolution|request_neighbor_tile|downlink_now",
  "reason": "...",
  "direction": "north|south|east|west"  (ONLY when action is request_neighbor_tile)
}
"""


def build_user_message(coord: TileCoord, context: PerPassContext) -> str:
    return (
        f"Tile {coord.coord_id} at lon={coord.lon:.4f}, lat={coord.lat:.4f}.\n"
        f"Stratum (sampling category): {coord.stratum}\n"
        f"Cloud cover (metadata, may be unreliable): {context.cloud_cover}\n"
        f"Captured: {context.captured_at}\n"
        f"Tile imagery issue: {context.tile_imagery_issue}\n"
        f"Pass budget: {context.budget_remaining_kb} of {context.budget_total_kb} KB remaining\n"
        f"Prior tiles downlinked this pass: {context.prior_tiles_downlinked}\n"
        f"Mission priors: {context.mission_priors}\n"
        f"Neighbor summary (structured):\n{json.dumps(context.neighbor_summary, indent=2)}\n"
    )


# --- Claude adjudicator --------------------------------------------------

ADJUDICATOR_MODEL = os.environ.get("ADJUDICATOR_MODEL", "claude-opus-4-7")

# Maps the labeling-prompt's tool-name vocabulary (what we ask Claude to emit)
# to the orchestrator's Action literal vocabulary (what _parse_json_response
# expects). The training JSONL is written using Action literals so it's
# directly compatible with the existing orchestrator schema.
_TOOL_NAME_TO_ACTION: dict[str, str] = {
    "discard": "discard",
    "flag_for_review": "flag",
    "request_higher_resolution": "request_hires",
    "request_neighbor_tile": "request_neighbor",
    "downlink_now": "downlink",
}

# JSON Schema enforced by the API via output_config.format. This guarantees
# Claude's response is parseable JSON conforming to our shape; no need for
# regex scraping or fallbacks. The enum on `action` constrains to the five
# tool names from LABELING_SYSTEM_PROMPT.
_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": list(_TOOL_NAME_TO_ACTION.keys()),
        },
        "reason": {"type": "string"},
        "direction": {
            "type": "string",
            "enum": ["north", "south", "east", "west"],
        },
    },
    "required": ["action", "reason"],
    "additionalProperties": False,
}


async def label_with_claude(
    rgb_bytes: bytes,
    swir_bytes: bytes,
    user_message: str,
) -> dict[str, Any]:
    """Call Claude as the labeling adjudicator on a single tile.

    Uses prompt caching on the system block — `LABELING_SYSTEM_PROMPT` is
    constant across all 1000 calls, so the cache write happens once on the
    first call and subsequent calls read the prefix at ~0.1× cost. Per-call
    cost is dominated by the two image tokens (~3000 tokens combined) plus
    a small per-tile context message. Output schema is enforced by the API
    via `output_config.format`.

    Returns:
        {
          "action": <Action literal: "discard" | "flag" | "request_hires" | "request_neighbor" | "downlink">,
          "tool_name": <original tool name Claude emitted>,
          "reason": <Claude's per-tile rationale>,
          "direction": <optional, only when action == "request_neighbor">,
          "raw_text": <Claude's full text response, for audit>,
          "usage": {input_tokens, cache_read_input_tokens, cache_creation_input_tokens, output_tokens},
        }

    Raises ``ValueError`` if Claude emits an action not in the labeling vocabulary
    (the schema enforcement should prevent this; the check is defensive).
    """
    from anthropic import AsyncAnthropic  # lazy import (large dep, only needed in this function)

    client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env

    rgb_b64 = base64.standard_b64encode(rgb_bytes).decode("ascii")
    swir_b64 = base64.standard_b64encode(swir_bytes).decode("ascii")

    response = await client.messages.create(
        model=ADJUDICATOR_MODEL,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": LABELING_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": rgb_b64,
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": swir_b64,
                        },
                    },
                    {"type": "text", "text": user_message},
                ],
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": _OUTPUT_SCHEMA,
            }
        },
    )

    raw_text = next(
        block.text for block in response.content if block.type == "text"
    )
    parsed = json.loads(raw_text)

    tool_name = parsed["action"]
    action = _TOOL_NAME_TO_ACTION.get(tool_name)
    if action is None:
        raise ValueError(
            f"Claude returned unknown action {tool_name!r}; "
            f"expected one of {list(_TOOL_NAME_TO_ACTION)}"
        )

    return {
        "action": action,
        "tool_name": tool_name,
        "reason": parsed["reason"],
        "direction": parsed.get("direction"),
        "raw_text": raw_text,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            "output_tokens": response.usage.output_tokens,
        },
    }


async def _label_one_cached_tile(tile_id: str) -> dict[str, Any]:
    """Standalone smoke test: load a cached tile from the orchestrator's
    benchmark cache, label it via Claude, return the result.

    Used to verify the labeling function works end-to-end on a single tile
    before scaling up. Uses the existing 6 Bibiani tiles + 13 diverse tiles
    we already have cached, so no new SimSat calls are needed.
    """
    cache_roots = [
        ROOT.parent / "orchestrator" / ".benchmark_cache" / tile_id,
        ROOT.parent / "orchestrator" / ".benchmark_cache" / "diverse_v1" / tile_id,
    ]
    tile_dir = next((r for r in cache_roots if r.exists()), None)
    if tile_dir is None:
        raise FileNotFoundError(
            f"tile {tile_id} not found in any cache. Searched: {cache_roots}"
        )

    rgb = (tile_dir / "rgb.png").read_bytes()
    swir = (tile_dir / "swir.png").read_bytes()
    meta = json.loads((tile_dir / "meta.json").read_text())

    coord = TileCoord(
        coord_id=tile_id,
        lon=meta["lon"],
        lat=meta["lat"],
        stratum=meta.get("label_hint", "mining").split("_")[0],  # crude derivation
        mission_priors=meta.get("label_hint", "(no mission prior)"),
    )
    context = synthesize_context(
        coord,
        cloud_cover=meta.get("cloud_cover"),
        captured_at=meta.get("captured_at"),
        tile_imagery_issue=None,
        rng=random.Random(42),
    )
    user_msg = build_user_message(coord, context)

    print(f"Calling Claude on tile {tile_id} (lon={coord.lon:.4f}, lat={coord.lat:.4f})...")
    result = await label_with_claude(rgb, swir, user_msg)

    print("\n--- Result ---")
    print(json.dumps(result, indent=2))

    usage = result["usage"]
    cache_read = usage["cache_read_input_tokens"]
    cache_write = usage["cache_creation_input_tokens"]
    fresh_input = usage["input_tokens"]
    output = usage["output_tokens"]
    total_input = cache_read + cache_write + fresh_input
    print(
        f"\n--- Usage ---\n"
        f"  total input: {total_input} tokens "
        f"(cache_read={cache_read}, cache_creation={cache_write}, fresh={fresh_input})\n"
        f"  output: {output} tokens"
    )
    if cache_read > 0:
        print(f"  cache hit: {cache_read} tokens served at ~0.1x cost")
    elif cache_write > 0:
        print(f"  cache miss (first call): {cache_write} tokens written at ~1.25x cost")

    return result


# --- JSONL writer --------------------------------------------------------

def labeled_to_jsonl_record(ex: LabeledExample) -> dict:
    """Convert one labeled example to a v9-format VLM-SFT JSONL record.

    The v9 prep scripts produced records of the form::

        {"messages": [
            {"role": "user", "content": [{"type": "image", "image": "..."},
                                          {"type": "image", "image": "..."},
                                          {"type": "text",  "text":  GROUNDING_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": "[{...bbox...}]"}]}
        ]}

    For the unified-VLM SFT we extend the assistant turn to be the JSON
    tool call instead of bounding boxes.
    """
    user_text = build_user_message(
        TileCoord(ex.coord_id, ex.lon, ex.lat, ex.stratum, ex.context.mission_priors),
        ex.context,
    )
    assistant_payload: dict[str, Any] = {"action": ex.action, "reason": ex.reason}
    if ex.direction:
        assistant_payload["direction"] = ex.direction

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ex.rgb_path},
                    {"type": "image", "image": ex.swir_path},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(assistant_payload)}],
            },
        ],
        # Sidecar metadata for stratification + audit.
        "_meta": {
            "coord_id": ex.coord_id,
            "lon": ex.lon,
            "lat": ex.lat,
            "stratum": ex.stratum,
            "context": asdict(ex.context),
            "adjudicator_model": ex.adjudicator_model,
            "raw_response": ex.raw_response,
        },
    }


# --- Modal entrypoint ----------------------------------------------------

@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    secrets=[modal.Secret.from_name("anthropic-api-key")],  # TODO: create this secret
    timeout=4 * 60 * 60,  # 4 hours, generous
)
async def generate_dataset(n: int = 1000, output: str = "unified_v1_train") -> dict:
    """Top-level Modal job: sample N coords, fetch + label + write JSONL.

    TODO: this is the integration point. Currently a no-op skeleton.
    """
    output_dir = Path(MODAL_MOUNT_POINT) / "data" / output
    output_dir.mkdir(parents=True, exist_ok=True)

    coords = sample_coordinates(n)
    logger.info("sampled %d coordinates across %d strata", len(coords), len(set(c.stratum for c in coords)))

    # TODO: for each coord:
    #   1. Fetch RGB + SWIR via SimSat (sequential, with retries)
    #   2. Save image bytes to a per-tile subdir
    #   3. Synthesize per-pass context
    #   4. Call label_with_claude
    #   5. Append LabeledExample to in-memory list
    # Then split by stratum + geography into train/val/test (800/100/100).
    # Then write JSONL files.
    # Then commit volume.

    raise NotImplementedError(
        "TODO: end-to-end loop. Components above are individually scaffolded."
    )


@app.local_entrypoint()
def main(n: int = 1000, output: str = "unified_v1_train") -> None:
    """Local entrypoint: ``uv run modal run scripts/prepare_unified_decision_dataset_modal.py``."""
    result = generate_dataset.remote(n=n, output=output)
    print(json.dumps(result, indent=2))


# --- Standalone smoke test ----------------------------------------------

if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1 and _sys.argv[1] == "label":
        # Live Claude API call. Requires ANTHROPIC_API_KEY in env and the
        # `anthropic` SDK installed. Will write to the cache (first call) and
        # incur a small cost (~$0.02 for one tile).
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not set in environment")
            _sys.exit(1)
        tile_id = _sys.argv[2] if len(_sys.argv) > 2 else "t000"
        asyncio.run(_label_one_cached_tile(tile_id))
    else:
        # Default smoke: sample 200 coordinates, verify stratum distribution.
        # No API calls, no cost.
        samples = sample_coordinates(n=200)
        counts: dict[str, int] = {}
        for c in samples:
            counts[c.stratum] = counts.get(c.stratum, 0) + 1
        print("Sample of 200 coordinates over Ghana:")
        for s, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            target = _STRATUM_WEIGHTS[s] * 200
            print(f"  {s:8s}: {n:>3d}  (target ~{target:>5.1f})")
        print("\nFirst 5 coords:")
        for c in samples[:5]:
            print(
                f"  {c.coord_id} ({c.lon:>+7.3f}, {c.lat:>+6.3f}) "
                f"stratum={c.stratum:8s} prior={c.mission_priors[:60]!r}"
            )
        print("\n[Hint] To smoke-test the Claude labeling on a real tile:")
        print("  ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/prepare_unified_decision_dataset_modal.py label t000")
