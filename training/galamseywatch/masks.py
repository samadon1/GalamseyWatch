"""Binary mask analysis and rich description generation.

`analyze_mask` runs connected-component labelling to extract per-pit geometry
from a SmallMinesDS binary mask. `generate_description` turns that geometry
into a varied natural-language sentence.

The v2 description generator uses real mask features (pit count, positions,
size distribution, adjacency) to produce thousands of distinct descriptions
per dataset, unlike the v1 generator which had only 2 base templates that
the model memorized trivially. See ARCHITECTURE.md §3.7.h for the overfitting
analysis that motivated this change.
"""

from __future__ import annotations

import random
from typing import TypedDict

import numpy as np
from scipy.ndimage import label

from galamseywatch.constants import HECTARES_PER_PIXEL


class PitStats(TypedDict):
    pit_count: int
    areas_ha: list[float]
    bboxes_normalized: list[list[float]]  # each [x1, y1, x2, y2] in 0-1
    total_area_ha: float


def analyze_mask(
    mask: np.ndarray,
    hectares_per_pixel: float = HECTARES_PER_PIXEL,
) -> PitStats:
    """Run connected-component labelling and extract per-pit geometry.

    Args:
        mask: 2D array of shape (H, W). Non-zero values are "mining" pixels.
        hectares_per_pixel: Ground area per pixel. Defaults to 0.01 ha for
            Sentinel-2's 10 m GSD.
    """
    binary = mask > 0
    labeled, n_pits = label(binary)

    if n_pits == 0:
        return {
            "pit_count": 0,
            "areas_ha": [],
            "bboxes_normalized": [],
            "total_area_ha": 0.0,
        }

    h, w = binary.shape
    areas_ha: list[float] = []
    bboxes_normalized: list[list[float]] = []

    for pit_id in range(1, n_pits + 1):
        pit_mask = labeled == pit_id
        ys, xs = np.where(pit_mask)
        if ys.size == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        bboxes_normalized.append([
            round(x1 / w, 4),
            round(y1 / h, 4),
            round(x2 / w, 4),
            round(y2 / h, 4),
        ])
        areas_ha.append(float(pit_mask.sum()) * hectares_per_pixel)

    return {
        "pit_count": len(areas_ha),
        "areas_ha": areas_ha,
        "bboxes_normalized": bboxes_normalized,
        "total_area_ha": float(sum(areas_ha)),
    }


# ---------------------------------------------------------------------------
# Description generation, v2 rich template bank
# ---------------------------------------------------------------------------

# Spatial region labels derived from a normalized (x, y) centroid.
# Keeping this as quadrants + center + edges gives us 9 distinct region names
# without being so precise that every image looks unique (we want *variety*,
# not random noise).


def _centroid_region(x: float, y: float) -> str:
    """Map a normalized (x, y) centroid in [0, 1]^2 to a qualitative region name."""
    # Horizontal band
    if x < 0.33:
        h = "left"
    elif x > 0.67:
        h = "right"
    else:
        h = "center"
    # Vertical band
    if y < 0.33:
        v = "upper"
    elif y > 0.67:
        v = "lower"
    else:
        v = "middle"

    # Special case: middle + center = "center of the scene"
    if h == "center" and v == "middle":
        return "center of the scene"
    if h == "center":
        return f"{v} portion of the scene"
    if v == "middle":
        return f"{h} side of the scene"
    return f"{v}-{h} quadrant"


def _bbox_centroid(bbox: list[float]) -> tuple[float, float]:
    """Return the (x, y) centroid of a normalized [x1, y1, x2, y2] bbox."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_size_category(bbox: list[float]) -> str:
    """Categorize a bbox by its area relative to the 128x128 patch."""
    x1, y1, x2, y2 = bbox
    area_fraction = max((x2 - x1) * (y2 - y1), 0.0)
    if area_fraction < 0.01:
        return "very small"
    if area_fraction < 0.04:
        return "small"
    if area_fraction < 0.15:
        return "moderate"
    if area_fraction < 0.40:
        return "large"
    return "very large"


def _size_distribution_phrase(areas_ha: list[float]) -> str:
    """Describe how pits are distributed in size."""
    if len(areas_ha) == 1:
        return ""
    mean_a = sum(areas_ha) / len(areas_ha)
    max_a = max(areas_ha)
    min_a = min(areas_ha)
    spread = (max_a - min_a) / max(mean_a, 1e-9)
    if spread < 0.4:
        return "with roughly uniform pit sizes"
    if spread < 1.5:
        return "with mixed pit sizes"
    return "dominated by one or two large pits among smaller satellite pits"


def _adjacency_phrase(bboxes: list[list[float]]) -> str:
    """Describe whether pits are clustered or scattered across the patch."""
    if len(bboxes) < 2:
        return ""
    centroids = [_bbox_centroid(b) for b in bboxes]
    xs = [c[0] for c in centroids]
    ys = [c[1] for c in centroids]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    total_spread = spread_x + spread_y
    if total_spread < 0.3:
        return "clustered together in a tight group"
    if total_spread < 0.7:
        return "spread across a connected area"
    return "scattered across the scene"


# Template banks, each sentence is a small DSL with {placeholders} that get
# slot-filled from mask statistics. Selection is randomized by a seeded RNG
# so the same input always produces the same output (reproducibility) while
# different inputs get different templates (variety).

NEGATIVE_TEMPLATES: list[str] = [
    "No signs of illegal mining activity visible. Vegetation appears intact.",
    "The scene shows intact vegetation with no evidence of excavation or exposed soil.",
    "No mining activity is visible in this tile. Land cover appears undisturbed.",
    "Vegetation and terrain look healthy with no indicators of small-scale gold mining.",
    "This patch shows no signs of galamsey. The land cover is consistent with undisturbed forest or farmland.",
    "No excavation pits, bare soil, or sediment plumes observed. The area appears undisturbed.",
    "Clean vegetation cover across the tile. No disturbance patterns suggesting mining.",
    "No visible signs of artisanal gold mining. Surface appears natural.",
    "Intact land cover. No excavation features or sediment signatures visible.",
    "The tile shows no mining-related disturbance. Vegetation and soil look consistent with the surrounding landscape.",
]

SINGLE_PIT_TEMPLATES: list[str] = [
    "A single {size} excavation pit visible in the {region}, covering approximately {area_ha:.2f} hectares.",
    "One {size} mining pit in the {region}. Affected area: ~{area_ha:.2f} hectares.",
    "Isolated {size} excavation feature in the {region} of the tile, about {area_ha:.2f} hectares in extent.",
    "Single pit ({size}) located in the {region}, {area_ha:.2f} ha of disturbed ground.",
    "One excavation pit visible, {size} in size, in the {region}. Approximately {area_ha:.2f} hectares affected.",
    "A {size} mining site is evident in the {region}. Disturbance covers roughly {area_ha:.2f} hectares.",
    "Solitary mining pit in the {region} of the patch, {area_ha:.2f} hectares of exposed subsurface.",
]

MULTI_PIT_TEMPLATES: list[str] = [
    "{count} excavation pits {adjacency}, with the largest in the {primary_region}. Total affected area approximately {total_area_ha:.2f} hectares {size_phrase}.",
    "{count} mining pits visible, {adjacency}. The largest pit is in the {primary_region}. Combined area: {total_area_ha:.2f} hectares.",
    "Multiple excavation features: {count} pits {adjacency}. Largest disturbance in the {primary_region}. Total ~{total_area_ha:.2f} hectares {size_phrase}.",
    "{count} active mining pits {adjacency}. The dominant pit is in the {primary_region}. Approximately {total_area_ha:.2f} hectares of disturbed ground {size_phrase}.",
    "The scene contains {count} excavation pits {adjacency}. Largest feature in the {primary_region}. Total affected area: {total_area_ha:.2f} hectares.",
    "{count} pit sites {adjacency}. The biggest pit sits in the {primary_region}. {total_area_ha:.2f} hectares of exposed or disturbed surface {size_phrase}.",
    "{count} mining pits identified, {adjacency}. The largest lies in the {primary_region}. {total_area_ha:.2f} total hectares affected.",
]


def generate_description(
    stats: PitStats,
    seed: int | None = None,
) -> str:
    """Generate a varied, channel-agnostic description from mask statistics.

    Variety guarantee: consecutive calls with different `seed` values (or no
    explicit seed, relying on the module-level `random` state) produce
    different template selections. This is the correct behavior because
    stats-identical patches (e.g., all empty masks, or two different 1-pit
    patches that happen to have the same total area) should still get
    *different* descriptions, otherwise the training set collapses large
    swaths of patches to identical text, which defeats the whole point of
    having multiple templates and encourages the model to overfit.

    Reproducibility: the caller is responsible for setting `random.seed(...)`
    at script start if reproducible JSONL generation is required. For
    unit tests, pass an explicit `seed` parameter.

    NB: an earlier version of this function used `seed = hash((pit_count,
    total_area_ha))` as a default, which silently collapsed all empty masks
    to a single description string and reduced the effective variety of the
    negative-template pool from 10 to 1. See ARCHITECTURE.md §3.4.7 and the
    4-connectivity ablation in the same session where this was caught.

    Args:
        stats: Output of `analyze_mask`.
        seed: Optional explicit RNG seed, when set, the function becomes
            deterministic per that seed. When None, uses the module-level
            `random` state (caller-controlled).
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    pit_count = stats["pit_count"]

    if pit_count == 0:
        return rng.choice(NEGATIVE_TEMPLATES)

    if pit_count == 1:
        bbox = stats["bboxes_normalized"][0]
        cx, cy = _bbox_centroid(bbox)
        region = _centroid_region(cx, cy)
        size = _bbox_size_category(bbox)
        area_ha = stats["areas_ha"][0]
        template = rng.choice(SINGLE_PIT_TEMPLATES)
        return template.format(size=size, region=region, area_ha=area_ha)

    # Multi-pit, use the largest pit as the "primary" location
    bboxes = stats["bboxes_normalized"]
    areas = stats["areas_ha"]
    largest_idx = max(range(len(areas)), key=lambda i: areas[i])
    primary_bbox = bboxes[largest_idx]
    primary_cx, primary_cy = _bbox_centroid(primary_bbox)
    primary_region = _centroid_region(primary_cx, primary_cy)

    adjacency = _adjacency_phrase(bboxes)
    size_phrase = _size_distribution_phrase(areas)

    template = rng.choice(MULTI_PIT_TEMPLATES)
    sentence = template.format(
        count=pit_count,
        adjacency=adjacency,
        primary_region=primary_region,
        total_area_ha=stats["total_area_ha"],
        size_phrase=size_phrase,
    )

    # Clean up any double-spaces from empty slot fills
    while "  " in sentence:
        sentence = sentence.replace("  ", " ")
    # Clean up " ." from missing trailing phrases
    sentence = sentence.replace(" .", ".").replace(" ,", ",")
    return sentence
