"""False-color composite rendering, per ARCHITECTURE.md §2.

Primary composite is SWIR2-SWIR1-NIR (B12, B11, B8A) rendered as a 3-channel
uint8 PNG with per-band 2-98 percentile stretch. That stretch is the single
highest-impact preprocessing step, skipping it produces a low-contrast gray
smear regardless of how well the model is trained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def percentile_stretch(
    band: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> np.ndarray:
    """Stretch a single Sentinel-2 band to fit [0, 255] uint8.

    Uses the 2nd and 98th percentiles of *this specific patch* (not global
    dataset statistics) as the clip range. Local stretching preserves contrast
    for every tile regardless of the scene's overall brightness.

    Args:
        band: 2D float array of raw reflectance values for a single band.
        low_pct: Lower percentile for the stretch range. Defaults to 2.0.
        high_pct: Upper percentile. Defaults to 98.0.

    Returns:
        2D uint8 array with values in [0, 255].
    """
    lo, hi = np.percentile(band, [low_pct, high_pct])
    clipped = np.clip(band, lo, hi)
    span = max(float(hi - lo), 1e-9)
    return ((clipped - lo) / span * 255.0).astype(np.uint8)


def compose_swir_false_color(
    bands_stack: np.ndarray,
    band_swir2: int,
    band_swir1: int,
    band_nir: int,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> np.ndarray:
    """Render a SWIR2-SWIR1-NIR false-color composite as HxWx3 uint8.

    Expects `bands_stack` in (C, H, W) order. If SmallMinesDS ships its arrays
    in (H, W, C) order instead, callers should transpose before calling.

    Args:
        bands_stack: 3D float array of shape (C, H, W) containing raw S2 bands.
        band_swir2: Index of B12 within `bands_stack`.
        band_swir1: Index of B11 within `bands_stack`.
        band_nir: Index of B8A within `bands_stack`.
        low_pct / high_pct: Percentile stretch bounds (see `percentile_stretch`).

    Returns:
        HxWx3 uint8 array suitable for `PIL.Image.fromarray`.
    """
    r = percentile_stretch(bands_stack[band_swir2], low_pct, high_pct)
    g = percentile_stretch(bands_stack[band_swir1], low_pct, high_pct)
    b = percentile_stretch(bands_stack[band_nir], low_pct, high_pct)
    return np.stack([r, g, b], axis=-1)


def encode_png(rgb_uint8: np.ndarray, out_path: str | Path) -> Path:
    """Save an HxWx3 uint8 array as a PNG.

    Creates parent directories if they don't exist. Returns the final path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_uint8).save(out_path, format="PNG")
    return out_path
