"""Prepare a tiny 20-sample slice of SmallMinesDS locally, then upload to Modal.

Runs on the user's Mac (CPU only, needs only HF auth which is already set up).
Streams a handful of samples from HuggingFace, renders them as simple 3-channel
PNGs, writes a JSONL conversation file, and uploads both to the `galamsey`
Modal volume via the `modal volume put` CLI.

This is intentionally the DUMBEST possible preparation, first 3 channels, no
SWIR composite tuning, trivial yes/no descriptions. Purpose is to smoke-test
the full leap-finetune + Modal training pipeline, not to produce anything
resembling a useful model.

Usage:
    cd training/
    uv run python scripts/prepare_tiny_dataset.py

After it completes, run:
    uv run leap-finetune configs/galamsey_tiny.yaml
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

# Local staging path. The Modal volume gets everything under `/galamsey/data/`,
# so we mirror that structure locally: tiny files live under data/tiny/ and are
# uploaded to /galamsey/data/tiny/ on the volume.
LOCAL_DATA_DIR = Path("./data/tiny")
LOCAL_IMAGES_DIR = LOCAL_DATA_DIR / "images"

MODAL_VOLUME_NAME = "galamsey"
MODAL_REMOTE_PATH = "/data/tiny"

N_SAMPLES = 20


def percentile_stretch(band: np.ndarray) -> np.ndarray:
    """2-98 percentile stretch, clip, normalize to uint8."""
    lo, hi = np.percentile(band, [2, 98])
    clipped = np.clip(band, lo, hi)
    span = max(float(hi - lo), 1e-9)
    return ((clipped - lo) / span * 255.0).astype(np.uint8)


def detect_keys(sample: dict) -> tuple[str, str]:
    keys = list(sample.keys())
    bands_key = next(
        (k for k in keys if any(term in k.lower() for term in ("image", "band", "patch", "raster"))),
        keys[0],
    )
    mask_key = next(
        (k for k in keys if any(term in k.lower() for term in ("mask", "label", "target"))),
        keys[-1],
    )
    return bands_key, mask_key


def main() -> None:
    # Clean any previous staging so re-runs don't mix stale files
    if LOCAL_DATA_DIR.exists():
        shutil.rmtree(LOCAL_DATA_DIR)
    LOCAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GalamseyWatch, tiny dataset prep (local)")
    print("=" * 60)
    print("\nLoading SmallMinesDS in streaming mode (~first shard only)...")
    ds = load_dataset("ellaampy/SmallMinesDS", split="train", streaming=True)

    # Inspect first sample, detect schema
    first_sample = next(iter(ds))
    print("\n=== Sample structure ===")
    for key, value in first_sample.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", type(value).__name__)
        print(f"  {key}: shape={shape}, dtype={dtype}")

    bands_key, mask_key = detect_keys(first_sample)
    print(f"\nUsing bands_key={bands_key!r}, mask_key={mask_key!r}")

    # Re-stream (the first() call above consumed one iterator item)
    ds = load_dataset("ellaampy/SmallMinesDS", split="train", streaming=True)

    samples = []
    for i, sample in enumerate(ds):
        if i >= N_SAMPLES:
            break

        bands = np.asarray(sample[bands_key])

        # Normalize channel axis: we want (C, H, W)
        if bands.ndim == 3 and bands.shape[0] > bands.shape[-1]:
            # First axis is already the channel dimension (more channels than spatial)
            pass
        elif bands.ndim == 3:
            bands = np.transpose(bands, (2, 0, 1))

        # Dumbest possible "composite": first three channels with percentile stretch
        r = percentile_stretch(bands[0])
        g = percentile_stretch(bands[1])
        b = percentile_stretch(bands[2])
        rgb = np.stack([r, g, b], axis=-1)

        png_name = f"tiny_{i:03d}.png"
        Image.fromarray(rgb).save(LOCAL_IMAGES_DIR / png_name)

        mask = np.asarray(sample[mask_key])
        has_mining = bool((mask > 0).sum() > 0)
        description = (
            "Mining activity visible in this Sentinel-2 patch."
            if has_mining
            else "No mining activity visible in this Sentinel-2 patch."
        )

        samples.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": png_name},
                            {"type": "text", "text": "Describe this Sentinel-2 patch."},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": description}],
                    },
                ]
            }
        )

    jsonl_path = LOCAL_DATA_DIR / "galamsey_tiny_train.jsonl"
    with jsonl_path.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\nWrote {len(samples)} samples to {jsonl_path}")
    n_positives = sum(1 for s in samples if "visible in" in s["messages"][1]["content"][0]["text"] and "No" not in s["messages"][1]["content"][0]["text"])
    n_negatives = len(samples) - n_positives
    print(f"  Positives: {n_positives}")
    print(f"  Negatives: {n_negatives}")

    # Upload to the Modal volume
    print(f"\nUploading to Modal volume '{MODAL_VOLUME_NAME}' at {MODAL_REMOTE_PATH}/...")
    result = subprocess.run(
        [
            sys.executable, "-m", "modal", "volume", "put",
            MODAL_VOLUME_NAME,
            str(LOCAL_DATA_DIR),
            MODAL_REMOTE_PATH,
            "--force",
        ],
        check=False,
    )
    if result.returncode != 0:
        print(f"\nUpload failed with exit code {result.returncode}.")
        sys.exit(result.returncode)

    print("\nUpload complete.")
    print(f"Next step: uv run leap-finetune configs/galamsey_tiny.yaml")


if __name__ == "__main__":
    main()
