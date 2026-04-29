"""Prepare a tiny 20-sample slice of SmallMinesDS on a Modal volume.

SmallMinesDS is distributed as a single ZIP file (`SmallMinesDS.zip`, ~1.65 GB)
containing a `YEAR/IMAGE/*.tif` and `YEAR/MASK/*.tif` folder hierarchy, plus
train/test split CSVs. It is NOT published as a HuggingFace parquet dataset -
the load_dataset() API only returns metadata columns (patch_name, class_percentage,
bin, split), not the actual pixel arrays. We have to download the zip, extract
it, and load individual GeoTIFFs with rasterio.

Canonical band order (from the dataset README):
  Index 0-9:   Sentinel-2 L2A [blue, green, red, rededge1, rededge2, rededge3,
                               nir, rededge4, swir1, swir2]
  Index 10-11: Sentinel-1 RTC [vv, vh]
  Index 12:    Copernicus DEM

For our SWIR2-SWIR1-NIR false-color composite:
  SWIR2 = index 9
  SWIR1 = index 8
  NIR   = index 6

Usage:
    uv run modal run scripts/prepare_tiny_dataset_modal.py
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"
MODAL_DATA_DIR = f"{MODAL_MOUNT_POINT}/data"

# Band indices within SmallMinesDS's 13-channel stack (confirmed from README)
BAND_SWIR2 = 9
BAND_SWIR1 = 8
BAND_NIR = 6

app = modal.App("galamsey-tiny-prep")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libexpat1")
    .pip_install(
        "huggingface_hub",
        "numpy>=2.0",
        "pillow>=11.0",
        "rasterio>=1.3",
    )
)


@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def prepare() -> dict:
    import json
    import zipfile
    from pathlib import Path

    import numpy as np
    import rasterio
    from huggingface_hub import hf_hub_download
    from PIL import Image

    data_dir = Path(MODAL_DATA_DIR)
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GalamseyWatch, tiny dataset prep on Modal")
    print("=" * 60)

    print("\nDownloading SmallMinesDS.zip from HuggingFace...")
    zip_path = hf_hub_download(
        repo_id="ellaampy/SmallMinesDS",
        filename="SmallMinesDS.zip",
        repo_type="dataset",
    )
    print(f"  Downloaded to: {zip_path}")

    def patch_id(entry_name: str) -> tuple[str, str]:
        """Extract (normalized_stem, year) from a TIFF entry path.

        Images and masks share the same suffix after their leading prefix, so we
        strip common prefixes (IMG_, MSK_, MASK_, etc.) and return the rest as
        the matching key. Year is extracted from the path (/2016/ or /2022/).
        """
        stem = Path(entry_name).stem
        for prefix in ("IMG_", "MSK_", "MASK_", "img_", "msk_", "mask_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        year = "unknown"
        for y in ("2016", "2022"):
            if f"/{y}/" in entry_name:
                year = y
                break
        return stem, year

    # Open the zip and scan for IMAGE/MASK pairs (extract only what we need)
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        print(f"\nZip contains {len(all_names)} entries")

        image_tiffs = sorted(
            n for n in all_names if "IMAGE" in n and n.lower().endswith((".tif", ".tiff"))
        )
        mask_tiffs = sorted(
            n for n in all_names if "MASK" in n and n.lower().endswith((".tif", ".tiff"))
        )

        print(f"\nFound {len(image_tiffs)} IMAGE tiffs and {len(mask_tiffs)} MASK tiffs")
        if image_tiffs:
            print(f"First image: {image_tiffs[0]}")
        if mask_tiffs:
            print(f"First mask:  {mask_tiffs[0]}")

        if not image_tiffs:
            raise RuntimeError(
                "No IMAGE tiffs found in the zip, layout may have changed."
            )

        # Build lookup by (normalized_stem, year) so we can pair images and masks
        # regardless of the leading prefix convention.
        mask_by_id = {patch_id(n): n for n in mask_tiffs}
        print(f"\nMask lookup has {len(mask_by_id)} unique (stem, year) keys.")

        def percentile_stretch(band: np.ndarray) -> np.ndarray:
            lo, hi = np.percentile(band, [2, 98])
            clipped = np.clip(band, lo, hi)
            span = max(float(hi - lo), 1e-9)
            return ((clipped - lo) / span * 255.0).astype(np.uint8)

        # Process the first 20 image/mask pairs
        samples = []
        skipped_no_mask = 0
        for i, image_name in enumerate(image_tiffs):
            if len(samples) >= 20:
                break

            key = patch_id(image_name)
            if key not in mask_by_id:
                skipped_no_mask += 1
                if skipped_no_mask <= 3:
                    print(f"  [skip] no mask for image {image_name} (key={key})")
                continue
            mask_name = mask_by_id[key]

            # Extract the image TIFF to a temp location and read with rasterio
            with zf.open(image_name) as fp, rasterio.MemoryFile(fp.read()) as memfile:
                with memfile.open() as src:
                    bands = src.read()  # shape (13, H, W)

            with zf.open(mask_name) as fp, rasterio.MemoryFile(fp.read()) as memfile:
                with memfile.open() as src:
                    mask = src.read(1)  # first and only band

            if i < 3:
                print(
                    f"  sample {i}: bands.shape={bands.shape} dtype={bands.dtype} "
                    f"mask.shape={mask.shape} mining_pixels={int((mask > 0).sum())}"
                )

            # Compose SWIR2-SWIR1-NIR and stretch
            r = percentile_stretch(bands[BAND_SWIR2])
            g = percentile_stretch(bands[BAND_SWIR1])
            b = percentile_stretch(bands[BAND_NIR])
            rgb = np.stack([r, g, b], axis=-1)

            png_name = f"tiny_{len(samples):03d}.png"
            Image.fromarray(rgb).save(images_dir / png_name)

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

    jsonl_path = data_dir / "galamsey_tiny_train.jsonl"
    with jsonl_path.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    n_positives = sum(
        1 for s in samples
        if "No mining" not in s["messages"][1]["content"][0]["text"]
    )
    n_negatives = len(samples) - n_positives

    print(f"\nWrote {len(samples)} samples to {jsonl_path}")
    print(f"  Positives: {n_positives}")
    print(f"  Negatives: {n_negatives}")
    print(f"Wrote {len(samples)} PNGs to {images_dir}")

    volume.commit()

    return {
        "n_samples": len(samples),
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "jsonl_path": str(jsonl_path),
    }


@app.local_entrypoint()
def main() -> None:
    print("Submitting tiny dataset prep to Modal...")
    result = prepare.remote()
    print(f"\nPrep complete: {result}")
    print("\nNext step: uv run leap-finetune configs/galamsey_tiny.yaml")
