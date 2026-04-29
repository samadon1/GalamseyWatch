"""Full-dataset SmallMinesDS → leap-finetune VLM SFT converter (Modal-resident).

Upgraded variant of `prepare_tiny_dataset_modal.py` that produces the real
training data for the v1 run. Key differences from the tiny prep:

  1. **Full dataset.** All 4,270 SmallMinesDS patches (2,175 per year), not 20.
  2. **Respects the published train/test splits** from `data_splits/train_test_splits_2016.csv`
     and `data_splits/train_test_splits_2022.csv`, so our eval set is the one
     the dataset authors intend, not a home-rolled slice.
  3. **Stratified negative sampling.** Keeps all positive (mining) patches and
     randomly downsamples negatives to 2× the positive count. Rationale: positives
     are precious (rare and informative), negatives are abundant and redundant.
     This prevents the loss from being dominated by trivial "no mining" predictions.
  4. **Real preprocessing.** Uses the galamseywatch primitives' logic inline
     (same as the primitives, but self-contained so Modal's container doesn't
     need to install the local package):
       - SWIR2-SWIR1-NIR composite from bands (9, 8, 6)
       - Per-band 2–98 percentile stretch
       - Connected-component mask analysis → pit count, bboxes, areas in hectares
       - Template description generation with real pit counts and areas
  5. **Four output JSONL files**, description + grounding × train + eval.
  6. **Progress reporting.** Scan pass takes ~10 min on Modal CPU; processing
     pass ~5 min. Prints progress every 500 patches so we know it's alive.

Usage:
    uv run modal run scripts/prepare_v1_dataset_modal.py
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"
MODAL_V1_DATA_DIR = f"{MODAL_MOUNT_POINT}/data/v1"

# Canonical band indices from SmallMinesDS README (confirmed in ARCHITECTURE.md §3.7.b)
BAND_SWIR2 = 9
BAND_SWIR1 = 8
BAND_NIR = 6

# Sentinel-2 at 10 m/pixel → 100 m² per pixel = 0.01 hectares
HECTARES_PER_PIXEL = 0.01

# Stratified sampling ratio: training set = N_positives + (NEG_RATIO × N_positives)
# 2× gives the model enough "not mining" examples to learn the contrast without
# drowning the positives in noise.
NEG_RATIO = 2

app = modal.App("galamsey-v1-prep")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libexpat1")
    .pip_install(
        "huggingface_hub",
        "numpy>=2.0",
        "pillow>=11.0",
        "rasterio>=1.3",
        "scipy>=1.13",
    )
)


@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    timeout=5400,  # 90 min, generous for first-pass scan + processing
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def prepare() -> dict:
    import csv
    import json
    import random
    import zipfile
    from pathlib import Path

    import numpy as np
    import rasterio
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from scipy.ndimage import label

    # ---------- inlined galamseywatch primitives ----------
    # Duplicated here so the Modal container doesn't need to install our local
    # package. These must stay in sync with galamseywatch/composites.py and
    # galamseywatch/masks.py, if you change one, change both.

    DESCRIPTION_PROMPT = (
        "You are analyzing a Sentinel-2 SWIR false-color composite (SWIR2, SWIR1, NIR) "
        "of southwestern Ghana. Describe any signs of illegal small-scale gold mining "
        "(galamsey) activity: exposed subsurface soil, excavation pits, sediment plumes, "
        "vegetation loss, and proximity to water bodies. If no mining is visible, say so."
    )
    GROUNDING_PROMPT = (
        "Inspect the image and detect any illegal small-scale gold mining pits. "
        'Provide result as a valid JSON: [{"label": str, "bbox": [x1,y1,x2,y2]}, ...]. '
        "Coordinates must be normalized to 0-1. If no pits are visible, return []."
    )

    def percentile_stretch(band: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(band, [2, 98])
        clipped = np.clip(band, lo, hi)
        span = max(float(hi - lo), 1e-9)
        return ((clipped - lo) / span * 255.0).astype(np.uint8)

    def compose_swir_false_color(bands: np.ndarray) -> np.ndarray:
        r = percentile_stretch(bands[BAND_SWIR2])
        g = percentile_stretch(bands[BAND_SWIR1])
        b = percentile_stretch(bands[BAND_NIR])
        return np.stack([r, g, b], axis=-1)

    def analyze_mask(mask: np.ndarray) -> dict:
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
        bboxes: list[list[float]] = []
        areas: list[float] = []
        for pit_id in range(1, n_pits + 1):
            pit_mask = labeled == pit_id
            ys, xs = np.where(pit_mask)
            if ys.size == 0:
                continue
            bboxes.append([
                round(int(xs.min()) / w, 4),
                round(int(ys.min()) / h, 4),
                round((int(xs.max()) + 1) / w, 4),
                round((int(ys.max()) + 1) / h, 4),
            ])
            areas.append(float(pit_mask.sum()) * HECTARES_PER_PIXEL)
        return {
            "pit_count": len(areas),
            "areas_ha": areas,
            "bboxes_normalized": bboxes,
            "total_area_ha": float(sum(areas)),
        }

    def generate_description(stats: dict) -> str:
        if stats["pit_count"] == 0:
            return (
                "No signs of illegal mining activity visible. "
                "Vegetation appears intact."
            )
        pit_word = "pit" if stats["pit_count"] == 1 else "pits"
        return (
            f"{stats['pit_count']} active excavation {pit_word} visible, "
            f"total affected area approximately {stats['total_area_ha']:.2f} hectares."
        )

    def make_vlm_message(
        image_filename: str, user_text: str, assistant_text: str
    ) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_filename},
                        {"type": "text", "text": user_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            ]
        }

    def patch_id(entry_name: str) -> tuple[int, str]:
        """Extract (numeric_patch_id, year) from any file path or CSV entry name.

        Works for all three observed formats:
          - `SmallMinesDS/2016/IMAGE/IMG_GH_0001_2016.tif`  → (1, "2016")
          - `SmallMinesDS/2016/MASK/MASK_GH_0001_2016.tif`  → (1, "2016")
          - `MASK_0001_2016.tif` (CSV patch_name)            → (1, "2016")

        Strategy: take the last two underscore-separated parts of the stem.
        The last part is the year; the second-to-last is the patch ID.
        This is robust to prefix/country-code variations.
        """
        stem = Path(entry_name).stem
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                patch_num = int(parts[-2])
                year = parts[-1]
                if year in ("2016", "2022"):
                    return (patch_num, year)
            except ValueError:
                pass
        return (-1, "unknown")

    # ---------- main work ----------

    data_dir = Path(MODAL_V1_DATA_DIR)
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GalamseyWatch, v1 full dataset prep on Modal")
    print("=" * 60)

    print("\n[1/4] Downloading SmallMinesDS artifacts from HuggingFace...")
    zip_path = hf_hub_download(
        repo_id="ellaampy/SmallMinesDS",
        filename="SmallMinesDS.zip",
        repo_type="dataset",
    )
    split_csv_paths: dict[str, str] = {}
    for year in ("2016", "2022"):
        split_csv_paths[year] = hf_hub_download(
            repo_id="ellaampy/SmallMinesDS",
            filename=f"data_splits/train_test_splits_{year}.csv",
            repo_type="dataset",
        )
    print(f"  zip: {zip_path}")
    print(f"  splits: {split_csv_paths}")

    # ---------- parse split CSVs ----------
    # Expected columns (from inspection during prior runs): patch_name,
    # class_percentage, bin, split. We defensively look up each column by a
    # list of aliases so the script survives minor schema drift.
    split_assignment: dict[tuple[int, str], str] = {}
    class_percentage_by_id: dict[tuple[int, str], float] = {}

    print("\n[2/4] Parsing split CSVs...")
    for year, csv_path in split_csv_paths.items():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            print(f"  {year} columns: {reader.fieldnames}")
            row_count = 0
            first_row_sample = None
            for row in reader:
                if first_row_sample is None:
                    first_row_sample = dict(row)

                pname = (
                    row.get("patch_name")
                    or row.get("name")
                    or row.get("patch")
                    or next(iter(row.values()), "")
                )
                split = (row.get("split") or row.get("set") or "train").strip().lower()
                try:
                    cp = float(row.get("class_percentage", 0) or 0)
                except (ValueError, TypeError):
                    cp = 0.0

                # Use the same patch_id function for both CSV entries and zip entries
                #, the (numeric_id, year) tuple is robust to prefix differences.
                key = patch_id(pname)
                if key[0] == -1:
                    continue  # unparseable row, skip
                split_assignment[key] = (
                    split if split in ("train", "test", "val") else "train"
                )
                class_percentage_by_id[key] = cp
                row_count += 1
            print(f"    first row sample: {first_row_sample}")
            print(f"    loaded {row_count} rows for {year}")

    print(f"\n  total split entries: {len(split_assignment)}")
    splits_seen = {s: sum(1 for v in split_assignment.values() if v == s) for s in ("train", "test", "val")}
    n_positive = sum(1 for cp in class_percentage_by_id.values() if cp > 0)
    print(f"  split distribution: {splits_seen}")
    print(f"  csv positives (class_percentage > 0): {n_positive}")

    # ---------- scan the zip and pair images with masks ----------
    print("\n[3/4] Scanning zip for image/mask pairs...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        image_tiffs = sorted(
            n for n in all_names
            if "IMAGE" in n and n.lower().endswith((".tif", ".tiff"))
        )
        mask_tiffs = sorted(
            n for n in all_names
            if "MASK" in n and n.lower().endswith((".tif", ".tiff"))
        )
        print(f"  {len(image_tiffs)} images, {len(mask_tiffs)} masks")

        mask_by_id = {patch_id(n): n for n in mask_tiffs}

        # Build the full roster of (key, image_name, mask_name, is_positive, split)
        # using the CSV's class_percentage when available, otherwise falling back
        # to "unknown" positivity (will default to negative).
        roster: list[dict] = []
        skipped_no_mask = 0
        for image_name in image_tiffs:
            key = patch_id(image_name)
            if key not in mask_by_id:
                skipped_no_mask += 1
                continue
            cp = class_percentage_by_id.get(key, 0.0)
            is_positive = cp > 0.0
            split = split_assignment.get(key, "train")
            roster.append(
                {
                    "key": key,
                    "image_name": image_name,
                    "mask_name": mask_by_id[key],
                    "is_positive": is_positive,
                    "class_percentage": cp,
                    "split": split,
                }
            )

        print(f"  roster: {len(roster)} patches (skipped {skipped_no_mask} unpaired)")
        n_train_pos = sum(1 for r in roster if r["split"] == "train" and r["is_positive"])
        n_train_neg = sum(1 for r in roster if r["split"] == "train" and not r["is_positive"])
        n_test_pos = sum(1 for r in roster if r["split"] == "test" and r["is_positive"])
        n_test_neg = sum(1 for r in roster if r["split"] == "test" and not r["is_positive"])
        print(
            f"  raw counts, train: {n_train_pos} pos / {n_train_neg} neg  "
            f"test: {n_test_pos} pos / {n_test_neg} neg"
        )

        # ---------- stratified sampling ----------
        rng = random.Random(42)
        train_pos = [r for r in roster if r["split"] == "train" and r["is_positive"]]
        train_neg_all = [r for r in roster if r["split"] == "train" and not r["is_positive"]]
        sample_size = min(len(train_neg_all), len(train_pos) * NEG_RATIO)
        train_neg = rng.sample(train_neg_all, sample_size)

        test_pos = [r for r in roster if r["split"] == "test" and r["is_positive"]]
        test_neg = [r for r in roster if r["split"] == "test" and not r["is_positive"]]

        print(
            f"  after stratification, train: {len(train_pos)} pos + {len(train_neg)} neg  "
            f"test: {len(test_pos)} pos + {len(test_neg)} neg"
        )

        train_samples_selected = train_pos + train_neg
        rng.shuffle(train_samples_selected)
        test_samples_selected = test_pos + test_neg

        # ---------- processing pass ----------
        print(f"\n[4/4] Processing {len(train_samples_selected) + len(test_samples_selected)} patches...")

        description_train: list[dict] = []
        description_eval: list[dict] = []
        grounding_train: list[dict] = []
        grounding_eval: list[dict] = []

        def process_patch(entry: dict, idx: int, is_train: bool) -> None:
            with zf.open(entry["image_name"]) as fp:
                with rasterio.MemoryFile(fp.read()) as memfile:
                    with memfile.open() as src:
                        bands = src.read()  # (13, 128, 128) float32

            with zf.open(entry["mask_name"]) as fp:
                with rasterio.MemoryFile(fp.read()) as memfile:
                    with memfile.open() as src:
                        mask = src.read(1)  # (128, 128)

            composite = compose_swir_false_color(bands)
            png_name = f"v1_{idx:06d}.png"
            Image.fromarray(composite).save(images_dir / png_name)

            stats = analyze_mask(mask)
            description = generate_description(stats)

            desc_msg = make_vlm_message(png_name, DESCRIPTION_PROMPT, description)
            grd_payload = [
                {"label": "mining_pit", "bbox": bb}
                for bb in stats["bboxes_normalized"]
            ]
            grd_msg = make_vlm_message(png_name, GROUNDING_PROMPT, json.dumps(grd_payload))

            if is_train:
                description_train.append(desc_msg)
                grounding_train.append(grd_msg)
            else:
                description_eval.append(desc_msg)
                grounding_eval.append(grd_msg)

        idx = 0
        for entry in train_samples_selected:
            process_patch(entry, idx, is_train=True)
            idx += 1
            if idx % 500 == 0:
                print(f"  processed {idx}/{len(train_samples_selected)} train")

        train_count = idx
        for entry in test_samples_selected:
            process_patch(entry, idx, is_train=False)
            idx += 1
            if (idx - train_count) % 200 == 0:
                print(f"  processed {idx - train_count}/{len(test_samples_selected)} eval")

    # ---------- write JSONL files ----------
    output_files = [
        ("galamsey_v1_description_train.jsonl", description_train),
        ("galamsey_v1_description_eval.jsonl", description_eval),
        ("galamsey_v1_grounding_train.jsonl", grounding_train),
        ("galamsey_v1_grounding_eval.jsonl", grounding_eval),
    ]
    for name, samples in output_files:
        path = data_dir / name
        with path.open("w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        print(f"  wrote {len(samples):>5} samples to {path}")

    volume.commit()

    return {
        "train_samples": len(description_train),
        "eval_samples": len(description_eval),
        "train_positives": len(train_pos),
        "train_negatives_sampled": len(train_neg),
        "train_negatives_available": len(train_neg_all),
        "eval_positives": len(test_pos),
        "eval_negatives": len(test_neg),
        "neg_ratio": NEG_RATIO,
    }


@app.local_entrypoint()
def main() -> None:
    print("Submitting v1 full dataset prep to Modal...")
    result = prepare.remote()
    print(f"\nPrep complete:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("\nNext steps:")
    print("  1. uv run leap-finetune configs/galamsey_probe.yaml   # cheap timing probe")
    print("  2. uv run leap-finetune configs/galamsey_v1_modal.yaml  # real v1 run")
