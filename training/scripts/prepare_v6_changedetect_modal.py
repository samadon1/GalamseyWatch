"""v6 full-dataset prep, change detection (2016 vs 2022 bitemporal pairs).

Each training sample pairs the 2016 and 2022 RGB composites of the SAME
geographic location. The ground truth is the CHANGE mask: pixels that are
mining in 2022 but NOT mining in 2016 = NEW galamsey. The model learns to
identify what CHANGED rather than what's mining in absolute terms.

Why this matters: v2/v4/v5 all fail on patches where mining disturbance is
visually ambiguous (diffuse or bare-soil-everywhere). With temporal context,
the model doesn't need to distinguish mining from farmland, it just needs
to spot what's NEW, which is a much easier visual task.

Key design:
  - Image 1: 2016 RGB composite (the "before")
  - Image 2: 2022 RGB composite (the "after")
  - Ground truth mask: (mask_2022 > 0) & (mask_2016 == 0) = NEW mining pixels
  - Descriptions: change-oriented ("new excavation appeared since 2016...")
  - Grounding bboxes: derived from the CHANGE mask only (not static 2022 mask)
  - Uses RGB composites (not SWIR) per v4's finding that RGB > SWIR for pretrained VLMs

Output path: `/galamsey/data/v6/`

Usage:
    uv run modal run scripts/prepare_v6_changedetect_modal.py
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"
MODAL_V6_DATA_DIR = f"{MODAL_MOUNT_POINT}/data/v6"

BAND_RED = 2
BAND_GREEN = 1
BAND_BLUE = 0

HECTARES_PER_PIXEL = 0.01
DATA_RNG_SEED = 42

app = modal.App("galamsey-v6-changedetect-prep")
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
    timeout=5400,
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

    random.seed(DATA_RNG_SEED)

    # ---------- prompts for change detection ----------
    DESCRIPTION_PROMPT = (
        "You are comparing two Sentinel-2 RGB satellite images of the same location "
        "in southwestern Ghana. The first image is from 2016, the second from 2022. "
        "Describe any NEW illegal small-scale gold mining (galamsey) activity visible "
        "in the 2022 image that was NOT present in the 2016 image. Focus on: new "
        "excavation pits, newly exposed soil, new sediment plumes, and recent "
        "vegetation loss. If no new mining activity is visible, say so."
    )
    GROUNDING_PROMPT = (
        "You are comparing two Sentinel-2 RGB images of the same location: 2016 (first) "
        "and 2022 (second). Detect any NEW illegal small-scale gold mining pits that "
        "appear in 2022 but were NOT present in 2016. "
        'Provide result as a valid JSON: [{"label": str, "bbox": [x1,y1,x2,y2]}, ...]. '
        "Coordinates must be normalized to 0-1. If no new pits are visible, return []."
    )

    # ---------- negative description templates (no new mining) ----------
    NEGATIVE_TEMPLATES = [
        "No new mining activity visible between 2016 and 2022. The landscape appears unchanged.",
        "The 2022 image shows no new galamsey compared to 2016. Land cover is consistent across both years.",
        "No new excavation pits or exposed soil appeared between 2016 and 2022.",
        "Comparing the two images, no new mining disturbance is evident. The area is stable.",
        "The scene appears unchanged between 2016 and 2022. No new galamsey activity detected.",
        "No temporal change in mining footprint. Both images show similar land cover.",
        "The 2022 image does not reveal any new small-scale mining compared to 2016.",
        "No new mining-related disturbance observed. Vegetation and soil patterns are consistent across years.",
    ]

    # ---------- positive change description templates ----------
    SINGLE_NEW_PIT_TEMPLATES = [
        "A new {size} excavation pit appeared in the {region} between 2016 and 2022, covering approximately {area_ha:.2f} hectares of previously undisturbed land.",
        "New mining activity in the {region}: one {size} pit ({area_ha:.2f} hectares) visible in 2022 that was not present in 2016.",
        "A {size} new galamsey pit has emerged in the {region} since 2016, disturbing approximately {area_ha:.2f} hectares.",
    ]

    MULTI_NEW_PIT_TEMPLATES = [
        "{count} new excavation pits appeared between 2016 and 2022, {adjacency}. The largest new disturbance is in the {primary_region}. Total newly affected area: {total_area_ha:.2f} hectares.",
        "New mining activity: {count} pits visible in 2022 that were absent in 2016, {adjacency}. Largest new pit in the {primary_region}. {total_area_ha:.2f} hectares of new disturbance.",
        "Comparing 2016 to 2022, {count} new galamsey pits are evident, {adjacency}. The primary new disturbance is in the {primary_region}, totaling {total_area_ha:.2f} hectares.",
    ]

    # ---------- helper functions ----------
    def percentile_stretch(band):
        lo, hi = np.percentile(band, [2, 98])
        clipped = np.clip(band, lo, hi)
        span = max(float(hi - lo), 1e-9)
        return ((clipped - lo) / span * 255.0).astype(np.uint8)

    def compose_rgb(bands):
        r = percentile_stretch(bands[BAND_RED])
        g = percentile_stretch(bands[BAND_GREEN])
        b = percentile_stretch(bands[BAND_BLUE])
        return np.stack([r, g, b], axis=-1)

    def analyze_mask(mask):
        binary = mask > 0
        labeled_arr, n_pits = label(binary)
        if n_pits == 0:
            return {"pit_count": 0, "areas_ha": [], "bboxes_normalized": [], "total_area_ha": 0.0}
        h, w = binary.shape
        bboxes, areas = [], []
        for pit_id in range(1, n_pits + 1):
            pit_mask = labeled_arr == pit_id
            ys, xs = np.where(pit_mask)
            if ys.size == 0:
                continue
            bboxes.append([
                round(int(xs.min()) / w, 4), round(int(ys.min()) / h, 4),
                round((int(xs.max()) + 1) / w, 4), round((int(ys.max()) + 1) / h, 4),
            ])
            areas.append(float(pit_mask.sum()) * HECTARES_PER_PIXEL)
        return {"pit_count": len(areas), "areas_ha": areas, "bboxes_normalized": bboxes, "total_area_ha": float(sum(areas))}

    def centroid_region(x, y):
        h = "left" if x < 0.33 else ("right" if x > 0.67 else "center")
        v = "upper" if y < 0.33 else ("lower" if y > 0.67 else "middle")
        if h == "center" and v == "middle":
            return "center of the scene"
        if h == "center":
            return f"{v} portion of the scene"
        if v == "middle":
            return f"{h} side of the scene"
        return f"{v}-{h} quadrant"

    def bbox_centroid(bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def bbox_size_category(bbox):
        area_fraction = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 0.0)
        if area_fraction < 0.01: return "very small"
        if area_fraction < 0.04: return "small"
        if area_fraction < 0.15: return "moderate"
        if area_fraction < 0.40: return "large"
        return "very large"

    def adjacency_phrase(bboxes):
        if len(bboxes) < 2: return ""
        centroids = [bbox_centroid(b) for b in bboxes]
        xs, ys = [c[0] for c in centroids], [c[1] for c in centroids]
        total_spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
        if total_spread < 0.3: return "clustered together"
        if total_spread < 0.7: return "spread across a connected area"
        return "scattered across the scene"

    def generate_change_description(stats):
        if stats["pit_count"] == 0:
            return random.choice(NEGATIVE_TEMPLATES)
        if stats["pit_count"] == 1:
            bbox = stats["bboxes_normalized"][0]
            cx, cy = bbox_centroid(bbox)
            return random.choice(SINGLE_NEW_PIT_TEMPLATES).format(
                size=bbox_size_category(bbox),
                region=centroid_region(cx, cy),
                area_ha=stats["areas_ha"][0],
            )
        bboxes = stats["bboxes_normalized"]
        areas = stats["areas_ha"]
        largest_idx = max(range(len(areas)), key=lambda i: areas[i])
        pcx, pcy = bbox_centroid(bboxes[largest_idx])
        return random.choice(MULTI_NEW_PIT_TEMPLATES).format(
            count=stats["pit_count"],
            adjacency=adjacency_phrase(bboxes),
            primary_region=centroid_region(pcx, pcy),
            total_area_ha=stats["total_area_ha"],
        )

    def make_vlm_message_multi(img1_filename, img2_filename, user_text, assistant_text):
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img1_filename},
                    {"type": "image", "image": img2_filename},
                    {"type": "text", "text": user_text},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": assistant_text},
                ]},
            ]
        }

    def patch_id_num(entry_name):
        stem = Path(entry_name).stem
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[-2])
            except ValueError:
                pass
        return -1

    # ---------- main work ----------
    data_dir = Path(MODAL_V6_DATA_DIR)
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GalamseyWatch, v6 change detection dataset prep on Modal")
    print("=" * 60)

    print("\n[1/5] Downloading SmallMinesDS artifacts...")
    zip_path = hf_hub_download("ellaampy/SmallMinesDS", "SmallMinesDS.zip", repo_type="dataset")
    split_csv_paths = {}
    for year in ("2016", "2022"):
        split_csv_paths[year] = hf_hub_download(
            "ellaampy/SmallMinesDS", f"data_splits/train_test_splits_{year}.csv", repo_type="dataset"
        )

    # ---------- parse splits per year ----------
    print("\n[2/5] Parsing split CSVs and building location pairs...")
    split_by_year = {}  # {(patch_num, year): split}
    for year, csv_path in split_csv_paths.items():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                pname = row.get("patch_name", "")
                pnum = patch_id_num(pname)
                if pnum == -1:
                    continue
                split = (row.get("split") or "train").strip().lower()
                split_by_year[(pnum, year)] = split if split in ("train", "test") else "train"

    # Find locations where BOTH years exist and have the SAME split
    all_patch_nums = set()
    for (pnum, year) in split_by_year:
        all_patch_nums.add(pnum)

    paired_locations = []
    split_mismatch = 0
    for pnum in sorted(all_patch_nums):
        s16 = split_by_year.get((pnum, "2016"))
        s22 = split_by_year.get((pnum, "2022"))
        if s16 is None or s22 is None:
            continue
        if s16 != s22:
            split_mismatch += 1
            continue
        paired_locations.append({"patch_num": pnum, "split": s16})

    print(f"  total locations with both years: {len(paired_locations)}")
    print(f"  split mismatches (dropped): {split_mismatch}")
    n_train = sum(1 for loc in paired_locations if loc["split"] == "train")
    n_test = sum(1 for loc in paired_locations if loc["split"] == "test")
    print(f"  train locations: {n_train}, test locations: {n_test}")

    # ---------- build file lookup ----------
    print("\n[3/5] Scanning zip for 2016/2022 image+mask pairs...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        # Index by (patch_num, year, type)
        file_index = {}
        for name in all_names:
            if not name.endswith(".tif"):
                continue
            pnum = patch_id_num(name)
            if pnum == -1:
                continue
            year = "2016" if "/2016/" in name else ("2022" if "/2022/" in name else None)
            ftype = "image" if "IMAGE" in name else ("mask" if "MASK" in name else None)
            if year and ftype:
                file_index[(pnum, year, ftype)] = name

        print(f"  indexed {len(file_index)} files")

        # ---------- process pairs ----------
        print(f"\n[4/5] Processing {len(paired_locations)} location pairs...")

        description_train, description_eval = [], []
        grounding_train, grounding_eval = [], []

        n_new_mining = 0
        n_stable = 0
        n_no_mining = 0
        n_reclaimed = 0

        for idx, loc in enumerate(paired_locations):
            pnum = loc["patch_num"]
            is_train = loc["split"] == "train"

            # Load all 4 files for this location
            keys_needed = [
                (pnum, "2016", "image"), (pnum, "2016", "mask"),
                (pnum, "2022", "image"), (pnum, "2022", "mask"),
            ]
            missing = [k for k in keys_needed if k not in file_index]
            if missing:
                continue

            def load_tiff(key):
                with zf.open(file_index[key]) as fp:
                    with rasterio.MemoryFile(fp.read()) as mf:
                        with mf.open() as src:
                            return src.read() if key[2] == "image" else src.read(1)

            bands_2016 = load_tiff((pnum, "2016", "image"))
            mask_2016 = load_tiff((pnum, "2016", "mask"))
            bands_2022 = load_tiff((pnum, "2022", "image"))
            mask_2022 = load_tiff((pnum, "2022", "mask"))

            # Render RGB composites for both years
            rgb_2016 = compose_rgb(bands_2016)
            rgb_2022 = compose_rgb(bands_2022)
            png_2016 = f"v6_2016_{idx:05d}.png"
            png_2022 = f"v6_2022_{idx:05d}.png"
            Image.fromarray(rgb_2016).save(images_dir / png_2016)
            Image.fromarray(rgb_2022).save(images_dir / png_2022)

            # Compute CHANGE mask: new mining = present in 2022 but NOT in 2016
            change_mask = ((mask_2022 > 0) & (mask_2016 == 0)).astype(np.uint8)

            # Classify the location
            has_2016 = (mask_2016 > 0).sum() > 0
            has_2022 = (mask_2022 > 0).sum() > 0
            has_new = change_mask.sum() > 0

            if has_new:
                n_new_mining += 1
            elif has_2016 and has_2022:
                n_stable += 1
            elif has_2016 and not has_2022:
                n_reclaimed += 1
            else:
                n_no_mining += 1

            # Analyze the CHANGE mask (not the static mask)
            stats = analyze_mask(change_mask)
            description = generate_change_description(stats)

            desc_target = description_train if is_train else description_eval
            grd_target = grounding_train if is_train else grounding_eval

            desc_target.append(make_vlm_message_multi(
                png_2016, png_2022, DESCRIPTION_PROMPT, description
            ))
            grd_payload = [{"label": "new_mining_pit", "bbox": bb} for bb in stats["bboxes_normalized"]]
            grd_target.append(make_vlm_message_multi(
                png_2016, png_2022, GROUNDING_PROMPT, json.dumps(grd_payload)
            ))

            if (idx + 1) % 200 == 0:
                print(f"  processed {idx+1}/{len(paired_locations)} pairs")

    print(f"\n  Location classification:")
    print(f"    new mining (change mask has pixels):  {n_new_mining}")
    print(f"    stable mining (both years):           {n_stable}")
    print(f"    reclaimed (2016 only):                {n_reclaimed}")
    print(f"    no mining (neither year):             {n_no_mining}")

    # ---------- multi-task JSONL ----------
    print(f"\n[5/5] Building multi-task JSONL...")
    multitask_train = description_train + grounding_train
    random.Random(DATA_RNG_SEED + 2).shuffle(multitask_train)

    output_files = [
        ("galamsey_v6_description_train.jsonl", description_train),
        ("galamsey_v6_description_eval.jsonl", description_eval),
        ("galamsey_v6_grounding_train.jsonl", grounding_train),
        ("galamsey_v6_grounding_eval.jsonl", grounding_eval),
        ("galamsey_v6_multitask_train.jsonl", multitask_train),
    ]
    for name, samples in output_files:
        path = data_dir / name
        with path.open("w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"  wrote {len(samples):>5} samples to {path}")

    unique_descs = len({s["messages"][1]["content"][0]["text"] for s in description_train})
    print(f"\n  unique training descriptions: {unique_descs}")

    volume.commit()

    return {
        "paired_locations": len(paired_locations),
        "split_mismatches": split_mismatch,
        "n_train": n_train,
        "n_test": n_test,
        "n_new_mining": n_new_mining,
        "n_stable": n_stable,
        "n_reclaimed": n_reclaimed,
        "n_no_mining": n_no_mining,
        "train_samples_multitask": len(multitask_train),
        "unique_descriptions": unique_descs,
    }


@app.local_entrypoint()
def main():
    print("Submitting v6 change detection dataset prep to Modal...")
    result = prepare.remote()
    print(f"\nPrep complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")
