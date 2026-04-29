"""v2 full-dataset prep, enriched descriptions + multi-task JSONL.

Differences from prepare_v1_dataset_modal.py:

1. **Fixed seeding bug.** v1 used `seed = hash((pit_count, total_area_ha))` as
   the default seed inside `generate_description`, which collapsed all empty
   masks to ONE identical description string. v2 calls `random.seed(42)` once
   at the start of the Modal function and then lets `generate_description` use
   the module-level `random` state, so each call advances the RNG. Same
   seed-42 convention means the output is still reproducible across reruns.
   See ARCHITECTURE.md §3.7.j for the full bug story.

2. **Enriched description templates.** The `galamsey_finetune/masks.py`
   `generate_description` function now slots in spatial quadrants, adjacency
   phrases, size categories, and primary-pit regions from the mask geometry.
   See ARCHITECTURE.md §3.4.2 for the full list of derived features.

3. **Multi-task training JSONL.** In addition to the separate description and
   grounding files, v2 emits `galamsey_v2_multitask_train.jsonl`, a shuffled
   combination of description + grounding samples that leap-finetune trains
   on as a single mixed task. This forces the model to learn both free-text
   description AND bbox-grounded detection in the same training run. Mirrors
   the pattern of Liquid's vrsbench_multitask_modal.yaml.

4. **Output path is `/galamsey/data/v2/`** so v1 data is preserved untouched.

Everything else (CSV parsing, stratification, composite rendering, bbox
derivation) is identical to v1.

Usage:
    uv run modal run scripts/prepare_v2_dataset_modal.py
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"
MODAL_V2_DATA_DIR = f"{MODAL_MOUNT_POINT}/data/v2"

# Canonical band indices from SmallMinesDS README (ARCHITECTURE.md §3.4.1)
BAND_SWIR2 = 9
BAND_SWIR1 = 8
BAND_NIR = 6

HECTARES_PER_PIXEL = 0.01  # Sentinel-2 10 m GSD

# v2 reproducibility seed, same value means same JSONL output across reruns.
# See ARCHITECTURE.md §3.7.j for why we cannot derive the seed from stats.
DATA_RNG_SEED = 42

# Stratification ratio, unchanged from v1 (all positives + 2x negatives cap)
NEG_RATIO = 2

app = modal.App("galamsey-v2-prep")
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

    # ---------- seed the RNG ONCE for the full run ----------
    # This fixes the v1 bug where per-call seeding collapsed all empty masks
    # to a single identical description string. Every rng.choice() advances
    # the module-level state from here, giving us real per-call variety.
    random.seed(DATA_RNG_SEED)
    stratify_rng = random.Random(DATA_RNG_SEED + 1)  # separate stream for stratification

    # ---------- inlined galamseywatch primitives (self-contained for Modal) ----------
    # Kept in sync with galamseywatch/{composites,masks,vlm_format}.py

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

    # --- enriched description template banks (mirrored from masks.py v2) ---
    NEGATIVE_TEMPLATES = [
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

    SINGLE_PIT_TEMPLATES = [
        "A single {size} excavation pit visible in the {region}, covering approximately {area_ha:.2f} hectares.",
        "One {size} mining pit in the {region}. Affected area: ~{area_ha:.2f} hectares.",
        "Isolated {size} excavation feature in the {region} of the tile, about {area_ha:.2f} hectares in extent.",
        "Single pit ({size}) located in the {region}, {area_ha:.2f} ha of disturbed ground.",
        "One excavation pit visible, {size} in size, in the {region}. Approximately {area_ha:.2f} hectares affected.",
        "A {size} mining site is evident in the {region}. Disturbance covers roughly {area_ha:.2f} hectares.",
        "Solitary mining pit in the {region} of the patch, {area_ha:.2f} hectares of exposed subsurface.",
    ]

    MULTI_PIT_TEMPLATES = [
        "{count} excavation pits {adjacency}, with the largest in the {primary_region}. Total affected area approximately {total_area_ha:.2f} hectares {size_phrase}.",
        "{count} mining pits visible, {adjacency}. The largest pit is in the {primary_region}. Combined area: {total_area_ha:.2f} hectares.",
        "Multiple excavation features: {count} pits {adjacency}. Largest disturbance in the {primary_region}. Total ~{total_area_ha:.2f} hectares {size_phrase}.",
        "{count} active mining pits {adjacency}. The dominant pit is in the {primary_region}. Approximately {total_area_ha:.2f} hectares of disturbed ground {size_phrase}.",
        "The scene contains {count} excavation pits {adjacency}. Largest feature in the {primary_region}. Total affected area: {total_area_ha:.2f} hectares.",
        "{count} pit sites {adjacency}. The biggest pit sits in the {primary_region}. {total_area_ha:.2f} hectares of exposed or disturbed surface {size_phrase}.",
        "{count} mining pits identified, {adjacency}. The largest lies in the {primary_region}. {total_area_ha:.2f} total hectares affected.",
    ]

    def centroid_region(x: float, y: float) -> str:
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
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def bbox_size_category(bbox):
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

    def size_distribution_phrase(areas_ha):
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

    def adjacency_phrase(bboxes):
        if len(bboxes) < 2:
            return ""
        centroids = [bbox_centroid(b) for b in bboxes]
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]
        total_spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
        if total_spread < 0.3:
            return "clustered together in a tight group"
        if total_spread < 0.7:
            return "spread across a connected area"
        return "scattered across the scene"

    def percentile_stretch(band):
        lo, hi = np.percentile(band, [2, 98])
        clipped = np.clip(band, lo, hi)
        span = max(float(hi - lo), 1e-9)
        return ((clipped - lo) / span * 255.0).astype(np.uint8)

    def compose_swir_false_color(bands):
        r = percentile_stretch(bands[BAND_SWIR2])
        g = percentile_stretch(bands[BAND_SWIR1])
        b = percentile_stretch(bands[BAND_NIR])
        return np.stack([r, g, b], axis=-1)

    def analyze_mask(mask):
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
        bboxes = []
        areas = []
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

    def generate_description(stats):
        """Enriched description using module-level random state, NO per-call seeding."""
        pit_count = stats["pit_count"]

        if pit_count == 0:
            return random.choice(NEGATIVE_TEMPLATES)

        if pit_count == 1:
            bbox = stats["bboxes_normalized"][0]
            cx, cy = bbox_centroid(bbox)
            region = centroid_region(cx, cy)
            size = bbox_size_category(bbox)
            area_ha = stats["areas_ha"][0]
            template = random.choice(SINGLE_PIT_TEMPLATES)
            return template.format(size=size, region=region, area_ha=area_ha)

        # Multi-pit
        bboxes = stats["bboxes_normalized"]
        areas = stats["areas_ha"]
        largest_idx = max(range(len(areas)), key=lambda i: areas[i])
        primary_bbox = bboxes[largest_idx]
        primary_cx, primary_cy = bbox_centroid(primary_bbox)
        primary_region = centroid_region(primary_cx, primary_cy)

        adjacency = adjacency_phrase(bboxes)
        size_phrase = size_distribution_phrase(areas)

        template = random.choice(MULTI_PIT_TEMPLATES)
        sentence = template.format(
            count=pit_count,
            adjacency=adjacency,
            primary_region=primary_region,
            total_area_ha=stats["total_area_ha"],
            size_phrase=size_phrase,
        )
        while "  " in sentence:
            sentence = sentence.replace("  ", " ")
        sentence = sentence.replace(" .", ".").replace(" ,", ",")
        return sentence

    def make_vlm_message(image_filename, user_text, assistant_text):
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

    def patch_id(entry_name):
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

    data_dir = Path(MODAL_V2_DATA_DIR)
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GalamseyWatch, v2 full dataset prep on Modal")
    print("=" * 60)
    print(f"RNG seed: {DATA_RNG_SEED}")

    print("\n[1/5] Downloading SmallMinesDS artifacts from HuggingFace...")
    zip_path = hf_hub_download(
        repo_id="ellaampy/SmallMinesDS",
        filename="SmallMinesDS.zip",
        repo_type="dataset",
    )
    split_csv_paths = {}
    for year in ("2016", "2022"):
        split_csv_paths[year] = hf_hub_download(
            repo_id="ellaampy/SmallMinesDS",
            filename=f"data_splits/train_test_splits_{year}.csv",
            repo_type="dataset",
        )
    print(f"  zip: {zip_path}")

    # ---------- parse split CSVs ----------
    split_assignment = {}
    class_percentage_by_id = {}

    print("\n[2/5] Parsing split CSVs...")
    for year, csv_path in split_csv_paths.items():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pname = row.get("patch_name") or next(iter(row.values()), "")
                split = (row.get("split") or "train").strip().lower()
                try:
                    cp = float(row.get("class_percentage", 0) or 0)
                except (ValueError, TypeError):
                    cp = 0.0
                key = patch_id(pname)
                if key[0] == -1:
                    continue
                split_assignment[key] = (
                    split if split in ("train", "test", "val") else "train"
                )
                class_percentage_by_id[key] = cp
    print(f"  total split entries: {len(split_assignment)}")

    # ---------- build roster + stratify ----------
    print("\n[3/5] Scanning zip for image/mask pairs...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        image_tiffs = sorted(
            n for n in all_names if "IMAGE" in n and n.lower().endswith((".tif", ".tiff"))
        )
        mask_tiffs = sorted(
            n for n in all_names if "MASK" in n and n.lower().endswith((".tif", ".tiff"))
        )
        print(f"  {len(image_tiffs)} images, {len(mask_tiffs)} masks")

        mask_by_id = {patch_id(n): n for n in mask_tiffs}

        roster = []
        for image_name in image_tiffs:
            key = patch_id(image_name)
            if key not in mask_by_id:
                continue
            cp = class_percentage_by_id.get(key, 0.0)
            roster.append({
                "key": key,
                "image_name": image_name,
                "mask_name": mask_by_id[key],
                "is_positive": cp > 0.0,
                "split": split_assignment.get(key, "train"),
            })
        print(f"  roster: {len(roster)} patches")

        # Stratified sampling, all train positives, 2x cap on negatives
        train_pos = [r for r in roster if r["split"] == "train" and r["is_positive"]]
        train_neg_all = [r for r in roster if r["split"] == "train" and not r["is_positive"]]
        sample_size = min(len(train_neg_all), len(train_pos) * NEG_RATIO)
        train_neg = stratify_rng.sample(train_neg_all, sample_size)

        test_pos = [r for r in roster if r["split"] == "test" and r["is_positive"]]
        test_neg = [r for r in roster if r["split"] == "test" and not r["is_positive"]]

        print(
            f"  train: {len(train_pos)} pos + {len(train_neg)} neg  "
            f"test: {len(test_pos)} pos + {len(test_neg)} neg"
        )

        train_selected = train_pos + train_neg
        stratify_rng.shuffle(train_selected)
        test_selected = test_pos + test_neg

        # ---------- processing pass ----------
        print(f"\n[4/5] Processing {len(train_selected) + len(test_selected)} patches...")

        description_train = []
        description_eval = []
        grounding_train = []
        grounding_eval = []

        def process_patch(entry, idx, is_train):
            with zf.open(entry["image_name"]) as fp:
                with rasterio.MemoryFile(fp.read()) as memfile:
                    with memfile.open() as src:
                        bands = src.read()

            with zf.open(entry["mask_name"]) as fp:
                with rasterio.MemoryFile(fp.read()) as memfile:
                    with memfile.open() as src:
                        mask = src.read(1)

            composite = compose_swir_false_color(bands)
            png_name = f"v2_{idx:06d}.png"
            Image.fromarray(composite).save(images_dir / png_name)

            stats = analyze_mask(mask)
            description = generate_description(stats)  # uses module-level random state

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
        for entry in train_selected:
            process_patch(entry, idx, is_train=True)
            idx += 1
            if idx % 500 == 0:
                print(f"  processed {idx}/{len(train_selected)} train")
        train_count = idx
        for entry in test_selected:
            process_patch(entry, idx, is_train=False)
            idx += 1
            if (idx - train_count) % 200 == 0:
                print(f"  processed {idx - train_count}/{len(test_selected)} eval")

    # ---------- build multi-task JSONL ----------
    print("\n[5/5] Building multi-task training JSONL (description + grounding shuffled)...")
    multitask_train = description_train + grounding_train
    multitask_rng = random.Random(DATA_RNG_SEED + 2)
    multitask_rng.shuffle(multitask_train)
    print(f"  multitask_train: {len(multitask_train)} samples "
          f"({len(description_train)} description + {len(grounding_train)} grounding)")

    # ---------- write JSONL files ----------
    output_files = [
        ("galamsey_v2_description_train.jsonl", description_train),
        ("galamsey_v2_description_eval.jsonl", description_eval),
        ("galamsey_v2_grounding_train.jsonl", grounding_train),
        ("galamsey_v2_grounding_eval.jsonl", grounding_eval),
        ("galamsey_v2_multitask_train.jsonl", multitask_train),
    ]
    for name, samples in output_files:
        path = data_dir / name
        with path.open("w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        print(f"  wrote {len(samples):>5} samples to {path}")

    # ---------- sanity check: count unique descriptions ----------
    unique_train_descs = len({s["messages"][1]["content"][0]["text"] for s in description_train})
    neg_count_train = sum(
        1 for s in description_train
        if "No " in s["messages"][1]["content"][0]["text"] or "intact" in s["messages"][1]["content"][0]["text"].lower()
    )
    print(f"\n  unique training descriptions: {unique_train_descs}")
    print(f"  training negatives (heuristic): {neg_count_train}")

    volume.commit()

    return {
        "train_samples_description": len(description_train),
        "eval_samples_description": len(description_eval),
        "train_samples_grounding": len(grounding_train),
        "train_samples_multitask": len(multitask_train),
        "unique_training_descriptions": unique_train_descs,
        "train_positives": len(train_pos),
        "train_negatives": len(train_neg),
    }


@app.local_entrypoint()
def main() -> None:
    print("Submitting v2 full dataset prep to Modal...")
    result = prepare.remote()
    print(f"\nPrep complete:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("\nNext step:")
    print("  uv run leap-finetune configs/galamsey_v2_modal.yaml")
