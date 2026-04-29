"""v10 production dataset prep, 450M WebGPU model, 4× augmentation, filtered bboxes.

The FINAL training data for the production WebGPU model. Two improvements over v7:
  1. 4× augmentation (original + hflip + vflip + hflip+vflip) instead of 2×
  2. Filter ground-truth bboxes < 5 pixels area to prevent sliver-bbox learning

Same RGB+SWIR multi-image format, same enriched descriptions, same multi-task mix.

Output path: `/galamsey/data/v10/`

Usage:
    uv run modal run scripts/prepare_v10_production_modal.py
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"
MODAL_V10_DATA_DIR = f"{MODAL_MOUNT_POINT}/data/v10"

BAND_RED = 2
BAND_GREEN = 1
BAND_BLUE = 0
BAND_SWIR2 = 9
BAND_SWIR1 = 8
BAND_NIR = 6

HECTARES_PER_PIXEL = 0.01
DATA_RNG_SEED = 42
NEG_RATIO = 2
MIN_PIT_PIXELS = 5  # filter bboxes from components smaller than this

app = modal.App("galamsey-v10-production-prep")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libexpat1")
    .pip_install(
        "huggingface_hub", "numpy>=2.0", "pillow>=11.0",
        "rasterio>=1.3", "scipy>=1.13",
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
    stratify_rng = random.Random(DATA_RNG_SEED + 1)

    DESCRIPTION_PROMPT = (
        "You are analyzing two views of the same Sentinel-2 patch of southwestern Ghana: "
        "the first image is a natural-color RGB composite, and the second is a SWIR "
        "false-color composite (SWIR2, SWIR1, NIR) where bright areas indicate exposed "
        "soil and mining disturbance. Using both views, describe any signs of illegal "
        "small-scale gold mining (galamsey) activity: exposed soil, excavation pits, "
        "sediment plumes, vegetation loss, and proximity to water bodies. "
        "If no mining is visible, say so."
    )
    GROUNDING_PROMPT = (
        "You are viewing two images of the same Sentinel-2 patch: a natural-color RGB "
        "composite and a SWIR false-color composite. Using both views, detect any "
        "illegal small-scale gold mining pits. "
        'Provide result as a valid JSON: [{"label": str, "bbox": [x1,y1,x2,y2]}, ...]. '
        "Coordinates must be normalized to 0-1. If no pits are visible, return []."
    )

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

    def centroid_region(x, y):
        h = "left" if x < 0.33 else ("right" if x > 0.67 else "center")
        v = "upper" if y < 0.33 else ("lower" if y > 0.67 else "middle")
        if h == "center" and v == "middle": return "center of the scene"
        if h == "center": return f"{v} portion of the scene"
        if v == "middle": return f"{h} side of the scene"
        return f"{v}-{h} quadrant"

    def bbox_centroid(bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def bbox_size_category(bbox):
        af = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 0.0)
        if af < 0.01: return "very small"
        if af < 0.04: return "small"
        if af < 0.15: return "moderate"
        if af < 0.40: return "large"
        return "very large"

    def size_distribution_phrase(areas_ha):
        if len(areas_ha) == 1: return ""
        mean_a = sum(areas_ha) / len(areas_ha)
        spread = (max(areas_ha) - min(areas_ha)) / max(mean_a, 1e-9)
        if spread < 0.4: return "with roughly uniform pit sizes"
        if spread < 1.5: return "with mixed pit sizes"
        return "dominated by one or two large pits among smaller satellite pits"

    def adjacency_phrase(bboxes):
        if len(bboxes) < 2: return ""
        cs = [bbox_centroid(b) for b in bboxes]
        spread = (max(c[0] for c in cs) - min(c[0] for c in cs)) + (max(c[1] for c in cs) - min(c[1] for c in cs))
        if spread < 0.3: return "clustered together in a tight group"
        if spread < 0.7: return "spread across a connected area"
        return "scattered across the scene"

    def percentile_stretch(band):
        lo, hi = np.percentile(band, [2, 98])
        clipped = np.clip(band, lo, hi)
        span = max(float(hi - lo), 1e-9)
        return ((clipped - lo) / span * 255.0).astype(np.uint8)

    def compose_rgb(bands):
        return np.stack([percentile_stretch(bands[BAND_RED]), percentile_stretch(bands[BAND_GREEN]), percentile_stretch(bands[BAND_BLUE])], axis=-1)

    def compose_swir(bands):
        return np.stack([percentile_stretch(bands[BAND_SWIR2]), percentile_stretch(bands[BAND_SWIR1]), percentile_stretch(bands[BAND_NIR])], axis=-1)

    def analyze_mask(mask, min_pixels=MIN_PIT_PIXELS):
        """Connected-component analysis with small-component filtering."""
        binary = mask > 0
        labeled_arr, n_pits = label(binary)
        if n_pits == 0:
            return {"pit_count": 0, "areas_ha": [], "bboxes_normalized": [], "total_area_ha": 0.0}
        h, w = binary.shape
        bboxes, areas = [], []
        for pid in range(1, n_pits + 1):
            pm = labeled_arr == pid
            pixel_count = int(pm.sum())
            if pixel_count < min_pixels:
                continue  # v10: skip tiny components
            ys, xs = np.where(pm)
            bboxes.append([
                round(int(xs.min()) / w, 4), round(int(ys.min()) / h, 4),
                round((int(xs.max()) + 1) / w, 4), round((int(ys.max()) + 1) / h, 4),
            ])
            areas.append(float(pixel_count) * HECTARES_PER_PIXEL)
        return {"pit_count": len(areas), "areas_ha": areas, "bboxes_normalized": bboxes, "total_area_ha": float(sum(areas))}

    def generate_description(stats):
        pc = stats["pit_count"]
        if pc == 0: return random.choice(NEGATIVE_TEMPLATES)
        if pc == 1:
            bb = stats["bboxes_normalized"][0]
            cx, cy = bbox_centroid(bb)
            return random.choice(SINGLE_PIT_TEMPLATES).format(
                size=bbox_size_category(bb), region=centroid_region(cx, cy), area_ha=stats["areas_ha"][0])
        bbs, areas = stats["bboxes_normalized"], stats["areas_ha"]
        li = max(range(len(areas)), key=lambda i: areas[i])
        pcx, pcy = bbox_centroid(bbs[li])
        s = random.choice(MULTI_PIT_TEMPLATES).format(
            count=pc, adjacency=adjacency_phrase(bbs),
            primary_region=centroid_region(pcx, pcy),
            total_area_ha=stats["total_area_ha"],
            size_phrase=size_distribution_phrase(areas))
        while "  " in s: s = s.replace("  ", " ")
        return s.replace(" .", ".").replace(" ,", ",")

    def make_vlm_message_multi(img1, img2, user_text, asst_text):
        return {"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": user_text},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": asst_text}]},
        ]}

    def patch_id(entry_name):
        stem = Path(entry_name).stem
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                pn = int(parts[-2])
                yr = parts[-1]
                if yr in ("2016", "2022"): return (pn, yr)
            except ValueError: pass
        return (-1, "unknown")

    # ---------- main ----------
    data_dir = Path(MODAL_V10_DATA_DIR)
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GalamseyWatch, v10 production dataset (4× aug, filtered bboxes)")
    print("=" * 60)

    zip_path = hf_hub_download("ellaampy/SmallMinesDS", "SmallMinesDS.zip", repo_type="dataset")
    split_csv_paths = {}
    for year in ("2016", "2022"):
        split_csv_paths[year] = hf_hub_download(
            "ellaampy/SmallMinesDS", f"data_splits/train_test_splits_{year}.csv", repo_type="dataset")

    split_assignment, class_percentage_by_id = {}, {}
    for year, csv_path in split_csv_paths.items():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                pname = row.get("patch_name") or ""
                split = (row.get("split") or "train").strip().lower()
                try: cp = float(row.get("class_percentage", 0) or 0)
                except: cp = 0.0
                key = patch_id(pname)
                if key[0] == -1: continue
                split_assignment[key] = split if split in ("train", "test") else "train"
                class_percentage_by_id[key] = cp

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        image_tiffs = sorted(n for n in all_names if "IMAGE" in n and n.endswith(".tif"))
        mask_tiffs = sorted(n for n in all_names if "MASK" in n and n.endswith(".tif"))
        mask_by_id = {patch_id(n): n for n in mask_tiffs}

        roster = []
        for image_name in image_tiffs:
            key = patch_id(image_name)
            if key not in mask_by_id: continue
            cp = class_percentage_by_id.get(key, 0.0)
            roster.append({"key": key, "image_name": image_name,
                          "mask_name": mask_by_id[key], "is_positive": cp > 0.0,
                          "split": split_assignment.get(key, "train")})

        train_pos = [r for r in roster if r["split"] == "train" and r["is_positive"]]
        train_neg_all = [r for r in roster if r["split"] == "train" and not r["is_positive"]]
        train_neg = stratify_rng.sample(train_neg_all, min(len(train_neg_all), len(train_pos) * NEG_RATIO))
        test_pos = [r for r in roster if r["split"] == "test" and r["is_positive"]]
        test_neg = [r for r in roster if r["split"] == "test" and not r["is_positive"]]

        train_selected = train_pos + train_neg
        stratify_rng.shuffle(train_selected)
        test_selected = test_pos + test_neg

        print(f"  train: {len(train_pos)} pos + {len(train_neg)} neg, test: {len(test_pos)} pos + {len(test_neg)} neg")

        # 8× augmentation, the full D4 dihedral group (all orientation-preserving
        # symmetries of a square). Satellite imagery has no up/down prior at nadir
        # geometry, so all 8 are valid label-preserving transforms.
        def rot(b, m, k):
            return np.rot90(b, k=k, axes=(1, 2)).copy(), np.rot90(m, k=k, axes=(0, 1)).copy()

        AUGMENTATIONS = [
            ("orig", lambda b, m: (b, m)),
            ("hflip", lambda b, m: (np.flip(b, axis=2).copy(), np.flip(m, axis=1).copy())),
            ("vflip", lambda b, m: (np.flip(b, axis=1).copy(), np.flip(m, axis=0).copy())),
            ("hvflip", lambda b, m: (np.flip(np.flip(b, axis=2), axis=1).copy(), np.flip(np.flip(m, axis=1), axis=0).copy())),
            ("rot90", lambda b, m: rot(b, m, 1)),
            ("rot270", lambda b, m: rot(b, m, 3)),
            ("rot90_hflip", lambda b, m: (np.flip(np.rot90(b, 1, (1, 2)), axis=2).copy(), np.flip(np.rot90(m, 1, (0, 1)), axis=1).copy())),
            ("rot90_vflip", lambda b, m: (np.flip(np.rot90(b, 1, (1, 2)), axis=1).copy(), np.flip(np.rot90(m, 1, (0, 1)), axis=0).copy())),
        ]

        description_train, description_eval = [], []
        grounding_train, grounding_eval = [], []
        n_filtered_bboxes = 0

        # Photometric variants, two per geometric variant: "normal" and "jittered".
        # Jitter applies independent brightness, contrast, and saturation shifts
        # sampled uniformly from a moderate range (±20% each). Label-preserving -
        # the pit masks don't move under color transforms.
        import PIL.ImageEnhance as ImgEnh

        def photometric_jitter(pil_img, seed_key):
            rng_local = random.Random(seed_key)
            bright = rng_local.uniform(0.8, 1.2)
            contrast = rng_local.uniform(0.8, 1.2)
            saturation = rng_local.uniform(0.8, 1.2)
            img = ImgEnh.Brightness(pil_img).enhance(bright)
            img = ImgEnh.Contrast(img).enhance(contrast)
            img = ImgEnh.Color(img).enhance(saturation)
            return img

        PHOTOMETRIC_MODES = ["orig", "jitter"]

        def process_patch(entry, idx, is_train):
            nonlocal n_filtered_bboxes
            with zf.open(entry["image_name"]) as fp:
                with rasterio.MemoryFile(fp.read()) as mf:
                    with mf.open() as src: bands = src.read()
            with zf.open(entry["mask_name"]) as fp:
                with rasterio.MemoryFile(fp.read()) as mf:
                    with mf.open() as src: mask = src.read(1)

            variants = AUGMENTATIONS if is_train else [AUGMENTATIONS[0]]  # eval: no augmentation
            photo_modes = PHOTOMETRIC_MODES if is_train else ["orig"]
            new_idx = idx
            for aug_name, aug_fn in variants:
                v_bands, v_mask = aug_fn(bands, mask)
                rgb_arr = compose_rgb(v_bands)
                swir_arr = compose_swir(v_bands)

                stats = analyze_mask(v_mask, min_pixels=MIN_PIT_PIXELS)
                stats_unfiltered = analyze_mask(v_mask, min_pixels=1)
                n_filtered_bboxes += stats_unfiltered["pit_count"] - stats["pit_count"]

                description = generate_description(stats)
                grd_payload = [{"label": "mining_pit", "bbox": bb} for bb in stats["bboxes_normalized"]]
                grd_json = json.dumps(grd_payload)

                for photo_mode in photo_modes:
                    rgb_pil = Image.fromarray(rgb_arr)
                    swir_pil = Image.fromarray(swir_arr)
                    if photo_mode == "jitter":
                        rgb_pil = photometric_jitter(rgb_pil, seed_key=f"rgb_{new_idx}")
                        swir_pil = photometric_jitter(swir_pil, seed_key=f"swir_{new_idx}")

                    rgb_name = f"v10_rgb_{new_idx:06d}.png"
                    swir_name = f"v10_swir_{new_idx:06d}.png"
                    rgb_pil.save(images_dir / rgb_name)
                    swir_pil.save(images_dir / swir_name)

                    desc_msg = make_vlm_message_multi(rgb_name, swir_name, DESCRIPTION_PROMPT, description)
                    grd_msg = make_vlm_message_multi(rgb_name, swir_name, GROUNDING_PROMPT, grd_json)

                    if is_train:
                        description_train.append(desc_msg)
                        grounding_train.append(grd_msg)
                    else:
                        description_eval.append(desc_msg)
                        grounding_eval.append(grd_msg)
                    new_idx += 1
            return new_idx

        print(f"\n  Processing train ({len(train_selected)} patches × 16 augmentations (8 geom × 2 photometric))...")
        idx = 0
        for entry in train_selected:
            idx = process_patch(entry, idx, is_train=True)
            if idx % 2000 == 0: print(f"    {idx} samples written")
        print(f"  Processing eval ({len(test_selected)} patches, no augmentation)...")
        train_count = idx
        for entry in test_selected:
            idx = process_patch(entry, idx, is_train=False)

        print(f"\n  Filtered {n_filtered_bboxes} tiny bboxes (< {MIN_PIT_PIXELS} pixels)")
        print(f"  Train: {len(description_train)} desc + {len(grounding_train)} grd")
        print(f"  Eval: {len(description_eval)} desc + {len(grounding_eval)} grd")

        # 50/50 multitask mix, same ratio as v9, for clean comparison on the aug variable alone.
        multitask_train = description_train + grounding_train
        random.Random(DATA_RNG_SEED + 2).shuffle(multitask_train)

        for name, samples in [
            ("galamsey_v10_description_train.jsonl", description_train),
            ("galamsey_v10_description_eval.jsonl", description_eval),
            ("galamsey_v10_grounding_train.jsonl", grounding_train),
            ("galamsey_v10_grounding_eval.jsonl", grounding_eval),
            ("galamsey_v10_multitask_train.jsonl", multitask_train),
        ]:
            path = data_dir / name
            with path.open("w") as f:
                for s in samples: f.write(json.dumps(s) + "\n")
            print(f"  wrote {len(samples):>6} to {path}")

        unique_descs = len({s["messages"][1]["content"][0]["text"] for s in description_train})
        print(f"\n  unique training descriptions: {unique_descs}")
        volume.commit()

        return {
            "train_desc": len(description_train),
            "eval_desc": len(description_eval),
            "multitask": len(multitask_train),
            "unique_descriptions": unique_descs,
            "filtered_tiny_bboxes": n_filtered_bboxes,
            "augmentation": "8x (D4 dihedral group)",
            "min_pit_pixels": MIN_PIT_PIXELS,
        }


@app.local_entrypoint()
def main():
    print("Submitting v10 production dataset prep to Modal...")
    result = prepare.remote()
    print(f"\nPrep complete:")
    for k, v in result.items(): print(f"  {k}: {v}")
