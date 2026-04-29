"""Generate a demo pack for the dashboard.

Picks a handful of real Ghana test patches, runs v8-e3 inference on each,
saves RGB.png + SWIR.png + detections.json, plus a manifest with lat/lng.
Output goes to /galamsey/demo_pack/ on the Modal volume; pull locally with
`modal volume get galamsey demo_pack ./app/public/`.

Usage:
    uv run modal run scripts/gen_demo_pack_modal.py
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

V8_CHECKPOINT = (
    "/galamsey/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-20260416_223020"
    "/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e2s17719-20260417_004324"
)  # v9-e3 (same variable name kept for simplicity)

BAND_RED, BAND_GREEN, BAND_BLUE = 2, 1, 0
BAND_SWIR2, BAND_SWIR1, BAND_NIR = 9, 8, 6

GROUNDING_PROMPT = (
    "You are viewing two images of the same Sentinel-2 patch: a natural-color RGB "
    "composite and a SWIR false-color composite. Using both views, detect any "
    "illegal small-scale gold mining pits. "
    'Provide result as a valid JSON: [{"label": str, "bbox": [x1,y1,x2,y2]}, ...]. '
    "Coordinates must be normalized to 0-1. If no pits are visible, return []."
)

DESCRIPTION_PROMPT = (
    "You are viewing a Sentinel-2 satellite patch of Ghana. Describe any illegal "
    "artisanal gold mining activity you observe: number of pits, their spatial "
    "distribution, and approximate affected area."
)

# Approximate lat/lng centers of SmallMinesDS patches, for demo pins on the map.
# Southwestern Ghana galamsey belt: ~5.5–7.0 N, -3.0 to -0.5 E.
# Without per-patch geocodes in the dataset, we spread demo patches across the
# known galamsey region so they land where the problem is real.
DEMO_LOCATIONS = [
    (6.05, -1.78),  # Obuasi (gold belt)
    (5.87, -2.18),  # Tarkwa
    (6.23, -1.42),  # Asante Akim
    (6.42, -2.05),  # Amansie West
    (5.75, -1.95),  # Dunkwa
    (6.15, -0.87),  # Atiwa forest
    (6.58, -1.62),  # Offinso
    (5.63, -2.40),  # Prestea
]

app = modal.App("galamsey-gen-demo-pack")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libexpat1")
    .pip_install(
        "torch>=2.5",
        "torchvision",
        "transformers>=4.51",
        "pillow",
        "huggingface_hub",
        "accelerate",
        "numpy>=2.0",
        "rasterio>=1.3",
    )
)


@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    gpu="H100",
    timeout=1200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_demo_pack() -> dict:
    import csv
    import json
    import re
    import zipfile
    from pathlib import Path

    import numpy as np
    import rasterio
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("Loading SmallMinesDS...")
    zip_path = hf_hub_download(
        repo_id="ellaampy/SmallMinesDS",
        filename="SmallMinesDS.zip",
        repo_type="dataset",
    )
    split_csv = hf_hub_download(
        repo_id="ellaampy/SmallMinesDS",
        filename="data_splits/train_test_splits_2022.csv",
        repo_type="dataset",
    )

    def patch_id(name):
        stem = Path(name).stem
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                return (int(parts[-2]), parts[-1])
            except ValueError:
                pass
        return (-1, "?")

    # Find 8 varied test positives
    test_pos_ids = set()
    with open(split_csv) as f:
        for row in csv.DictReader(f):
            if row.get("split", "").strip().lower() == "test":
                try:
                    cp = float(row.get("class_percentage", 0))
                except (ValueError, TypeError):
                    cp = 0.0
                if cp > 0.0:
                    test_pos_ids.add(patch_id(row["patch_name"]))

    print(f"  {len(test_pos_ids)} test positives in 2022 split")

    # Pull 8 patches of varying pit density
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        image_names = sorted(n for n in names if "IMAGE" in n and n.lower().endswith((".tif", ".tiff")))
        mask_names = sorted(n for n in names if "MASK" in n and n.lower().endswith((".tif", ".tiff")))
        mask_by_id = {patch_id(n): n for n in mask_names}

        selected = []
        step = max(1, len(test_pos_ids) // 8)
        sorted_pos = sorted(test_pos_ids)
        for i, key in enumerate(sorted_pos[::step][:8]):
            img_name = next((n for n in image_names if patch_id(n) == key), None)
            if img_name is None:
                continue
            selected.append({"key": key, "image": img_name, "mask": mask_by_id.get(key)})

        print(f"Selected {len(selected)} demo patches")

    # Composite helpers
    def stretch(b):
        lo, hi = np.percentile(b, [2, 98])
        return ((np.clip(b, lo, hi) - lo) / max(float(hi - lo), 1e-9) * 255.0).astype(np.uint8)

    def rgb(bands):
        return np.stack([stretch(bands[BAND_RED]), stretch(bands[BAND_GREEN]), stretch(bands[BAND_BLUE])], axis=-1)

    def swir(bands):
        return np.stack([stretch(bands[BAND_SWIR2]), stretch(bands[BAND_SWIR1]), stretch(bands[BAND_NIR])], axis=-1)

    def parse_bboxes(text):
        m = re.search(r"\[.*\]", text.strip(), re.DOTALL)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        out = []
        for item in parsed if isinstance(parsed, list) else []:
            if isinstance(item, dict) and isinstance(item.get("bbox"), list) and len(item["bbox"]) == 4:
                try:
                    c = [max(0.0, min(1.0, float(x))) for x in item["bbox"]]
                    if c[2] > c[0] and c[3] > c[1]:
                        out.append({"label": item.get("label", "mining_pit"), "bbox": c})
                except (ValueError, TypeError):
                    pass
        return out

    # Load model
    print(f"Loading v8-e3 from {V8_CHECKPOINT}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(V8_CHECKPOINT, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        V8_CHECKPOINT, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    out_dir = Path("/galamsey/demo_pack")
    out_dir.mkdir(exist_ok=True, parents=True)

    manifest = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        with torch.no_grad():
            for i, entry in enumerate(selected):
                with zf.open(entry["image"]) as fp:
                    with rasterio.MemoryFile(fp.read()) as mf:
                        with mf.open() as src:
                            bands = src.read()

                rgb_img = Image.fromarray(rgb(bands))
                swir_img = Image.fromarray(swir(bands))

                patch_slug = f"patch_{i:02d}"
                rgb_path = out_dir / f"{patch_slug}_rgb.png"
                swir_path = out_dir / f"{patch_slug}_swir.png"
                rgb_img.save(rgb_path)
                swir_img.save(swir_path)

                # Run both description and grounding
                def run(prompt):
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": rgb_img},
                            {"type": "image", "image": swir_img},
                            {"type": "text", "text": prompt},
                        ],
                    }]
                    inputs = processor.apply_chat_template(
                        [messages], tokenize=True, return_dict=True,
                        return_tensors="pt", add_generation_prompt=True,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
                    plen = inputs["input_ids"].shape[1]
                    out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                    return processor.tokenizer.decode(out_ids[0, plen:], skip_special_tokens=True).strip()

                description = run(DESCRIPTION_PROMPT)
                grounding_raw = run(GROUNDING_PROMPT)
                bboxes = parse_bboxes(grounding_raw)

                lat, lng = DEMO_LOCATIONS[i % len(DEMO_LOCATIONS)]
                manifest.append({
                    "slug": patch_slug,
                    "rgb": f"{patch_slug}_rgb.png",
                    "swir": f"{patch_slug}_swir.png",
                    "lat": lat,
                    "lng": lng,
                    "description": description,
                    "bboxes": bboxes,
                })
                print(f"  [{i+1}/{len(selected)}] {patch_slug}: {len(bboxes)} bboxes, {description[:80]}...")

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    volume.commit()
    print(f"\nWrote {len(manifest)} demo patches to {out_dir}")
    return {"n_patches": len(manifest), "manifest": manifest}


@app.local_entrypoint()
def main():
    print("Generating demo pack on Modal...")
    result = generate_demo_pack.remote()
    print(f"\nDone. {result['n_patches']} patches generated.")
    print("Pull locally with:")
    print("  modal volume get galamsey demo_pack ./app/public/")
