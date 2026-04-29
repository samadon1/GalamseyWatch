"""Batch-verify v10-e5 against earthrise-media's 685 Ghana mining sites.

For each earthrise polygon centroid:
  1. Fetch a 1.28 km Sentinel-2 RGB+SWIR tile from our Cloud Run SimSat
  2. Run v10-e5 inference
  3. Record whether v10 confirms or misses

Output: `/galamsey/earthrise_verification_v10.json` with per-site status.

Usage:
    uv run modal run scripts/verify_earthrise_sites_v10_modal.py
"""

from __future__ import annotations

import json
import pathlib

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"
SIMSAT_URL = "https://simsat-sim-943572188770.us-central1.run.app"

V10_CHECKPOINT = (
    "/galamsey/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-20260418_165633"
    "/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e5s59065-20260418_165633"
)

GROUNDING_PROMPT = (
    "You are viewing two images of the same Sentinel-2 patch: a natural-color RGB "
    "composite and a SWIR false-color composite. Using both views, detect any "
    "illegal small-scale gold mining pits. "
    'Provide result as a valid JSON: [{"label": str, "bbox": [x1,y1,x2,y2]}, ...]. '
    "Coordinates must be normalized to 0-1. If no pits are visible, return []."
)

# Read earthrise centroids locally; pass to the Modal function as JSON string.
def _load_earthrise_centroids(path: pathlib.Path) -> list[dict]:
    d = json.loads(path.read_text())
    sites = []
    for i, f in enumerate(d["features"]):
        g = f["geometry"]
        if g["type"] == "Polygon":
            rings = g["coordinates"]
        elif g["type"] == "MultiPolygon":
            rings = [ring for poly in g["coordinates"] for ring in poly]
        else:
            continue
        all_coords = [pt for ring in rings for pt in ring]
        if not all_coords:
            continue
        lons = [c[0] for c in all_coords]
        lats = [c[1] for c in all_coords]
        sites.append({
            "id": i,
            "lng": round(sum(lons) / len(lons), 5),
            "lat": round(sum(lats) / len(lats), 5),
        })
    return sites


app = modal.App("galamsey-verify-earthrise-v10")
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
        "requests>=2.31",
    )
)


@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    gpu="H100",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def verify_sites(sites_json: str) -> dict:
    import io
    import json as pyjson
    import re
    import time

    import requests
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    sites = pyjson.loads(sites_json)
    print(f"Verifying {len(sites)} earthrise sites against v10-e5…")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(V10_CHECKPOINT, trust_remote_code=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            V10_CHECKPOINT, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .to(device)
        .eval()
    )

    def parse_bboxes(text: str):
        m = re.search(r"\[.*\]", text.strip(), re.DOTALL)
        if not m:
            return []
        try:
            parsed = pyjson.loads(m.group(0))
        except pyjson.JSONDecodeError:
            return []
        out = []
        for it in parsed if isinstance(parsed, list) else []:
            if isinstance(it, dict) and isinstance(it.get("bbox"), list) and len(it["bbox"]) == 4:
                try:
                    c = [max(0.0, min(1.0, float(x))) for x in it["bbox"]]
                    if c[2] > c[0] and c[3] > c[1]:
                        out.append({"label": it.get("label", "mining_pit"), "bbox": c})
                except (ValueError, TypeError):
                    pass
        return out

    def fetch_tile(lon: float, lat: float, bands: str) -> Image.Image | None:
        url = f"{SIMSAT_URL}/data/image/sentinel"
        params = {
            "lon": lon,
            "lat": lat,
            "timestamp": "2024-01-15T00:00:00Z",
            "size_km": 1.28,
            "window_seconds": 730 * 24 * 60 * 60,
            "return_type": "png",
        }
        # Multi-value band param
        band_tuples = [("spectral_bands", b) for b in bands.split(",")]
        try:
            r = requests.get(url, params=[*params.items(), *band_tuples], timeout=60)
            if r.status_code != 200:
                return None
            return Image.open(io.BytesIO(r.content))
        except Exception as e:
            print(f"  fetch failed: {e}")
            return None

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=4)

    results = []
    start = time.time()
    with torch.no_grad():
        for i, s in enumerate(sites):
            lng, lat = s["lng"], s["lat"]
            # Parallelize the 2 SimSat fetches, halves fetch latency per site
            fut_rgb = pool.submit(fetch_tile, lng, lat, "red,green,blue")
            fut_swir = pool.submit(fetch_tile, lng, lat, "swir22,swir16,nir")
            rgb = fut_rgb.result()
            swir = fut_swir.result()
            if rgb is None or swir is None:
                results.append({**s, "status": "fetch_error", "bboxes": []})
                continue

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb},
                    {"type": "image", "image": swir},
                    {"type": "text", "text": GROUNDING_PROMPT},
                ],
            }]
            inputs = processor.apply_chat_template(
                [messages],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
            prompt_len = inputs["input_ids"].shape[1]
            out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            text = processor.tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=True).strip()

            bboxes = parse_bboxes(text)
            results.append({
                **s,
                "status": "confirmed" if bboxes else "empty",
                "bboxes": bboxes,
            })

            if (i + 1) % 25 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta_min = (len(sites) - (i + 1)) / rate / 60
                n_conf = sum(1 for r in results if r["status"] == "confirmed")
                print(f"  [{i+1}/{len(sites)}] rate={rate:.2f}/s eta={eta_min:.1f}m confirmed={n_conf}", flush=True)
                # Checkpoint save so a timeout doesn't lose everything
                pathlib.Path("/galamsey/earthrise_verification_v10.json").write_text(
                    pyjson.dumps(results, indent=2)
                )
                volume.commit()

    n_confirmed = sum(1 for r in results if r["status"] == "confirmed")
    n_empty = sum(1 for r in results if r["status"] == "empty")
    n_fetch_err = sum(1 for r in results if r["status"] == "fetch_error")
    print(f"\nDone. confirmed={n_confirmed} empty={n_empty} fetch_error={n_fetch_err}")

    out_path = pathlib.Path("/galamsey/earthrise_verification_v10.json")
    out_path.write_text(pyjson.dumps(results, indent=2))
    volume.commit()
    print(f"Saved to {out_path}")

    return {
        "total": len(results),
        "confirmed": n_confirmed,
        "empty": n_empty,
        "fetch_error": n_fetch_err,
    }


@app.local_entrypoint()
def main():
    # Source the 685 centroids straight from app/public/mining_sites.json rather
    # than re-parsing the earthrise GeoJSON, the public JSON was derived from
    # that same GeoJSON and carries the exact (id, lng, lat) tuples we need.
    sites_path = pathlib.Path(__file__).resolve().parents[2] / "app" / "public" / "mining_sites.json"
    raw = json.loads(sites_path.read_text())
    sites = [{"id": s["id"], "lng": s["lng"], "lat": s["lat"]} for s in raw]
    print(f"Loaded {len(sites)} sites from {sites_path}")
    result = verify_sites.remote(json.dumps(sites))
    print("\nResult:", result)
