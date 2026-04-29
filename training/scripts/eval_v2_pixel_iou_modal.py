"""v2 pixel-IoU eval on the full SmallMinesDS test split.

Puts v2 on the same scoreboard as Ofori-Ampofo et al. 2025's Table I:
- U-Net + ResNet-50 (ImageNet pretrained, RGB): IoU 0.7539
- Prithvi-EO-2.0 (300M, RGB+): IoU 0.7579
- SAM-2 (fine-tuned, RGB): IoU 0.4261
- Random Forest (RGB+): IoU 0.5027

Methodology:
1. Download SmallMinesDS.zip and split CSVs (cached from prior runs)
2. Build the test patch roster (same as v2 prep: test split, all 1,287 patches)
3. For each test patch:
   a. Read 13-band TIFF, render SWIR2-SWIR1-NIR composite with 2-98 percentile stretch
   b. Save composite to a temp PNG (v2's processor expects an image file/PIL)
   c. Read the GT mask TIFF (128x128 uint8, 1 = mining)
   d. Run v2 inference on the grounding prompt
   e. Parse the generated JSON output into a list of [x1,y1,x2,y2] bboxes
   f. Rasterize bboxes → 128x128 binary prediction mask (1 inside any bbox)
   g. Accumulate TP / FP / FN against the GT mask
4. Aggregate: compute IoU, Precision, Recall, SDC over all pixels in the test split
5. Also report:
   - Per-patch mean IoU (arithmetic mean of per-patch IoU scores)
   - Count of patches with unparseable output
   - Count of patches where v2 correctly returned "[]" on negatives

Cost estimate: 1,287 patches × ~1 sec/inference ≈ 22 min on H100 ≈ $1.50

Usage:
    uv run modal run scripts/eval_v2_pixel_iou_modal.py
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

V2_CHECKPOINT = (
    "/galamsey/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-20260415_231743"
    "/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s2217-20260415_231743"
)

BAND_SWIR2 = 9
BAND_SWIR1 = 8
BAND_NIR = 6

GROUNDING_PROMPT = (
    "Inspect the image and detect any illegal small-scale gold mining pits. "
    'Provide result as a valid JSON: [{"label": str, "bbox": [x1,y1,x2,y2]}, ...]. '
    "Coordinates must be normalized to 0-1. If no pits are visible, return []."
)

app = modal.App("galamsey-v2-pixel-iou")
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
        "scipy>=1.13",
    )
)


@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    gpu="H100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_pixel_iou_eval() -> dict:
    import csv
    import io
    import json
    import re
    import time
    import zipfile
    from pathlib import Path

    import numpy as np
    import rasterio
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("=" * 72)
    print("v2 PIXEL-IoU EVAL, full SmallMinesDS test split (1,287 patches)")
    print("Direct scoreboard comparison to Ofori-Ampofo et al. 2025 Table I")
    print("=" * 72)

    # ---------- Load SmallMinesDS artifacts ----------
    print("\n[1/4] Loading SmallMinesDS artifacts...")
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
    print(f"  zip cached at: {zip_path}")

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

    # Build test split lookup
    split_assignment = {}
    class_percentage_by_id = {}
    for year, csv_path in split_csv_paths.items():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pname = row.get("patch_name") or ""
                split = (row.get("split") or "train").strip().lower()
                try:
                    cp = float(row.get("class_percentage", 0) or 0)
                except (ValueError, TypeError):
                    cp = 0.0
                key = patch_id(pname)
                if key[0] == -1:
                    continue
                split_assignment[key] = split if split in ("train", "test", "val") else "train"
                class_percentage_by_id[key] = cp

    # Build test roster, same order as v2 prep (sorted, test split, positives first then negatives)
    with zipfile.ZipFile(zip_path, "r") as zf_scan:
        all_names = zf_scan.namelist()
        image_tiffs = sorted(
            n for n in all_names if "IMAGE" in n and n.lower().endswith((".tif", ".tiff"))
        )
        mask_tiffs = sorted(
            n for n in all_names if "MASK" in n and n.lower().endswith((".tif", ".tiff"))
        )
        mask_by_id = {patch_id(n): n for n in mask_tiffs}

        roster = []
        for image_name in image_tiffs:
            key = patch_id(image_name)
            if key not in mask_by_id:
                continue
            if split_assignment.get(key, "train") != "test":
                continue
            cp = class_percentage_by_id.get(key, 0.0)
            roster.append({
                "key": key,
                "image_name": image_name,
                "mask_name": mask_by_id[key],
                "is_positive": cp > 0.0,
                "class_percentage": cp,
            })

    # Reorder to match v2 prep: positives first, then negatives
    test_pos = [r for r in roster if r["is_positive"]]
    test_neg = [r for r in roster if not r["is_positive"]]
    test_selected = test_pos + test_neg
    print(f"  test roster: {len(test_selected)} patches ({len(test_pos)} pos + {len(test_neg)} neg)")

    # ---------- Load v2 checkpoint ----------
    print(f"\n[2/4] Loading v2 checkpoint...")
    print(f"  {V2_CHECKPOINT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(V2_CHECKPOINT, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        V2_CHECKPOINT,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"  device: {device}")

    # ---------- Helpers ----------
    def percentile_stretch(band):
        lo, hi = np.percentile(band, [2, 98])
        clipped = np.clip(band, lo, hi)
        span = max(float(hi - lo), 1e-9)
        return ((clipped - lo) / span * 255.0).astype(np.uint8)

    def compose_swir(bands):
        r = percentile_stretch(bands[BAND_SWIR2])
        g = percentile_stretch(bands[BAND_SWIR1])
        b = percentile_stretch(bands[BAND_NIR])
        return np.stack([r, g, b], axis=-1)

    def parse_bboxes(text: str) -> list[list[float]]:
        """Extract a list of [x1,y1,x2,y2] from the model's JSON output.

        Defensive, handles: valid JSON arrays, empty arrays [], lists of dicts
        with a 'bbox' key, malformed output. Returns empty list on parse failure.
        """
        text = text.strip()
        if not text or text == "[]":
            return []
        # Try to extract the first JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        bboxes = []
        for item in parsed:
            if isinstance(item, dict) and "bbox" in item:
                bb = item["bbox"]
                if isinstance(bb, list) and len(bb) == 4:
                    try:
                        coords = [float(c) for c in bb]
                        # Clamp to [0, 1], some outputs have slightly out-of-range values
                        coords = [max(0.0, min(1.0, c)) for c in coords]
                        # Ensure x1 < x2 and y1 < y2
                        x1, y1, x2, y2 = coords
                        if x2 > x1 and y2 > y1:
                            bboxes.append([x1, y1, x2, y2])
                    except (ValueError, TypeError):
                        continue
        return bboxes

    def rasterize_bboxes(bboxes: list[list[float]], h: int, w: int) -> np.ndarray:
        """Convert list of normalized [x1,y1,x2,y2] to a 0/1 pixel mask (h, w)."""
        mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in bboxes:
            px1 = int(round(x1 * w))
            py1 = int(round(y1 * h))
            px2 = int(round(x2 * w))
            py2 = int(round(y2 * h))
            # Clamp to image bounds
            px1 = max(0, min(w, px1))
            px2 = max(0, min(w, px2))
            py1 = max(0, min(h, py1))
            py2 = max(0, min(h, py2))
            if px2 > px1 and py2 > py1:
                mask[py1:py2, px1:px2] = 1
        return mask

    # ---------- Main eval loop ----------
    print(f"\n[3/4] Running inference on {len(test_selected)} test patches...")

    # Pixel-level aggregators
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    # Per-patch aggregators
    per_patch_iou: list[float] = []
    n_unparseable = 0
    n_empty_pred_on_neg = 0
    n_empty_pred_on_pos = 0  # false negative on positive = predicted empty
    n_true_positive_class = 0  # GT has mining AND pred has any bbox
    n_true_negative_class = 0  # GT has no mining AND pred has no bbox
    n_false_positive_class = 0  # GT has no mining BUT pred has bboxes
    n_false_negative_class = 0  # GT has mining BUT pred has no bboxes

    start_time = time.time()

    with zipfile.ZipFile(zip_path, "r") as zf:
        with torch.no_grad():
            for i, entry in enumerate(test_selected):
                # Load image + mask
                with zf.open(entry["image_name"]) as fp:
                    with rasterio.MemoryFile(fp.read()) as memfile:
                        with memfile.open() as src:
                            bands = src.read()
                with zf.open(entry["mask_name"]) as fp:
                    with rasterio.MemoryFile(fp.read()) as memfile:
                        with memfile.open() as src:
                            gt_mask = src.read(1)
                gt_binary = (gt_mask > 0).astype(np.uint8)

                # Render composite → PIL
                composite = compose_swir(bands)
                composite_pil = Image.fromarray(composite)

                # Build VLM conversation
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": composite_pil},
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

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
                pred_text = processor.tokenizer.decode(
                    output_ids[0, prompt_len:], skip_special_tokens=True
                ).strip()

                # Parse bboxes
                pred_bboxes = parse_bboxes(pred_text)
                was_parseable = (pred_text == "[]") or (pred_bboxes) or ("[" in pred_text)
                if not was_parseable and pred_text != "[]":
                    n_unparseable += 1

                pred_binary = rasterize_bboxes(pred_bboxes, 128, 128)

                # Pixel-level TP/FP/FN/TN
                tp = int(((pred_binary == 1) & (gt_binary == 1)).sum())
                fp = int(((pred_binary == 1) & (gt_binary == 0)).sum())
                fn = int(((pred_binary == 0) & (gt_binary == 1)).sum())
                tn = int(((pred_binary == 0) & (gt_binary == 0)).sum())
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                # Per-patch IoU (skip patches with empty GT for mean IoU)
                if tp + fp + fn > 0:
                    patch_iou = tp / (tp + fp + fn)
                    per_patch_iou.append(patch_iou)

                # Patch-level classification stats
                gt_has_mining = gt_binary.sum() > 0
                pred_has_mining = len(pred_bboxes) > 0
                if gt_has_mining and pred_has_mining:
                    n_true_positive_class += 1
                elif not gt_has_mining and not pred_has_mining:
                    n_true_negative_class += 1
                    n_empty_pred_on_neg += 1
                elif not gt_has_mining and pred_has_mining:
                    n_false_positive_class += 1
                else:  # gt_has_mining and not pred_has_mining
                    n_false_negative_class += 1
                    n_empty_pred_on_pos += 1

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta_sec = (len(test_selected) - (i + 1)) / rate
                    running_iou = total_tp / max(total_tp + total_fp + total_fn, 1)
                    print(
                        f"  [{i+1:>4}/{len(test_selected)}] "
                        f"rate={rate:.2f}/s eta={eta_sec/60:.1f}m "
                        f"running IoU={running_iou:.4f}"
                    )

    elapsed = time.time() - start_time

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    # ---------- Theoretical bbox ceiling ----------
    # Compute IoU of GT-bboxes-rasterized vs GT-pixel-masks to find the
    # theoretical maximum any bbox-based approach could achieve on this dataset.
    # This is free (no inference, just mask analysis on the GT we already loaded).
    print(f"\n[4/5] Computing theoretical bbox ceiling...")

    from scipy.ndimage import label as scipy_label

    ceiling_tp = 0
    ceiling_fp = 0
    ceiling_fn = 0

    with zipfile.ZipFile(zip_path, "r") as zf_ceil:
        for entry in test_selected:
            with zf_ceil.open(entry["mask_name"]) as fp:
                with rasterio.MemoryFile(fp.read()) as memfile:
                    with memfile.open() as src:
                        gt_mask = src.read(1)
            gt_binary = (gt_mask > 0).astype(np.uint8)

            # Derive GT bboxes via connected components (same as training pipeline)
            labeled, n_pits = scipy_label(gt_binary)
            gt_bboxes = []
            h, w = gt_binary.shape
            for pit_id in range(1, n_pits + 1):
                pit_mask = labeled == pit_id
                ys, xs = np.where(pit_mask)
                if ys.size == 0:
                    continue
                gt_bboxes.append([
                    int(xs.min()) / w,
                    int(ys.min()) / h,
                    (int(xs.max()) + 1) / w,
                    (int(ys.max()) + 1) / h,
                ])

            # Rasterize GT bboxes into a pixel mask
            ceil_mask = rasterize_bboxes(gt_bboxes, 128, 128)

            ceiling_tp += int(((ceil_mask == 1) & (gt_binary == 1)).sum())
            ceiling_fp += int(((ceil_mask == 1) & (gt_binary == 0)).sum())
            ceiling_fn += int(((ceil_mask == 0) & (gt_binary == 1)).sum())

    ceiling_iou = safe_div(ceiling_tp, ceiling_tp + ceiling_fp + ceiling_fn)
    ceiling_precision = safe_div(ceiling_tp, ceiling_tp + ceiling_fp)
    ceiling_recall = safe_div(ceiling_tp, ceiling_tp + ceiling_fn)
    ceiling_sdc = safe_div(2 * ceiling_tp, 2 * ceiling_tp + ceiling_fp + ceiling_fn)

    print(f"  Theoretical bbox ceiling:")
    print(f"    IoU       = {ceiling_iou:.4f}")
    print(f"    Precision = {ceiling_precision:.4f}")
    print(f"    Recall    = {ceiling_recall:.4f} (should be 1.0, every GT pixel is inside its own bbox)")
    print(f"    SDC       = {ceiling_sdc:.4f}")

    # ---------- Aggregate metrics ----------
    print(f"\n[5/5] Aggregating metrics...")

    iou = safe_div(total_tp, total_tp + total_fp + total_fn)
    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    sdc = safe_div(2 * total_tp, 2 * total_tp + total_fp + total_fn)
    accuracy = safe_div(total_tp + total_tn, total_tp + total_fp + total_fn + total_tn)

    mean_per_patch_iou = sum(per_patch_iou) / len(per_patch_iou) if per_patch_iou else 0.0

    results = {
        "n_test_patches": len(test_selected),
        "n_positives": len(test_pos),
        "n_negatives": len(test_neg),
        "total_inference_time_sec": round(elapsed, 1),
        "mean_inference_time_sec": round(elapsed / len(test_selected), 3),
        # Pixel-level (directly comparable to paper's Table I)
        "pixel_iou": round(iou, 4),
        "pixel_precision": round(precision, 4),
        "pixel_recall": round(recall, 4),
        "pixel_sdc_f1": round(sdc, 4),
        "pixel_accuracy": round(accuracy, 4),
        # Per-patch mean (robust to outliers)
        "mean_per_patch_iou": round(mean_per_patch_iou, 4),
        "n_patches_with_nonempty_union": len(per_patch_iou),
        # Patch-level classification
        "patch_tp": n_true_positive_class,
        "patch_fp": n_false_positive_class,
        "patch_fn": n_false_negative_class,
        "patch_tn": n_true_negative_class,
        "patch_accuracy": round(
            (n_true_positive_class + n_true_negative_class) / len(test_selected), 4
        ),
        # Output hygiene
        "n_unparseable_outputs": n_unparseable,
        "n_empty_pred_on_negative": n_empty_pred_on_neg,
        "n_empty_pred_on_positive_false_neg": n_empty_pred_on_pos,
        # Theoretical bbox ceiling
        "ceiling_iou": round(ceiling_iou, 4),
        "ceiling_precision": round(ceiling_precision, 4),
        "ceiling_recall": round(ceiling_recall, 4),
        "ceiling_sdc": round(ceiling_sdc, 4),
    }

    print("\n" + "=" * 72)
    print("RESULTS, v2 on full SmallMinesDS test split")
    print("=" * 72)
    for key, value in results.items():
        print(f"  {key:<40} {value}")

    print("\n" + "=" * 72)
    print("COMPARISON TO OFORI-AMPOFO ET AL. 2025 TABLE I (mining class, same split)")
    print("=" * 72)
    print(f"  {'Model':<35} {'IoU':>8} {'Precision':>11} {'Recall':>8} {'SDC':>8}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 11} {'-' * 8} {'-' * 8}")
    print(f"  {'Random Forest (RGB)':<35} {0.4109:>8.4f} {0.9238:>11.4f} {0.4253:>8.4f} {0.5825:>8.4f}")
    print(f"  {'SAM-2 fine-tuned (RGB)':<35} {0.4261:>8.4f} {0.7416:>11.4f} {0.5004:>8.4f} {0.5976:>8.4f}")
    print(f"  {'U-Net pretrained (RGB)':<35} {0.7539:>8.4f} {0.9278:>11.4f} {0.8009:>8.4f} {0.8597:>8.4f}")
    print(f"  {'Random Forest (RGB+)':<35} {0.5027:>8.4f} {0.9531:>11.4f} {0.5155:>8.4f} {0.6691:>8.4f}")
    print(f"  {'U-Net from scratch (RGB+)':<35} {0.6513:>8.4f} {0.9642:>11.4f} {0.6675:>8.4f} {0.7889:>8.4f}")
    print(f"  {'Prithvi-EO-2.0 300M (RGB+)':<35} {0.7579:>8.4f} {0.8558:>11.4f} {0.8689:>8.4f} {0.8623:>8.4f}")
    print(f"  {'Prithvi-EO-2.0 600M (RGB+)':<35} {0.7560:>8.4f} {0.8457:>11.4f} {0.8769:>8.4f} {0.8610:>8.4f}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 11} {'-' * 8} {'-' * 8}")
    print(f"  {'v2 (LFM2.5-VL-450M, SWIR2-1-NIR)':<35} {iou:>8.4f} {precision:>11.4f} {recall:>8.4f} {sdc:>8.4f}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 11} {'-' * 8} {'-' * 8}")
    print(f"  {'THEORETICAL BBOX CEILING':<35} {ceiling_iou:>8.4f} {ceiling_precision:>11.4f} {ceiling_recall:>8.4f} {ceiling_sdc:>8.4f}")
    print(f"\n  (Bbox ceiling = IoU if our bboxes perfectly matched the GT connected-component bboxes."
          f"\n   Any bbox-based method is bounded above by this ceiling on this dataset."
          f"\n   v2 / ceiling = {iou/ceiling_iou:.1%} of theoretical maximum.)" if ceiling_iou > 0 else "")

    return results


@app.local_entrypoint()
def main() -> None:
    print("Submitting v2 pixel-IoU eval to Modal...")
    result = run_pixel_iou_eval.remote()
    print(f"\nDone. Final results:")
    import json
    print(json.dumps(result, indent=2))
