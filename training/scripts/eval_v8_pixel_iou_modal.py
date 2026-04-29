"""v8 (450M + 4x aug RGB+SWIR) pixel-IoU eval, epoch 2 vs epoch 3 comparison.

Runs inference with both e2 and e3 checkpoints on the full SmallMinesDS test
split. Purpose: detect overfitting in epoch 3 (train loss dropped 0.22 -> 0.072
across the last epoch, which could be genuine learning or memorization).

Usage:
    uv run modal run scripts/eval_v8_pixel_iou_modal.py
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

V8_RUN_DIR = (
    "/galamsey/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-20260416_173647"
)
V8_CHECKPOINT_E2 = (
    f"{V8_RUN_DIR}/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e2s5908-20260416_173647"
)
V8_CHECKPOINT_E3 = (
    f"{V8_RUN_DIR}/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e2s8860-20260416_191804"
)

BAND_RED = 2
BAND_GREEN = 1
BAND_BLUE = 0
BAND_SWIR2 = 9
BAND_SWIR1 = 8
BAND_NIR = 6

GROUNDING_PROMPT = (
    "You are viewing two images of the same Sentinel-2 patch: a natural-color RGB "
    "composite and a SWIR false-color composite. Using both views, detect any "
    "illegal small-scale gold mining pits. "
    'Provide result as a valid JSON: [{"label": str, "bbox": [x1,y1,x2,y2]}, ...]. '
    "Coordinates must be normalized to 0-1. If no pits are visible, return []."
)

app = modal.App("galamsey-v8-pixel-iou")
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
    timeout=5400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_pixel_iou_eval() -> dict:
    import csv
    import gc
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
    print("v8 PIXEL-IoU EVAL, e2 vs e3 on full SmallMinesDS test split")
    print("=" * 72)

    # ---------- Load SmallMinesDS artifacts ----------
    print("\n[1/5] Loading SmallMinesDS artifacts...")
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

    test_pos = [r for r in roster if r["is_positive"]]
    test_neg = [r for r in roster if not r["is_positive"]]
    test_selected = test_pos + test_neg
    print(f"  test roster: {len(test_selected)} patches ({len(test_pos)} pos + {len(test_neg)} neg)")

    # ---------- Helpers ----------
    def percentile_stretch(band):
        lo, hi = np.percentile(band, [2, 98])
        clipped = np.clip(band, lo, hi)
        span = max(float(hi - lo), 1e-9)
        return ((clipped - lo) / span * 255.0).astype(np.uint8)

    def compose_rgb(bands):
        return np.stack([
            percentile_stretch(bands[BAND_RED]),
            percentile_stretch(bands[BAND_GREEN]),
            percentile_stretch(bands[BAND_BLUE]),
        ], axis=-1)

    def compose_swir(bands):
        return np.stack([
            percentile_stretch(bands[BAND_SWIR2]),
            percentile_stretch(bands[BAND_SWIR1]),
            percentile_stretch(bands[BAND_NIR]),
        ], axis=-1)

    def parse_bboxes(text: str) -> list[list[float]]:
        text = text.strip()
        if not text or text == "[]":
            return []
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
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
                        coords = [max(0.0, min(1.0, float(c))) for c in bb]
                        x1, y1, x2, y2 = coords
                        if x2 > x1 and y2 > y1:
                            bboxes.append([x1, y1, x2, y2])
                    except (ValueError, TypeError):
                        continue
        return bboxes

    def rasterize_bboxes(bboxes, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in bboxes:
            px1 = max(0, min(w, int(round(x1 * w))))
            px2 = max(0, min(w, int(round(x2 * w))))
            py1 = max(0, min(h, int(round(y1 * h))))
            py2 = max(0, min(h, int(round(y2 * h))))
            if px2 > px1 and py2 > py1:
                mask[py1:py2, px1:px2] = 1
        return mask

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    # ---------- Pre-load images + masks once ----------
    print(f"\n[2/5] Pre-loading {len(test_selected)} images and masks...")
    preloaded = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in test_selected:
            with zf.open(entry["image_name"]) as fp:
                with rasterio.MemoryFile(fp.read()) as memfile:
                    with memfile.open() as src:
                        bands = src.read()
            with zf.open(entry["mask_name"]) as fp:
                with rasterio.MemoryFile(fp.read()) as memfile:
                    with memfile.open() as src:
                        gt_mask = src.read(1)
            preloaded.append({
                "rgb": Image.fromarray(compose_rgb(bands)),
                "swir": Image.fromarray(compose_swir(bands)),
                "gt_binary": (gt_mask > 0).astype(np.uint8),
            })

    # ---------- Run inference for one checkpoint ----------
    def eval_checkpoint(name: str, ckpt_path: str) -> dict:
        print(f"\n{'=' * 72}")
        print(f"Evaluating {name}")
        print(f"  {ckpt_path}")
        print("=" * 72)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            ckpt_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        model.eval()

        total_tp = total_fp = total_fn = total_tn = 0
        per_patch_iou = []
        n_unparseable = 0
        n_tp_class = n_fp_class = n_fn_class = n_tn_class = 0

        start = time.time()

        with torch.no_grad():
            for i, pre in enumerate(preloaded):
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pre["rgb"]},
                        {"type": "image", "image": pre["swir"]},
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

                output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                pred_text = processor.tokenizer.decode(
                    output_ids[0, prompt_len:], skip_special_tokens=True
                ).strip()

                pred_bboxes = parse_bboxes(pred_text)
                if pred_text and pred_text != "[]" and not pred_bboxes and "[" not in pred_text:
                    n_unparseable += 1

                pred_binary = rasterize_bboxes(pred_bboxes, 128, 128)
                gt_binary = pre["gt_binary"]

                tp = int(((pred_binary == 1) & (gt_binary == 1)).sum())
                fp = int(((pred_binary == 1) & (gt_binary == 0)).sum())
                fn = int(((pred_binary == 0) & (gt_binary == 1)).sum())
                tn = int(((pred_binary == 0) & (gt_binary == 0)).sum())
                total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

                if tp + fp + fn > 0:
                    per_patch_iou.append(tp / (tp + fp + fn))

                gt_has = gt_binary.sum() > 0
                pred_has = len(pred_bboxes) > 0
                if gt_has and pred_has: n_tp_class += 1
                elif not gt_has and not pred_has: n_tn_class += 1
                elif not gt_has and pred_has: n_fp_class += 1
                else: n_fn_class += 1

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    eta = (len(preloaded) - (i + 1)) / rate
                    running_iou = safe_div(total_tp, total_tp + total_fp + total_fn)
                    print(
                        f"  [{i+1:>4}/{len(preloaded)}] "
                        f"rate={rate:.2f}/s eta={eta/60:.1f}m "
                        f"running_iou={running_iou:.4f}",
                        flush=True,
                    )

        elapsed = time.time() - start
        results = {
            "checkpoint": name,
            "pixel_iou": round(safe_div(total_tp, total_tp + total_fp + total_fn), 4),
            "pixel_precision": round(safe_div(total_tp, total_tp + total_fp), 4),
            "pixel_recall": round(safe_div(total_tp, total_tp + total_fn), 4),
            "pixel_sdc_f1": round(safe_div(2 * total_tp, 2 * total_tp + total_fp + total_fn), 4),
            "pixel_accuracy": round(safe_div(total_tp + total_tn, total_tp + total_fp + total_fn + total_tn), 4),
            "mean_per_patch_iou": round(sum(per_patch_iou) / len(per_patch_iou), 4) if per_patch_iou else 0.0,
            "patch_tp": n_tp_class,
            "patch_fp": n_fp_class,
            "patch_fn": n_fn_class,
            "patch_tn": n_tn_class,
            "patch_accuracy": round((n_tp_class + n_tn_class) / len(preloaded), 4),
            "n_unparseable": n_unparseable,
            "inference_time_sec": round(elapsed, 1),
        }
        print(f"\n  {name}: pixel_IoU={results['pixel_iou']:.4f} "
              f"P={results['pixel_precision']:.4f} "
              f"R={results['pixel_recall']:.4f} "
              f"SDC={results['pixel_sdc_f1']:.4f}")
        print(f"  time: {elapsed/60:.1f} min")

        # Free GPU before loading next checkpoint
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

        return results

    # ---------- Eval both checkpoints ----------
    e2_results = eval_checkpoint("v8-e2 (step 5908)", V8_CHECKPOINT_E2)
    e3_results = eval_checkpoint("v8-e3 (step 8860)", V8_CHECKPOINT_E3)

    print("\n" + "=" * 72)
    print("v8 e2 vs e3 COMPARISON")
    print("=" * 72)
    print(f"  {'Metric':<25} {'e2':>10} {'e3':>10} {'Δ (e3-e2)':>12}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 12}")
    for metric in ["pixel_iou", "pixel_precision", "pixel_recall", "pixel_sdc_f1",
                   "pixel_accuracy", "mean_per_patch_iou", "patch_accuracy"]:
        e2v, e3v = e2_results[metric], e3_results[metric]
        delta = e3v - e2v
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {metric:<25} {e2v:>10.4f} {e3v:>10.4f} {delta:>+10.4f} {arrow}")

    winner = "e3" if e3_results["pixel_iou"] > e2_results["pixel_iou"] else "e2"
    print(f"\n  Winner (by pixel IoU): v8-{winner}")

    return {"e2": e2_results, "e3": e3_results, "winner": winner}


@app.local_entrypoint()
def main() -> None:
    print("Submitting v8 pixel-IoU eval (e2 vs e3) to Modal...")
    result = run_pixel_iou_eval.remote()
    print("\n" + "=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)
    import json
    print(json.dumps(result, indent=2))
