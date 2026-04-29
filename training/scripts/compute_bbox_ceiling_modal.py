"""Compute the theoretical bbox ceiling on SmallMinesDS test split.

Answers: "What's the highest pixel IoU any bbox-based method could achieve?"
by rasterizing the GT-derived connected-component bboxes and comparing against
the GT pixel masks. No model inference, just mask analysis.

CPU-only, costs pennies.

Usage:
    uv run modal run scripts/compute_bbox_ceiling_modal.py
"""

from __future__ import annotations

import modal

app = modal.App("galamsey-bbox-ceiling")
volume = modal.Volume.from_name("galamsey", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libexpat1")
    .pip_install("huggingface_hub", "numpy>=2.0", "rasterio>=1.3", "scipy>=1.13")
)


@app.function(
    image=image,
    volumes={"/galamsey": volume},
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def compute_ceiling() -> dict:
    import csv
    import zipfile
    from pathlib import Path

    import numpy as np
    import rasterio
    from huggingface_hub import hf_hub_download
    from scipy.ndimage import label

    zip_path = hf_hub_download("ellaampy/SmallMinesDS", "SmallMinesDS.zip", repo_type="dataset")
    split_csv_paths = {}
    for year in ("2016", "2022"):
        split_csv_paths[year] = hf_hub_download(
            "ellaampy/SmallMinesDS", f"data_splits/train_test_splits_{year}.csv", repo_type="dataset"
        )

    def patch_id(entry_name):
        stem = Path(entry_name).stem
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                return (int(parts[-2]), parts[-1])
            except ValueError:
                pass
        return (-1, "unknown")

    split_assignment = {}
    for year, csv_path in split_csv_paths.items():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                key = patch_id(row.get("patch_name", ""))
                if key[0] == -1:
                    continue
                split_assignment[key] = (row.get("split") or "train").strip().lower()

    with zipfile.ZipFile(zip_path) as zf:
        mask_tiffs = sorted(n for n in zf.namelist() if "MASK" in n and n.endswith(".tif"))
        test_masks = [n for n in mask_tiffs if split_assignment.get(patch_id(n)) == "test"]
        print(f"Test masks: {len(test_masks)}")

        tp = fp = fn = 0
        per_patch_iou = []

        for i, name in enumerate(test_masks):
            with zf.open(name) as f:
                with rasterio.MemoryFile(f.read()) as mf:
                    with mf.open() as src:
                        gt = (src.read(1) > 0).astype(np.uint8)

            labeled, n_pits = label(gt)
            h, w = gt.shape
            bbox_mask = np.zeros_like(gt)
            for pid in range(1, n_pits + 1):
                ys, xs = np.where(labeled == pid)
                if ys.size == 0:
                    continue
                bbox_mask[ys.min():ys.max()+1, xs.min():xs.max()+1] = 1

            t = int(((bbox_mask == 1) & (gt == 1)).sum())
            f_p = int(((bbox_mask == 1) & (gt == 0)).sum())
            f_n = int(((bbox_mask == 0) & (gt == 1)).sum())
            tp += t
            fp += f_p
            fn += f_n
            if t + f_p + f_n > 0:
                per_patch_iou.append(t / (t + f_p + f_n))

            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(test_masks)}] running ceiling IoU = {tp / max(tp+fp+fn, 1):.4f}")

        iou = tp / max(tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        sdc = 2 * tp / max(2 * tp + fp + fn, 1)
        mean_iou = sum(per_patch_iou) / len(per_patch_iou) if per_patch_iou else 0

        print(f"\nTheoretical bbox ceiling on {len(test_masks)} test masks:")
        print(f"  IoU       = {iou:.4f}")
        print(f"  Precision = {prec:.4f}")
        print(f"  Recall    = {rec:.4f} (should be ~1.0)")
        print(f"  SDC       = {sdc:.4f}")
        print(f"  Mean per-patch IoU = {mean_iou:.4f}")

        return {
            "n_test_masks": len(test_masks),
            "ceiling_iou": round(iou, 4),
            "ceiling_precision": round(prec, 4),
            "ceiling_recall": round(rec, 4),
            "ceiling_sdc": round(sdc, 4),
            "ceiling_mean_per_patch_iou": round(mean_iou, 4),
        }


@app.local_entrypoint()
def main():
    result = compute_ceiling.remote()
    print(f"\nCeiling: {result}")
