"""Swap our fine-tuned v8-e3 weights into LiquidAI's base ONNX structure.

LiquidAI/LFM2.5-VL-450M-ONNX uses the exact same architecture as our fine-tuned
v8-e3 checkpoint, only the parameter values differ. Since optimum-cli doesn't
yet support lfm2_vl, we re-use Liquid's pre-exported ONNX graphs and replace
their weight initializers in-place with our fine-tuned tensors.

Output: modified ONNX files + config uploaded to `{HF_USERNAME}/galamsey-v8-e3-onnx`.

Usage:
    uv run modal run scripts/swap_v8_weights_into_base_onnx_modal.py
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

V8_CHECKPOINT = (
    "/galamsey/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-20260416_173647"
    "/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e2s8860-20260416_191804"
)
BASE_ONNX_REPO = "LiquidAI/LFM2.5-VL-450M-ONNX"
HF_REPO_ID = "samwell/galamsey-v8-e3-onnx"

app = modal.App("galamsey-swap-onnx-weights")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.5",
        "safetensors>=0.4",
        "onnx>=1.17",
        "huggingface_hub",
        "numpy>=2.0",
    )
)


@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    timeout=3600,
    cpu=4,
    memory=16384,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def swap_weights() -> dict:
    import os
    import shutil
    from pathlib import Path

    import numpy as np
    import onnx
    from huggingface_hub import create_repo, snapshot_download, upload_folder
    from onnx import numpy_helper
    from safetensors.torch import load_file as load_safetensors

    print("=" * 70)
    print("v8-e3 weight-swap into LiquidAI base ONNX")
    print("=" * 70)

    # 1. Download base ONNX files
    print(f"\n[1/4] Downloading {BASE_ONNX_REPO}…")
    base_dir = snapshot_download(
        repo_id=BASE_ONNX_REPO,
        repo_type="model",
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  cached at {base_dir}")

    # 2. Load our fine-tuned safetensors
    print(f"\n[2/4] Loading v8-e3 safetensors…")
    ckpt_path = Path(V8_CHECKPOINT) / "model.safetensors"
    ft_weights = load_safetensors(str(ckpt_path))
    print(f"  {len(ft_weights)} tensors loaded")

    # Build a lookup by tensor name with various normalizations to improve match rate
    ft_by_suffix: dict[str, tuple[str, "object"]] = {}
    for name, tensor in ft_weights.items():
        arr = tensor.detach().cpu().float().numpy()
        ft_by_suffix[name] = (name, arr)
        # Also index by suffix components so ONNX's slash-based paths find a match
        parts = name.split(".")
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            ft_by_suffix.setdefault(suffix, (name, arr))
            # Slash variant
            ft_by_suffix.setdefault("/".join(parts[i:]), (name, arr))

    # 3. Walk each ONNX file, swap matching initializers
    out_dir = Path("/galamsey/onnx_exports/galamsey-v8-e3-onnx")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-model files (config, tokenizer, preprocessor, chat template, etc.)
    for item in Path(base_dir).iterdir():
        if item.is_file():
            shutil.copy2(item, out_dir / item.name)

    total_init = 0
    total_swapped = 0
    total_shape_mismatch = 0
    unmatched_samples: list[str] = []

    onnx_src_dir = Path(base_dir) / "onnx"
    onnx_dst_dir = out_dir / "onnx"
    onnx_dst_dir.mkdir(parents=True, exist_ok=True)

    for onnx_file in sorted(onnx_src_dir.glob("*.onnx")):
        # Skip the quantized variants, only modify the fp32 references so we can
        # re-quantize cleanly afterwards. The fp16/q4/q8 variants are skipped.
        name = onnx_file.name
        if any(q in name for q in ("_fp16", "_q4", "_q8")):
            print(f"\n[3/4] Skipping {name} (quantized variant, will regenerate later)")
            continue

        print(f"\n[3/4] Processing {name}…")
        model = onnx.load(str(onnx_file))

        file_swapped = 0
        file_total = 0
        for init in model.graph.initializer:
            file_total += 1
            total_init += 1

            # Try several candidate lookups
            candidates = [init.name]
            # Strip common ONNX prefixes
            stripped = init.name.lstrip("/").replace("/", ".")
            candidates.append(stripped)
            # Last-N components
            parts = init.name.replace("/", ".").lstrip(".").split(".")
            for i in range(len(parts)):
                candidates.append(".".join(parts[i:]))
                candidates.append("/".join(parts[i:]))

            hit = None
            for cand in candidates:
                if cand in ft_by_suffix:
                    hit = ft_by_suffix[cand]
                    break

            if hit is None:
                if len(unmatched_samples) < 10:
                    unmatched_samples.append(init.name)
                continue

            ft_name, ft_arr = hit
            onnx_arr = numpy_helper.to_array(init)
            if ft_arr.shape != onnx_arr.shape:
                total_shape_mismatch += 1
                if len(unmatched_samples) < 20:
                    unmatched_samples.append(f"{init.name} shape {onnx_arr.shape} vs {ft_name} {ft_arr.shape}")
                continue

            # Replace
            new_init = numpy_helper.from_array(ft_arr.astype(onnx_arr.dtype), init.name)
            init.CopyFrom(new_init)
            file_swapped += 1
            total_swapped += 1

        print(f"    {file_swapped}/{file_total} initializers swapped")
        onnx.save(model, str(onnx_dst_dir / name), save_as_external_data=False)

    print(f"\n[summary] swapped {total_swapped}/{total_init} initializers across all ONNX files")
    print(f"  shape mismatches: {total_shape_mismatch}")
    print(f"  unmatched samples (first 10):")
    for s in unmatched_samples[:10]:
        print(f"    - {s}")

    if total_swapped == 0:
        raise RuntimeError("No initializers swapped, naming convention mismatch. Need a different approach.")

    volume.commit()

    # 4. Upload to HF
    print(f"\n[4/4] Uploading to {HF_REPO_ID}…")
    token = os.environ.get("HF_TOKEN")
    try:
        create_repo(HF_REPO_ID, exist_ok=True, repo_type="model", token=token, private=False)
        upload_folder(
            folder_path=str(out_dir),
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=token,
            commit_message=f"v8-e3 weight-swap ({total_swapped}/{total_init} initializers)",
        )
        print(f"  ✓ https://huggingface.co/{HF_REPO_ID}")
        return {
            "status": "uploaded",
            "repo_id": HF_REPO_ID,
            "swapped": total_swapped,
            "total": total_init,
            "shape_mismatches": total_shape_mismatch,
        }
    except Exception as e:
        print(f"  Upload failed: {e}")
        return {
            "status": "swapped_not_uploaded",
            "swapped": total_swapped,
            "total": total_init,
            "error": str(e),
        }


@app.local_entrypoint()
def main():
    print("Submitting weight-swap to Modal…")
    result = swap_weights.remote()
    print("\nDone:", result)
