"""Export fine-tuned v8-e3 to ONNX for browser WebGPU inference.

Produces the same 3-model layout Liquid AI publishes for the base model:
  onnx/
    embed_tokens.onnx           (fp32)  + embed_tokens_fp16.onnx
    vision_encoder.onnx         (fp32)  + _fp16 + _q4 + _q8
    decoder_model_merged.onnx   (fp32)  + _fp16 + _q4 + _q8

Uses optimum-cli's `image-text-to-text` task. LFM2.5-VL registers its own
ONNX config via trust_remote_code, so the standard exporter works.

Output goes to:
  /galamsey/onnx_exports/galamsey-v8-e3-onnx/

Then pushes to HuggingFace Hub as `{HF_USERNAME}/galamsey-v8-e3-onnx` if
HF_TOKEN (from the huggingface-secret) has write scope.

Usage:
    uv run modal run scripts/export_v8_onnx_modal.py
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

V8_CHECKPOINT = (
    "/galamsey/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-20260416_173647"
    "/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e2s8860-20260416_191804"
)

# Target HF repo. Will be created as a public repo if it doesn't exist.
HF_REPO_ID = "samwell/galamsey-v8-e3-onnx"

app = modal.App("galamsey-export-v8-onnx")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libexpat1")
    .pip_install(
        "torch>=2.5",
        "torchvision",
        "transformers>=4.51",
        "accelerate",
        "pillow",
        "huggingface_hub",
        "numpy>=2.0",
        "optimum[onnxruntime]>=1.24",
        "onnx>=1.17",
        "onnxruntime>=1.19",
        "onnxconverter-common",
    )
)


@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    gpu="H100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def export_to_onnx() -> dict:
    import os
    import shutil
    import subprocess
    from pathlib import Path

    from huggingface_hub import HfApi, create_repo, upload_folder

    out_dir = Path("/galamsey/onnx_exports/galamsey-v8-e3-onnx")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {V8_CHECKPOINT} to ONNX via optimum-cli...")
    print(f"  Output: {out_dir}")

    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        V8_CHECKPOINT,
        "--task",
        "image-text-to-text",
        "--trust-remote-code",
        str(out_dir),
    ]
    print("  Command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT (last 80 lines):")
    print("\n".join(result.stdout.splitlines()[-80:]))
    if result.returncode != 0:
        print("STDERR:")
        print(result.stderr[-4000:])
        raise RuntimeError(f"optimum-cli export failed with code {result.returncode}")

    # List what we got
    print("\nExport contents:")
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out_dir)}  ({p.stat().st_size/1e6:.1f} MB)")

    volume.commit()

    # Upload to HF
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"status": "exported_not_uploaded", "reason": "no HF_TOKEN in env"}

    print(f"\nUploading to HuggingFace: {HF_REPO_ID}")
    try:
        create_repo(HF_REPO_ID, exist_ok=True, repo_type="model", token=token, private=False)
        upload_folder(
            folder_path=str(out_dir),
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=token,
            commit_message="v8-e3 ONNX export, galamsey VLM fine-tuned on SmallMinesDS",
        )
        print(f"  ✓ Uploaded to https://huggingface.co/{HF_REPO_ID}")
        return {"status": "uploaded", "repo_id": HF_REPO_ID}
    except Exception as e:
        print(f"  Upload failed: {e}")
        return {"status": "exported_but_upload_failed", "error": str(e)}


@app.local_entrypoint()
def main():
    print("Submitting ONNX export to Modal...")
    result = export_to_onnx.remote()
    print("\nDone:", result)
