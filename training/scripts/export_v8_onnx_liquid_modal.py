"""Export fine-tuned v8-e3 to ONNX using Liquid AI's official onnx-export repo.

Liquid4All/onnx-export ships `lfm2-vl-export`, the same CLI they use to produce
their own published ONNX builds (including the LFM2.5-VL-450M-ONNX on HF).
Since it wraps `transformers.from_pretrained()`, it accepts a local checkpoint
directory path just as well as an HF repo ID.

We run it on an H100, point it at our v8-e3 checkpoint on the galamsey volume,
and push the result to `samwell/galamsey-v8-e3-onnx`.

Usage:
    uv run modal run scripts/export_v8_onnx_liquid_modal.py
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

V8_CHECKPOINT = (
    "/galamsey/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-20260416_173647"
    "/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e2s8860-20260416_191804"
)
HF_REPO_ID = "samwell/galamsey-v8-e3-onnx"

app = modal.App("galamsey-export-onnx-liquid")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

# Base image with git + uv so we can clone Liquid's repo and resolve its deps.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .run_commands(
        "pip install uv",
        "git clone --depth 1 https://github.com/Liquid4All/onnx-export.git /opt/onnx-export",
        "cd /opt/onnx-export && uv sync --extra gpu",
    )
    .pip_install("huggingface_hub")
)


@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    gpu="H100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def export_v8_onnx() -> dict:
    import os
    import shutil
    import subprocess
    from pathlib import Path

    from huggingface_hub import create_repo, upload_folder

    out_dir = Path("/galamsey/onnx_exports/galamsey-v8-e3-onnx")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {V8_CHECKPOINT}")
    print(f"  via Liquid4All/onnx-export (lfm2-vl-export)")
    print(f"  output → {out_dir}")

    cmd = [
        "uv",
        "run",
        "lfm2-vl-export",
        V8_CHECKPOINT,
        "--precision",          # produce fp32 + fp16 + q4 + q8 variants
        "--output-dir",
        str(out_dir.parent),    # parent; tool creates its own subfolder
    ]
    print("  Command:", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd="/opt/onnx-export",
        capture_output=True,
        text=True,
        env={**os.environ, "HF_TOKEN": os.environ.get("HF_TOKEN", "")},
    )
    print("--- STDOUT (last 120 lines) ---")
    print("\n".join(proc.stdout.splitlines()[-120:]))
    if proc.returncode != 0:
        print("--- STDERR ---")
        print(proc.stderr[-6000:])
        # Before raising, see if the tool wrote anything useful somewhere else
        for p in Path("/opt/onnx-export").rglob("*-ONNX"):
            print(f"  found possible output: {p}")
        raise RuntimeError(f"lfm2-vl-export exited with {proc.returncode}")

    # Figure out the actual output dir (the CLI names it based on model path)
    candidates = sorted(out_dir.parent.glob("*-ONNX")) + sorted(
        Path("/opt/onnx-export/exports").glob("*-ONNX") if Path("/opt/onnx-export/exports").exists() else []
    )
    print(f"\nOutput candidates: {candidates}")
    actual_dir = candidates[-1] if candidates else out_dir

    print(f"\nContents of {actual_dir}:")
    for p in sorted(actual_dir.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.relative_to(actual_dir)}  ({size_mb:.1f} MB)")

    volume.commit()

    # Upload to HF
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"status": "exported_not_uploaded", "local_dir": str(actual_dir)}

    print(f"\nUploading to {HF_REPO_ID}…")
    try:
        create_repo(HF_REPO_ID, exist_ok=True, repo_type="model", token=token, private=False)
        upload_folder(
            folder_path=str(actual_dir),
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=token,
            commit_message="v8-e3 ONNX export via Liquid4All/onnx-export",
        )
        print(f"  ✓ https://huggingface.co/{HF_REPO_ID}")
        return {"status": "uploaded", "repo_id": HF_REPO_ID, "local_dir": str(actual_dir)}
    except Exception as e:
        print(f"  Upload failed: {e}")
        return {"status": "exported_but_upload_failed", "error": str(e), "local_dir": str(actual_dir)}


@app.local_entrypoint()
def main():
    print("Submitting Liquid4All onnx-export on Modal…")
    result = export_v8_onnx.remote()
    print("\nDone:", result)
