"""Push v3 merged checkpoint to samwell/galamsey-unified-v3 on HuggingFace.

Runs on Modal so we don't have to download ~900MB of weights to local just
to upload them again. Loads the merged checkpoint from the galamsey volume,
attaches the model card from docs/hf_model_card_unified_v3.md, pushes via
push_to_hub.

Usage:
    cd training && uv run modal run scripts/push_unified_v3_to_hf_modal.py
"""
from __future__ import annotations
from pathlib import Path

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

V3_CHECKPOINT = (
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
    "-vlm_sft-galamsey_u-all-lr2em05-w0p1-lora_a-20260505_080020/"
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
    "-vlm_sft-galamsey_u-all-lr2em05-w0p1-lora_m-20260505_080020"
)

HF_REPO_ID = "samwell/galamsey-unified-v3"

# Mount the model card from the local docs dir into the container.
LOCAL_REPO = Path(__file__).resolve().parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4", "torchvision>=0.19",
        "transformers>=4.46", "huggingface_hub>=0.27",
        "pillow>=11.0", "safetensors>=0.4",
    )
    .add_local_file(
        str(LOCAL_REPO / "docs" / "hf_model_card_unified_v3.md"),
        "/root/MODEL_CARD.md",
    )
)

app = modal.App("galamsey-push-unified-v3")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    gpu=None,  # CPU-only; we only load + push, no inference
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def push() -> dict:
    import os
    from pathlib import Path
    from huggingface_hub import HfApi
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt_path = Path(MODAL_MOUNT_POINT) / V3_CHECKPOINT
    print(f"Loading checkpoint: {ckpt_path}")
    processor = AutoProcessor.from_pretrained(str(ckpt_path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(ckpt_path), trust_remote_code=True,
    )
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    print(f"Pushing to {HF_REPO_ID}")
    model.push_to_hub(HF_REPO_ID, private=False)
    processor.push_to_hub(HF_REPO_ID, private=False)

    # Attach the model card as README.md
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(
        path_or_fileobj="/root/MODEL_CARD.md",
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Add model card",
    )
    print(f"Done. View at https://huggingface.co/{HF_REPO_ID}")
    return {"repo_id": HF_REPO_ID}


@app.local_entrypoint()
def main() -> None:
    result = push.remote()
    print(f"\nPushed: https://huggingface.co/{result['repo_id']}")
