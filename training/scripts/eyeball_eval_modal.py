"""Minimal eyeball eval, compare v1 fine-tuned checkpoint vs base LFM2.5-VL-450M.

Bypasses leap-finetune's benchmark callback entirely. Loads both models directly
with transformers, runs inference on 10 samples from our held-out eval JSONL,
and prints the ground truth alongside each model's prediction.

The goal is to answer: did v1 actually learn anything, or does it parrot one of
the two training templates regardless of image content?

Usage:
    uv run modal run scripts/eyeball_eval_modal.py

Expected cost: ~$0.20 (runs on H100 for ~3-5 min).
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

V1_CHECKPOINT = (
    "/galamsey/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-20260415_213456"
    "/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s1110-20260415_213456"
)
BASE_MODEL = "LiquidAI/LFM2.5-VL-450M"
EVAL_JSONL = "/galamsey/data/v1/galamsey_v1_description_eval.jsonl"
IMAGE_ROOT = "/galamsey/data/v1/images"
N_SAMPLES = 10

app = modal.App("galamsey-eyeball-eval")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.5",
        "torchvision",  # needed by Lfm2VlImageProcessor
        "transformers>=4.51",
        "pillow",
        "huggingface_hub",
        "accelerate",
    )
)


@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    gpu="H100",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_eyeball() -> dict:
    import copy
    import json
    from pathlib import Path

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("=" * 72)
    print("GalamseyWatch, Eyeball eval: v1 fine-tuned vs base LFM2.5-VL-450M")
    print("=" * 72)

    # Load eval samples
    samples: list[dict] = []
    with open(EVAL_JSONL) as f:
        for i, line in enumerate(f):
            if i >= N_SAMPLES:
                break
            samples.append(json.loads(line))
    print(f"\nLoaded {len(samples)} eval samples from {EVAL_JSONL}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    def prepare_inputs(messages: list[dict], processor) -> tuple[dict, str]:
        """Strip the last (assistant) turn, load images as PIL, tokenize.

        Returns (tokenized inputs on device, ground truth string).
        """
        messages = copy.deepcopy(messages)
        ground_truth = messages[-1]["content"][0]["text"]
        prompt_messages = messages[:-1]

        # Load image paths into PIL objects
        for msg in prompt_messages:
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "image":
                    img_ref = item["image"]
                    if not img_ref.startswith("/"):
                        img_ref = f"{IMAGE_ROOT}/{img_ref}"
                    item["image"] = Image.open(img_ref).convert("RGB")

        inputs = processor.apply_chat_template(
            [prompt_messages],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
        return inputs, ground_truth

    def run_model(model_name_or_path: str, label: str) -> list[str]:
        print(f"\n{'=' * 72}")
        print(f"Loading {label}: {model_name_or_path}")
        print("=" * 72)

        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        model.eval()

        predictions: list[str] = []
        with torch.no_grad():
            for i, sample in enumerate(samples):
                inputs, ground_truth = prepare_inputs(sample["messages"], processor)
                prompt_len = inputs["input_ids"].shape[1]

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
                prediction = processor.tokenizer.decode(
                    output_ids[0, prompt_len:], skip_special_tokens=True
                ).strip()
                predictions.append(prediction)
                print(f"\n--- [{label}] sample {i} ---")
                print(f"GT   : {ground_truth}")
                print(f"PRED : {prediction}")

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()
        return predictions

    v1_preds = run_model(V1_CHECKPOINT, "v1 fine-tuned")
    base_preds = run_model(BASE_MODEL, "base LFM2.5-VL-450M")

    # Side-by-side summary
    print("\n" + "=" * 72)
    print("SUMMARY, ground truth vs v1 vs base (all 10 samples)")
    print("=" * 72)
    for i, sample in enumerate(samples):
        gt = sample["messages"][-1]["content"][0]["text"]
        print(f"\n[{i}]")
        print(f"  GT  : {gt}")
        print(f"  v1  : {v1_preds[i]}")
        print(f"  base: {base_preds[i]}")

    # Quick heuristic: how many unique predictions did each model produce?
    v1_unique = len(set(v1_preds))
    base_unique = len(set(base_preds))
    print(f"\nUnique predictions across {N_SAMPLES} samples:")
    print(f"  v1 fine-tuned: {v1_unique}")
    print(f"  base:          {base_unique}")
    print("(higher = more varied / responsive to image content)")

    return {
        "n_samples": len(samples),
        "v1_unique_predictions": v1_unique,
        "base_unique_predictions": base_unique,
    }


@app.local_entrypoint()
def main() -> None:
    print("Submitting eyeball eval to Modal...")
    result = run_eyeball.remote()
    print(f"\nDone. Summary: {result}")
