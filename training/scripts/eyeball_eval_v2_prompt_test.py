"""Step B diagnostic: test whether v2's single-pit false negatives respond
to a stronger system prompt.

v2 failed on samples 0 and 2 in the earlier eyeball eval. Both failing samples
have unusual visual patterns:
  - Sample 0: faint/diffuse disturbance in a mostly-vegetated scene
  - Sample 2: tiny labeled mining region embedded in a mostly-bare-soil scene
Neither matches the "dark background + discrete bright pit spots" pattern the
model learned from training.

This script runs v2 inference on the same 10 description samples but prepends
a strong system message asking the model to look for subtle/ambiguous cases.
If samples 0 and 2 flip positive with the stronger prompt, we've learned the
failures were correctable by better prompting. If they don't, we've confirmed
v2 has a genuine visual capability limit that won't be fixed by prompting.

Usage:
    uv run modal run scripts/eyeball_eval_v2_prompt_test.py

Expected cost: ~$0.20 (loads 1 model, 10 inferences, ~3 min H100).
"""

from __future__ import annotations

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

V2_CHECKPOINT = (
    "/galamsey/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-20260415_231743"
    "/lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s2217-20260415_231743"
)

DESC_EVAL = "/galamsey/data/v2/galamsey_v2_description_eval.jsonl"
IMAGE_ROOT = "/galamsey/data/v2/images"

N_SAMPLES = 10

# Stronger system message, explicitly tells the model to look for subtle
# and ambiguous cases, including the two failure modes we identified.
SYSTEM_MESSAGE = (
    "You are an expert remote sensing analyst specializing in illegal gold mining "
    "detection in southwestern Ghana. You are analyzing Sentinel-2 SWIR false-color "
    "composites where bright yellow or cream colors indicate exposed soil and "
    "mining disturbance, dark blue-green indicates vegetation, and dark pixels "
    "indicate water. "
    "Look carefully for ALL signs of mining activity, including: "
    "(a) scenes where a SINGLE large uniform disturbance covers a significant "
    "portion of the image, these are still mining even without multiple discrete pits; "
    "(b) scenes with SMALL or FAINT pits that blend with surrounding vegetation or soil, "
    "these are still mining even if the signal is subtle; "
    "(c) scenes with extensive bare soil where only part of the exposed area is "
    "actually mining, identify the mining portion if any. "
    "Do not dismiss subtle or ambiguous signals. Describe what you observe honestly, "
    "erring on the side of reporting potential mining rather than missing it."
)

app = modal.App("galamsey-v2-prompt-test")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.5",
        "torchvision",
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
    timeout=1200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_prompt_test() -> dict:
    import copy
    import json
    from pathlib import Path

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("=" * 72)
    print("GalamseyWatch, v2 stronger-prompt diagnostic")
    print("=" * 72)

    # Load the same 10 description samples as the earlier eyeball eval
    samples = []
    with open(DESC_EVAL) as f:
        for i, line in enumerate(f):
            if i >= N_SAMPLES:
                break
            samples.append(json.loads(line))
    print(f"\nLoaded {len(samples)} description samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"\nStronger system message:\n{SYSTEM_MESSAGE}\n")

    print(f"\n{'=' * 72}")
    print(f"Loading v2 checkpoint: {V2_CHECKPOINT}")
    print("=" * 72)

    processor = AutoProcessor.from_pretrained(V2_CHECKPOINT, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        V2_CHECKPOINT,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    def prepare_inputs_with_system(sample: dict) -> tuple[dict, str]:
        messages = copy.deepcopy(sample["messages"])
        ground_truth = messages[-1]["content"][0]["text"]
        prompt_messages = messages[:-1]

        # Prepend a system message
        system_msg = {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        }
        prompt_messages = [system_msg] + prompt_messages

        # Load images
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

    results: list[tuple[str, str, bool]] = []  # (gt, pred, is_failing_sample)
    FAILING_INDICES = {0, 2}  # from the earlier eyeball eval

    with torch.no_grad():
        for i, sample in enumerate(samples):
            inputs, ground_truth = prepare_inputs_with_system(sample)
            prompt_len = inputs["input_ids"].shape[1]
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
            prediction = processor.tokenizer.decode(
                output_ids[0, prompt_len:], skip_special_tokens=True
            ).strip()
            is_failing = i in FAILING_INDICES
            results.append((ground_truth, prediction, is_failing))

            marker = "  ← was failing in earlier eyeball eval" if is_failing else ""
            print(f"\n[{i}]{marker}")
            print(f"  GT  : {ground_truth}")
            print(f"  v2+sys: {prediction}")

    # Check whether the previously-failing samples flipped positive
    print("\n" + "=" * 72)
    print("VERDICT, did the stronger prompt fix the failing samples?")
    print("=" * 72)

    prev_fail_fixed = 0
    for i in sorted(FAILING_INDICES):
        gt, pred, _ = results[i]
        # Heuristic: did the new prediction acknowledge mining activity?
        # Look for positive-indicator phrases or absence of negative phrases
        pred_lower = pred.lower()
        negative_phrases = [
            "no signs",
            "no visible signs",
            "no mining",
            "no evidence",
            "not visible",
            "not currently under",
            "no illegal",
            "appears intact",
            "appears undisturbed",
            "no excavation",
            "land cover appears",
        ]
        looks_positive = not any(phrase in pred_lower for phrase in negative_phrases)
        status = "✅ FIXED (now positive)" if looks_positive else "❌ STILL FAILING"
        if looks_positive:
            prev_fail_fixed += 1
        print(f"\n  sample {i}: {status}")
        print(f"    GT  : {gt}")
        print(f"    PRED: {pred}")

    print(f"\nPreviously-failing samples fixed: {prev_fail_fixed}/{len(FAILING_INDICES)}")
    return {
        "n_samples": len(samples),
        "prev_fail_fixed": prev_fail_fixed,
        "total_prev_fail": len(FAILING_INDICES),
    }


@app.local_entrypoint()
def main() -> None:
    print("Submitting v2 stronger-prompt diagnostic to Modal...")
    result = run_prompt_test.remote()
    print(f"\nDone. Summary: {result}")
