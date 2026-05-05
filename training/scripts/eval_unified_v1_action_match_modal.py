"""Phase A.4 head-to-head: action-match accuracy for the unified VLM.

Loads the merged unified-v1 LoRA checkpoint (LFM2.5-VL-450M + action-policy
LoRA merged into the base weights), runs generation against the 39-example
held-out eval JSONL, parses the JSON action out of each completion, and
compares to the gold action.

The BLEU benchmark that ran during training was uninformative because the
model's `reason` field (long, free-form) varies even when the `action`
field (the thing we actually care about) is correct. This script measures
the right thing: action-match accuracy + per-class confusion.

Outputs:
    /galamsey/data/unified_v1/predictions_unified_v1.jsonl  (per-tile preds)
    stdout: confusion matrix + accuracy

Usage:
    cd training && uv run modal run scripts/eval_unified_v1_action_match_modal.py
"""
from __future__ import annotations

from pathlib import Path

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

CHECKPOINT_DIR = (
    "lfm2.5-VL-450M-vlm_sft-galamsey_u-all-lr2em05-w0p1-lora_a-20260505_004327/"
    "lfm2.5-VL-450M-vlm_sft-galamsey_u-all-lr2em05-w0p1-lora_m-20260505_004327"
)
EVAL_JSONL = "data/unified_v1/galamsey_unified_v1_eval.jsonl"
IMAGE_ROOT = "data/unified_v1/images"
PREDICTIONS_OUT = "data/unified_v1/predictions_unified_v1.jsonl"

ACTIONS = ["discard", "flag_for_review", "request_higher_resolution",
           "request_neighbor_tile", "downlink_now"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "torchvision>=0.19",
        "transformers>=4.46",
        "pillow>=11.0",
        "accelerate>=1.0",
        "safetensors>=0.4",
    )
)

app = modal.App("galamsey-unified-v1-eval")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    gpu="H100",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate() -> dict:
    import json
    import time
    from collections import defaultdict

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt_path = Path(MODAL_MOUNT_POINT) / CHECKPOINT_DIR
    eval_path = Path(MODAL_MOUNT_POINT) / EVAL_JSONL
    image_root = Path(MODAL_MOUNT_POINT) / IMAGE_ROOT
    out_path = Path(MODAL_MOUNT_POINT) / PREDICTIONS_OUT

    print(f"Loading model from: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(str(ckpt_path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(ckpt_path), torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"Model loaded on {device}")

    eval_rows = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    print(f"Eval set: {len(eval_rows)} examples")

    # Confusion matrix: rows = gold, cols = pred
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_correct = 0
    n_unparseable = 0
    n_invalid_action = 0
    predictions: list[dict] = []

    start = time.time()
    with torch.no_grad():
        for i, row in enumerate(eval_rows):
            # Reconstruct the prompt as the model was trained: system + user (image, image, text).
            # We strip the assistant turn (that's what we're predicting) and re-add the system
            # prompt + user turn through the chat template with `add_generation_prompt=True`.
            user_content = row["messages"][1]["content"]
            image_paths = [c["image"] for c in user_content if c["type"] == "image"]
            rgb_rel, swir_rel = image_paths[0], image_paths[1]
            user_text = next(c["text"] for c in user_content if c["type"] == "text")
            system_text = row["messages"][0]["content"][0]["text"]

            rgb = Image.open(image_root / rgb_rel).convert("RGB")
            swir = Image.open(image_root / swir_rel).convert("RGB")

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [
                    {"type": "image", "image": rgb},
                    {"type": "image", "image": swir},
                    {"type": "text", "text": user_text},
                ]},
            ]

            inputs = processor.apply_chat_template(
                [messages], tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
            prompt_len = inputs["input_ids"].shape[1]

            output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            pred_text = processor.tokenizer.decode(
                output_ids[0, prompt_len:], skip_special_tokens=True,
            ).strip()

            # Parse JSON action
            gold_text = row["messages"][2]["content"][0]["text"]
            gold_action = json.loads(gold_text)["action"]

            pred_action = "<unparseable>"
            try:
                # Be lenient: find the first { ... } block
                start_idx = pred_text.find("{")
                end_idx = pred_text.rfind("}")
                if start_idx >= 0 and end_idx > start_idx:
                    parsed = json.loads(pred_text[start_idx : end_idx + 1])
                    pred_action = parsed.get("action", "<no_action_field>")
                else:
                    n_unparseable += 1
            except (json.JSONDecodeError, ValueError):
                n_unparseable += 1

            if pred_action not in ACTIONS and pred_action not in ("<unparseable>", "<no_action_field>"):
                n_invalid_action += 1

            confusion[gold_action][pred_action] += 1
            if pred_action == gold_action:
                n_correct += 1

            predictions.append({
                "rgb_path": rgb_rel,
                "swir_path": swir_rel,
                "gold_action": gold_action,
                "pred_action": pred_action,
                "pred_text_full": pred_text,
                "match": pred_action == gold_action,
            })

            if (i + 1) % 5 == 0 or i == len(eval_rows) - 1:
                elapsed = time.time() - start
                acc_so_far = n_correct / (i + 1)
                print(f"  [{i+1:>3}/{len(eval_rows)}] acc={acc_so_far:.3f}  "
                      f"unparseable={n_unparseable}  invalid={n_invalid_action}  "
                      f"({elapsed:.1f}s)")

    accuracy = n_correct / len(eval_rows) if eval_rows else 0.0
    elapsed = time.time() - start

    # Write per-tile predictions
    with out_path.open("w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"\nWrote per-tile predictions to {out_path.relative_to(MODAL_MOUNT_POINT)}")

    # Print confusion matrix
    print("\n" + "=" * 72)
    print(f"Action-match accuracy: {n_correct}/{len(eval_rows)} = {accuracy:.4f}")
    print(f"Unparseable outputs: {n_unparseable}/{len(eval_rows)}")
    print(f"Invalid action labels: {n_invalid_action}/{len(eval_rows)}")
    print(f"Inference time: {elapsed:.1f}s ({elapsed/len(eval_rows):.2f}s/sample)")
    print("=" * 72)

    print("\nConfusion matrix (rows=gold, cols=pred):")
    pred_keys_seen = sorted({p for d in confusion.values() for p in d.keys()})
    header = "gold \\ pred".ljust(30) + "".join(p[:13].ljust(14) for p in pred_keys_seen) + "total"
    print(header)
    for gold in ACTIONS:
        if gold not in confusion:
            continue
        row_total = sum(confusion[gold].values())
        cells = "".join(str(confusion[gold].get(p, 0)).ljust(14) for p in pred_keys_seen)
        print(gold.ljust(30) + cells + str(row_total))

    print("\nPer-class breakdown:")
    print(f"{'action':30s}{'precision':12s}{'recall':12s}{'support':10s}")
    for action in ACTIONS:
        # Recall: TP / (TP + FN) where FN = gold says action, pred says other
        gold_count = sum(confusion[action].values()) if action in confusion else 0
        if gold_count == 0:
            continue
        tp = confusion[action].get(action, 0)
        recall = tp / gold_count
        # Precision: TP / (TP + FP) where FP = pred says action, gold says other
        pred_count = sum(confusion[g].get(action, 0) for g in confusion)
        precision = tp / pred_count if pred_count > 0 else 0.0
        print(f"{action:30s}{precision:.3f}       {recall:.3f}       {gold_count}")

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": len(eval_rows),
        "n_unparseable": n_unparseable,
        "n_invalid": n_invalid_action,
        "elapsed_sec": elapsed,
        "confusion": {g: dict(d) for g, d in confusion.items()},
    }


@app.local_entrypoint()
def main() -> None:
    result = evaluate.remote()
    print(f"\nFinal: accuracy={result['accuracy']:.4f} on {result['n_total']} eval examples.")
