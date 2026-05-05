"""Phase A.4 head-to-head: action-match accuracy for the TWO-LAYER baseline.

Two-layer baseline = galamsey-v9-e3 perception VLM (450M, full FT) + LFM2-2.6B
policy agent. For each of the same 39 held-out eval examples used by the
unified v2 eval:
  1. Run v9-e3 on (RGB, SWIR) -> bounding boxes + scene description.
  2. Build the LFM2 prompt from boxes/description + scalar context.
  3. Run LFM2-2.6B -> Pythonic-bracket tool call -> action.
  4. Compare to gold action.

Outputs same shape as eval_unified_v2: confusion matrix + per-class metrics
+ per-tile predictions JSONL. Direct comparison with v2 numbers.

Note: the LFM2 baseline only sees `bandwidth_remaining_kb` and the perception
output — it does NOT see `mission_priors` or `neighbor_summary` because the
existing build_tile_prompt was designed before those fields existed. This is
the apples-to-apples comparison vs v1 of the project, but unified gets richer
context. Worth a follow-up "rich-context two-layer" eval to isolate the
architecture effect from the context effect.

Usage:
    cd training && uv run modal run scripts/eval_two_layer_baseline_modal.py
"""
from __future__ import annotations

from pathlib import Path

import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

PERCEPTION_CKPT = (
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-20260418_165633/"
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
)
AGENT_MODEL_ID = "LiquidAI/LFM2-2.6B"

# Same eval set the unified v2 was scored on, so numbers are comparable.
EVAL_JSONL = "data/unified_v2/galamsey_unified_v2_eval_expanded.jsonl"
IMAGE_ROOT = "data/unified_v2/images"
PREDICTIONS_OUT = "data/unified_v2/predictions_two_layer_baseline_expanded.jsonl"

ACTIONS = ["discard", "flag_for_review", "request_higher_resolution",
           "request_neighbor_tile", "downlink_now"]
ACTION_TO_NAME = {
    "discard": "discard",
    "flag": "flag_for_review",
    "request_hires": "request_higher_resolution",
    "request_neighbor": "request_neighbor_tile",
    "downlink": "downlink_now",
}

# Vendored from orchestrator/agentic_eo/models/agent.py — keeping the Modal
# script self-contained.
TOOLS = [
    {"type": "function", "function": {"name": "discard",
        "description": "Skip this tile entirely. THIS IS THE DEFAULT FOR MOST TILES — forest, water, undisturbed land, heavy cloud, or ocean. Use whenever the VLM found 0 boxes or clearly described undisturbed terrain.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
    {"type": "function", "function": {"name": "flag_for_review",
        "description": "Add this tile to the end-of-pass TEXT summary — no image downlink, just a brief log entry. Use for moderate-confidence detections (1-2 small boxes, ambiguous description) worth recording but not worth bandwidth.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
    {"type": "function", "function": {"name": "request_higher_resolution",
        "description": "Request a higher-resolution recapture of this same tile next pass. Use when the VLM found a SMALL candidate (1 tiny box) that needs more pixels to confirm.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
    {"type": "function", "function": {"name": "request_neighbor_tile",
        "description": "Fetch a tile in a given compass direction. Use only when the VLM described a feature that likely continues into the adjacent tile (e.g., sediment plume extending off-frame to the east).",
        "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["north", "south", "east", "west"]}, "reason": {"type": "string"}}, "required": ["direction", "reason"]}}},
    {"type": "function", "function": {"name": "downlink_now",
        "description": "Use the precious downlink budget to send THIS tile's image to ground during the current pass. RESERVE FOR HIGH-CONFIDENCE DETECTIONS ONLY: 2+ clear bounding boxes, confidence >= 0.85, AND the description explicitly mentions active pits, exposed soil, or sediment plumes. If in doubt, prefer flag_for_review — it's text-only and far cheaper.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
]
SYSTEM_PROMPT = (
    "You are an on-orbit Earth-observation agent on a satellite-class compute platform during a "
    "Sentinel-2 pass over southwestern Ghana. Mission: detect illegal small-scale gold mining (galamsey).\n\n"
    "For each tile, the on-board VLM has given you bounding boxes + a description. Reply with "
    "EXACTLY ONE tool call — no preamble, no alternatives, no questions.\n\n"
    "Three actions, each with a clear trigger:\n"
    "- **discard**: 0 boxes, OR description says forest/water/undisturbed/cloud-obscured.\n"
    "- **flag_for_review**: 1-2 boxes with low/moderate confidence, ambiguous descriptions, OR boundary cases. Cheap (text-only).\n"
    "- **downlink_now**: confidence >= 0.85 AND (>=2 boxes OR a single large box) AND the description names active galamsey features (pits, sediment plumes, exposed soil, turbid water). When these conditions are met you MUST call downlink_now."
)

# Vendored few-shot examples from orchestrator/agentic_eo/models/agent.py.
# Without these LFM2-2.6B falls back to plain-text action names instead of
# emitting the wrapped <|tool_call_start|>...<|tool_call_end|> format the
# parser expects. Production agent uses these too.
FEW_SHOT: list[tuple[str, str]] = [
    ("Tile e001 at lon=-1.5000, lat=5.0000.\n"
     "Cloud cover: 5%. Captured: 2024-01-15T10:39:00Z.\n\n"
     "VLM detection:\n- 0 mining-pit candidate bounding box(es)\n"
     '- Description: "Continuous tropical forest canopy. No visible disturbance."\n'
     "- Derived confidence: 0.00\n\nPass budget: 480 KB of 512 KB remaining.\n\n"
     "Choose the appropriate tool.",
     '<|tool_call_start|>[discard(reason="forest canopy, no detection")]<|tool_call_end|>'),
    ("Tile e002 at lon=-1.9900, lat=5.3000.\n"
     "Cloud cover: 5%. Captured: 2024-01-15T10:39:00Z.\n\n"
     "VLM detection:\n- 4 mining-pit candidate bounding box(es) (largest area = 0.180)\n"
     '- Description: "Multiple active excavation pits with sediment plumes and exposed lateritic soil."\n'
     "- Derived confidence: 0.95\n\nPass budget: 400 KB of 512 KB remaining.\n\n"
     "Choose the appropriate tool.",
     '<|tool_call_start|>[downlink_now(reason="4 high-confidence pits with sediment plumes — clear active galamsey")]<|tool_call_end|>'),
    ("Tile e003 at lon=-1.7000, lat=5.4000.\n"
     "Cloud cover: 20%. Captured: 2024-01-15T10:39:00Z.\n\n"
     "VLM detection:\n- 1 mining-pit candidate bounding box(es) (largest area = 0.012)\n"
     '- Description: "A small bright patch near a stream bend; could be exposed soil or a sandbar."\n'
     "- Derived confidence: 0.70\n\nPass budget: 200 KB of 512 KB remaining.\n\n"
     "Choose the appropriate tool.",
     '<|tool_call_start|>[flag_for_review(reason="single ambiguous small candidate; not worth bandwidth")]<|tool_call_end|>'),
    ("Tile e004 at lon=-2.0500, lat=5.4500.\n"
     "Cloud cover: 8%. Captured: 2024-01-15T10:39:00Z.\n\n"
     "VLM detection:\n- 2 mining-pit candidate bounding box(es) (largest area = 0.090)\n"
     '- Description: "Two excavation pits with sediment-laden water; exposed soil clearly visible."\n'
     "- Derived confidence: 0.92\n\nPass budget: 350 KB of 512 KB remaining.\n\n"
     "Choose the appropriate tool.",
     '<|tool_call_start|>[downlink_now(reason="2 high-confidence pits with sediment-laden water — active galamsey")]<|tool_call_end|>'),
]
GROUNDING_PROMPT = (
    "You are viewing two images of the same Sentinel-2 patch: a natural-color RGB "
    "composite and a SWIR false-color composite. Using both views, detect any "
    "illegal small-scale gold mining pits. Include any exposed soil, excavation, "
    "or sediment-laden water even if you are uncertain — err toward detection. "
    'Provide result as a valid JSON: [{"label": str, "bbox": [x1,y1,x2,y2]}, ...]. '
    "Coordinates must be normalized to 0-1. Only return [] if the scene is entirely "
    "pristine forest, clean water, or urban built-up area with no disturbance."
)
DESCRIPTION_PROMPT = (
    "You are analyzing two views of the same Sentinel-2 patch of southwestern Ghana: "
    "the first image is a natural-color RGB composite, and the second is a SWIR "
    "false-color composite (SWIR2, SWIR1, NIR) where bright areas indicate exposed "
    "soil and mining disturbance. Using both views, describe any signs of illegal "
    "small-scale gold mining (galamsey) activity: exposed soil, excavation pits, "
    "sediment plumes, vegetation loss, and proximity to water bodies. "
    "If no mining is visible, say so."
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4", "torchvision>=0.19", "transformers>=4.46",
        "pillow>=11.0", "accelerate>=1.0", "safetensors>=0.4",
    )
)

app = modal.App("galamsey-two-layer-baseline-eval-expanded")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image, volumes={MODAL_MOUNT_POINT: volume}, gpu="H100", timeout=2400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate() -> dict:
    import ast
    import json
    import re
    import time
    from collections import defaultdict

    import torch
    from PIL import Image as PilImage
    from transformers import (
        AutoModelForCausalLM, AutoModelForImageTextToText,
        AutoProcessor, AutoTokenizer,
    )

    perception_path = Path(MODAL_MOUNT_POINT) / PERCEPTION_CKPT
    eval_path = Path(MODAL_MOUNT_POINT) / EVAL_JSONL
    image_root = Path(MODAL_MOUNT_POINT) / IMAGE_ROOT
    out_path = Path(MODAL_MOUNT_POINT) / PREDICTIONS_OUT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading perception VLM: {perception_path}")
    p_processor = AutoProcessor.from_pretrained(str(perception_path), trust_remote_code=True)
    p_model = AutoModelForImageTextToText.from_pretrained(
        str(perception_path), torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    print(f"Loading LFM2-2.6B agent: {AGENT_MODEL_ID}")
    a_tokenizer = AutoTokenizer.from_pretrained(AGENT_MODEL_ID)
    a_model = AutoModelForCausalLM.from_pretrained(
        AGENT_MODEL_ID, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # Helpers
    bbox_re = re.compile(r'\[\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\]')
    tool_call_re = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL)

    def parse_boxes(raw: str) -> list[dict]:
        # Try strict JSON first; fall back to regex
        try:
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end+1])
                if isinstance(parsed, list):
                    return [b for b in parsed if isinstance(b, dict) and "bbox" in b]
        except (json.JSONDecodeError, ValueError):
            pass
        return [{"label": "?", "bbox": [0, 0, 0, 0]} for _ in bbox_re.findall(raw)]

    def perception(rgb: "PilImage.Image", swir: "PilImage.Image") -> tuple[list[dict], str]:
        # Grounding pass: boxes
        for prompt, max_tokens in [(GROUNDING_PROMPT, 256), (DESCRIPTION_PROMPT, 128)]:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": rgb}, {"type": "image", "image": swir},
                {"type": "text", "text": prompt}]}]
            inputs = p_processor.apply_chat_template(
                [messages], tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
            prompt_len = inputs["input_ids"].shape[1]
            out_ids = p_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            text = p_processor.tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=True).strip()
            if prompt is GROUNDING_PROMPT:
                boxes_raw = text
            else:
                description = text
        boxes = parse_boxes(boxes_raw)
        return boxes, description

    def build_tile_prompt(*, tile_id, lon, lat, cloud_cover, captured_at,
                          boxes_count, max_area, description, overall_confidence,
                          bandwidth_remaining_kb, bandwidth_total_kb):
        cc = f"{cloud_cover * 100:.0f}%" if cloud_cover is not None else "n/a"
        return (
            f"Tile {tile_id} at lon={lon:.4f}, lat={lat:.4f}.\n"
            f"Cloud cover: {cc}. Captured: {captured_at or 'unknown'}.\n\n"
            f"VLM detection:\n"
            f"- {boxes_count} mining-pit candidate bounding box(es)"
            + (f" (largest area = {max_area:.3f})" if boxes_count > 0 else "") + "\n"
            f'- Description: "{description.strip()}"\n'
            f"- Derived confidence: {overall_confidence:.2f}\n\n"
            f"Pass budget: {bandwidth_remaining_kb} KB of {bandwidth_total_kb} KB remaining.\n\n"
            "Choose the appropriate tool."
        )

    def policy(user_prompt: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u, a in FEW_SHOT:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_prompt})
        prompt = a_tokenizer.apply_chat_template(
            messages, tools=TOOLS, add_generation_prompt=True, tokenize=False,
        )
        inputs = a_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.inference_mode():
            out = a_model.generate(
                **inputs, do_sample=True, temperature=0.4, top_p=0.9,
                max_new_tokens=192, pad_token_id=a_tokenizer.eos_token_id,
            )
        return a_tokenizer.batch_decode(out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=False)[0]

    def parse_action(raw: str) -> str:
        # Accept both the wrapped format (<|tool_call_start|>[fn(...)]<|tool_call_end|>)
        # AND the bare format LFM2 actually emits without few-shot priming
        # (e.g. `discard(reason="...")<|im_end|>`).
        m = tool_call_re.search(raw)
        if m:
            inner = m.group(1).strip()
            if inner.startswith("[") and inner.endswith("]"):
                inner = inner[1:-1].strip()
        else:
            # Strip special tokens, take everything before <|im_end|> if present
            text = raw.split("<|im_end|>", 1)[0].strip()
            # Pull out the first call-like substring fn_name(...)
            call_match = re.search(r"(\w+)\s*\([^)]*\)", text, re.DOTALL)
            if not call_match:
                return "<no_call>"
            inner = call_match.group(0).strip()
        try:
            tree = ast.parse(inner, mode="eval")
            if isinstance(tree.body, ast.Call) and isinstance(tree.body.func, ast.Name):
                return tree.body.func.id
        except SyntaxError:
            pass
        return "<unparseable>"

    def confidence(boxes: list[dict]) -> float:
        if not boxes:
            return 0.0
        # Same heuristic as the orchestrator: 0.4 base + 0.15 per box, capped at 0.95
        return min(0.95, 0.4 + 0.15 * len(boxes))

    eval_rows = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    print(f"Eval set: {len(eval_rows)} examples")

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_correct = n_unparseable = 0
    predictions: list[dict] = []
    start = time.time()

    with torch.no_grad():
        for i, row in enumerate(eval_rows):
            user_content = row["messages"][1]["content"]
            image_paths = [c["image"] for c in user_content if c["type"] == "image"]
            rgb_rel, swir_rel = image_paths[0], image_paths[1]
            user_text = next(c["text"] for c in user_content if c["type"] == "text")

            # Pull tile_id, lon, lat, budget, etc out of the user_text (regex)
            tile_id = re.search(r"Tile (\w+)", user_text).group(1)
            lon = float(re.search(r"lon=(-?\d+\.\d+)", user_text).group(1))
            lat = float(re.search(r"lat=(-?\d+\.\d+)", user_text).group(1))
            cc_match = re.search(r"Cloud cover.*?:\s*(\d+\.?\d*)", user_text)
            cloud_cover = float(cc_match.group(1)) if cc_match else None
            cap_match = re.search(r"Captured: (\S+)", user_text)
            captured_at = cap_match.group(1) if cap_match else None
            budget_match = re.search(r"Pass budget: (\d+) of (\d+) KB", user_text)
            budget_remaining = int(budget_match.group(1)) if budget_match else 256
            budget_total = int(budget_match.group(2)) if budget_match else 512

            rgb = PilImage.open(image_root / rgb_rel).convert("RGB")
            swir = PilImage.open(image_root / swir_rel).convert("RGB")

            # Run perception
            boxes, description = perception(rgb, swir)
            max_area = max(((b["bbox"][2]-b["bbox"][0]) * (b["bbox"][3]-b["bbox"][1])
                            for b in boxes), default=0.0)
            tile_prompt = build_tile_prompt(
                tile_id=tile_id, lon=lon, lat=lat, cloud_cover=cloud_cover,
                captured_at=captured_at, boxes_count=len(boxes), max_area=max_area,
                description=description, overall_confidence=confidence(boxes),
                bandwidth_remaining_kb=budget_remaining, bandwidth_total_kb=budget_total,
            )

            # Run policy
            agent_raw = policy(tile_prompt)
            pred_action = parse_action(agent_raw)
            if pred_action.startswith("<"):
                n_unparseable += 1

            gold_action = json.loads(row["messages"][2]["content"][0]["text"])["action"]

            confusion[gold_action][pred_action] += 1
            if pred_action == gold_action:
                n_correct += 1

            predictions.append({
                "tile_id": tile_id, "rgb_path": rgb_rel, "swir_path": swir_rel,
                "boxes_count": len(boxes), "description": description,
                "agent_raw": agent_raw, "pred_action": pred_action,
                "gold_action": gold_action, "match": pred_action == gold_action,
            })

            if (i + 1) % 5 == 0 or i == len(eval_rows) - 1:
                elapsed = time.time() - start
                print(f"  [{i+1:>3}/{len(eval_rows)}] acc={n_correct/(i+1):.3f}  "
                      f"unparse={n_unparseable}  ({elapsed:.1f}s)")

    accuracy = n_correct / len(eval_rows) if eval_rows else 0.0
    elapsed = time.time() - start

    with out_path.open("w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"\nWrote per-tile predictions to {out_path.relative_to(MODAL_MOUNT_POINT)}")

    print("\n" + "=" * 72)
    print(f"TWO-LAYER BASELINE accuracy: {n_correct}/{len(eval_rows)} = {accuracy:.4f}")
    print(f"Unparseable outputs: {n_unparseable}/{len(eval_rows)}")
    print(f"Inference time: {elapsed:.1f}s ({elapsed/len(eval_rows):.2f}s/sample)")
    print("=" * 72)

    print("\nConfusion matrix (rows=gold, cols=pred):")
    pred_keys = sorted({p for d in confusion.values() for p in d.keys()})
    print("gold \\ pred".ljust(30) + "".join(p[:13].ljust(14) for p in pred_keys) + "total")
    for gold in ACTIONS:
        if gold not in confusion:
            continue
        row_total = sum(confusion[gold].values())
        cells = "".join(str(confusion[gold].get(p, 0)).ljust(14) for p in pred_keys)
        print(gold.ljust(30) + cells + str(row_total))

    print("\nPer-class:")
    print(f"{'action':30s}{'precision':12s}{'recall':12s}{'support':10s}")
    for action in ACTIONS:
        gold_count = sum(confusion[action].values()) if action in confusion else 0
        if gold_count == 0:
            continue
        tp = confusion[action].get(action, 0)
        recall = tp / gold_count
        pred_count = sum(confusion[g].get(action, 0) for g in confusion)
        precision = tp / pred_count if pred_count > 0 else 0.0
        print(f"{action:30s}{precision:.3f}       {recall:.3f}       {gold_count}")

    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": len(eval_rows),
            "n_unparseable": n_unparseable, "elapsed_sec": elapsed,
            "confusion": {g: dict(d) for g, d in confusion.items()}}


@app.local_entrypoint()
def main() -> None:
    result = evaluate.remote()
    print(f"\nFinal: two-layer accuracy={result['accuracy']:.4f} on {result['n_total']} examples.")
