"""Extract real perception outputs for the blog "outputs gallery" figure.

Tests an open question: does the v3 unified model retain v9-e3's
grounding/description capability when given those prompts, even though
its LoRA was trained on action-only target?

For each of three tiles (u0050 discard, u0078 flag, u0152 downlink) we run
BOTH:
  - v3 with the grounding prompt + the description prompt + the action prompt
  - v9-e3 with the grounding prompt + the description prompt

If v3 produces clean grounding/description outputs, we use v3 only in the
blog figure (one model genuinely doing all three jobs). If v3 collapses
(emits action JSON regardless of prompt, or produces malformed boxes), we
fall back to v9-e3 outputs for the perception side and use v3 only for the
action side.

Output: /galamsey/data/unified_v2/v9_outputs_for_blog.json with both
sets of outputs side by side.
"""
from __future__ import annotations
from pathlib import Path
import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

V9_CKPT = (
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-20260418_165633/"
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
)
V3_CKPT = (
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
    "-vlm_sft-galamsey_u-all-lr2em05-w0p1-lora_a-20260505_080020/"
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
    "-vlm_sft-galamsey_u-all-lr2em05-w0p1-lora_m-20260505_080020"
)
ACTION_SYSTEM_PROMPT = (
    "You are an on-orbit Earth-observation policy adjudicator. Given two views of "
    "a Sentinel-2 patch (natural-color RGB + SWIR false-color composite) and the "
    "per-pass operational context, decide which ONE of the five tools to call. "
    "Reply with EXACTLY ONE tool call as a JSON object: "
    '{"action": "discard|flag_for_review|request_higher_resolution|request_neighbor_tile|downlink_now"}.'
)

TILES = ["u0105", "u0231", "u0226"]
IMAGE_ROOT = "data/unified_v2/images"
OUT_PATH = "/galamsey/data/unified_v2/v9_outputs_for_blog.json"

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
        "torch>=2.4", "torchvision>=0.19",
        "transformers>=4.46", "pillow>=11.0",
        "accelerate>=1.0", "safetensors>=0.4",
    )
)

app = modal.App("galamsey-extract-v9-outputs")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=False)


@app.function(image=image, volumes={MODAL_MOUNT_POINT: volume},
              gpu="H100", timeout=900,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def extract() -> dict:
    import json
    import re
    import torch
    from PIL import Image as PilImage
    from transformers import AutoModelForImageTextToText, AutoProcessor

    image_root = Path(MODAL_MOUNT_POINT) / IMAGE_ROOT
    out_path = Path(OUT_PATH)
    device = torch.device("cuda")

    def parse_boxes(raw: str) -> list[dict]:
        try:
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end + 1])
                if isinstance(parsed, list):
                    return [b for b in parsed if isinstance(b, dict) and "bbox" in b]
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    def gen(model, processor, msgs: list, max_tok: int) -> str:
        inputs = processor.apply_chat_template(
            [msgs], tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False)
        return processor.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()

    def user_msg(rgb, swir, prompt: str) -> list:
        return [{"role": "user", "content": [
            {"type": "image", "image": rgb}, {"type": "image", "image": swir},
            {"type": "text", "text": prompt},
        ]}]

    results: dict[str, dict] = {cid: {} for cid in TILES}

    # Pre-load all tile images
    tiles = {cid: (
        PilImage.open(image_root / cid / "rgb.png").convert("RGB"),
        PilImage.open(image_root / cid / "swir.png").convert("RGB"),
    ) for cid in TILES}

    # Run v3 first (the experiment): does it retain perception outputs?
    print(f"\n=== v3 unified ===\nLoading from {Path(MODAL_MOUNT_POINT) / V3_CKPT}")
    v3_proc = AutoProcessor.from_pretrained(str(Path(MODAL_MOUNT_POINT) / V3_CKPT), trust_remote_code=True)
    v3 = AutoModelForImageTextToText.from_pretrained(
        str(Path(MODAL_MOUNT_POINT) / V3_CKPT), torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    with torch.no_grad():
        for cid, (rgb, swir) in tiles.items():
            grounding = gen(v3, v3_proc, user_msg(rgb, swir, GROUNDING_PROMPT), 256)
            description = gen(v3, v3_proc, user_msg(rgb, swir, DESCRIPTION_PROMPT), 128)
            results[cid]["v3_grounding_raw"] = grounding
            results[cid]["v3_description"] = description
            results[cid]["v3_boxes"] = parse_boxes(grounding)
            print(f"  [{cid}] v3 boxes={len(results[cid]['v3_boxes'])}  desc: {description[:80]}...")
    del v3, v3_proc
    torch.cuda.empty_cache()

    # Run v9-e3 (known-good fallback)
    print(f"\n=== v9-e3 perception ===\nLoading from {Path(MODAL_MOUNT_POINT) / V9_CKPT}")
    v9_proc = AutoProcessor.from_pretrained(str(Path(MODAL_MOUNT_POINT) / V9_CKPT), trust_remote_code=True)
    v9 = AutoModelForImageTextToText.from_pretrained(
        str(Path(MODAL_MOUNT_POINT) / V9_CKPT), torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    with torch.no_grad():
        for cid, (rgb, swir) in tiles.items():
            grounding = gen(v9, v9_proc, user_msg(rgb, swir, GROUNDING_PROMPT), 512)
            description = gen(v9, v9_proc, user_msg(rgb, swir, DESCRIPTION_PROMPT), 128)
            results[cid]["v9_grounding_raw"] = grounding
            results[cid]["v9_description"] = description
            results[cid]["v9_boxes"] = parse_boxes(grounding)
            print(f"  [{cid}] v9 boxes={len(results[cid]['v9_boxes'])}  desc: {description[:80]}...")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    return results


@app.local_entrypoint()
def main() -> None:
    results = extract.remote()
    import json
    print(json.dumps(results, indent=2))
