"""Extract v4.1 unified outputs for the blog gallery.

v4.1 is a single LoRA on v9-e3 trained with a multitask mixture (action +
grounding + description). For each blog tile we run v4.1 THREE times — once
per prompt — and capture all three outputs from the same weight set.

This replaces the earlier v9-e3-perception + v3-action setup with a genuine
single-model gallery: one model emits boxes, description, and the policy
decision.
"""
from __future__ import annotations
from pathlib import Path
import modal

MODAL_VOLUME_NAME = "galamsey"
MOUNT = "/galamsey"

V4_1_CKPT = (
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
    "-vlm_sft-galamsey_u-all-lr2em05-w0p1-lora_a-20260505_154237/"
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
    "-vlm_sft-galamsey_u-all-lr2em05-w0p1-lora_m-20260505_154237"
)

TILES = ["u0105", "u0231", "u0226"]
IMAGE_ROOT = "data/unified_v2/images"
OUT_PATH = "/galamsey/data/unified_v4_1_multitask/v4_1_outputs_for_blog.json"

# Prompts — same wording as the v9 perception SFT and the v2 action SFT
GROUNDING_PROMPT = (
    "You are viewing two images of the same Sentinel-2 patch: a natural-color RGB "
    "composite and a SWIR false-color composite. Using both views, detect any "
    "illegal small-scale gold mining pits. Include any exposed soil, excavation, "
    "or sediment-laden water even if you are uncertain - err toward detection. "
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

app = modal.App("galamsey-extract-v4-1-outputs")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=False)


@app.function(image=image, volumes={MOUNT: volume},
              gpu="H100", timeout=900,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def extract() -> dict:
    import json
    import torch
    from PIL import Image as PilImage
    from transformers import AutoModelForImageTextToText, AutoProcessor

    image_root = Path(MOUNT) / IMAGE_ROOT
    out_path = Path(OUT_PATH)
    device = torch.device("cuda")

    def parse_boxes(raw: str) -> list[dict]:
        try:
            s, e = raw.find("["), raw.rfind("]")
            if s >= 0 and e > s:
                parsed = json.loads(raw[s:e + 1])
                if isinstance(parsed, list):
                    return [b for b in parsed if isinstance(b, dict) and "bbox" in b]
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    def gen(model, processor, msgs, max_tok):
        inputs = processor.apply_chat_template(
            [msgs], tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
        plen = inputs["input_ids"].shape[1]
        out = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False)
        return processor.tokenizer.decode(out[0, plen:], skip_special_tokens=True).strip()

    def user_msg(rgb, swir, prompt):
        return [{"role": "user", "content": [
            {"type": "image", "image": rgb}, {"type": "image", "image": swir},
            {"type": "text", "text": prompt},
        ]}]

    # Load action eval JSONL so we can use the EXACT system+user prompt the
    # model was trained on (with full per-pass scalar context). The blog gallery
    # should reflect actual deployment conditions.
    eval_jsonl = Path(MOUNT) / "data/unified_v2/galamsey_unified_v2_eval_expanded.jsonl"
    eval_rows_by_cid: dict[str, dict] = {}
    for line in eval_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        first_image = next(c["image"] for c in r["messages"][1]["content"] if c["type"] == "image")
        cid = first_image.split("/")[0]
        eval_rows_by_cid[cid] = r

    tiles = {cid: (
        PilImage.open(image_root / cid / "rgb.png").convert("RGB"),
        PilImage.open(image_root / cid / "swir.png").convert("RGB"),
    ) for cid in TILES}

    print(f"\n=== v4.1 multitask ===\nLoading {Path(MOUNT) / V4_1_CKPT}")
    proc = AutoProcessor.from_pretrained(str(Path(MOUNT) / V4_1_CKPT), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(Path(MOUNT) / V4_1_CKPT), torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    results: dict[str, dict] = {}
    with torch.no_grad():
        for cid, (rgb, swir) in tiles.items():
            grounding = gen(model, proc, user_msg(rgb, swir, GROUNDING_PROMPT), 512)
            description = gen(model, proc, user_msg(rgb, swir, DESCRIPTION_PROMPT), 128)

            # Action: use the actual system+user prompt the model was trained on
            row = eval_rows_by_cid[cid]
            sys_text = row["messages"][0]["content"][0]["text"]
            user_text = next(c["text"] for c in row["messages"][1]["content"] if c["type"] == "text")
            action_msgs = [
                {"role": "system", "content": [{"type": "text", "text": sys_text}]},
                {"role": "user", "content": [
                    {"type": "image", "image": rgb}, {"type": "image", "image": swir},
                    {"type": "text", "text": user_text},
                ]},
            ]
            action_raw = gen(model, proc, action_msgs, 64)

            try:
                s, e = action_raw.find("{"), action_raw.rfind("}")
                action_obj = json.loads(action_raw[s:e+1]) if s >= 0 and e > s else {}
            except (json.JSONDecodeError, ValueError):
                action_obj = {}

            gold_action = json.loads(row["messages"][2]["content"][0]["text"])["action"]

            results[cid] = {
                "grounding_raw": grounding,
                "description": description,
                "action_raw": action_raw,
                "action": action_obj.get("action", "<unparseable>"),
                "boxes": parse_boxes(grounding),
                "gold_action": gold_action,
            }
            n_boxes = len(results[cid]["boxes"])
            print(f"  [{cid}] action={results[cid]['action']:20s} (gold={gold_action})"
                  f"  boxes={n_boxes}  desc: {description[:70]}...")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    return results


@app.local_entrypoint()
def main() -> None:
    res = extract.remote()
    import json
    print(json.dumps(res, indent=2))
