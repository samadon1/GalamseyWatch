"""Evaluate v4 multitask vs v9-e3 on perception holdouts (head-to-head).

Goal: did v4's multitask SFT preserve v9-e3's grounding + description ability?

For each of N tiles drawn from the v9 grounding/description eval JSONLs:
  - Run BOTH v4 and v9-e3 with the same prompt
  - Grounding: parse predicted boxes, compute mean IoU vs gold boxes
  - Description: compute corpus BLEU vs gold description

Output: stdout summary + JSON of per-tile predictions.

Usage:
    cd training && uv run modal run scripts/eval_unified_v4_perception_modal.py
"""
from __future__ import annotations
from pathlib import Path
import modal

MODAL_VOLUME_NAME = "galamsey"
MOUNT = "/galamsey"

V9_CKPT = (
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-20260418_165633/"
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
)
V4_CKPT = (
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
    "-vlm_sft-galamsey_u-all-lr2em05-w0p1-lora_a-20260505_154237/"
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
    "-vlm_sft-galamsey_u-all-lr2em05-w0p1-lora_m-20260505_154237"
)

GROUNDING_JSONL = "data/v9/galamsey_v9_grounding_eval.jsonl"
DESCRIPTION_JSONL = "data/v9/galamsey_v9_description_eval.jsonl"
IMAGE_ROOT = "data/v9/images"
N_GROUNDING = 100
N_DESCRIPTION = 100
OUT_PATH = "data/unified_v4_1_multitask/perception_eval_v4_1_vs_v9.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4", "torchvision>=0.19",
        "transformers>=4.46", "pillow>=11.0",
        "accelerate>=1.0", "safetensors>=0.4",
        "sacrebleu>=2.4",
    )
)
app = modal.App("galamsey-unified-v4-1-perception-eval")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=False)


@app.function(image=image, volumes={MOUNT: volume},
              gpu="H100", timeout=3600,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def evaluate() -> dict:
    import json, time, statistics
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import sacrebleu

    root = Path(MOUNT)
    image_root = root / IMAGE_ROOT
    device = torch.device("cuda")

    def parse_boxes(raw: str) -> list[list[float]]:
        try:
            s, e = raw.find("["), raw.rfind("]")
            if s < 0 or e <= s:
                return []
            parsed = json.loads(raw[s:e + 1])
            if not isinstance(parsed, list):
                return []
            out = []
            for b in parsed:
                if isinstance(b, dict) and "bbox" in b and isinstance(b["bbox"], list) and len(b["bbox"]) == 4:
                    out.append([float(x) for x in b["bbox"]])
            return out
        except (json.JSONDecodeError, ValueError, TypeError):
            return []

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = a_area + b_area - inter
        return inter / union if union > 0 else 0.0

    def best_iou_match(pred_boxes, gold_boxes):
        """For each gold box, find max IoU over pred boxes; return mean."""
        if not gold_boxes:
            return 1.0 if not pred_boxes else 0.0
        if not pred_boxes:
            return 0.0
        scores = []
        for gb in gold_boxes:
            best = max((iou(pb, gb) for pb in pred_boxes), default=0.0)
            scores.append(best)
        return sum(scores) / len(scores)

    def gen(model, processor, msgs, max_tok):
        inputs = processor.apply_chat_template(
            [msgs], tokenize=True, return_dict=True,
            return_tensors="pt", add_generation_prompt=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
        plen = inputs["input_ids"].shape[1]
        out = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False)
        return processor.tokenizer.decode(out[0, plen:], skip_special_tokens=True).strip()

    grounding_rows = [
        json.loads(l) for l in (root / GROUNDING_JSONL).read_text().splitlines() if l.strip()
    ][:N_GROUNDING]
    description_rows = [
        json.loads(l) for l in (root / DESCRIPTION_JSONL).read_text().splitlines() if l.strip()
    ][:N_DESCRIPTION]

    print(f"Grounding rows: {len(grounding_rows)}  Description rows: {len(description_rows)}")
    results = {"grounding": [], "description": []}

    for label, ckpt in [("v9", V9_CKPT), ("v4", V4_CKPT)]:
        print(f"\n=== {label} ===\nLoading {root / ckpt}")
        proc = AutoProcessor.from_pretrained(str(root / ckpt), trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            str(root / ckpt), torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device).eval()

        with torch.no_grad():
            t0 = time.time()
            for i, row in enumerate(grounding_rows):
                user = row["messages"][0]["content"]
                images = [c["image"] for c in user if c["type"] == "image"]
                user_text = next(c["text"] for c in user if c["type"] == "text")
                pil_imgs = [Image.open(image_root / fn).convert("RGB") for fn in images]
                content = [{"type": "image", "image": pi} for pi in pil_imgs] + [{"type": "text", "text": user_text}]
                msgs = [{"role": "user", "content": content}]
                pred = gen(model, proc, msgs, 256)

                gold_text = row["messages"][1]["content"][0]["text"]
                gold_boxes = parse_boxes(gold_text)
                pred_boxes = parse_boxes(pred)
                row_iou = best_iou_match(pred_boxes, gold_boxes)
                results["grounding"].append({
                    "model": label, "i": i,
                    "n_gold": len(gold_boxes), "n_pred": len(pred_boxes),
                    "iou": row_iou, "pred_raw": pred[:200], "gold_raw": gold_text[:200],
                })
                if (i + 1) % 20 == 0:
                    cur = [r["iou"] for r in results["grounding"] if r["model"] == label]
                    print(f"  grounding [{i+1}/{len(grounding_rows)}] mIoU={statistics.mean(cur):.3f}  ({time.time()-t0:.0f}s)")

            t1 = time.time()
            for i, row in enumerate(description_rows):
                user = row["messages"][0]["content"]
                images = [c["image"] for c in user if c["type"] == "image"]
                user_text = next(c["text"] for c in user if c["type"] == "text")
                pil_imgs = [Image.open(image_root / fn).convert("RGB") for fn in images]
                content = [{"type": "image", "image": pi} for pi in pil_imgs] + [{"type": "text", "text": user_text}]
                msgs = [{"role": "user", "content": content}]
                pred = gen(model, proc, msgs, 256)
                gold = row["messages"][1]["content"][0]["text"]
                results["description"].append({"model": label, "i": i,
                                               "pred": pred[:300], "gold": gold[:300]})
                if (i + 1) % 20 == 0:
                    print(f"  description [{i+1}/{len(description_rows)}]  ({time.time()-t1:.0f}s)")

        del model, proc
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    summary = {}
    for label in ["v9", "v4"]:
        g = [r for r in results["grounding"] if r["model"] == label]
        d = [r for r in results["description"] if r["model"] == label]
        mean_iou = statistics.mean(r["iou"] for r in g) if g else 0.0
        cnt_match = sum(1 for r in g if r["n_pred"] == r["n_gold"]) / len(g) if g else 0.0
        bleu = sacrebleu.corpus_bleu([r["pred"] for r in d],
                                     [[r["gold"] for r in d]]).score if d else 0.0
        empty = sum(1 for r in d if not r["pred"].strip()) if d else 0
        summary[label] = {
            "grounding_mean_iou": round(mean_iou, 4),
            "grounding_box_count_match_rate": round(cnt_match, 4),
            "description_bleu": round(bleu, 2),
            "n_empty_descriptions": empty,
        }
        print(f"{label:5s}  mIoU={mean_iou:.3f}  box_count_match={cnt_match:.2%}"
              f"  desc_BLEU={bleu:.2f}  empty_desc={empty}")

    (root / OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / OUT_PATH).write_text(json.dumps({"summary": summary, "per_row": results}, indent=2))
    print(f"\nWrote {OUT_PATH}")
    return summary


@app.local_entrypoint()
def main() -> None:
    s = evaluate.remote()
    import json
    print(json.dumps(s, indent=2))
