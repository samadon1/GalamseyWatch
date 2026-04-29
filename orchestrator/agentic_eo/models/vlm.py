"""VLM perception layer.

Wraps the v9-e3 fine-tuned LFM2.5-VL-450M for in-process perception.
Reuses the exact prompts, dual-image input, and generation params from
``app/src/lib/inference.ts`` so the orchestrator and the browser path
are the same model with the same input contract.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = os.environ.get(
    "GALAMSEY_VLM_PATH",
    str(Path(__file__).resolve().parents[2] / "checkpoints" / "galamsey-v9-e3"),
)

# Verbatim from app/src/lib/inference.ts — same model, same prompt strings,
# so any drift between browser and orchestrator behavior comes from runtime
# (ONNX vs torch) not prompt design.
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

# Mirror inference.ts post-processing
NMS_IOU_THRESHOLD = 0.5
MIN_BBOX_AREA = 0.0001
GROUNDING_MAX_TOKENS = 256  # JSON output up to ~180 tokens in practice
DESCRIPTION_MAX_TOKENS = 128  # NL prose typically ~35 tokens; cap for speed


@dataclass
class VlmBox:
    label: str
    bbox: list[float]  # [x0, y0, x1, y1] normalized
    area: float


@dataclass
class VlmResult:
    boxes: list[VlmBox]
    raw_grounding: str
    description: str
    inference_ms: int


class Lfm2VlGalamseyProvider:
    """Process-level singleton. ``ensure_loaded()`` is idempotent and lazy."""

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT) -> None:
        self.checkpoint = checkpoint
        self._model = None
        self._processor = None
        self._device: str | None = None
        self._lock = asyncio.Lock()

    def _ensure_loaded_sync(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        logger.info(
            "loading VLM %s on %s (dtype=%s)", self.checkpoint, device, dtype
        )
        self._processor = AutoProcessor.from_pretrained(self.checkpoint)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint,
            dtype=dtype,
        ).to(device)
        self._model.eval()
        self._device = device

    def _generate_sync(
        self,
        rgb: Image.Image,
        swir: Image.Image,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        import torch

        self._ensure_loaded_sync()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(
            images=[rgb, swir],
            text=chat_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        input_len = inputs["input_ids"].shape[-1]
        gen = out[:, input_len:]
        return self._processor.batch_decode(gen, skip_special_tokens=True)[0]

    async def detect(self, rgb_png: bytes, swir_png: bytes) -> VlmResult:
        rgb = Image.open(BytesIO(rgb_png)).convert("RGB")
        swir = Image.open(BytesIO(swir_png)).convert("RGB")

        # Serialize VLM calls so two parallel passes don't interleave on the
        # same model state. Generation itself runs in a worker thread so the
        # asyncio loop stays responsive.
        async with self._lock:
            started = perf_counter()
            grounding = await asyncio.to_thread(
                self._generate_sync, rgb, swir, GROUNDING_PROMPT, GROUNDING_MAX_TOKENS
            )
            description = await asyncio.to_thread(
                self._generate_sync, rgb, swir, DESCRIPTION_PROMPT, DESCRIPTION_MAX_TOKENS
            )
            elapsed_ms = int((perf_counter() - started) * 1000)

        return VlmResult(
            boxes=_parse_and_filter_boxes(grounding),
            raw_grounding=grounding,
            description=description.strip(),
            inference_ms=elapsed_ms,
        )


# Process-wide singleton — instantiated cheaply, weights load on first use.
VLM = Lfm2VlGalamseyProvider()


# --- Box parsing (mirrors inference.ts) ----------------------------------

def _parse_and_filter_boxes(text: str) -> list[VlmBox]:
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    raw: list[VlmBox] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            c = [max(0.0, min(1.0, float(x))) for x in bbox]
        except (TypeError, ValueError):
            continue
        if not (c[2] > c[0]) or not (c[3] > c[1]):
            continue
        area = (c[2] - c[0]) * (c[3] - c[1])
        if area < MIN_BBOX_AREA:
            continue
        raw.append(VlmBox(label=str(item.get("label", "mining_pit")), bbox=c, area=area))

    raw.sort(key=lambda b: -b.area)
    kept: list[VlmBox] = []
    for b in raw:
        if any(_iou(b.bbox, k.bbox) >= NMS_IOU_THRESHOLD for k in kept):
            continue
        kept.append(b)
    return kept


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)
