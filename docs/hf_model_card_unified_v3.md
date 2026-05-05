---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/main/LICENSE
base_model: samwell/galamsey-v9-e3
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - earth-observation
  - sentinel-2
  - galamsey
  - illegal-mining
  - lfm2-vl
  - lfm2.5-vl
  - agentic-eo
  - on-orbit-policy
  - tool-calling
language:
  - en
---

# galamsey-unified-v3

Single-model **on-orbit decision policy** for bandwidth-aware satellite tasking, fine-tuned from [`samwell/galamsey-v9-e3`](https://huggingface.co/samwell/galamsey-v9-e3) (which itself is a perception fine-tune of [`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)). One forward pass through one 450M VLM picks the action - no separate perception + policy models, no text bottleneck between layers.

Used as the unified policy in [**GalamseyWatch**](https://github.com/samadon1/GalamseyWatch). Trained as Phase A of the agentic-EO research plan ([`docs/UNIFIED_VLM_PLAN.md`](https://github.com/samadon1/GalamseyWatch/blob/main/docs/UNIFIED_VLM_PLAN.md)) to test whether a single LFM2.5-VL-450M can replace a 3.05 B two-layer (perception VLM + LFM2 text policy) reference system on action selection.

## What it does

Given a Sentinel-2 patch over southwestern Ghana, the model returns one of five satellite-tasking actions for the tile:

| Action | Meaning |
|---|---|
| `discard` | skip the tile entirely (forest, water, urban, cloud, no signal) |
| `flag_for_review` | log as text only - no image downlink, cheap follow-up |
| `request_higher_resolution` | request a higher-res recapture next pass |
| `request_neighbor_tile` | fetch the adjacent tile in a given direction |
| `downlink_now` | spend the downlink budget to send this tile to ground |

**Input per tile:**
- Two images: a natural-color **RGB** composite and a **SWIR false-color** composite (SWIR2 + SWIR1 + NIR), both 256×256 PNGs of a 1.28 km Sentinel-2 patch
- A short text block carrying scalar context: downlink budget remaining (KB), prior tiles already downlinked this pass, AOI mission prior text, cloud cover, capture time, and a structured neighbor summary describing what each adjacent tile decided this pass

**Output:**
A JSON object `{"action": "<action_name>"}`, optionally with a `direction` field for `request_neighbor_tile`.

See [`orchestrator/agentic_eo/models/agent.py`](https://github.com/samadon1/GalamseyWatch/blob/main/orchestrator/agentic_eo/models/agent.py) for the exact prompt schema and parsing.

## Why a unified model

The earlier GalamseyWatch architecture used a **two-layer** pipeline: a 450M perception VLM (`galamsey-v9-e3`) emitted bounding boxes + a scene description, then a 2.6B text-only LFM2 policy read those plus a scalar context block and picked the action. The split was well-motivated when perception and policy come from different communities, but the description string is a real bottleneck - visual cues that the perception VLM doesn't surface in prose can't reach the policy.

`galamsey-unified-v3` collapses both jobs into one model. It reads the pixels and the scalar context jointly, attends across them, and emits the action directly. No intermediate text. The architectural advantage is concentrated where it should be - the `flag_for_review` class, where ambiguous tiles need joint pixel+context reasoning that text descriptions tend to flatten.

## Results

Evaluated on a held-out 99-tile expanded set (39 original + 60 newly labeled at indices the training set never saw, all from the same deterministic sampler at seed 42). Action-match accuracy:

| System | Total params | 99-tile accuracy |
|---|---:|---:|
| Always-discard floor | - | 59.6 % |
| Two-layer (bare; perception + budget context only) | 3.05 B | 65.7 % |
| Two-layer (rich-context; + mission_priors + neighbor_summary) | 3.05 B | 63.6 % |
| Unified v2 (LoRA on bare base) | 450 M | 70.7 % |
| **galamsey-unified-v3 (LoRA on `galamsey-v9-e3`)** | **450 M** | **76.8 %** |

**+11.1 pp over the strongest baseline at 6.8× fewer parameters.** Per-class recall on the 99-tile eval:

| Action | Bare two-layer | Rich-context two-layer | **galamsey-unified-v3** |
|---|---:|---:|---:|
| `discard` (n = 59) | 1.00 | 0.78 | 0.80 |
| `flag_for_review` (n = 18) | 0.11 | 0.56 | **0.89** |
| `downlink_now` (n = 21) | 0.19 | 0.33 | **0.62** |
| `request_higher_resolution` (n = 1) | 0.00 | 0.00 | 0.00 |

The 78 pp `flag_for_review` recall gap (0.89 vs 0.11) is the cleanest single-class evidence of the architectural advantage. Full evaluation methodology, confusion matrices, and the per-tile predictions JSONL are documented in [`docs/UNIFIED_VLM_RESULTS.md`](https://github.com/samadon1/GalamseyWatch/blob/main/docs/UNIFIED_VLM_RESULTS.md).

## Honest ceiling

- **`request_higher_resolution` and `request_neighbor_tile`** sit at 0 % recall. Only 2 hires examples in the train set (oversampled to 80 by repetition, which the model memorised rather than generalised); zero neighbor examples. These two actions are structurally hard to elicit from naturalistic Sentinel-2 imagery and need deliberate hand-construction.
- **Eval set is same-distribution.** The 99 held-out tiles come from the same 15 hand-curated AOIs as the training set; we have not yet evaluated on disjoint AOIs. Robustness across Ghana isn't proven, only across held-out indices within the sampled regions.
- **Single training run, no hyperparameter sweep.** LoRA rank, learning rate, oversampling ratio chosen by analogy to the v9-e3 perception fine-tune. Likely room for improvement.
- **GRPO post-SFT is documented as a follow-up with three identified `trl` integration gaps**; not in this checkpoint. See the results doc for the findings.

## Training recipe

- **Base:** `samwell/galamsey-v9-e3` (full fine-tune of LFM2.5-VL-450M on SmallMinesDS for galamsey perception, 4× D4 augmentation)
- **Adapter:** LoRA, r = 16, α = 32, on `q_proj`, `k_proj`, `v_proj`, `out_proj`, `in_proj`, plus vision-tower fc1/fc2 and multimodal projector
- **Data:** 327 train rows (151 unique tiles, oversampled across 4 actions: 87 discard / 80 flag / 80 downlink / 80 hires; neighbor class absent from labels). 99 held-out eval tiles. Hand-labeled over multiple sessions following the validated labeling protocol in [`docs/UNIFIED_VLM_VALIDATION.md`](https://github.com/samadon1/GalamseyWatch/blob/main/docs/UNIFIED_VLM_VALIDATION.md).
- **Target:** action-only - assistant emits just `{"action": "<action_name>"}` (~15 tokens). Concentrates LM loss on the prediction we care about. (An earlier v1 attempt with a JSON+reason target collapsed to {discard, flag} only and scored below the always-discard floor - see the results doc.)
- **Optim:** AdamW with cosine LR schedule, peak `2e-5`, warmup 5 %, 15 epochs, batch 4 × grad-accum 2, bf16. Training ran ~9 minutes on Modal H100.
- **Config + scripts:** [`training/configs/galamsey_unified_v3_modal.yaml`](https://github.com/samadon1/GalamseyWatch/blob/main/training/configs/galamsey_unified_v3_modal.yaml) and [`training/scripts/build_unified_v2_sft_dataset.py`](https://github.com/samadon1/GalamseyWatch/blob/main/training/scripts/build_unified_v2_sft_dataset.py) (v2 and v3 share the same training set; only the LoRA base differs).

## Quickstart

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import json

model = AutoModelForImageTextToText.from_pretrained(
    "samwell/galamsey-unified-v3",
    torch_dtype="bfloat16",
    trust_remote_code=True,
).cuda().eval()
processor = AutoProcessor.from_pretrained(
    "samwell/galamsey-unified-v3", trust_remote_code=True,
)

rgb = Image.open("rgb.png").convert("RGB")
swir = Image.open("swir.png").convert("RGB")

system_prompt = "You are an on-orbit Earth-observation policy adjudicator. ..."  # see docs/UNIFIED_VLM_PLAN.md
user_text = (
    "Tile u0001 at lon=-2.7500, lat=5.6400.\n"
    "Cloud cover (metadata, may be unreliable): 0.001\n"
    "Pass budget: 320 of 512 KB remaining\n"
    "Prior tiles downlinked this pass: 2\n"
    "Mission priors: Bibiani: known active galamsey cluster\n"
    'Neighbor summary (structured):\n{"north": null, ...}\n'
)

messages = [
    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    {"role": "user", "content": [
        {"type": "image", "image": rgb},
        {"type": "image", "image": swir},
        {"type": "text", "text": user_text},
    ]},
]

inputs = processor.apply_chat_template(
    [messages], tokenize=True, return_dict=True, return_tensors="pt",
    add_generation_prompt=True,
)
inputs = {k: v.cuda() for k, v in inputs.items() if v is not None}
out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
text = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
action = json.loads(text[text.find("{"):text.rfind("}")+1])["action"]
print(action)  # one of: discard, flag_for_review, request_higher_resolution, request_neighbor_tile, downlink_now
```

For the full prompt template (system message, user-text builder, parsing), see [`orchestrator/agentic_eo/models/agent.py`](https://github.com/samadon1/GalamseyWatch/blob/main/orchestrator/agentic_eo/models/agent.py).

## Related artifacts

- **Perception base:** [`samwell/galamsey-v9-e3`](https://huggingface.co/samwell/galamsey-v9-e3) - the perception fine-tune this model stacks on
- **Browser/WebGPU sibling of the perception model:** [`samwell/galamsey-v9-e3-onnx`](https://huggingface.co/samwell/galamsey-v9-e3-onnx)
- **Training data:** [`samwell/galamsey-unified-decisions`](https://huggingface.co/datasets/samwell/galamsey-unified-decisions) (250 hand-labeled Sentinel-2 tiles + scalar context + 5-action targets)
- **Repo:** [`samadon1/GalamseyWatch`](https://github.com/samadon1/GalamseyWatch)
- **Live demo (perception only, browser/WebGPU):** [galamseywatch.vercel.app](https://galamseywatch.vercel.app)

## License

[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/main/LICENSE), inherited from the base model.

## Citation

If you use this model, please cite the GalamseyWatch repository:

```bibtex
@misc{galamseywatch2026,
  author = {Donkor, Samuel},
  title = {GalamseyWatch: agentic Earth observation for galamsey detection},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/samadon1/GalamseyWatch}
}
```
