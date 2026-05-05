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
  - multitask
language:
  - en
---

# galamsey-unified-v4-1

Single-model unified VLM for **bandwidth-aware satellite tasking AND on-ground analyst review**, fine-tuned from [`samwell/galamsey-v9-e3`](https://huggingface.co/samwell/galamsey-v9-e3) (which itself is a perception fine-tune of [`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)). One weight set produces three outputs from three different prompts:

* `{"action": ...}` for the on-orbit policy decision
* bounding-box JSON for grounding
* free-form text for scene description

Used as the unified policy in [**GalamseyWatch**](https://github.com/samadon1/GalamseyWatch). Trained as the multitask follow-up to [`samwell/galamsey-unified-v3`](https://huggingface.co/samwell/galamsey-unified-v3): same base, same LoRA hyperparams, but a multitask training mixture that lets a single 450 M model serve all three jobs without losing v9-e3's perception ability or v3's action accuracy.

## What it does

Given a Sentinel-2 patch over southwestern Ghana (RGB + SWIR composites) plus a scalar context block, the model can be prompted three different ways:

| Prompt | Output |
|---|---|
| Action / policy prompt | One of `discard`, `flag_for_review`, `request_higher_resolution`, `request_neighbor_tile`, `downlink_now` |
| Grounding prompt | JSON list of bounding boxes: `[{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]` (normalised 0-1) |
| Description prompt | Free-form scene description |

In the GalamseyWatch deployment, the action prompt runs on orbit (cheap, 1 forward pass over a 450M model) and the grounding/description prompts run on the ground for the small subset of tiles the policy chose to downlink.

## Why a unified, multitask model

The earlier two-layer architecture used a 450 M perception VLM (`galamsey-v9-e3`) to emit boxes + a scene description, then a 2.6 B text-only LFM2 policy read those plus the scalar context and picked the action. The text description between the two layers is a real bottleneck - visual cues that the perception VLM doesn't surface in prose can't reach the policy.

`galamsey-unified-v3` collapsed both jobs into a single 450 M model with one LoRA on top of v9-e3, with action-only target. That gave +11.1 pp over the strongest baseline at 6.8 x fewer parameters, but it had a wart: training on action-only target partially overwrote v9-e3's perception ability, so the on-ground analyst-review step still needed a separate v9-e3 forward pass.

`galamsey-unified-v4-1` is the same recipe with a multitask data mixture that re-introduces a small, regulariser-sized dose of perception examples during the LoRA fine-tune. Result: perception is preserved at v9-e3 quality, action accuracy slightly exceeds v3, one weight set covers everything.

## Results

### Action eval (99-tile held-out set, expanded)

| System | Total params | 99-tile accuracy |
|---|---:|---:|
| Always-discard floor | - | 59.6 % |
| Two-layer (bare; perception + budget context only) | 3.05 B | 65.7 % |
| Two-layer (rich-context; + mission_priors + neighbor_summary) | 3.05 B | 63.6 % |
| Unified v2 (LoRA on bare base) | 450 M | 70.7 % |
| Unified v3 (action-only LoRA on v9-e3) | 450 M | 76.8 % |
| **galamsey-unified-v4-1 (multitask LoRA on v9-e3)** | **450 M** | **77.8 %** |

**+12.1 pp over the strongest baseline at 6.8 x fewer parameters.** Per-class recall on the 99-tile eval:

| Action | Bare two-layer | Rich-context two-layer | **galamsey-unified-v4-1** |
|---|---:|---:|---:|
| `discard` (n = 59) | 1.00 | 0.78 | 0.86 |
| `flag_for_review` (n = 18) | 0.11 | 0.56 | **0.89** |
| `downlink_now` (n = 21) | 0.19 | 0.33 | **0.48** |
| `request_higher_resolution` (n = 1) | 0.00 | 0.00 | 0.00 |

The 78 pp `flag_for_review` recall gap (0.89 vs 0.11) is the cleanest single-class evidence of the architectural advantage.

### Perception eval (100-tile head-to-head vs `galamsey-v9-e3`)

| Metric | `galamsey-v9-e3` (specialist) | **galamsey-unified-v4-1 (multitask)** |
|---|---:|---:|
| Grounding mean IoU | 0.337 | **0.334** |
| Box-count match rate | 25 % | **30 %** |
| Description BLEU | 34.13 | **33.18** |

v4.1 retains v9-e3's perception ability within noise on the same 100-tile sample drawn from the v9 grounding/description eval JSONLs.

## Honest ceiling

* **`request_higher_resolution` and `request_neighbor_tile`** sit at 0 % recall. Only 2 hires examples in the train set (oversampled to 80 by repetition, which the model memorised rather than generalised); zero neighbor examples. These two actions need deliberate hand-construction.
* **`downlink_now` recall is 0.48** (down from v3's 0.62). The multitask mixture makes the model slightly more cautious about emitting `downlink_now` because it now also sees perception examples that disambiguate "bright soil != necessarily mining". On a high-bandwidth pass this trades favourably (fewer wasted downlinks); on a low-bandwidth pass it could miss real signals. For deployments that prioritise downlink recall over per-pass economy, prefer v3.
* **Eval set is same-distribution.** The 99 held-out tiles come from the same 15 hand-curated AOIs as the training set; we have not yet evaluated on disjoint AOIs.
* **Single training run, no hyperparameter sweep.** Mixture ratio (327 action / 125 grounding / 125 description) was the second guess after a 1000-perception run that drowned the action signal. Likely room for refinement.
* **GRPO post-SFT remains a documented follow-up** with three identified `trl` integration gaps; not in this checkpoint.

## Training recipe

* **Base:** `samwell/galamsey-v9-e3` (full fine-tune of LFM2.5-VL-450M on SmallMinesDS for galamsey perception, 4 x D4 augmentation).
* **Adapter:** LoRA, r = 16, α = 32, on `q_proj`, `k_proj`, `v_proj`, `out_proj`, `in_proj`, plus vision-tower fc1/fc2 and multimodal projector.
* **Data:** 577 mixed rows total - 327 action examples (the same v2 oversampled corpus used by v3), plus 125 grounding rows and 125 description rows subsampled from the v9 perception SFT corpus. Action share 56.7 %. The perception subset acts as a regulariser keeping the LoRA from overwriting v9-e3's perception knowledge - we don't need to re-teach it from scratch, just remind it.
* **Target:** task-appropriate per row (action JSON, grounding JSON, or free-form description).
* **Optim:** AdamW with cosine LR schedule, peak `2e-5`, warmup 5 %, 15 epochs, batch 4 x grad-accum 2, bf16. Training ran ~13 minutes on Modal H100.
* **Config + scripts:** [`training/configs/galamsey_unified_v4_1_multitask_modal.yaml`](https://github.com/samadon1/GalamseyWatch/blob/main/training/configs/galamsey_unified_v4_1_multitask_modal.yaml) and [`training/scripts/build_unified_v4_1_multitask_dataset_modal.py`](https://github.com/samadon1/GalamseyWatch/blob/main/training/scripts/build_unified_v4_1_multitask_dataset_modal.py).

## Quickstart - all three prompts

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import json

model = AutoModelForImageTextToText.from_pretrained(
    "samwell/galamsey-unified-v4-1",
    torch_dtype="bfloat16",
    trust_remote_code=True,
).cuda().eval()
processor = AutoProcessor.from_pretrained(
    "samwell/galamsey-unified-v4-1", trust_remote_code=True,
)

rgb = Image.open("rgb.png").convert("RGB")
swir = Image.open("swir.png").convert("RGB")

def run(messages, max_new_tokens=128):
    inputs = processor.apply_chat_template(
        [messages], tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = {k: v.cuda() for k, v in inputs.items() if v is not None}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()

# 1) Action prompt (the on-orbit decision)
system_prompt = "You are an on-orbit Earth-observation policy adjudicator. ..."  # see repo
user_text = "Tile u0001 at lon=-2.75, lat=5.64. Cloud cover: 0.001. Pass budget: 320 of 512 KB remaining. ..."
action_msgs = [
    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    {"role": "user", "content": [
        {"type": "image", "image": rgb},
        {"type": "image", "image": swir},
        {"type": "text", "text": user_text},
    ]},
]
action_text = run(action_msgs, max_new_tokens=32)
action = json.loads(action_text[action_text.find("{"):action_text.rfind("}")+1])["action"]
print("ACTION:", action)

# 2) Grounding prompt (boxes for analyst review)
grounding_prompt = (
    "You are viewing two images of the same Sentinel-2 patch: a natural-color RGB "
    "composite and a SWIR false-color composite. Detect any illegal small-scale gold "
    'mining pits. Provide result as a valid JSON: [{"label": str, "bbox": [x1,y1,x2,y2]}, ...]. '
    "Coordinates must be normalized to 0-1. If no pits are visible, return []."
)
grounding_msgs = [{"role": "user", "content": [
    {"type": "image", "image": rgb},
    {"type": "image", "image": swir},
    {"type": "text", "text": grounding_prompt},
]}]
print("BOXES:", run(grounding_msgs, max_new_tokens=512))

# 3) Description prompt (free-form scene summary)
description_prompt = (
    "You are analyzing two views of the same Sentinel-2 patch of southwestern Ghana. "
    "Describe any signs of illegal small-scale gold mining (galamsey) activity: "
    "exposed soil, excavation pits, sediment plumes, vegetation loss, proximity to water. "
    "If no mining is visible, say so."
)
desc_msgs = [{"role": "user", "content": [
    {"type": "image", "image": rgb},
    {"type": "image", "image": swir},
    {"type": "text", "text": description_prompt},
]}]
print("DESCRIPTION:", run(desc_msgs, max_new_tokens=128))
```

## Related artifacts

* **Predecessor (action-only):** [`samwell/galamsey-unified-v3`](https://huggingface.co/samwell/galamsey-unified-v3) - same base, same LoRA, action-only target. Slightly higher `downlink_now` recall, no perception preserved.
* **Perception base:** [`samwell/galamsey-v9-e3`](https://huggingface.co/samwell/galamsey-v9-e3) - the perception fine-tune this model stacks on.
* **Browser/WebGPU sibling of the perception model:** [`samwell/galamsey-v9-e3-onnx`](https://huggingface.co/samwell/galamsey-v9-e3-onnx)
* **Training data:** [`samwell/galamsey-unified-decisions`](https://huggingface.co/datasets/samwell/galamsey-unified-decisions) - 250 hand-labeled Sentinel-2 tiles + scalar context + 5-action targets
* **Repo:** [`samadon1/GalamseyWatch`](https://github.com/samadon1/GalamseyWatch)
* **Live demo (perception only, browser/WebGPU):** [galamseywatch.vercel.app](https://galamseywatch.vercel.app)

## License

[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/main/LICENSE), inherited from the base model.

## Citation

```bibtex
@misc{galamseywatch2026,
  author = {Donkor, Samuel},
  title = {GalamseyWatch: agentic Earth observation for galamsey detection},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/samadon1/GalamseyWatch}
}
```
