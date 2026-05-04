# GalamseyWatch — training

Fine-tuning [LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) on [SmallMinesDS](https://huggingface.co/datasets/ellaampy/SmallMinesDS) (Ofori-Ampofo et al., 2025) for galamsey detection in 10 m/px Sentinel-2 imagery over Ghana. The model produced here is the perception layer of the [GalamseyWatch agentic pipeline](../README.md).

## Headline result

| Metric | Base LFM2.5-VL-450M | galamsey-v9-e3 (this fine-tune) | Δ |
|---|---:|---:|---:|
| Pixel IoU | 0.069 | **0.332** | **+0.263** (~4.8×) |

Evaluated on the SmallMinesDS test split, RGB + SWIR two-image prompt, full pixel-IoU pipeline (bbox-to-mask scoring). The fine-tuned model sits at **71% of the achievable bbox ceiling** (0.469) for any axis-aligned-bbox method on this benchmark.

Recall, F1, and patch accuracy for the fine-tuned model: **0.592 / 0.499 / 0.795** respectively. These were not separately recorded for the base run.

Public weights:

- PyTorch: [samwell/galamsey-v9-e3](https://huggingface.co/samwell/galamsey-v9-e3)
- ONNX (fp16, browser/WebGPU): [samwell/galamsey-v9-e3-onnx](https://huggingface.co/samwell/galamsey-v9-e3-onnx)

## Production recipe

The shipped model (`galamsey-v9-e3`) was trained with:

| | |
|---|---|
| **Config** | [`configs/galamsey_v9_450m_aug_modal.yaml`](configs/galamsey_v9_450m_aug_modal.yaml) |
| **Base** | LiquidAI/LFM2.5-VL-450M |
| **Method** | Full fine-tuning, no LoRA |
| **Dataset** | SmallMinesDS, 4,270 labeled Ghana patches, RGB + SWIR two-image inputs |
| **Augmentation** | 4× D4 dihedral group (flips + rotations) |
| **Schedule** | 3 epochs (17,719 steps), batch 4 × grad-accum 2, cosine LR, warmup 0.03 |
| **Learning rate** | 2e-5, with separate rates for LM / projector / vision tower |
| **Hardware** | 1× NVIDIA H100 via [Modal](https://modal.com) |
| **Final training loss** | 0.175 (from 2.10 at step 1) |

The v9 config resumes from a v8-e2 checkpoint, so v9-e3 is effectively v8 + one extra epoch with the augmentation policy held fixed. The intermediate `outputs/` directory (~10 GB of checkpoints + HF cache) is gitignored; pull from Modal or HuggingFace if you want the raw artifacts.

## Reproducing the headline numbers

The same framework that produced the fine-tune metrics also runs the base-model baseline, so the comparison is apples-to-apples (same prompt, same eval set, same scoring code).

| | |
|---|---|
| **Fine-tuned eval** | [`scripts/eval_v9_pixel_iou_modal.py`](scripts/eval_v9_pixel_iou_modal.py) |
| **Base-model eval** | [`scripts/eval_base_pixel_iou_modal.py`](scripts/eval_base_pixel_iou_modal.py) |

Both scripts run on Modal against the SmallMinesDS test split. The v9 script also prints a side-by-side comparison against the prior v8 run for ablation.

## Directory layout

```
training/
├── README.md                    , this file
├── pyproject.toml               , uv-managed Python project + local galamseywatch package
├── galamseywatch/               , shared preprocessing primitives (composites, masks, prompts, JSONL writers)
├── configs/                     , YAML hyperparameter sets for v1..v10 (the version progression IS the methodology)
└── scripts/                     , Modal entrypoints
    ├── prepare_smallminesds.py        , pull the upstream dataset
    ├── prepare_v{N}_dataset_modal.py  , convert to multitask JSONL with paired RGB + SWIR composites
    ├── eval_v{N}_pixel_iou_modal.py   , pixel-IoU eval per checkpoint (v9 ships)
    ├── eval_base_pixel_iou_modal.py   , same eval against the un-fine-tuned base model
    └── export_v8_onnx_*.py            , ONNX (fp16) export for browser/WebGPU runtime
```

The `notebooks/` directory holds the dev notebooks used during iteration; it is gitignored to keep the public surface focused on the production scripts. `data/`, `models/`, and `outputs/` are gitignored because they're large (multi-GB checkpoints + intermediate caches); fetch from HuggingFace and Modal as needed.

## Why so many config versions?

`configs/galamsey_v1.yaml` through `galamsey_v10_450m_aug_modal.yaml` are the iteration history. A few notable ones:

- `v4_rgb_modal.yaml` — RGB-only training. Flipped a prior assumption: pretrained VLMs care more about ImageNet-shaped pixels than spectral content. Counterintuitively beat earlier SWIR-only runs on this dataset's distribution. SWIR was added back as a *paired* second image in v5+ and that's what shipped.
- `v8_450m_aug_modal.yaml` — first version with 4× D4 augmentation + small-bbox filtering. Reached pixel IoU 0.295 (62.8% of bbox ceiling).
- `v9_450m_aug_modal.yaml` — v8 + 1 epoch under the same recipe, 4× D4 aug. Pixel IoU 0.332, **the production model.**
- `v10_450m_aug_modal.yaml` — attempted to push further. 5 epochs (59,065 steps) vs v9's 3 epochs (17,719 steps). Marginal gain (+0.005 pixel IoU). Returns clearly diminished, v9 stayed shipped.

## Cross-references

- **Parent repo:** [github.com/samadon1/GalamseyWatch](https://github.com/samadon1/GalamseyWatch)
- **Live demo:** [galamseywatch.vercel.app](https://galamseywatch.vercel.app)
- **Model card (PyTorch):** [`samwell/galamsey-v9-e3`](https://huggingface.co/samwell/galamsey-v9-e3)
- **Model card (ONNX):** [`samwell/galamsey-v9-e3-onnx`](https://huggingface.co/samwell/galamsey-v9-e3-onnx)
- **Dataset:** [`ellaampy/SmallMinesDS`](https://huggingface.co/datasets/ellaampy/SmallMinesDS) (Ofori-Ampofo et al., 2025)
- **Base model:** [`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)
