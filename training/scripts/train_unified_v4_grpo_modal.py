"""GRPO post-SFT from the v3 stacked-LoRA checkpoint.

STATUS: incomplete — see "Integration findings" below. The training script
runs but does not converge end-to-end on LFM2.5-VL with trl 1.3.0. We
preserve it here as the artifact behind the documented finding rather than
a working recipe.

Setup attempted:
  - Base: v3 merged checkpoint (LoRA on v9-e3, then merged)
  - Reward: 1.0 if parsed action equals gold, 0.0 otherwise
  - Group size: 4 completions per prompt
  - Train data: 327-row oversampled set (same as v2/v3 SFT)
  - PEFT: new LoRA on top of v3 merged weights

Integration findings (LFM2.5-VL ↔ trl 1.3.0 multimodal GRPO):

  1. SOLVED. trl's GRPOTrainer has an explicit allowlist of multimodal
     kwargs in `_get_per_token_logps_and_entropies` and
     `_get_last_hidden_state` — `pixel_values`, `image_grid_thw`,
     `pixel_attention_mask`, `image_sizes`, `image_position_ids`,
     `token_type_ids`, `mm_token_type_ids`. **`spatial_shapes` is not in
     it.** LFM2.5-VL's NaFlex SigLIP2 encoder produces this kwarg, so
     forward fails with `TypeError: unexpected keyword argument
     'spatial_shapes'`. The fix is the GRPOTrainerLfm2Vl subclass below
     (~80 lines, copies the parent method bodies and adds spatial_shapes
     to the model_inputs dict with the same per-image slicing the other
     vision kwargs use).

  2. PARTIALLY MITIGATED. SigLIP2's SDPA attention path triggers a cuDNN
     graph error under GRPO's variable batch shapes (K=4 sampled
     completions per prompt change the batch composition between forward
     passes). `attn_implementation="eager"` avoids it.

  3. UNSOLVED. After (1) and (2), the next failure is a CUDA device-side
     assert in `transformers/masking_utils.py::create_bidirectional_mask`
     during the SigLIP2 encoder's attention-mask construction. Likely an
     alignment issue between how GRPO expands the batch with K=4
     completions and how LFM2.5-VL's image-token attention mask is built.
     This needs a deeper fix — either an upstream transformers patch or a
     custom collator that pre-aligns the mask shapes before the trainer
     sees them. We did not pursue further; the v3 SFT result is already
     statistically robust at +11.1 pp over the best baseline on 99 tiles
     (UNIFIED_VLM_RESULTS.md), so GRPO is polish rather than headline.

Worth filing upstream with trl as part of a fuller LFM2.5-VL multimodal
GRPO support PR — finding (1) is a small clean fix, finding (3) needs more
investigation but the failure mode is reproducible on this script.

Usage (will fail at issue 3 above):
    cd training && uv run modal run scripts/train_unified_v4_grpo_modal.py
"""
from __future__ import annotations
from pathlib import Path
import modal

MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"

V3_CHECKPOINT = (
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
    "-vlm_sft-galamsey_u-all-lr2em05-w0p1-lora_a-20260505_080020/"
    "lfm2.5-VL-450M-vlm_sft-galamsey_v-all-lr2em05-w0p0-no_lora-e3s35439-20260418_165633"
    "-vlm_sft-galamsey_u-all-lr2em05-w0p1-lora_m-20260505_080020"
)
TRAIN_JSONL = "data/unified_v2/galamsey_unified_v2_train.jsonl"
IMAGE_ROOT = "data/unified_v2/images"
OUTPUT_DIR = "v4_grpo_runs"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4", "torchvision>=0.19",
        "transformers>=4.46", "trl>=0.18",
        "peft>=0.13", "accelerate>=1.0",
        "datasets>=3.0", "pillow>=11.0",
        "safetensors>=0.4", "bitsandbytes>=0.43",
    )
)

app = modal.App("galamsey-unified-v4-grpo")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=False)


@app.function(
    image=image,
    volumes={MODAL_MOUNT_POINT: volume},
    gpu="H100",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train() -> dict:
    import json
    import time
    from pathlib import Path

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import GRPOConfig, GRPOTrainer
    from trl.extras.profiling import profiling_decorator
    from accelerate.utils import is_peft_model
    from trl.trainer.utils import selective_log_softmax, entropy_from_logits

    # --- LFM2.5-VL compat patch -----------------------------------------
    # trl 1.3.0's GRPOTrainer has an explicit allowlist of VLM kwargs in
    # _get_per_token_logps_and_entropies + _get_last_hidden_state. The
    # allowlist covers Qwen-VL (image_grid_thw), LLaVa-Next (image_sizes),
    # SmolVLM2 (pixel_attention_mask), Gemma (pixel_values) etc. but NOT
    # `spatial_shapes`, which LFM2.5-VL's NaFlex SigLIP encoder produces.
    # Without these overrides the trainer crashes immediately on the first
    # forward with TypeError: unexpected kwarg 'spatial_shapes'. We extend
    # both methods to thread spatial_shapes through model_inputs with the
    # same per-image slicing the existing kwargs use.

    class GRPOTrainerLfm2Vl(GRPOTrainer):
        @profiling_decorator
        def _get_per_token_logps_and_entropies(
            self, model, input_ids, attention_mask, logits_to_keep,
            batch_size=None, compute_entropy=False,
            pixel_values=None, image_grid_thw=None, num_images=None,
            pixel_attention_mask=None, image_sizes=None,
            token_type_ids=None, mm_token_type_ids=None,
            image_position_ids=None,
            spatial_shapes=None,  # added for LFM2.5-VL
        ):
            batch_size = batch_size or input_ids.size(0)
            all_logps, all_entropies = [], []
            for start in range(0, input_ids.size(0), batch_size):
                input_ids_batch = input_ids[start:start + batch_size]
                attention_mask_batch = attention_mask[start:start + batch_size]
                model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

                if image_grid_thw is not None and pixel_values is not None:
                    rows_per_image = image_grid_thw.prod(dim=-1)
                    rows_per_sample = torch.split(rows_per_image, num_images)
                    rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                    cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                    row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                    model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                    cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                    img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                    model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
                elif spatial_shapes is not None and pixel_values is not None and num_images is not None:
                    # LFM2.5-VL: slice pixel_values + spatial_shapes by image count
                    cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                    img_start, img_end = cum_imgs[start].item(), cum_imgs[start + batch_size].item()
                    model_inputs["pixel_values"] = pixel_values[img_start:img_end]
                    model_inputs["spatial_shapes"] = spatial_shapes[img_start:img_end]
                elif image_position_ids is not None and pixel_values is not None:
                    cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                    img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                    model_inputs["pixel_values"] = pixel_values[img_start:img_end]
                    model_inputs["image_position_ids"] = image_position_ids[img_start:img_end]
                elif pixel_values is not None:
                    model_inputs["pixel_values"] = pixel_values[start:start + batch_size]
                if pixel_attention_mask is not None:
                    model_inputs["pixel_attention_mask"] = pixel_attention_mask[start:start + batch_size]
                if image_sizes is not None:
                    model_inputs["image_sizes"] = image_sizes[start:start + batch_size]
                if token_type_ids is not None:
                    model_inputs["token_type_ids"] = token_type_ids[start:start + batch_size]
                if mm_token_type_ids is not None:
                    model_inputs["mm_token_type_ids"] = mm_token_type_ids[start:start + batch_size]

                if "logits_to_keep" in self.model_kwarg_keys:
                    model_inputs["logits_to_keep"] = logits_to_keep + 1
                model_inputs["use_cache"] = False

                logits = model(**model_inputs).logits
                logits = logits[:, :-1, :]
                logits = logits[:, -logits_to_keep:, :]
                logits.div_(self.temperature)
                completion_ids = input_ids_batch[:, -logits_to_keep:]
                logps = selective_log_softmax(logits, completion_ids)
                all_logps.append(logps)
                if compute_entropy:
                    with torch.no_grad():
                        entropies = entropy_from_logits(logits)
                    all_entropies.append(entropies)

            logps = torch.cat(all_logps, dim=0)
            entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
            return logps, entropies

        @profiling_decorator
        def _get_last_hidden_state(
            self, unwrapped_model, input_ids, attention_mask, logits_to_keep,
            pixel_values=None, image_grid_thw=None,
            pixel_attention_mask=None, image_sizes=None, image_position_ids=None,
            spatial_shapes=None,  # added for LFM2.5-VL
        ):
            if is_peft_model(unwrapped_model):
                unwrapped_model = unwrapped_model.base_model.model
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if image_grid_thw is not None and pixel_values is not None:
                model_inputs["image_grid_thw"] = image_grid_thw
            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes
            if image_position_ids is not None:
                model_inputs["image_position_ids"] = image_position_ids
            if spatial_shapes is not None:
                model_inputs["spatial_shapes"] = spatial_shapes
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            model_inputs["use_cache"] = False
            outputs = unwrapped_model(**model_inputs, output_hidden_states=True)
            return outputs.hidden_states[-1]
    # --- end LFM2.5-VL compat patch -------------------------------------

    ckpt_path = Path(MODAL_MOUNT_POINT) / V3_CHECKPOINT
    train_path = Path(MODAL_MOUNT_POINT) / TRAIN_JSONL
    image_root = Path(MODAL_MOUNT_POINT) / IMAGE_ROOT
    output_dir = Path(MODAL_MOUNT_POINT) / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading v3 merged checkpoint: {ckpt_path}")
    processor = AutoProcessor.from_pretrained(
        str(ckpt_path), trust_remote_code=True,
        truncation_side="left", padding_side="left",
    )
    # attn_implementation="eager" avoids a known cuDNN SDPA graph error
    # in SigLIP2's attention path that triggers with GRPO's variable
    # batch shapes (K=4 sampled completions per prompt).
    model = AutoModelForImageTextToText.from_pretrained(
        str(ckpt_path), torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    )

    # New LoRA on top of v3 merged so we don't catastrophically forget v3's prior.
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    print("Loading train data + images")
    train_rows = [json.loads(l) for l in train_path.read_text().splitlines() if l.strip()]
    print(f"  {len(train_rows)} train rows")

    def row_to_grpo(row: dict) -> dict:
        # row: {"messages": [system, user (image,image,text), assistant {"action":"..."}]}
        user_content = row["messages"][1]["content"]
        image_paths = [c["image"] for c in user_content if c["type"] == "image"]
        user_text = next(c["text"] for c in user_content if c["type"] == "text")
        system_text = row["messages"][0]["content"][0]["text"]
        gold_action = json.loads(row["messages"][2]["content"][0]["text"])["action"]
        return {
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [
                    {"type": "image"}, {"type": "image"},
                    {"type": "text", "text": user_text},
                ]},
            ],
            "images": [
                Image.open(image_root / image_paths[0]).convert("RGB"),
                Image.open(image_root / image_paths[1]).convert("RGB"),
            ],
            "gold_action": gold_action,
        }

    print("Building HF Dataset (will preload images into memory)")
    grpo_rows = [row_to_grpo(r) for r in train_rows]
    train_dataset = Dataset.from_list(grpo_rows)
    print(f"  Dataset built, {len(train_dataset)} rows")

    def reward_action_match(prompts, completions, **kwargs) -> list[float]:
        """1.0 if generated action equals gold, 0.0 otherwise.

        kwargs contains the dataset's `gold_action` column at the same indices.
        """
        gold = kwargs.get("gold_action", [])
        rewards = []
        for completion, g in zip(completions, gold):
            # completions in conversational format are list of messages; in
            # standard format they're strings. Handle both.
            if isinstance(completion, list):
                text = completion[-1]["content"] if completion else ""
                if isinstance(text, list):
                    text = next((c["text"] for c in text if c.get("type") == "text"), "")
            else:
                text = completion
            try:
                start = text.find("{")
                end = text.rfind("}")
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end + 1])
                    pred = parsed.get("action", "")
                    rewards.append(1.0 if pred == g else 0.0)
                else:
                    rewards.append(0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_completion_length=32,  # action JSON is ~15-20 tokens
        num_generations=4,         # group size for GRPO
        beta=0.04,                 # KL coefficient
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )

    print("Building GRPOTrainerLfm2Vl (compat-patched for spatial_shapes)")
    trainer = GRPOTrainerLfm2Vl(
        model=model,
        reward_funcs=[reward_action_match],
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=lora_config,
    )

    print(f"Starting GRPO training (epochs={grpo_config.num_train_epochs})")
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start

    print(f"\nGRPO training complete in {elapsed:.1f}s")
    print(f"Final train metrics: {train_result.metrics}")

    print(f"Saving final model to {output_dir / 'final'}")
    trainer.save_model(str(output_dir / "final"))

    return {
        "elapsed_sec": elapsed,
        "metrics": {k: float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in train_result.metrics.items()},
    }


@app.local_entrypoint()
def main() -> None:
    result = train.remote()
    print(f"\nFinal: GRPO complete in {result['elapsed_sec']:.1f}s")
    print(f"Metrics: {result['metrics']}")
