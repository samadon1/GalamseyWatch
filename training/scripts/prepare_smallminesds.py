"""
SmallMinesDS -> leap-finetune VLM SFT format converter.

Downloads SmallMinesDS from Hugging Face and converts each (patch, mask) pair
into the JSONL conversation format consumed by leap-finetune's `vlm_sft`
training type. Mirrors the structure of Liquid's `prepare_vrsbench.py`.

All preprocessing primitives live in the `galamseywatch` package so notebooks
and this script share a single implementation. When a notebook surfaces a fix,
the fix lands in the package and this script picks it up automatically.

Usage:
    # Local dry run on a small subset (no Modal credit burned):
    uv run python scripts/prepare_smallminesds.py --task description --limit 50

    # Full prep on Modal volume:
    uv run python scripts/prepare_smallminesds.py --task all --modal

STATUS: main conversion loop depends on SmallMinesDS feature-schema details
that are resolved in `notebooks/01_dataset_schema.ipynb`. Do not run
end-to-end before that notebook has been completed, the core loop raises
NotImplementedError until the dataset key names are confirmed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from galamseywatch import (
    BAND_NIR,
    BAND_SWIR1,
    BAND_SWIR2,
    DESCRIPTION_PROMPT,
    GROUNDING_PROMPT,
    HF_DATASET_ID,
    MODAL_DATA_DIR,
    MODAL_MOUNT_POINT,
    MODAL_VOLUME_NAME,
    analyze_mask,
    compose_swir_false_color,
    encode_png,
    generate_description,
    make_vlm_message,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Main conversion loop
# ---------------------------------------------------------------------------


def convert_smallminesds(
    output_dir: Path,
    task: str,
    limit: int | None,
) -> None:
    """Load SmallMinesDS from HF and convert to JSONL + PNG.

    Writes:
      - <output_dir>/images/<split>_<idx>.png                        (composite PNGs)
      - <output_dir>/galamsey_description_train.jsonl                (if task in {description, all})
      - <output_dir>/galamsey_description_eval.jsonl
      - <output_dir>/galamsey_grounding_train.jsonl                  (if task in {grounding, all})
      - <output_dir>/galamsey_grounding_eval.jsonl
    """
    if any(idx is None for idx in (BAND_SWIR2, BAND_SWIR1, BAND_NIR)):
        raise NotImplementedError(
            "Band indices are not yet resolved. Run notebooks/01_dataset_schema.ipynb "
            "first and fill in galamseywatch/constants.py with integer values for "
            "BAND_SWIR2, BAND_SWIR1, BAND_NIR before running this script."
        )

    from datasets import load_dataset  # local import, keeps --help fast
    import numpy as np

    ds = load_dataset(HF_DATASET_ID)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    description_train: list[dict] = []
    description_eval: list[dict] = []
    grounding_train: list[dict] = []
    grounding_eval: list[dict] = []

    for split_name, split in ds.items():
        target_desc = description_eval if "test" in split_name else description_train
        target_grd = grounding_eval if "test" in split_name else grounding_train

        for i, sample in enumerate(split):
            if limit is not None and i >= limit:
                break

            # TODO: replace hard-coded key names ("image", "mask") with the
            # real SmallMinesDS keys as resolved in notebook 01. These are the
            # placeholders used throughout the notebooks, keep them in sync.
            bands = np.asarray(sample["image"])
            if bands.ndim == 3 and bands.shape[0] > bands.shape[-1]:
                bands = np.transpose(bands, (2, 0, 1))
            mask = np.asarray(sample["mask"])

            composite = compose_swir_false_color(
                bands, BAND_SWIR2, BAND_SWIR1, BAND_NIR
            )
            png_filename = f"{split_name}_{i:06d}.png"
            encode_png(composite, images_dir / png_filename)

            stats = analyze_mask(mask)
            description = generate_description(stats)

            if task in ("description", "all"):
                target_desc.append(
                    make_vlm_message(png_filename, DESCRIPTION_PROMPT, description)
                )

            if task in ("grounding", "all"):
                payload = [
                    {"label": "mining_pit", "bbox": bbox}
                    for bbox in stats["bboxes_normalized"]
                ]
                target_grd.append(
                    make_vlm_message(png_filename, GROUNDING_PROMPT, json.dumps(payload))
                )

    # Shuffle training sets deterministically (eval sets stay in dataset order)
    rng = random.Random(42)
    rng.shuffle(description_train)
    rng.shuffle(grounding_train)

    if task in ("description", "all"):
        write_jsonl(description_train, output_dir / "galamsey_description_train.jsonl")
        write_jsonl(description_eval, output_dir / "galamsey_description_eval.jsonl")

    if task in ("grounding", "all"):
        write_jsonl(grounding_train, output_dir / "galamsey_grounding_train.jsonl")
        write_jsonl(grounding_eval, output_dir / "galamsey_grounding_eval.jsonl")

    print(
        f"Done. description train/eval: {len(description_train)}/{len(description_eval)}  "
        f"grounding train/eval: {len(grounding_train)}/{len(grounding_eval)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SmallMinesDS to leap-finetune VLM SFT format"
    )
    parser.add_argument(
        "--task",
        choices=["description", "grounding", "all"],
        required=True,
        help="Task to convert. description=free-text galamsey description, "
        "grounding=JSON bbox detection, all=both.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data/smallminesds",
        help="Directory to write output JSONL/PNG (default: ./data/smallminesds)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per split (useful for local dry runs)",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help=(
            f"Run data preparation on Modal (serverless cloud). "
            f"Writes output to the Modal volume '{MODAL_VOLUME_NAME}' at {MODAL_MOUNT_POINT}/."
        ),
    )
    args = parser.parse_args()

    if args.modal:
        _run_on_modal(args)
        return

    output_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    convert_smallminesds(output_dir, args.task, args.limit)


def _run_on_modal(args: argparse.Namespace) -> None:
    """Run the data preparation pipeline on Modal (mirrors prepare_vrsbench.py)."""
    import modal

    app = modal.App("galamsey-data-prep")
    volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "datasets>=3.0",
            "rasterio>=1.3",
            "numpy>=2.0",
            "scipy>=1.13",
            "pillow>=11.0",
            "huggingface_hub",
            "tqdm",
        )
        .add_local_dir("galamseywatch", "/app/galamseywatch", copy=True)
        .add_local_file(__file__, "/app/prepare_smallminesds.py", copy=True)
    )

    @app.function(
        image=image,
        volumes={MODAL_MOUNT_POINT: volume},
        timeout=3600,
        serialized=True,
    )
    def prepare(task: str, limit: int | None) -> None:
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "/app/prepare_smallminesds.py",
            "--task", task,
            "--data-dir", MODAL_DATA_DIR,
        ]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        subprocess.run(cmd, check=True, cwd="/app")
        volume.commit()

    print(f"Preparing SmallMinesDS on Modal (volume: '{MODAL_VOLUME_NAME}')...")
    with modal.enable_output():
        with app.run():
            prepare.remote(args.task, args.limit)

    print(f"\nData ready in Modal volume '{MODAL_VOLUME_NAME}'.")
    print("Next step: uv run leap-finetune ../configs/galamsey_v1_modal.yaml")


if __name__ == "__main__":
    main()
