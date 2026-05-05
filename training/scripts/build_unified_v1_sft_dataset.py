"""Convert inline-labeled unified_v1/labels.jsonl into v9-format SFT JSONLs.

Reads the 190 production labels we generated via inline Claude Code labeling,
emits two leap-finetune-compatible JSONL files (train + eval) and stages the
RGB+SWIR PNGs under a flat `images/` directory so the resulting layout is the
same shape leap-finetune expects:

    training/data/unified_v1/
        galamsey_unified_v1_train.jsonl
        galamsey_unified_v1_eval.jsonl
        images/
            u0000/rgb.png
            u0000/swir.png
            ...
            u0189/rgb.png
            u0189/swir.png

Each JSONL row is a HF-VLM-SFT message conversation:
    {"messages": [
        {"role": "system", "content": [{"type": "text", "text": LABELING_SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "image", "image": "u0000/rgb.png"},
                                        {"type": "image", "image": "u0000/swir.png"},
                                        {"type": "text",  "text":  "<context block>"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "{...JSON action...}"}]}
    ]}

The split is stratified by `label.action` so each rare class (`request_hires`,
`request_neighbor`) lands in both splits when at least 2 examples exist.
Deterministic given seed=42.

Usage:
    cd training && uv run python scripts/build_unified_v1_sft_dataset.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.prepare_unified_decision_dataset_modal import (  # noqa: E402
    LABELING_SYSTEM_PROMPT,
    PerPassContext,
    TileCoord,
    build_user_message,
)

LABELS_JSONL = ROOT / "data" / "unified_v1" / "labels.jsonl"
CACHE_DIR = ROOT / "data" / "unified_v1_cache"
OUT_DIR = ROOT / "data" / "unified_v1"
OUT_IMAGES_DIR = OUT_DIR / "images"
OUT_TRAIN = OUT_DIR / "galamsey_unified_v1_train.jsonl"
OUT_EVAL = OUT_DIR / "galamsey_unified_v1_eval.jsonl"

# Map our orchestrator action literals back to the labeling prompt's tool names
# so the assistant target matches the system prompt's vocabulary.
_ACTION_TO_TOOL_NAME: dict[str, str] = {
    "discard": "discard",
    "flag": "flag_for_review",
    "request_hires": "request_higher_resolution",
    "request_neighbor": "request_neighbor_tile",
    "downlink": "downlink_now",
}

EVAL_FRACTION = 0.20  # 152 train / 38 eval at n=190
SEED = 42


def to_messages_record(row: dict) -> dict:
    """One labels.jsonl row -> one v9-format messages record."""
    coord = TileCoord(
        coord_id=row["coord_id"],
        lon=row["lon"],
        lat=row["lat"],
        stratum=row["stratum"],
        mission_priors=row["context"]["mission_priors"],
    )
    context = PerPassContext(**row["context"])
    user_text = build_user_message(coord, context)

    label = row["label"]
    tool_name = _ACTION_TO_TOOL_NAME[label["action"]]
    assistant_payload: dict = {"action": tool_name, "reason": label["reason"]}
    if label.get("direction"):
        assistant_payload["direction"] = label["direction"]
    assistant_text = json.dumps(assistant_payload)

    rgb_rel = f"{coord.coord_id}/rgb.png"
    swir_rel = f"{coord.coord_id}/swir.png"

    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": LABELING_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": rgb_rel},
                {"type": "image", "image": swir_rel},
                {"type": "text", "text": user_text},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ],
    }


def stage_image(coord_id: str) -> None:
    """Copy cached PNGs into images/<coord_id>/ under OUT_DIR."""
    src_rgb = CACHE_DIR / coord_id / "rgb.png"
    src_swir = CACHE_DIR / coord_id / "swir.png"
    dst_dir = OUT_IMAGES_DIR / coord_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_rgb, dst_dir / "rgb.png")
    shutil.copy2(src_swir, dst_dir / "swir.png")


def stratified_split(rows: list[dict], eval_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split rows into (train, eval), stratified by `label.action`.

    For classes with only 1 example, that example goes to train (eval needs
    ≥1 example per class only if we want per-class eval, otherwise train).
    Deterministic given seed.
    """
    rng = random.Random(seed)
    by_action: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_action[row["label"]["action"]].append(row)

    train: list[dict] = []
    eval_: list[dict] = []
    for action, group in by_action.items():
        rng.shuffle(group)
        n_eval = max(1, round(len(group) * eval_frac)) if len(group) >= 2 else 0
        eval_.extend(group[:n_eval])
        train.extend(group[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def main() -> None:
    assert LABELS_JSONL.exists(), f"missing {LABELS_JSONL}"
    rows = [json.loads(l) for l in LABELS_JSONL.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} labeled rows from {LABELS_JSONL.relative_to(ROOT)}")

    train_rows, eval_rows = stratified_split(rows, EVAL_FRACTION, SEED)
    print(f"Stratified split (seed={SEED}, eval_frac={EVAL_FRACTION}): "
          f"{len(train_rows)} train / {len(eval_rows)} eval")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Staging images -> {OUT_IMAGES_DIR.relative_to(ROOT)}/")
    for row in rows:
        stage_image(row["coord_id"])

    print(f"Writing {OUT_TRAIN.relative_to(ROOT)}")
    with OUT_TRAIN.open("w") as f:
        for row in train_rows:
            f.write(json.dumps(to_messages_record(row)) + "\n")

    print(f"Writing {OUT_EVAL.relative_to(ROOT)}")
    with OUT_EVAL.open("w") as f:
        for row in eval_rows:
            f.write(json.dumps(to_messages_record(row)) + "\n")

    print("\nClass distribution:")
    for split_name, split_rows in [("train", train_rows), ("eval", eval_rows)]:
        actions: dict[str, int] = defaultdict(int)
        for r in split_rows:
            actions[r["label"]["action"]] += 1
        print(f"  {split_name:5s}: " + ", ".join(f"{a}={n}" for a, n in sorted(actions.items())))

    print("\nNext step:")
    print("  modal volume put galamsey data/unified_v1 /data/unified_v1")
    print("  uv run leap-finetune configs/galamsey_unified_v1_modal.yaml")


if __name__ == "__main__":
    main()
