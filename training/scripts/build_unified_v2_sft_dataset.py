"""Build unified v2 SFT dataset: action-only target + class-balanced oversampling.

v1 produced a model that collapsed to {discard, flag_for_review, unparseable}
with 38.5% accuracy on the held-out 39 (below the 56% always-discard baseline).
The two structural causes:
  1. Loss diluted across ~150 tokens of free-form `reason` text — the action
     token was a tiny fraction of the gradient signal.
  2. Class imbalance (87 discard / 35 flag / 27 downlink / 2 hires / 0 neighbor)
     made "always discard" a strong local minimum.

v2 changes:
  - Assistant target is just `{"action": "<action>"}` — no reason. Concentrates
    100% of loss on the prediction we actually care about.
  - Oversample rare classes so each class is closer to balanced. Eval set is
    untouched (same 39 held-out examples) so accuracy comparisons stay valid.

Output:
    training/data/unified_v2/
        galamsey_unified_v2_train.jsonl
        galamsey_unified_v2_eval.jsonl
        images/  (symlinks/copies pointing to the same PNGs as v1)

Usage:
    cd training && uv run python scripts/build_unified_v2_sft_dataset.py
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
OUT_DIR = ROOT / "data" / "unified_v2"
OUT_IMAGES_DIR = OUT_DIR / "images"
OUT_TRAIN = OUT_DIR / "galamsey_unified_v2_train.jsonl"
OUT_EVAL = OUT_DIR / "galamsey_unified_v2_eval.jsonl"

_ACTION_TO_TOOL_NAME: dict[str, str] = {
    "discard": "discard",
    "flag": "flag_for_review",
    "request_hires": "request_higher_resolution",
    "request_neighbor": "request_neighbor_tile",
    "downlink": "downlink_now",
}

EVAL_FRACTION = 0.20
SEED = 42

# Target per-class count after oversampling. Set to discard's natural count
# so we don't have to discard any examples; rare classes are repeated.
TARGET_PER_CLASS = 80


def to_messages_record(row: dict) -> dict:
    """One labels.jsonl row -> v2-format messages record (action-only target)."""
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
    assistant_payload: dict = {"action": tool_name}
    if label.get("direction"):
        assistant_payload["direction"] = label["direction"]
    assistant_text = json.dumps(assistant_payload)  # short — just action+direction

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


def stratified_split(rows: list[dict], eval_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
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


def oversample(train_rows: list[dict], target: int, seed: int) -> list[dict]:
    """Oversample rare classes to bring each up to (close to) `target` count."""
    rng = random.Random(seed + 1)
    by_action: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows:
        by_action[row["label"]["action"]].append(row)

    out: list[dict] = []
    for action, group in by_action.items():
        if len(group) >= target:
            out.extend(group)
            continue
        # Repeat full group as many times as needed, then sample remainder
        n_full_copies, remainder = divmod(target, len(group))
        repeated = group * n_full_copies + rng.sample(group, remainder)
        out.extend(repeated)
        print(f"  oversample {action}: {len(group)} -> {len(repeated)}")

    rng.shuffle(out)
    return out


def stage_image(coord_id: str) -> None:
    src_rgb = CACHE_DIR / coord_id / "rgb.png"
    src_swir = CACHE_DIR / coord_id / "swir.png"
    dst_dir = OUT_IMAGES_DIR / coord_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_rgb, dst_dir / "rgb.png")
    shutil.copy2(src_swir, dst_dir / "swir.png")


def main() -> None:
    rows = [json.loads(l) for l in LABELS_JSONL.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} labeled rows")

    train_rows, eval_rows = stratified_split(rows, EVAL_FRACTION, SEED)
    print(f"Stratified split: {len(train_rows)} train / {len(eval_rows)} eval")
    print("\nPre-oversampling train distribution:")
    actions: dict[str, int] = defaultdict(int)
    for r in train_rows:
        actions[r["label"]["action"]] += 1
    for a, n in sorted(actions.items()):
        print(f"  {a}: {n}")

    print(f"\nOversampling to target ~{TARGET_PER_CLASS} per class:")
    train_rows_balanced = oversample(train_rows, TARGET_PER_CLASS, SEED)

    print(f"\nPost-oversampling train: {len(train_rows_balanced)}")
    actions = defaultdict(int)
    for r in train_rows_balanced:
        actions[r["label"]["action"]] += 1
    for a, n in sorted(actions.items()):
        print(f"  {a}: {n}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nStaging images -> {OUT_IMAGES_DIR.relative_to(ROOT)}/")
    for row in rows:
        stage_image(row["coord_id"])

    print(f"Writing {OUT_TRAIN.relative_to(ROOT)}")
    with OUT_TRAIN.open("w") as f:
        for row in train_rows_balanced:
            f.write(json.dumps(to_messages_record(row)) + "\n")

    print(f"Writing {OUT_EVAL.relative_to(ROOT)}")
    with OUT_EVAL.open("w") as f:
        for row in eval_rows:
            f.write(json.dumps(to_messages_record(row)) + "\n")

    print(f"\nEval distribution (untouched):")
    actions = defaultdict(int)
    for r in eval_rows:
        actions[r["label"]["action"]] += 1
    for a, n in sorted(actions.items()):
        print(f"  {a}: {n}")

    print("\nNext step:")
    print("  modal volume put galamsey data/unified_v2 /data/unified_v2")
    print("  uv run leap-finetune configs/galamsey_unified_v2_modal.yaml")


if __name__ == "__main__":
    main()
