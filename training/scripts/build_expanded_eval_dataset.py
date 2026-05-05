"""Build the 99-tile expanded eval JSONL.

Combines the original 39-tile v2 eval (u0000-u0189 stratified split) with
the 60 new eval tiles (u0190-u0249, all not in any training set) into a
single JSONL of 99 examples in the v2 action-only format. Same format as
build_unified_v2_sft_dataset.py emits, so the existing eval scripts work
unchanged — just point at the new file.

Output:
    training/data/unified_v2/galamsey_unified_v2_eval_expanded.jsonl
    training/data/unified_v2_eval_expanded/images/
        (symlinks/copies of cache PNGs for the new tiles)

Usage:
    cd training && uv run python scripts/build_expanded_eval_dataset.py
"""
from __future__ import annotations
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.prepare_unified_decision_dataset_modal import (  # noqa: E402
    LABELING_SYSTEM_PROMPT, PerPassContext, TileCoord, build_user_message,
)

LABELS_JSONL = ROOT / "data" / "unified_v1" / "labels.jsonl"
CACHE_DIR = ROOT / "data" / "unified_v1_cache"
EXISTING_EVAL = ROOT / "data" / "unified_v2" / "galamsey_unified_v2_eval.jsonl"
OUT_EVAL = ROOT / "data" / "unified_v2" / "galamsey_unified_v2_eval_expanded.jsonl"
OUT_IMAGES = ROOT / "data" / "unified_v2" / "images"  # already populated, will add new tiles

_ACTION_TO_TOOL_NAME = {
    "discard": "discard", "flag": "flag_for_review",
    "request_hires": "request_higher_resolution",
    "request_neighbor": "request_neighbor_tile", "downlink": "downlink_now",
}


def to_messages_record(row: dict) -> dict:
    coord = TileCoord(
        coord_id=row["coord_id"], lon=row["lon"], lat=row["lat"],
        stratum=row["stratum"], mission_priors=row["context"]["mission_priors"],
    )
    context = PerPassContext(**row["context"])
    user_text = build_user_message(coord, context)
    label = row["label"]
    tool_name = _ACTION_TO_TOOL_NAME[label["action"]]
    payload = {"action": tool_name}
    if label.get("direction"):
        payload["direction"] = label["direction"]
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
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(payload)}]},
        ],
    }


def stage_image(coord_id: str) -> None:
    src_rgb = CACHE_DIR / coord_id / "rgb.png"
    src_swir = CACHE_DIR / coord_id / "swir.png"
    dst_dir = OUT_IMAGES / coord_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not (dst_dir / "rgb.png").exists():
        shutil.copy2(src_rgb, dst_dir / "rgb.png")
    if not (dst_dir / "swir.png").exists():
        shutil.copy2(src_swir, dst_dir / "swir.png")


def main() -> None:
    rows = [json.loads(l) for l in LABELS_JSONL.read_text().splitlines() if l.strip()]
    new_rows = [r for r in rows if "u0190" <= r["coord_id"] <= "u0249"]
    print(f"New eval rows (u0190-u0249): {len(new_rows)}")

    # Stage images for new tiles
    print(f"Staging images -> {OUT_IMAGES.relative_to(ROOT)}/")
    for r in new_rows:
        stage_image(r["coord_id"])

    # Read existing eval
    existing = [json.loads(l) for l in EXISTING_EVAL.read_text().splitlines() if l.strip()]
    print(f"Existing v2 eval rows: {len(existing)}")

    new_records = [to_messages_record(r) for r in new_rows]

    # Concatenate; existing first, new second (deterministic ordering)
    combined = existing + new_records
    with OUT_EVAL.open("w") as f:
        for rec in combined:
            f.write(json.dumps(rec) + "\n")
    print(f"\nWrote {OUT_EVAL.relative_to(ROOT)} ({len(combined)} rows)")

    # Distribution
    print("\nClass distribution (expanded eval):")
    counts: dict[str, int] = defaultdict(int)
    for rec in combined:
        action = json.loads(rec["messages"][2]["content"][0]["text"])["action"]
        counts[action] += 1
    for a, n in sorted(counts.items()):
        print(f"  {a}: {n}")


if __name__ == "__main__":
    main()
