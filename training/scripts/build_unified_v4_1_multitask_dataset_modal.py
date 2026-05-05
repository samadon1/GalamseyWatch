"""Build unified v4 multitask SFT dataset (Modal-resident).

v3 trained an action-only LoRA on v9-e3 — that LoRA partially overwrote v9-e3's
grounding/description capability (confirmed empirically when extracting outputs
for the blog: v3 emits malformed JSON for grounding prompts and hallucinates
for description prompts).

v4 trains a fresh LoRA on the same v9-e3 base with a MULTITASK mixture so the
model preserves all three abilities in one weight set:
  - action target (decision policy)
  - grounding target (bounding boxes)
  - description target (free-form prose)

Mixture (kept small + balanced so the action signal isn't drowned):
  - 327 action rows (the entire v2 oversampled train set, unchanged)
  - 500 grounding rows (random subsample of v9's 23,864 grounding rows)
  - 500 description rows (random subsample of v9's 23,864 description rows)

Total: 1,327 mixed rows. Shuffled, written as a single JSONL.

Image root: a merged directory containing both the u0xxx subtree (action data)
and the v9_*.png flat files (perception data). The relative paths inside the
JSONL already match the layout from each source corpus so we just symlink.

Output (on the galamsey volume):
  /galamsey/data/unified_v4_1_multitask/
      galamsey_unified_v4_1_multitask_train.jsonl
      images/u0xxx/{rgb,swir}.png   (symlinks to data/unified_v2/images/...)
      images/v9_rgb_NNNNNN.png      (symlinks to data/v9/images/...)
      images/v9_swir_NNNNNN.png     (symlinks to data/v9/images/...)
"""
from __future__ import annotations
from pathlib import Path
import modal

MODAL_VOLUME_NAME = "galamsey"
MOUNT = "/galamsey"

# v4.1: rebalanced mixture. v4 had 500+500=1000 perception vs 327 action
# (action only 24.6% of training). v4 perception was preserved cleanly but
# action dropped 7.1pp vs v3. v4.1 keeps just enough perception to act as a
# regularizer on top of v9-e3's already-strong perception, lifts action share.
PERCEPTION_PER_TASK = 125
SEED = 42

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tqdm")
)
app = modal.App("galamsey-build-unified-v4-1-multitask")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=False)


@app.function(image=image, volumes={MOUNT: volume}, timeout=900)
def build() -> dict:
    import json, random, os
    from collections import defaultdict

    root = Path(MOUNT)
    v9_jsonl = root / "data/v9/galamsey_v9_multitask_train.jsonl"
    v9_images = root / "data/v9/images"
    action_jsonl = root / "data/unified_v2/galamsey_unified_v2_train.jsonl"
    action_images = root / "data/unified_v2/images"

    out_dir = root / "data/unified_v4_1_multitask"
    out_jsonl = out_dir / "galamsey_unified_v4_1_multitask_train.jsonl"
    out_images = out_dir / "images"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    # 1) Load + classify v9 multitask rows
    print(f"Reading {v9_jsonl}")
    grounding, description = [], []
    with v9_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            text = r["messages"][0]["content"][-1]["text"]
            if "Provide result as a valid JSON" in text or ("detect" in text.lower() and "bbox" in text.lower()):
                grounding.append(r)
            elif "describe" in text.lower():
                description.append(r)
    print(f"  v9 grounding={len(grounding)} description={len(description)}")

    rng = random.Random(SEED)
    grounding_sample = rng.sample(grounding, min(PERCEPTION_PER_TASK, len(grounding)))
    description_sample = rng.sample(description, min(PERCEPTION_PER_TASK, len(description)))
    print(f"  subsampled grounding={len(grounding_sample)} description={len(description_sample)}")

    # 2) Load action rows (already oversampled by v2 builder)
    print(f"Reading {action_jsonl}")
    action_rows = [json.loads(l) for l in action_jsonl.read_text().splitlines() if l.strip()]
    print(f"  action rows={len(action_rows)}")

    # 3) Combine + shuffle
    all_rows = grounding_sample + description_sample + action_rows
    rng.shuffle(all_rows)
    print(f"Total mixed rows: {len(all_rows)}")

    # 4) Write merged JSONL
    with out_jsonl.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {out_jsonl}")

    # 5) Build merged image_root via symlinks (cheap, no copies)
    #    Action data: u0xxx/{rgb,swir}.png — symlink each u0xxx dir
    #    Perception data: v9_{rgb,swir}_NNNNNN.png — symlink only the files
    #    referenced by the sampled rows (not all 47k worth of files)
    print(f"Setting up image_root at {out_images}")
    perception_files: set[str] = set()
    for r in grounding_sample + description_sample:
        for c in r["messages"][0]["content"]:
            if c.get("type") == "image":
                perception_files.add(c["image"])

    action_dirs: set[str] = set()
    for r in action_rows:
        for c in r["messages"][1]["content"] if r["messages"][0]["role"] == "system" else r["messages"][0]["content"]:
            if c.get("type") == "image":
                action_dirs.add(c["image"].split("/")[0])

    n_linked_action = 0
    for d in action_dirs:
        src = action_images / d
        dst = out_images / d
        if not dst.exists() and src.exists():
            os.symlink(src.resolve(), dst)
            n_linked_action += 1
    print(f"  linked {n_linked_action} action subdirs")

    n_linked_perception = 0
    for fn in perception_files:
        src = v9_images / fn
        dst = out_images / fn
        if not dst.exists() and src.exists():
            os.symlink(src.resolve(), dst)
            n_linked_perception += 1
    print(f"  linked {n_linked_perception} perception files")

    # Sanity check: count distribution
    by_task = defaultdict(int)
    for r in all_rows:
        msgs = r["messages"]
        text = msgs[1]["content"][-1]["text"] if msgs[0]["role"] == "system" else msgs[0]["content"][-1]["text"]
        target = msgs[-1]["content"][0]["text"]
        if target.startswith("{\"action\""):
            by_task["action"] += 1
        elif target.startswith("[") or "bbox" in target:
            by_task["grounding"] += 1
        else:
            by_task["description"] += 1

    print("\nFinal task distribution:")
    for k, v in sorted(by_task.items()):
        print(f"  {k}: {v}")

    volume.commit()
    return dict(by_task)


@app.local_entrypoint()
def main() -> None:
    out = build.remote()
    print(out)
