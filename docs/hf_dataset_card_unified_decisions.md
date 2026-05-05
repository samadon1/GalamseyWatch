---
license: cc-by-sa-4.0
language:
  - en
tags:
  - earth-observation
  - sentinel-2
  - galamsey
  - illegal-mining
  - agentic-eo
  - on-orbit-policy
  - tool-calling
  - decision-making
size_categories:
  - n<1K
task_categories:
  - image-text-to-text
pretty_name: GalamseyWatch unified-decisions corpus
---

# galamsey-unified-decisions

Hand-labeled corpus of 250 Sentinel-2 tiles over Ghana, each annotated with one of five **on-orbit satellite-tasking actions** plus a per-pass scalar context block (downlink budget, mission priors, structured neighbor summary). Used to train [`samwell/galamsey-unified-v3`](https://huggingface.co/samwell/galamsey-unified-v3), a 450M unified vision-language model that picks bandwidth-aware actions per tile.

The corpus is the data half of the [GalamseyWatch](https://github.com/samadon1/GalamseyWatch) project - an agentic Earth-observation system for detecting illegal small-scale gold mining ("galamsey") in southwestern Ghana.

## What's in here

Each row in `labels.jsonl` corresponds to one Sentinel-2 patch (1.28 km × 1.28 km, 10 m/pixel) over a hand-curated AOI in Ghana, with:

- **Two image composites** in `images/<coord_id>/`:
  - `rgb.png` - natural-color RGB (bands B4 + B3 + B2)
  - `swir.png` - SWIR false-color (bands B12 + B11 + B8); bright SWIR indicates exposed soil and mining disturbance
- **Per-tile metadata:** `coord_id`, `lon`, `lat`, `stratum`, `mission_priors` (AOI-specific operational note)
- **Per-pass scalar context:** `budget_remaining_kb`, `budget_total_kb`, `prior_tiles_downlinked`, `cloud_cover`, `captured_at`, `tile_imagery_issue`, and a `neighbor_summary` JSON object describing what each adjacent tile decided this pass
- **A target action** (the label): one of five tool names plus an optional reason

### Action vocabulary

| Action | Meaning |
|---|---|
| `discard` | skip the tile entirely (forest, water, urban, cloud, no signal) |
| `flag_for_review` | log as text only - no image downlink, cheap follow-up |
| `request_higher_resolution` | request a higher-res recapture next pass |
| `request_neighbor_tile` | fetch the adjacent tile in a given direction |
| `downlink_now` | spend the downlink budget to send this tile to ground |

### Class distribution (250 labeled rows)

| Action | Count | % |
|---:|---:|---:|
| `discard` | 146 | 58 % |
| `flag_for_review` | 53 | 21 % |
| `downlink_now` | 47 | 19 % |
| `request_higher_resolution` | 4 | 2 % |
| `request_neighbor_tile` | 0 | 0 % |

Discard dominates because Ghana's actual surface composition is mostly forest, water, savanna, urban, and farmland - the deterministic sampler reflects that base rate. The `request_neighbor_tile` class is structurally hard to elicit from naturalistic Sentinel-2 imagery and is absent from the labels (open follow-up).

## How it was built

**Sampling.** A deterministic stratified sampler (seed = 42) over 15 hand-curated AOIs across Ghana spanning all relevant strata:

- *Mining hotspots:* Bibiani, Pra basin (Bogoso), Ankobra basin (Prestea), Obuasi, Asutifi
- *Forest reserves:* Atewa, Kakum, Bia
- *Water:* Lake Bosumtwi (crater lake), Lake Volta
- *Urban:* Accra, Kumasi
- *Agricultural mosaic:* Northern savanna, central farmland (cocoa belt)
- *Cloud-prone edge:* Coastal Axim

The sampler picks a stratum per fixed weights, an AOI within the stratum, and adds Gaussian jitter within each AOI's radius. Reproducible end-to-end via [`training/scripts/fetch_unified_corpus.py`](https://github.com/samadon1/GalamseyWatch/blob/main/training/scripts/fetch_unified_corpus.py).

**Imagery fetch.** RGB + SWIR composites pulled from the [DPhi SimSat](https://github.com/DPhi-Space/SimSat) simulator (Sentinel-2 imagery served as a satellite-pass simulator); sequential to be polite to the upstream service.

**Labeling.** Hand-labeled over multiple sessions, following the labeling protocol locked in three validation rounds (documented in [`docs/UNIFIED_VLM_VALIDATION.md`](https://github.com/samadon1/GalamseyWatch/blob/main/docs/UNIFIED_VLM_VALIDATION.md)). For each tile the labeler reads both image composites plus the scalar context, applies the disambiguation rules from the labeling system prompt (e.g. *"SWIR brightness in the absence of exposed-soil patterns is more likely infrastructure, not mining"*, *"rectilinear field patterns are agriculture, not mining"*), and emits the action with a reason string.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("samwell/galamsey-unified-decisions", split="train")
row = ds[0]
print(row["coord_id"], row["label"]["action"], row["context"]["budget_remaining_kb"])
# row["rgb"] and row["swir"] are PIL Images
```

For the train/eval splits used in [`samwell/galamsey-unified-v3`](https://huggingface.co/samwell/galamsey-unified-v3) training (151 train, oversampled to 327, plus 99 held-out eval), see [`training/scripts/build_unified_v2_sft_dataset.py`](https://github.com/samadon1/GalamseyWatch/blob/main/training/scripts/build_unified_v2_sft_dataset.py) and [`training/scripts/build_expanded_eval_dataset.py`](https://github.com/samadon1/GalamseyWatch/blob/main/training/scripts/build_expanded_eval_dataset.py) in the GalamseyWatch repo.

## Recurring data anomalies

Two SimSat-side defects are documented in the labels (`tile_imagery_issue` field is `null` in the JSONL because the synthesis pipeline didn't catch them - the labeler-side reasoning string flags them):

- **`cloud_cover = 1.403499`** appears on every Coastal Axim tile in the corpus (18 tiles total). The expected range is `[0, 1]`; the bit-identical out-of-range value across all 18 tiles indicates a deterministic upstream pipeline bug specific to this AOI's scene definition. **Mitigation:** clamp `cloud_cover` to `[0, 1]` before consuming the field.
- **Partial-tile black-bottom imagery** appears on 5 Kakum tiles (u0126, u0148, u0192, u0212, u0235). Bottom 40-60 % of the tile is solid black; top portion is valid forest. Also AOI-specific. **Mitigation:** detect via pixel statistics (fraction of `RGB = 0,0,0`) and treat as a `tile_imagery_issue`.

Both are useful as out-of-distribution / data-quality test cases for downstream consumers.

## Limitations

- **Same-distribution.** The 99-tile held-out eval is drawn from the same 15 AOIs as the train set, just at indices the train set never saw. Robustness across the rest of Ghana is not proven.
- **Ghana-specific.** All AOIs are in Ghana; geographies, land cover, and mining patterns elsewhere may differ.
- **Class imbalance.** Two of the five classes (`request_higher_resolution`, `request_neighbor_tile`) are essentially absent (4 / 0 examples respectively). Models trained on this corpus will not learn those actions without additional hand-constructed examples.
- **Single labeler.** Each labeling round was produced by a single labeler, not multi-rater. Inter-rater reliability is not measured.

## License and citation

Released under [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/), matching the upstream perception dataset ([SmallMinesDS](https://huggingface.co/datasets/ellaampy/SmallMinesDS), Ofori-Ampofo et al., 2025).

```bibtex
@misc{galamsey_unified_decisions_2026,
  author = {Donkor, Samuel},
  title = {galamsey-unified-decisions: hand-labeled Sentinel-2 tiles for on-orbit satellite-tasking decisions},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/samwell/galamsey-unified-decisions}
}
```

Upstream Sentinel-2 imagery is © European Union / Copernicus Programme. The SimSat simulator is © DPhi Space.
