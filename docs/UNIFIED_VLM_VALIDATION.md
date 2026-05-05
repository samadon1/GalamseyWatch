# Unified VLM Plan — Phase A.1 Validation

**Date:** 2026-05-04 (post-hackathon-submission day).
**Goal:** validate the labeling prompt from `UNIFIED_VLM_PLAN.md` Section 2 on real Sentinel-2 tiles before scaling to 1000 examples.
**Adjudicator:** Claude (this session, inline reading of cached PNGs and metadata).
**Source tiles:** 6 cached tiles from `orchestrator/.benchmark_cache/` (Bibiani cluster, lat 5.63-5.65, lon -2.78 to -2.72), captured 2023-12-20 from a SimSat pass.

## Per-pass simulated state

Tiles labeled in t000→t005 order, single 512 KB pass, 80 KB per downlink, 0% cloud throughout. Captured neighbor / budget context evolves as the pass progresses.

## Tile-by-tile labels

### t000 (lon=−2.777, lat=5.63) — pass start, budget 512/512 KB

| | |
|---|---|
| **VLM perception** | 5 mining pits, "largest in lower-left", 12.46 hectares, conf 1.0 |
| **RGB** | dense canopy with curved chain of bright orange/tan exposed-soil patches running through the center, classic galamsey pit signature along a stream |
| **SWIR** | strong cyan/white patches confirming exposed soil and water-saturated mining disturbance |
| **Label** | `{"action": "downlink", "reason": "5+ visible active mining pits along a stream chain, strong SWIR exposed-soil signature, 12 hectares affected"}` |

### t001 (lon=−2.75, lat=5.63) — east of t000, budget 432 KB

| | |
|---|---|
| **VLM perception** | 1 box (description says 2), 8.12 hectares, conf 0.99 |
| **RGB** | scattered bright patches and channel features in lower portion, cleared areas continuing the t000 cluster eastward |
| **SWIR** | pronounced bright disturbance in lower-left, contiguous with t000 signature |
| **Label** | `{"action": "downlink", "reason": "Mining activity continues from t000 westward neighbor, lower-portion exposed soil and SWIR signature, ~8 hectares"}` |

### t002 (lon=−2.723, lat=5.63) — east of t001, budget 352 KB

| | |
|---|---|
| **VLM perception** | 6 boxes (description says 4), 6.61 hectares, conf 1.0 |
| **RGB** | multiple small dispersed bright orange patches across the tile, dispersed-pit galamsey pattern |
| **SWIR** | scattered bright spots matching RGB features |
| **Label** | `{"action": "downlink", "reason": "6 dispersed pits across the tile, SWIR confirms multiple active sites, distinct cluster from t000-t001 line"}` |

### t003 (lon=−2.777, lat=5.65) — north of t000, budget 272 KB

| | |
|---|---|
| **VLM perception** | 6 boxes ("7 mining pits"), 9.89 hectares, conf 1.0 |
| **RGB** | scattered exposed-soil patches throughout, continuing t000 cluster northward |
| **SWIR** | bright disturbance patches across the tile |
| **Label** | `{"action": "downlink", "reason": "Cluster continues from t000 to the north, 6+ pits visible, ~10 hectares; spatial extent of cluster justifies the bandwidth"}` |

### t004 (lon=−2.75, lat=5.65) — north of t001, budget 192 KB

| | |
|---|---|
| **VLM perception** | 0 boxes, "no mining activity visible", conf 0.0 |
| **RGB** | uniform dark green forest canopy, no bright patches |
| **SWIR** | uniform blue forest signature, no disturbance |
| **Label** | `{"action": "discard", "reason": "Uniform forest canopy in both RGB and SWIR, no mining activity, VLM correctly returned 0 boxes"}` |

### t005 (lon=−2.723, lat=5.65) — north of t002, east of t004, budget 192 KB

| | |
|---|---|
| **VLM perception** | 3 boxes (largest covers ~75% of tile), 7.51 hectares, conf 1.0 |
| **RGB** | branching stream network with extensive sediment exposure, heaviest visible activity in the pass |
| **SWIR** | continuous bright signal across most of the tile, classic riverbed-mining pattern |
| **Label** | `{"action": "downlink", "reason": "Extensive riverbed mining: branching stream network with major sediment exposure, continuous SWIR disturbance, possibly the heaviest activity site in the AOI"}` |

## End-of-pass summary

| Tile | Action | KB used |
|---|---|---:|
| t000 | downlink | 80 |
| t001 | downlink | 80 |
| t002 | downlink | 80 |
| t003 | downlink | 80 |
| t004 | discard | 0 |
| t005 | downlink | 80 |
| **Total** | **5 downlinks, 1 discard** | **400 / 512** |

**Match against LFM2-2.6B benchmark.** The earlier benchmark on these same 6 tiles produced 5 downlinks + 1 discard for 400 KB used (see `orchestrator/scripts/baseline_results.csv`, currently gitignored). Action-correctness rate vs adjudicator: **6/6 (100%)** on this sample. This is a positive sanity check that the labeling prompt produces decisions calibrated to the existing two-layer pipeline.

Caveat: 6 tiles, all from a known-active cluster, all under cloud cover 0%. This is **not** a generalization claim. It says the prompt doesn't fail catastrophically on the easy distribution; we need diverse-AOI follow-up before scaling to 1000.

## What worked in the labeling prompt

- **Five-tool space was sufficient** for this sample. Used `downlink` (5×) and `discard` (1×). The other three tools (`flag_for_review`, `request_higher_resolution`, `request_neighbor_tile`) weren't called but are available for ambiguous / partial-detection cases not present in this AOI.
- **Per-pass scalar context did real work.** t003 was on the edge ("budget allows" was the swing factor); without budget context the decision logic would have been the same but for different reasons. The context isn't decoration.
- **JSON output is unambiguous** and round-trips cleanly through the new `_parse_json_response` parser added in the Phase 0 refactor.
- **Both RGB and SWIR images contributed** to decisions, especially t005 where SWIR makes the riverbed pattern obvious in a way RGB alone wouldn't.

## Refinements made before scaling up

The four refinements identified during this exercise have been merged into `UNIFIED_VLM_PLAN.md` Section 2:

1. **Structure the neighbor summary** as an object (compass direction → `{action, boxes}`), not free-text. Reduces labeler variance.
2. **Clarify the `direction` enum** for `request_neighbor` in the labeling prompt template (`"north" | "south" | "east" | "west"` explicitly).
3. **Add a scene-type hint to neighbor context** ("cluster continuation" vs "isolated forest") so the model can use spatial coherence as a feature.
4. **Source examples across the full AOI distribution** including rural / forest / cloud-only tiles, not just hotspots. The 1000-example training corpus must NOT be skewed toward `downlink`, otherwise the unified VLM will overfit.

## Open question for Phase A.2 mini

Does the prompt hold up on edge-case tiles (forest, ocean, cloud-heavy, partial detections)? The 6-tile sample only had 1 negative example (t004 forest). Phase A.2 mini will sample 10 diverse tiles deliberately mixing terrain types and label inline before committing to the full pipeline.

---

# Phase A.2 Mini — 10 Diverse-Terrain Tiles (2026-05-04)

**Goal:** stress-test the labeling prompt against deliberately-mixed terrain (mining hotspots, pristine forest, water, urban, coastal) before scaling to a 1000-example training corpus.

**Source tiles:** 10 fetched from self-hosted Cloud Run SimSat into `orchestrator/.benchmark_cache/diverse_v1/d000-d009/` via `orchestrator/scripts/fetch_diverse_validation_tiles.py`. Sequential fetch (initial concurrent attempt timed out under cold-start scaling). Coordinates picked across Ghana to span: 2 mining hotspots (Pra basin, Ankobra basin), 1 threatened reserve (Atewa), 2 pristine forests (Kakum, Bia), 2 water bodies (Bosumtwi, Volta), 2 urban (Accra, Kumasi), 1 coastal (Axim).

## Per-pass simulated state

Tiles labeled in d000→d009 order, single 512 KB pass budget, 80 KB per downlink. `cloud_cover` reported as ≤0.012 across all tiles (questionable on d009, see Finding 4).

## Tile-by-tile labels

### d000 — Pra basin near Bogoso, "mining_pra_basin_bogoso"

| | |
|---|---|
| **RGB** | left half solid black (Sentinel-2 swath edge / no-data); right half dark forest with scattered tan disturbance |
| **SWIR** | same split, blue forest with cyan disturbance spots on right |
| **Edge case** | partial imagery — `image_available=true` but half the tile is no-data |
| **Label** | `{"action": "flag_for_review", "reason": "partial Sentinel-2 swath coverage, left half no-data; right portion shows possible disturbance but tile is unreliable for downlink"}` |

### d001 — Ankobra basin near Prestea, "mining_ankobra_basin_prestea"

| | |
|---|---|
| **RGB** | forest with scattered light brown/tan exposed-soil patches throughout, dispersed galamsey pattern |
| **SWIR** | bright cyan/yellow patches matching, strong galamsey signature |
| **Label** | `{"action": "downlink", "reason": "Dispersed active mining across the tile, exposed soil in RGB, strong SWIR signature"}` |

### d002 — Atewa Reserve, "forest_atewa_reserve_threatened"

| | |
|---|---|
| **RGB** | predominantly dark green forest, very small scattered orange/tan patches |
| **SWIR** | bright cyan disturbance scattered across, more pronounced than RGB suggests |
| **Mission context** | Atewa is famously threatened by galamsey encroachment |
| **Label** | `{"action": "downlink", "reason": "SWIR shows real disturbance even where RGB is subtle, consistent with documented galamsey encroachment in Atewa; high mission priority"}` |

### d003 — Kakum National Park, "forest_kakum_pristine"

| | |
|---|---|
| **RGB** | uniform dark green canopy with small clearing in upper-right (looks like a ranger station + access road) |
| **SWIR** | scattered brightness, but matches RGB infrastructure rather than indicating pit clusters |
| **Edge case** | infrastructure SWIR signature mimics mining brightness |
| **Label** | `{"action": "discard", "reason": "Protected forest, visible signature is park infrastructure (building/road), not mining; no exposed-soil chains or sediment plumes"}` |

### d004 — Bia National Park, "forest_bia_pristine"

| | |
|---|---|
| **RGB** | mostly dark canopy with one notable bright clearing in upper-left, lighter green patches scattered |
| **SWIR** | bright disturbance especially upper-left matching the clearing |
| **Edge case** | ambiguous-origin clearing in a buffer zone |
| **Label** | `{"action": "flag_for_review", "reason": "Visible clearing in upper-left of ambiguous origin (buffer-zone farming vs disturbance); worth logging but not bandwidth"}` |

### d005 — Lake Bosumtwi, "water_lake_bosumtwi"

| | |
|---|---|
| **RGB** | uniform near-black |
| **SWIR** | uniform dark blue (water signature) |
| **Label** | `{"action": "discard", "reason": "Lake Bosumtwi crater lake, uniform water in both RGB and SWIR, no mining possible"}` |

### d006 — Lake Volta region, "water_lake_volta"

| | |
|---|---|
| **RGB** | mixed land/water, brown patchwork agriculture on left/top, lake on right |
| **SWIR** | bright multicolored land portion, water gradient |
| **Edge case** | agricultural patchwork must be distinguished from mining |
| **Label** | `{"action": "discard", "reason": "Agricultural patchwork (rectilinear fields) plus lake; no exposed-soil pit pattern, agriculture's geometric signature is distinct from galamsey"}` |

### d007 — Accra, "urban_accra"

| | |
|---|---|
| **RGB** | dense rectilinear pattern of buildings, roads, small green patches |
| **SWIR** | very bright multicolored urban signature |
| **Edge case** | urban built environment SWIR is bright, looks superficially like mining |
| **Label** | `{"action": "discard", "reason": "Urban infrastructure, SWIR brightness is asphalt/building reflectance not exposed soil"}` |

### d008 — Kumasi, "urban_kumasi"

| | |
|---|---|
| **RGB** | similar urban mosaic, slightly different texture |
| **SWIR** | bright multicolored urban signature |
| **Label** | `{"action": "discard", "reason": "Urban Kumasi, bright SWIR is built environment not mining"}` |

### d009 — Coastal Axim, "coastal_axim_cloudprone"

| | |
|---|---|
| **RGB** | dark forest with bright white puffy clouds over ~30-40% of the tile (visually) |
| **SWIR** | bright cyan/white where clouds are, dark blue forest underneath |
| **Edge case** | metadata reports `cloud_cover=0.001` (0.1%) but visual cloud cover is clearly much higher |
| **Label** | `{"action": "discard", "reason": "Cloud-occluded ~30-40% of tile despite metadata reporting low cloud cover; visible non-cloud forest shows no mining"}` |

## End-of-pass summary

| Action | Count | KB used |
|---|---:|---:|
| downlink | 2 | 160 |
| flag_for_review | 2 | 0 |
| discard | 6 | 0 |
| **Total** | **10** | **160 / 512** |

## Class distribution achieved

| Tool | Bibiani (n=6) | Diverse (n=10) | Combined (n=16) | Plan target |
|---|---:|---:|---:|---:|
| downlink | 5 (83%) | 2 (20%) | 7 (44%) | ~30% |
| flag_for_review | 0 | 2 (20%) | 2 (12%) | ~20% |
| discard | 1 (17%) | 6 (60%) | 7 (44%) | ~25% |
| request_higher_resolution | 0 | 0 | 0 | ~15% |
| request_neighbor_tile | 0 | 0 | 0 | ~10% |

Diverse sample is much closer to a defensible class distribution than Bibiani alone, but **`request_higher_resolution` and `request_neighbor_tile` were not exercised on either sample.** The 1000-example training set must deliberately include tiles that should call these — small ambiguous candidates and edge-of-cluster tiles.

## Findings (merged into `UNIFIED_VLM_PLAN.md` Section 2 and Section 5)

**1. Partial swath tiles need explicit handling.** d000 returned `image_available=true` but with half the tile being no-data (Sentinel-2 swath edge artifact). The labeling prompt didn't anticipate this. *Resolution: extend the per-tile context with a `tile_imagery_issue` field; flag_for_review covers the action, but the labeler needs the signal that the tile is partially unreliable.*

**2. Infrastructure SWIR signature mimics mining.** d003 (Kakum) showed bright SWIR patches from the ranger station + access road that, taken naively, look like exposed-soil disturbance. *Resolution: add explicit guidance to the labeling prompt — "SWIR brightness in the absence of exposed-soil patterns or sediment plumes is more likely infrastructure (rooftops, asphalt, cleared paths), not mining."*

**3. Mission priors should be in the labeling context.** d002 (Atewa) was a downlink call partly because Atewa is documented as a galamsey encroachment frontier. The current prompt doesn't surface this kind of operational context. *Resolution: add a `mission_priors` field listing AOI-specific known sensitivities ("Atewa: known encroachment frontier"; "Kakum: pristine, low expected disturbance"). Reduces judgment-call variance.*

**4. SimSat cloud-cover metadata is unreliable on edge cases.** d009 reported `cloud_cover=0.001` but visually had ~30-40% cloud cover. *Resolution: either (a) add a visual-cloud-detection sanity check to the data pipeline, or (b) train the unified VLM to be robust to cloud cover regardless of the reported value. Preference for (b) — robustness is a real production constraint.*

**5. Two of five tools never exercised on 16 tiles.** `request_higher_resolution` and `request_neighbor_tile` are in the prompt but didn't get called once. *Resolution: when sampling the 1000-example training set, deliberately include (a) tiles with a single small candidate that should `request_higher_resolution`, and (b) tiles where mining clearly continues off-frame, to seed `request_neighbor_tile` calls. Otherwise the unified VLM will only learn 3 of 5 tools.*

## Verdict

The labeling prompt is solid for clear positives and clear negatives. The five edge cases above are real but addressable through the refinements proposed. With those merged, scale to 1000 is unblocked.

---

# Phase A.2 Mini Round 2 — Positive Controls for HR + Neighbor (2026-05-04)

**Goal:** the first 16 tiles never exercised `request_higher_resolution` (HR) or `request_neighbor_tile`. Engineer 3 specific Ghana coordinates designed to force these tools to fire, confirm the prompt elicits them. Without positive controls, the unified VLM will never learn 2 of 5 tools.

**Source tiles:** 3 fetched into `orchestrator/.benchmark_cache/diverse_v1/d010-d012/` via the same script (extended).

## Tile-by-tile labels

### d010 — west of Bibiani cluster, target: `request_neighbor_tile` east

| | |
|---|---|
| **Coordinate rationale** | lon=-2.81, lat=5.63; t000 (heaviest Bibiani cluster) at lon=-2.78. Hypothesis: cluster extends westward, this tile catches the western edge with the cluster cutting off east |
| **RGB** | scattered orange/tan exposed-soil patches concentrated in the lower-right quadrant, trailing toward the eastern boundary |
| **SWIR** | corresponding bright cyan/yellow patches in lower-right |
| **Outcome** | NOT a clean positive control. The mining is in *this* tile (multiple visible pits in lower-right, ~5+ patches), independently downlink-worthy |
| **Label** | `{"action": "downlink", "reason": "Active mining cluster in lower-right quadrant, multiple exposed-soil patches with SWIR confirmation, ~5+ pits visible in this tile alone"}` |

### d011 — Atewa periphery, target: `request_higher_resolution`

| | |
|---|---|
| **Coordinate rationale** | lon=-0.60, lat=6.18; just south of d002 (Atewa center, which had scattered SWIR). Hypothesis: periphery shows a single ambiguous candidate worth a higher-res look |
| **RGB** | uniform dark green forest canopy, no visible exposed-soil or disturbance |
| **SWIR** | mostly uniform blue with natural texture variance, no localized bright disturbance |
| **Outcome** | NOT a positive control either. The tile is genuinely pristine — no candidate to be ambiguous about |
| **Label** | `{"action": "discard", "reason": "Uniform forest canopy in RGB, no localized exposed-soil patches; SWIR variance is natural canopy texture not disturbance"}` |

### d012 — Pra tributary, target: `request_higher_resolution` (or `request_neighbor_tile`)

| | |
|---|---|
| **Coordinate rationale** | lon=-1.95, lat=5.40; downstream of Bogoso. Hypothesis: dispersed small candidates worth disambiguation, possibly with sediment plume continuing off-frame |
| **RGB** | scattered small brown/tan patches across the tile, plus a curving linear feature (road or stream) |
| **SWIR** | scattered bright patches, including a yellow-green diagonal streak in right/lower portion (sediment-laden water) |
| **Outcome** | **HR fires cleanly.** Patches are dispersed and individually small — could be small pits, agricultural clearings, or settlement infrastructure. Higher resolution would actually disambiguate |
| **Label** | `{"action": "request_higher_resolution", "reason": "Multiple small candidates dispersed across the tile, individually too small to confidently classify as mining vs agriculture/settlement; higher resolution needed to disambiguate"}` |

## Cumulative class distribution after 19 tiles

| Tool | Bibiani (n=6) | Diverse (n=10) | Pos-controls (n=3) | All (n=19) | Plan target |
|---|---:|---:|---:|---:|---:|
| downlink | 5 | 2 | 1 | **8 (42%)** | ~30% |
| flag_for_review | 0 | 2 | 0 | **2 (11%)** | ~20% |
| discard | 1 | 6 | 1 | **8 (42%)** | ~25% |
| request_higher_resolution | 0 | 0 | 1 | **1 (5%)** | ~15% |
| request_neighbor_tile | 0 | 0 | 0 | **0 (0%)** | ~10% |

## New findings (merged into `UNIFIED_VLM_PLAN.md`)

**6. `request_higher_resolution` works as designed (validated by d012).** When dispersed small candidates need disambiguation, the prompt cleanly elicits HR. The 1000-example training corpus needs more tiles like d012 — dispersed sub-resolution candidates with mixed possible interpretations.

**7. `request_neighbor_tile` is order-dependent and hard to elicit naturally.** It fires when (a) a feature clearly continues off-frame, (b) the tile itself isn't independently downlink-worthy, and (c) the relevant neighbor hasn't been processed yet. d010 had (a) and (c) but failed (b) — its in-frame mining was independently worth downlinking. This is a structural challenge, not a coordinate-picking problem. Most "feature continues off-frame" tiles will *also* have enough in-frame activity to call downlink.

**8. Mitigation for the `request_neighbor_tile` gap:** when generating the 1000-example training corpus, deliberately synthesize the regime by (a) choosing tile coordinates so that a known cluster cuts off at exactly one edge with the rest of the tile being forest/water, OR (b) accepting the gap and documenting it as a known limitation in the paper ("the unified VLM has fewer training examples for `request_neighbor_tile` and consequently lower coverage of this action"). Preference: do both — try to engineer some synthetic positive controls AND document the limitation honestly.

## Updated verdict

The labeling prompt is validated end-to-end for 4 of 5 tools across 19 tiles spanning mining hotspots, pristine forests, water, urban, coastal, and engineered positive controls. The 5th tool (`request_neighbor_tile`) has a known structural elicitation challenge with an explicit mitigation strategy.

**Phase A.2 mini is closed.** The next milestone is the 1000-example pipeline scale-up.

---

# Phase A.2 — Batch 1 (2026-05-04, 10 tiles)

**Goal:** first batch of training-corpus labeling via the deterministic sampler. This is the production labeling path going forward — `sample_coordinates(seed=42)` + SimSat fetch + Claude Code inline labeling, with results landing in `training/data/unified_v1/labels.jsonl`.

**Source:** 10 tiles from `sample_coordinates(n=10, seed=42)`, fetched into `training/data/unified_v1_cache/u0000-u0009/` via `training/scripts/fetch_unified_corpus.py 10`. Stratum distribution from the sampler: 4 forest, 3 mining, 2 water, 1 mixed.

## Per-tile labels (compact)

Full per-tile context (mission priors, neighbor summary, budget state) and full reasoning are in `training/data/unified_v1/labels.jsonl`. Summary table here:

| Tile | (lon, lat) | Stratum | Mission prior | Label | One-line reason |
|---|---|---|---|---:|---|
| u0000 | (−1.421, +6.485) | water | Lake Bosumtwi crater lake | discard | uniform water in both bands |
| u0001 | (−2.773, +5.593) | mining | Bibiani active cluster | **downlink** | scattered exposed-soil patches + classic SWIR signature |
| u0002 | (−1.639, +6.302) | mining | Obuasi periphery | flag | mixed urban + ambiguous reddish patches mid-tile |
| u0003 | (−2.430, +6.953) | mining | Asutifi expanding frontier | **request_hires** | dispersed sub-resolution candidates, prior raises suspicion |
| u0004 | (+0.075, +7.631) | water | Lake Volta | discard | open-water surface |
| u0005 | (−0.995, +9.412) | mixed | N. savanna agriculture | discard | rectilinear field patterns + dry-savanna SWIR |
| u0006 | (−1.324, +5.433) | forest | Kakum protected park | discard | uniform forest canopy |
| u0007 | (−0.584, +6.232) | forest | Atewa encroachment frontier | discard | tile is pristine despite prior; encroachment is in S/E neighbors |
| u0008 | (−1.357, +5.289) | forest | Kakum buffer zone (~14% cloud) | flag | road + clearings, ambiguous origin, partial cloud |
| u0009 | (−1.351, +5.366) | forest | Kakum protected park | discard | uniform forest canopy |

## Class distribution this batch

| Action | n | % | Plan target |
|---|---:|---:|---:|
| discard | 6 | 60% | ~25% (over) |
| flag_for_review | 2 | 20% | ~20% ✓ |
| downlink | 1 | 10% | ~30% (under) |
| request_higher_resolution | 1 | 10% | ~15% (under) |
| request_neighbor_tile | 0 | 0% | ~10% (gap) |

Heavy on discard because the deterministic sampler delivered 4 forest + 2 water tiles in the first 10 draws. The class balance averages out as the corpus grows, but it's worth tracking — if discard stays >50% across the next batches, we'll need to either bias the sampler toward the rarer strata or accept the imbalance as honest reflection of Ghana's surface (mostly not-mining).

## Cumulative state across all labeling rounds

| Action | Bibiani (n=6) | Diverse (n=10) | Pos-controls (n=3) | Batch 1 (n=10) | All (n=29) | Plan target |
|---|---:|---:|---:|---:|---:|---:|
| downlink | 5 | 2 | 1 | 1 | **9 (31%)** | ~30% ✓ |
| discard | 1 | 6 | 1 | 6 | **14 (48%)** | ~25% (over) |
| flag_for_review | 0 | 2 | 0 | 2 | **4 (14%)** | ~20% (under) |
| request_higher_resolution | 0 | 0 | 1 | 1 | **2 (7%)** | ~15% (under) |
| request_neighbor_tile | 0 | 0 | 0 | 0 | **0 (0%)** | ~10% (gap) |

`request_neighbor_tile` remains at 0 across 29 tiles. Per the Phase A.1 finding (`UNIFIED_VLM_VALIDATION.md` Findings 7-8), this tool is structurally hard to elicit because most "feature continues off-frame" tiles independently warrant downlink. Mitigation strategy stays: synthesize positive controls when sampling, and document the lower coverage as a known limitation in the paper.

## Notable observations from this batch

1. **Mission priors don't override visual evidence (u0007).** Atewa is documented as a "known galamsey encroachment frontier, threatened reserve." That's a strong prior — but tile u0007 itself was uniform pristine forest in both RGB and SWIR. The encroachment is happening in adjacent tiles (per the structured neighbor summary: south and east both downlinked clusters). Calling `discard` despite the prior is the right call — the prior raises *attention*, not the action when visual evidence is clean. Worth noting in the paper as evidence the structured neighbor summary is doing real work.

2. **Dry-savanna SWIR (u0005) is a real false-positive risk.** Northern Ghana savanna in dry season produces very bright SWIR signatures (high reflectance from dry vegetation + bare soil) that visually resemble mining disturbance. Without the regional prior + the rectilinear-field disambiguation rule, this tile would be a tempting downlink. The labeling prompt's existing rules handled it correctly, but the magnitude of the SWIR brightness was striking — far more than typical mining tiles.

3. **Cloud-cover metadata still unreliable (u0008).** Metadata reported `cloud_cover=0.143` (14%) but the visual cloud cover looks more moderate. The `flag_for_review` call was driven by other features (visible road + clearings) more than the cloud. Still a reminder that metadata cloud_cover should not be a single-source-of-truth filter.

4. **`request_higher_resolution` validated again (u0003).** The Asutifi tile fits the same pattern as d012 (Pra tributary) — dispersed sub-resolution candidates that need more pixels to confirm. Two HR examples in 29 tiles is light; need more to teach the model this action robustly.

## Pipeline notes

- **Fetch infrastructure works.** Sequential SimSat fetches for 10 tiles took ~1-2 minutes; no timeouts under sequential mode. Cloud Run scaling is adequate for this rate.
- **Cache layout (`u<NNNN>/{rgb.png, swir.png, meta.json, context.json}`)** is clean and machine-readable for the next labeling session.
- **Manifest JSON** is a useful index for batch labeling — lets a fresh Claude Code session read all tile paths at once instead of guessing the structure.
- **JSONL format** captures both the SFT-relevant fields (paths + context + assistant payload) and audit metadata (labeler ID, date, full reasoning) in one record. Single source of truth for both training and audit.

## Next session

Run `uv run python scripts/fetch_unified_corpus.py N` (skips already-cached, fetches the next batch). Open a fresh Claude Code conversation, label inline, append to `labels.jsonl`. Repeat until corpus reaches ~150-200 minimum (5-7 more sessions) before attempting a real SFT run.

In parallel: the training infrastructure (LoRA config, chat-template plumbing verification, pre-training baseline) is independent of corpus size and worth doing while labels accrue.

---

# Production labeling progress (rolling)

Per-tile coords, context, full reasoning live in `training/data/unified_v1/labels.jsonl` — that's the source of truth. This section tracks rolling totals, per-batch summaries, and methodological observations only. Future batches add a one-line journal entry; full per-tile tables stay in the JSONL.

## Cumulative state

| Metric | Value |
|---|---|
| Validation tiles labeled (Phase A.1) | 19 |
| Production tiles labeled (Phase A.2) | 190 |
| **Total labels** | **209** |
| Target (minimum for first SFT) | ~150-200 ✓ achieved |
| Target (paper-grade) | ~500-800 |

Combined class distribution across all 209 tiles:

| Action | Validation (n=19) | Production (n=190) | All (n=209) | Plan target |
|---|---:|---:|---:|---:|
| downlink | 8 | 34 | **42 (20%)** | ~30% (under, but trending closer) |
| discard | 8 | 109 | **117 (56%)** | ~25% (still over but improving) |
| flag_for_review | 2 | 44 | **46 (22%)** | ~20% ✓ |
| request_higher_resolution | 1 | 3 | **4 (2%)** | ~15% (under) |
| request_neighbor_tile | 0 | 0 | **0 (0%)** | ~10% (gap) |

We're at 209 labels — past the SFT minimum threshold and ready for Phase A.3. The class balance has shifted noticeably between rounds: round 2 (u0120-u0189) added 33 discard / 15 downlink / 22 flag / 0 hires / 0 neighbor — so discard's share of the corpus has come down (60% → 56%) and downlink's has come up (19% → 20%) just from continuing to label without changing the sampler. Flag is now at the plan target (22% vs ~20%). Discard is still over because the deterministic sampler keeps drawing forest, water, savanna, urban, and farmland tiles — these AOIs span a large fraction of Ghana's actual surface. For paper-grade balance the next round still needs class-aware oversampling: explicit upweighting of the mining-active strata (Bibiani, Obuasi, Pra basin, Ankobra, Asutifi) and a deliberate `request_neighbor_tile` collection sweep — that class is structurally hard to elicit through the current pass-runner protocol because we have no in-loop neighbor-fetch mechanism, so it will likely need to be hand-constructed from cluster-edge tiles. But for the first SFT run, the corpus is sufficient.

## Per-batch journal

| Batch | Date | Range | n | Class distribution | Notable |
|---|---|---|---:|---|---|
| 1 | 2026-05-04 | u0000-u0009 | 10 | 6D / 2F / 1DL / 1HR / 0N | Production labeling kickoff. Full table preserved above. |
| 2 | 2026-05-04 | u0010-u0019 | 10 | 7D / 2DL / 1F / 0HR / 0N | Two textbook downlinks (Bibiani u0014 stream-chain, Pra basin u0011); u0016 cloud_cover metadata anomaly (see observation below). |
| 3 | 2026-05-04 | u0020-u0049 | 30 | 15D / 8F / 6DL / 1HR / 0N | Confirmed SimSat `cloud_cover=1.40` is a recurring bug, not a one-off (u0045 second occurrence after u0016, both Coastal Axim). Strong downlink examples (u0036 Asutifi diagonal exposed-soil swath; u0014 stream-chain pattern repeats elsewhere). u0026 (Asutifi center) pristine despite being surrounded by an active cluster — same "interior holdout" pattern as u0007 Atewa, reinforces that mission priors don't override visual evidence. u0044 Obuasi settlement+peripheral-mining ambiguity hit the flag rule cleanly. |
| 4 | 2026-05-04 | u0050-u0059 | 10 | 7D / 2DL / 1F / 0HR / 0N | Nuclear run wave 1 of 7. u0053+u0054 active Bibiani/Obuasi cluster downlinks. u0055 third occurrence of `cloud_cover=1.403499` (Coastal Axim, same bit-identical out-of-range value), elevating the SimSat AOI bug from "recurring" to "persistent — every Coastal Axim tile in the sampler returns this value". |
| 5 | 2026-05-04 | u0060-u0069 | 10 | 7D / 2DL / 1HR / 0N | u0061 textbook stream-following Bibiani downlink, u0066 Obuasi multi-feature cluster downlink. u0067 (Asutifi frontier) earned a `request_higher_resolution` — subtle in-tile signal but adjacent to a 7-box cluster, the disambiguation-before-spending case the rare-class plan calls for. u0069 another "interior holdout" — pristine despite 8-box neighbor cluster. |
| 6 | 2026-05-04 | u0070-u0079 | 10 | 6D / 1DL / 3F / 0HR / 0N | Heavy Coastal Axim cloud streak (u0070, u0071, u0075, u0077 all 1.40). u0072 Bibiani cluster downlink with low budget — high-signal tile justifies spending despite tight budget. u0078 Atewa settlement-edge encroachment flag. |
| 7 | 2026-05-04 | u0080-u0089 | 10 | 7D / 2DL / 1F / 0HR / 0N | u0084 Pra basin near Bogoso fully surrounded by 4 active downlinked neighbors with extensive in-tile disturbance — the cluster-interior gold-standard downlink. u0087 first **budget-yields-to-reality** case — clear Bibiani signal but only 9kb budget left, flag rather than waste a useless thumbnail downlink. New training pattern: budget-aware downgrade. |
| 8 | 2026-05-04 | u0090-u0099 | 10 | 5D / 2DL / 3F / 0HR / 0N | u0093 Ankobra basin near Prestea scattered-disturbance downlink, u0099 Ankobra stream-following downlink. u0094 Kakum protected-park encroachment flag (small clearing inside protected forest is worth flagging even when subtle). |
| 9 | 2026-05-04 | u0100-u0109 | 10 | 6D / 1DL / 3F / 0HR / 0N | u0106 Bia protected park ENCROACHMENT downlink — visible road+settlement+structures inside the park, surrounded by 7 and 8-box neighbor clusters. u0100 and u0103 BOTH triggered budget-yields cases (21kb and 16kb remaining respectively, strong signals downgraded to flag). u0107 tenth Coastal Axim 1.40 cloud_cover. |
| 10 | 2026-05-04 | u0110-u0119 | 10 | 10D / 0DL / 0F / 0HR / 0N | Pure-discard streak. The deterministic sampler drew an all-negative-class wave (Volta shore, Accra coast, Central farmland mosaic, Kumasi urban, Bia pristine). Not a quality issue — reflects how often the sampler hits "nothing to see" tiles in Ghana's actual surface composition. The training set needs this base rate to teach the model when *not* to spend budget. |
| 11 | 2026-05-04 | u0120-u0129 | 10 | 3D / 1DL / 6F / 0HR / 0N | Round-2 nuclear run wave 1. Heavy flag concentration (6/10) — many tiles were subtle-signal-in-cluster cases where in-tile evidence was borderline but cluster bracketing strong. u0126 first **partial-tile imagery issue**: bottom ~50% of tile is solid black, possible SimSat data anomaly distinct from cloud_cover bug. u0129 Atewa settlement+road encroachment downlink (clean encroachment-into-frontier signature). |
| 12 | 2026-05-04 | u0130-u0139 | 10 | 6D / 2DL / 2F / 0HR / 0N | u0133 Kakum protected-park encroachment downlink (clear settlement INSIDE protected park, second protected-park encroachment after u0106 Bia). u0136 Bibiani in-tile-strong downlink without immediate cluster bracketing (signal alone in known-active AOI sufficient). u0132 sixth confirmed interior-holdout. Coastal Axim 1.40 hits 11th and 12th occurrences (u0130, u0139). |
| 13 | 2026-05-04 | u0140-u0149 | 10 | 7D / 1DL / 2F / 0HR / 0N | u0140 Asutifi distinctive access-road but budget=20 → flag (budget yields). u0148 SECOND partial-tile imagery issue (after u0126) — confirms partial-tile black-bottom defect is a recurring SimSat anomaly, not one-off; both occurrences in Kakum AOI. u0146 seventh interior-holdout. u0142 Ankobra fresh-budget cluster-bracketed downlink. |
| 14 | 2026-05-04 | u0150-u0159 | 10 | 4D / 3DL / 3F / 0HR / 0N | High-yield wave for downlinks. u0152 Pra basin TEXTBOOK pit-cluster downlink. u0156 Asutifi textbook galamsey scene downlink (large exposed-soil patch with bright SWIR). u0158 Atewa scattered-clearings cluster downlink. u0155 budget-yields case (56kb, heavy 3-side cluster bracketing → flag). u0150 Coastal Axim 1.40 (13th). |
| 15 | 2026-05-04 | u0160-u0169 | 10 | 5D / 2DL / 3F / 0HR / 0N | u0163 EXTREME budget-yields (only 8kb remaining + visible signal). u0166 Bibiani disturbance cluster downlink. u0169 Obuasi mining infrastructure downlink (fresh budget, AngloGold-style facility layout in tile, flagged on three sides). u0168 budget-yields at 52kb threshold — same threshold below which downlink quality becomes marginal. u0164 Coastal Axim 1.40 (14th). |
| 16 | 2026-05-05 | u0170-u0179 | 10 | 3D / 3DL / 4F / 0HR / 0N | Highest-yield wave for active-mining downlinks. u0170 + u0171 + u0179 are three Obuasi/Pra-basin downlinks: u0170 fresh-budget Obuasi disturbance, u0171 Obuasi road+clearings disturbance, u0179 textbook winding stream-following galamsey. u0177 Bia buffer-zone-farming flag (consistent with mission prior 'buffer-zone farming nearby'). u0178 budget=9kb extreme yields. u0176 Coastal Axim 1.40 (15th). |
| 17 | 2026-05-05 | u0180-u0189 | 10 | 5D / 3DL / 2F / 0HR / 0N | u0180 Obuasi heavy-disturbance cluster downlink (north+west BOTH 6-box). u0181 Ankobra subtle in-tile but heavy cluster bracketing (south 5, east 7) → downlink. u0188 Bibiani pit + cluster downlink. u0182 Bia protected-park road encroachment flag. u0185 cloud-vs-pit ambiguity flag — the disambiguation case where higher-resolution would help (and we don't have a `request_higher_resolution` neighbor for this exact scenario). u0186 Coastal Axim 1.40 (16th — running total). |

## Observations from production batches

1. **SimSat cloud_cover=1.40 anomaly is a recurring bug, not a one-off.** Tile u0016 (Coastal Axim, Batch 2) and tile u0045 (Coastal Axim, Batch 3) both returned `cloud_cover=1.403499` — *bit-for-bit identical out-of-range value, two different tiles, same AOI*. The expected range is 0-1 (cloud fraction). This is a stronger version of the existing "cloud metadata can be unreliable" finding from validation tiles d009 and u0008 — those were *underestimates* (low metadata, high visual cloud). u0016 and u0045 are out of range entirely. The repeat occurrence with the identical numeric signature on the Coastal Axim AOI looks like a SimSat upstream pipeline bug specific to that scene rather than a transient transport error. **Mitigation for the training data:** clamp cloud_cover values to [0, 1] when synthesizing per-tile context, and consider treating any anomalous metadata as a `tile_imagery_issue` flag. Don't filter the training set on cloud_cover alone — let the model see the visual reality regardless of metadata.

2. **Stream-following galamsey is the cleanest training signal.** Tile u0014 (Bibiani) showed a textbook winding chain of bright orange/tan exposed-soil patches following a stream through dense forest, with corresponding bright SWIR. This is the high-confidence `downlink` archetype. Batch 3 added u0036 (Asutifi) as a complementary archetype — a strong diagonal exposed-soil swath rather than a stream chain — broadening the visual vocabulary of "obvious downlink." The training set needs more like both — the v9-e3 perception fine-tune already learned these patterns; the unified model needs to learn that the same patterns → `downlink_now` action under bandwidth budget.

3. **Mission priors don't override visual evidence — the "interior holdout" pattern.** Confirmed instances now: u0007 (Atewa, Batch 1), u0026 (Asutifi center, Batch 3), u0069 (Asutifi, Wave 2), u0081 (Atewa, Wave 4), u0095 (Asutifi, Wave 5), u0132 (Atewa, Wave 12), u0146 (Asutifi, Wave 13) — seven confirmed cases now spanning Atewa and Asutifi, sometimes with neighbor cluster bracketing as heavy as 6-8 boxes downlinked. In all cases the actual imagery was pristine forest with no visible disturbance, all labeled `discard`. This is the most important test of the labeling discipline: we are training the model to read pixels, not narratives. The "interior holdout" is now a stable named pattern with broad representation in the training set — every prior-active AOI generates these negative examples, and they belong in `discard` to teach the model that priors and neighbor context inform but do not determine the action.

4. **Budget-aware downgrade: high-signal tiles can yield to budget reality.** Confirmed budget-yields cases: u0087 (Bibiani, budget=9kb), u0100 (Bia near cluster, budget=21kb), u0103 (Atewa with strong disturbance, budget=16kb), u0140 (Asutifi access road, budget=20kb), u0155 (Pra basin heavy cluster bracketing, budget=56kb), u0163 (Pra basin, budget=8kb), u0168 (Bibiani strong stream-following pattern, budget=52kb), u0178 (Pra basin, budget=9kb). Eight confirmed cases now spanning every active-mining AOI. The threshold where downlink quality becomes marginal sits around 50-55kb — below that, even strong in-tile signals get downgraded to `flag_for_review`. In each case the visual+context strongly supported downlink, but the remaining bandwidth was insufficient to capture useful imagery — flagging preserves the lead for the next pass when budget refreshes. This is a critical training pattern: the action is not a pure function of (image, neighbor_summary, mission_prior) — `budget_remaining_kb` is load-bearing context that can downgrade an otherwise-confident downlink to a flag. The two-layer system gets this through the LFM2-2.6B policy reasoning over the scalar context; the unified VLM needs to learn the same trade-off through the per-tile budget signal in its training prompts. **Implication for SFT:** these budget-yields tiles are exactly the disambiguation examples that justify the unified-VLM hypothesis — if the unified model can't learn this, the two-layer system is genuinely doing extra work.

5. **Protected-park encroachment is a distinct training signal from mining-AOI clusters.** Two confirmed protected-park encroachment downlinks now: u0106 (Bia protected park — road + settlement + structures inside the park, surrounded by 7 and 8-box active neighbor clusters) and u0133 (Kakum protected park — clear settlement/clearing in central tile, neighbors all `discard`, signal-driven downlink without cluster bracketing). Both labeled `downlink_now` despite "low expected disturbance" priors. These differ from cluster-interior downlinks like u0014 (Bibiani) or u0084 (Pra basin) because the AOI-level prior says "protected park, low expected disturbance" rather than "active galamsey cluster" — the high-confidence downlink is driven by *visible encroachment INTO the protected zone*, which is a higher-stakes signal than cluster continuation in a known-active area. The flag-level versions of the same pattern: u0094 (Kakum subtle clearing), u0046 (Atewa scattered disturbance), u0128 (Kakum multiple subtle clearings), u0143 (Bia track encroachment), u0172 (Bia track + cluster), u0177 (Bia buffer-zone farming), u0182 (Bia road network in protected). Training should preserve the asymmetry: same visual feature in a protected AOI vs an active-cluster AOI may warrant different actions, and signal-strength gradient (track → road network → settlement → settlement+structures) maps to action-strength gradient (`flag` → `flag` → `flag/downlink edge` → `downlink`).

6. **Persistent SimSat `cloud_cover=1.403499` AOI bug — confirmed across 16 tiles.** All sixteen Coastal Axim tiles drawn by the deterministic sampler returned the exact same out-of-range value `1.403499`: u0016, u0045, u0055, u0058, u0068, u0070, u0071, u0075, u0077, u0107, u0130, u0139, u0150, u0164, u0176, u0186. The bit-for-bit identical numeric signature across sixteen different geographic coordinates within the same AOI confirms this is a deterministic upstream pipeline bug specific to the Coastal Axim scene definition rather than a transient transport error. The sample size now makes this finding paper-grade — every Coastal Axim tile in our 209-tile corpus exhibits this anomaly, suggesting the bug is in the scene-level cloud-cover computation rather than per-tile. **Mitigation for the training data:** clamp cloud_cover to [0, 1] when synthesizing per-tile context; for the unified-VLM SFT corpus this means training prompts for these 16 tiles will show a clamped value (e.g. 1.0) — the model never sees the raw out-of-range value. **Mitigation for the orchestrator:** add a sanity-check guard in `synthesize_context` that warns when cloud_cover falls outside [0, 1] and clamps before serialization; track recurrence per AOI in the manifest. **Upstream fix:** worth filing a SimSat issue noting the Coastal Axim AOI returns out-of-range cloud_cover deterministically, since other downstream consumers will hit the same bug.

7. **Partial-tile imagery anomaly — second class of SimSat data defect.** Two tiles show solid-black bottom halves (~40-50% of the tile area is invalid pixels): u0126 (Kakum protected) and u0148 (Kakum protected). Both occurrences are in the same AOI (Kakum), suggesting another scene-level data anomaly distinct from the cloud_cover=1.40 bug. The `tile_imagery_issue` field in the synthesized context was `null` in both cases — context synthesis is not detecting the partial-tile defect. **Mitigation for the training data:** treat as `discard` and add a sanity check on image pixel statistics (e.g. fraction of pixels at exactly RGB=0,0,0) when synthesizing context; if above a threshold, set `tile_imagery_issue: "partial_tile"`. **Implication for SFT:** these are pathological inputs that the model should learn to handle gracefully (discard with low confidence) rather than try to extract signal from. **Upstream fix:** file with SimSat alongside the cloud_cover bug; both anomalies point to AOI-specific scene-definition issues.

8. **The disambiguation-class gap (`request_higher_resolution` and `request_neighbor_tile`) is structural, not data-quality.** After 209 labels we have only 4 `request_higher_resolution` and 0 `request_neighbor_tile` examples. These are the two rare classes the plan called out. The cause is structural: the synthesized context already supplies neighbor summaries, so the labeler rarely needs to *request* a neighbor tile (the value is in the context). And `request_higher_resolution` only fires when the in-tile signal is *just* below the downlink threshold AND the cluster context warrants the spend AND the budget allows it — a narrow Goldilocks zone. The cloud-vs-pit ambiguity in u0185 (Ankobra basin, white feature could be cloud or pit, west neighbor downlinked 8 boxes) was the textbook `request_higher_resolution` case but defaulted to `flag` instead — flag is the cheaper "I don't know" action when the model has no native way to escalate. **For training:** we may need to hand-construct synthetic prompts for these two classes, e.g. by taking known-mining tiles, blurring them, and instructing the labeler to demand resolution. Or by deliberately cropping cluster-interior tiles to leave the visible disturbance at the tile edge, eliciting a `request_neighbor_tile` to follow it. Without these, the unified VLM will likely learn to treat the rare actions as essentially out-of-distribution.
