# Unified VLM — Working Document

A living journal of the unified-VLM research arc spun out of the GalamseyWatch hackathon submission. Read top-to-bottom for the story; jump to the reference sections at the bottom for hyperparameters, citations, and codebase changes.

**Owner:** Samuel A. Donkor.
**Started:** 2026-05-04 (the day the hackathon submission shipped).
**Target:** arxiv submission, ~5-7 weeks from start.
**Parent project:** [GalamseyWatch](https://github.com/samadon1/GalamseyWatch).

---

## The story so far

GalamseyWatch is a two-layer agentic Earth-observation system: a fine-tuned LFM2.5-VL-450M does perception over Sentinel-2 RGB + SWIR composites, then hands its structured output (boxes, description, confidence) to an LFM2-2.6B tool-calling policy that decides per-tile what to do under a bandwidth budget. We shipped that to the Liquid AI × DPhi Space "AI in Space" hackathon on **2026-05-04**, then opened a PR to the Liquid4All cookbook the same day.

Hours later, on the cookbook PR thread, Pau Labarta Bajo (Liquid staff) asked the question that started this experiment:

> *"why are you using a separate text-to-text model (LFM2-2.6B in this case) to do the tool calling? Have you tried using/fine-tuning the LFM2.5-VL to do the whole thing end-2-end (perception + tool calling)?"*

The honest answer was no — we hadn't. The two-layer split came from how the project evolved (started with a click-to-detect dashboard, walked back when that didn't match real on-orbit ops) more than from a designed architecture choice. But Pau's question pointed at a real, unresolved engineering question: **is the perception/policy split necessary at the 450M VLM scale, or can a single fine-tuned VLM carry both jobs?** This document is the plan to find out.

### Why this is publishable

Three parallel research streams on **2026-05-04** (literature, LFM2.5-VL specifics, codebase audit) converged on a sharper picture than I'd expected:

- **Image-conditioned tool calling via SFT in sub-1B VLMs is genuinely open territory.** [VisionThink](https://arxiv.org/abs/2507.13348) does it via RL on a single self-referential tool. [SpotAgent](https://arxiv.org/abs/2602.09463) does it on larger VLMs. The Liquid4All cookbook ships text-only tool callers (`flight-search-assistant`, `home-assistant`) and VL fine-tunes without tools (`invoice-parser`, `satellite-vlm`). No public SFT recipe exists for what Pau suggested. We'd be the first.
- **LFM2.5-VL-450M is mechanically capable of it.** The tool-calling head lives in the LM backbone (LFM2.5-350M, BFCLv4=21.86); the VL post-training preserves it (BFCLv4=21.08, only 0.78pt loss). The `<|tool_call_start|>` / `<|tool_call_end|>` special tokens are already in the tokenizer/embedding table. The "text-only" caveat in the model card reflects training-data composition + chat-template plumbing, not an architectural barrier. Vision tokens project into the same LM token stream.
- **The bandwidth-aware on-orbit framing is unclaimed.** [Grace](https://arxiv.org/abs/2510.24242) and [Geo-OLM](https://arxiv.org/abs/2504.04319) are nearest neighbors but operate in different regimes (satellite-ground LVLM and ground-side agentic, respectively). On-orbit + sub-1B + tool-call-via-SFT is a defensible novel intersection.

So the contribution is: **the first published SFT recipe for image-conditioned tool calling in a sub-1B VLM, applied to bandwidth-aware on-orbit downlink decisions.** Both directions of empirical result land somewhere publishable. If the unified VLM matches the two-layer system, the headline is "the split is not necessary at 450M, here's the recipe." If it doesn't, the headline is "compression boundaries are load-bearing at 450M, here's the analysis." Either is a real contribution.

### The research question, locked

> Is the perception/policy split necessary at the 450M VLM scale, or can a single fine-tuned VLM match a 450M-perception + 2.6B-policy two-layer system on (a) action-correctness and (b) bandwidth-recall AUC, given the same per-pass scalar context?

---

## What we've done so far

### Phase 0 — slot refactor (done, local only)

The current orchestrator code couples the LFM2-2.6B agent into the pass loop via a singleton import. To plug in a unified-VLM agent later, that needed to be a Protocol with multiple implementations. On 2026-05-04 we:

- Extracted a `PolicyAgent` Protocol in `orchestrator/agentic_eo/models/agent.py`. `LFM2Agent` formally conforms.
- Renamed `_parse_response` → `_parse_lfm2_response` (Pythonic-bracket parsing, unchanged behavior) and added a sibling `_parse_json_response` for the future unified-VLM JSON output format. The new parser handles the five action literals + an optional `direction` field.
- Added a `get_default_agent()` factory that selects via a `POLICY_AGENT` env var. Default is the existing LFM2-2.6B path, so production behavior is identical.
- Updated `pass_runner.py` and `benchmark_policies.py` to consume the factory.
- Smoke-tested all of it: existing LFM2 path works, new JSON parser works on representative inputs, env-var dispatch works including unknown-value validation.

**Status:** sitting in the working tree, not committed. The plan is to commit Phase 0 + the unified-VLM agent in one coherent push when both exist — empty contracts (a Protocol with one implementation) aren't a meaningful signal to push to a public repo on their own.

### Phase A.1 — labeling-prompt validation (done)

Before scaling to 1000 labels, we needed evidence that the labeling prompt produces calibrated decisions. On 2026-05-04 we labeled 19 tiles inline via Claude Code, in three rounds:

1. **Six Bibiani tiles** (already cached from the hackathon benchmark). All six were known-active galamsey terrain. Labels: 5 `downlink` + 1 `discard`. **6/6 match against the existing LFM2-2.6B baseline** from `orchestrator/scripts/baseline_results.csv` — positive sanity check that the prompt is calibrated to the existing two-layer pipeline.

2. **Ten deliberately-diverse tiles** (mining hotspots, pristine forest, water, urban, coastal cloud-prone). Fetched from the self-hosted Cloud Run SimSat. This sample produced **20% downlink, 20% flag, 60% discard** — much closer to the target class balance and a real stress test of the disambiguation rules. The findings drove four refinements that are now baked into the canonical labeling prompt below: structured neighbor summary, explicit `direction` enum for `request_neighbor_tile`, scene-hint enum, and the class-balance requirement.

3. **Three positive controls** for the two tools never exercised on the first 16 (`request_higher_resolution`, `request_neighbor_tile`). One landed cleanly: tile `d012` (Pra tributary) elicited HR on a tile of dispersed sub-resolution candidates. Two failed to elicit `request_neighbor_tile` — we discovered that this tool is structurally hard to elicit, because most "feature continues off-frame" tiles also have enough in-frame activity to call `downlink` independently. That's now an explicit known limitation with documented mitigations.

Per-tile labels and reasoning are captured in [`UNIFIED_VLM_VALIDATION.md`](UNIFIED_VLM_VALIDATION.md). It's worth reading alongside this doc — it shows the prompt actually working on real imagery, edge cases included.

### The cost decision: Claude Code, not the API

The original plan called for a Modal pipeline that runs the labeling prompt against the Anthropic API on 1000 tiles. We built the scaffold (`training/scripts/prepare_unified_decision_dataset_modal.py`) including a fully implemented `label_with_claude` function with prompt caching on the system block (validated against the official `claude-api` skill).

Then we hit a real consideration: the GalamseyWatch developer has a Pro Max Claude subscription that covers Claude Code, but the Anthropic API is separately billed. ~$15-100 for 1000 tiles via API versus $0 marginal cost via Claude Code (interactive labeling like we did for the 19 validation tiles).

Decision: **defer the API path. Use Claude Code for labeling.** The same Claude does the work either way — the API path's only real advantage is reproducibility ("we labeled with Opus 4.7 pinned to date X via prompt caching"). The Claude Code path's advantage is $0 cost and an existing pattern that already works for us at the 19-tile scale. The methodology defense for the paper is honest: *labels were generated by Claude (Opus 4.x) reading RGB + SWIR composites in interactive sessions, with stratified manual audit*.

The Modal-pipeline scaffold stays in the codebase as forward-compatible infrastructure. If we ever decide to scale beyond what Claude Code can do interactively, we already have `label_with_claude` ready to wire up.

### Phase A.2 setup — current state (in progress)

Built `training/scripts/fetch_unified_corpus.py`: takes N, samples N coordinates deterministically (seed=42) over 15 hand-curated AOIs, fetches RGB + SWIR via SimSat, synthesizes per-pass scalar context, writes everything to `training/data/unified_v1_cache/<coord_id>/{rgb.png, swir.png, meta.json, context.json}` and emits a manifest.

The deterministic sampler matters for paper reproducibility — a reviewer asking "how did you pick these 250 coordinates?" gets a clean answer (stratified random sample over the documented AOIs with seed=42), and a future replicator runs the same script and gets the same coordinates.

**Up next:** run the fetcher for N=20 as a first batch, then I (Claude Code in this conversation) read the cached tiles and emit JSON labels into `training/data/unified_v1/labels.jsonl`. We'll see how many we can get through per session, then iterate.

---

## What's next

In rough priority:

1. **Run Phase A.2 in batches.** Fetch ~250 coords, label in interactive Claude Code sessions of ~50 tiles each (5 sessions). Track stratum distribution + class distribution as labels accrue. Stop and audit at 100 if anything looks weird.
2. **Phase A.3 SFT on whatever we have.** Even 100 examples can produce a feasibility-prove run. Stacked LoRA on top of the existing v9-e3 weights, using the recipe in §Training. If the unified VLM shows signal at 100, expand the corpus. If it doesn't, that's itself a finding.
3. **Phase A.4 head-to-head eval.** Reuse the existing `orchestrator/scripts/benchmark_policies.py` harness — it's already abstracted over `PolicyFn`. Slot in the unified VLM as a third policy alongside the existing two-layer system. Same eval set, same per-pass context.
4. **Writeup.** Frame as the contribution claim above. Reference the citation neighborhood at the bottom of this doc.

Decisions consciously deferred until results are in: paper title, abstract framing, figure choices, whether to scale up to a 1000-example corpus via the API path. Make those after Phase A.4 lands.

---

## Reference: architecture under test

Two architectures, head-to-head on the same eval set:

| ID | Architecture | Implementation status |
|---|---|---|
| **Current** | Two-layer: galamsey-v9-e3 (450M VL, perception) + LFM2-2.6B (general instruct, policy) | shipped |
| **Unified** | One model doing both: SFT'd galamsey-v9-e3 emits tool calls directly given image + per-pass scalar context | Phase A build |

The current two-layer split has a specific information topology:

```
RGB+SWIR pixels  →  VLM (450M)  →  {boxes, description, confidence}  →  LLM (2.6B)  →  {tool, reason}
                                          ↑
                                          + scalar context: {budget, cloud_cover, captured_at, neighbors, mission_priors}
```

The agent doesn't see pixels; it reasons over the *text* representation of perception plus per-pass scalars. The unified architecture collapses this:

```
RGB+SWIR pixels  +  scalar context (in prompt)  →  unified VLM (450M)  →  {tool, reason}
```

The empirical question is whether one model at 450M can carry both jobs, or whether the boundary is load-bearing.

**Naming precision for the paper:** the codebase uses `LiquidAI/LFM2-2.6B` as the policy, which is the general instruct model, not a dedicated tool variant. There is no LFM2-2.6B-Tool. The dedicated tool callers in the Liquid lineup are LFM2-1.2B-Tool. The two-layer baseline in this experiment is therefore "450M VL perception + 2.6B general instruct as policy," not "450M VL + 2.6B-Tool."

**One thing we considered and cut:** an earlier draft included a third architecture — keep the two-layer split but swap LFM2-2.6B for base LFM2.5-VL-450M in text-only mode as the policy. We removed it from the critical path. It answers a different question (*what if the policy model is smaller?*) than the one Pau asked (*is the split necessary at all?*), and the paper is cleaner with the head-to-head comparison. Worth running later as a separate ablation if curiosity demands it.

---

## Reference: data design

### Source

- **Geographic distribution:** stratified over Ghana — southwestern (Bibiani, training-distribution), eastern + Volta (out-of-distribution per `notebooks/05_eval_analysis.ipynb` Section 3).
- **Site list:** sample from SmallMinesDS test split + earthrise-media held-out Ghana coordinates (already wired via `training/scripts/verify_earthrise_sites_*.py`).
- **Tile fetch:** existing SimSat client (`/data/image/sentinel`), RGB + SWIR composites per tile, sequential to avoid Cloud Run cold-start cascade.
- **Class balance (critical):** the 1000 examples must NOT be skewed toward `downlink`. Deliberately mix mining hotspots with forest reserves, lakes/rivers, urban areas, cloud-heavy patches. The Phase A.1 validation sample (Bibiani only) was 5/6 `downlink` — that is *not* the production training distribution. The diverse 10-tile follow-up landed at 20% downlink, 20% flag, 60% discard. Target rough class balance: ~30% downlink, ~25% discard, ~20% flag, ~15% request_hires, ~10% request_neighbor.
- **Must exercise all five tools.** The 16-tile validation set never called `request_higher_resolution` or `request_neighbor_tile`. When sampling, deliberately include (a) tiles with a single small candidate that should `request_higher_resolution`, and (b) tiles where mining clearly continues off-frame. `request_higher_resolution` was validated 2026-05-04 by tile d012 (Pra tributary). `request_neighbor_tile` is structurally hard to elicit; mitigation is to synthesize coordinates where a known cluster cuts off at exactly one edge with the rest being forest/water, AND to document the lower coverage as a known limitation.

### Synthesized per-tile context

For each tile, generate plausible per-pass scalar context (do **not** uniform-sample, real distributions are skewed):

```python
{
  "tile_id":         "u0042",
  "lon":             -2.7400,
  "lat":             5.6300,
  "cloud_cover":     0.05,        # log-normal; most tiles have <10% but some 50%+
  "captured_at":     "2024-01-15T10:39:00Z",
  "budget_remaining_kb":  256,    # uniform across [0, 512]
  "budget_total_kb":      512,
  "prior_tiles_downlinked":  3,
  "tile_imagery_issue": null,     # null | "partial_swath" | "high_visual_cloud"
  "neighbor_summary": {           # structured per-direction state, NOT free-text
    "north": {"action": "downlink", "boxes": 5, "scene_hint": "cluster"},
    "south": {"action": "discard",  "boxes": 0, "scene_hint": "forest"},
    "east":  null,                # not yet visited / out of AOI
    "west":  {"action": "downlink", "boxes": 6, "scene_hint": "cluster_continuation"},
  },
}
```

`scene_hint` enum: `cluster` | `cluster_continuation` | `forest` | `water` | `cloud` | `urban` | `mixed`.

### Total size

- **Train:** 800 examples
- **Val:** 100 examples (held out by geography, not random)
- **Test:** 100 examples (held out by geography, **disjoint from val**)

LoRA recipes need less data than full FT. 800 examples should be enough to demonstrate or refute the unified approach. We're starting small (~250) via Claude Code and scaling only if Phase A.3 results justify it.

### Labeling prompt (validated 2026-05-04)

```
You are an on-orbit Earth-observation policy adjudicator. Given two views
of a Sentinel-2 patch (natural-color RGB + SWIR false-color composite) and
the per-pass operational context, decide which ONE of the five tools to
call.

Tools:
- discard: Skip this tile. Default for forest, water, cloud, or
  undisturbed land.
- flag_for_review: Log as text only. Use for moderate-confidence
  detections (1-2 small candidates, ambiguous descriptions) worth
  recording but not bandwidth.
- request_higher_resolution: Ask for a higher-res recapture next
  pass. Use when you see a small candidate (1 tiny box, or dispersed
  sub-resolution candidates) that needs more pixels to confirm.
- request_neighbor_tile: Fetch an adjacent tile when a feature visibly
  continues off-frame. Requires a "direction" field, one of:
  "north" | "south" | "east" | "west".
- downlink_now: Use the precious downlink budget to send THIS tile's
  image to ground. Reserve for high-confidence detections: 2+ clear
  pits, OR a single large pit, AND active galamsey indicators (sediment
  plumes, exposed lateritic soil, turbid water). If in doubt, prefer
  flag_for_review.

Disambiguation rules:
- SWIR brightness in the absence of exposed-soil patterns or sediment
  plumes is more likely infrastructure (rooftops, asphalt, cleared
  paths), not mining. Urban tiles have very bright SWIR but no pits.
- Rectilinear field patterns are agriculture, not mining. Galamsey pits
  are amorphous and cluster near water.
- If imagery is partial (large no-data regions, swath-edge artifacts) or
  the tile is heavily cloud-occluded regardless of metadata cloud_cover
  value, prefer flag_for_review over downlink. Don't burn bandwidth on
  unreliable input.

Operational context:
- Tile: {tile_id} at lon={lon}, lat={lat}
- Cloud cover (metadata, may be unreliable): {cloud_cover}
- Captured: {captured_at}
- Pass budget: {budget_remaining_kb} of {budget_total_kb} KB remaining
- Prior tiles downlinked this pass: {prior_tiles_downlinked}
- Tile imagery issue: {tile_imagery_issue}
- Mission priors: {mission_priors}
- Neighbor summary (structured): {neighbor_summary_json}
  - Each compass direction is either null (not visited / out of AOI)
    or an object: {"action": <tool>, "boxes": <int>, "scene_hint": <enum>}
  - scene_hint enum: cluster | cluster_continuation | forest | water |
    cloud | urban | mixed.

Reply with EXACTLY ONE tool call in this JSON format and nothing else:
{"action": "discard|flag_for_review|request_higher_resolution|request_neighbor_tile|downlink_now",
 "reason": "...",
 "direction": "north|south|east|west"  (ONLY when action is request_neighbor_tile)
}
```

### Validation protocol

Per [Iskander et al. EMNLP 2024](https://arxiv.org/abs/2409.16341), GPT-generated tool-call data has documented quality issues. Counter:

1. **Stratified manual audit** of ~10% of the corpus, distributed across all five tool labels and across geographic strata.
2. **Report agreement rate** with the adjudicator. If <70%, the experiment is contaminated and the data must be regenerated.
3. **Track failure modes** as a taxonomy in the audit report: schema invalid, wrong tool, plausible-but-debatable, etc.

### Mix-in: text-only tool-call samples

Per the [LFM2 Technical Report](https://arxiv.org/abs/2511.23404), the dense-model SFT mix is 10.1% tool use, mixed with everything else. To preserve baseline BFCLv4 behavior in our unified VLM, mix in **~10% text-only tool-call samples** from the Liquid cookbook recipes (`home-assistant`, `flight-search-assistant`) as a regularization signal. Without this mix, the unified VLM may forget that text-only tool calling exists, which (a) hurts the public model artifact and (b) prevents fair comparison against the (deferred) text-only baseline.

---

## Reference: training recipe

### Strategy: stacked LoRA

Naive single LoRA can also catastrophically forget ([Zhai et al. 2024](https://arxiv.org/abs/2402.18865); SMoLoRA, ICCV 2025; ColPro, ACL Findings 2025). We're doing **stacked LoRA**: leave the v9-e3 perception fine-tune in place, train a separate tool-call adapter on top.

The v9-e3 fine-tune was full FT (`peft_config.use_peft: false` in `training/configs/galamsey_v9_450m_aug_modal.yaml`), so in practice we're stacking a fresh LoRA on top of the v9-e3 weights as the new base. That's fine — the perception capability is in the weights; the LoRA learns the tool-call task without rewriting the visual head.

**Fallback if the stacked approach underfits:** merge perception adapter into the base, train fresh LoRA on tool-call only. Loses ablate-perception-alone but gives more capacity for the task.

### Hyperparameters (starting point)

```yaml
lora:
  r: 8                  # try 4 first if overfitting, 16 if underfitting
  alpha: 16             # 2 * r
  dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
    # if underfitting, add: k_proj, o_proj, gate_proj, up_proj, down_proj

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2  # effective batch 8, matches v9
  learning_rate: 1e-4             # higher than v9's 2e-5; LoRA needs more LR
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  eval_on_start: true             # capture pre-training baseline this time

hardware:
  gpu: H100
  dtype: bfloat16
```

### Custom chat template

`processor.apply_chat_template` doesn't propagate `tools=` for the VL path; only `processor.tokenizer.apply_chat_template` does. Two ways to handle:

1. Patch the chat template JSON in the processor config to inject the `tools=` list into the system message. Cleaner long-term but requires reading the template Jinja.
2. Build prompts manually: `system_message = "You are an on-orbit EO agent. " + json.dumps(tools)`, then standard chat-template flow.

Start with (2). Move to (1) if the format becomes finicky.

### Output format: JSON, not Pythonic-bracket

The current LFM2-2.6B agent emits Pythonic calls (`<|tool_call_start|>[discard(reason="...")]<|tool_call_end|>`). For the unified VLM, train it to emit JSON instead:

```json
{"action": "discard", "reason": "forest canopy, no detection"}
```

Easier to parse robustly (single `json.loads`, no AST gymnastics), forces commit to one of the five action literals, and decouples from LFM2 special tokens. The Phase 0 parser already handles both formats (Pythonic for the existing agent, JSON for the unified one).

### Compute budget

Per v9-e3 cost (~3 H100-hours full FT, 17,719 steps): LoRA SFT on ~250 examples × 3 epochs ≈ 0.5-1 H100-hour per run. Across 5-10 hyperparameter iterations: **~5-10 H100-hours total, $15-30 in Modal compute.** Lower than the original 1000-example estimate because we're starting smaller.

---

## Reference: evaluation

**Eval set:** the held-out test split from the labeled corpus (geographic, disjoint from val), plus the existing Bibiani 6-tile + earthrise-media held-out coordinates so we can compare against the existing baseline numbers in `orchestrator/scripts/baseline_results.csv`.

| Metric | What it measures | Why it matters |
|---|---|---|
| **Action-correctness** | Fraction of tiles where the model's action matches the ground-truth label | Headline: did the right thing happen? |
| **Per-action precision/recall** | F1 per tool (`downlink`, `flag`, `discard`, etc.) | Where does each policy fail? |
| **Bandwidth-recall AUC** | Curve of `pits_downlinked / pits_total` vs. `bandwidth_used_kb`, integrated | The actual production-relevant metric |
| **Parser failure rate** | Fraction of outputs that didn't parse as a valid tool call | Surfaces model brittleness |
| **Decision latency** | ms per decision | Practical satellite constraint |

**Honest comparison rules:** both architectures see the same operational context (same budget, same neighbor summary). The unified VLM gets the perception input as images; the two-layer architecture gets perception via the v9-e3 VLM first. Both consume the same upstream visual signal, just at different points in the pipeline. No per-architecture prompt tuning beyond the obvious format differences.

---

## Reference: risks and mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Synthetic data quality is bad (audit agreement <70%) | Medium | Stratified audit upfront; report rate honestly; regenerate if below threshold. |
| Stacked LoRA underfits at r=8 | Medium | Increase rank (r=16), expand target modules, or fall back to merging perception + fresh LoRA. |
| Unified VLM forgets text-only tool calling | High without mix-in | The 10% text-only mix-in is for exactly this. Verify by running BFCLv4 (or proxy) on the trained model. |
| Custom chat template breaks at inference | Medium | Test the manual prompt construction in Phase 0 before any training. Start with manual-injection (cleaner code path). |
| LFM2.5-VL-450M is too small to do both jobs | Medium | This is itself a publishable finding: "compression boundary is necessary at 450M." Frame the paper accordingly if results trend that way. |
| Geographic stratification too aggressive (test set uniformly OOD) | Low | If unified looks worse on test but better on val, recheck the split. |
| Modal compute exceeds budget | Low | At ~$15-30 estimated, well within hackathon-credit territory. |
| **SimSat cloud-cover metadata is unreliable.** Validated 2026-05-04 on tile d009 (Coastal Axim): metadata reported `cloud_cover=0.001` but visual cloud cover was ~30-40%. | Medium | Train the unified VLM to be robust to cloud regardless of reported metadata. Don't filter the training set on metadata `cloud_cover` alone; let the model see the visual variance. |

---

## Reference: codebase changes

| File | Status | Change |
|---|---|---|
| `orchestrator/agentic_eo/models/agent.py` | Done (local) | Added `PolicyAgent` Protocol; `LFM2Agent` conforms; `_parse_response` → `_parse_lfm2_response`; new `_parse_json_response`; `get_default_agent()` factory. |
| `orchestrator/agentic_eo/pass_runner.py` | Done (local) | Uses the factory; no behavior change at default. |
| `orchestrator/scripts/benchmark_policies.py` | Done (local) | Uses the factory. Existing `PolicyFn` accepts the unified agent when added. |
| `training/scripts/prepare_unified_decision_dataset_modal.py` | Done | Modal pipeline scaffold + `label_with_claude` (Anthropic API path, deferred). |
| `training/scripts/fetch_unified_corpus.py` | Done | Deterministic sampler + SimSat fetcher for the Claude Code labeling path. |
| `training/configs/galamsey_unified_v1.yaml` | TODO | Stacked-LoRA SFT config. |
| `training/scripts/audit_unified_decisions_modal.py` | TODO | Stratified manual-audit helper. |
| `training/scripts/eval_unified_modal.py` | TODO | Eval against the SmallMinesDS test split for continuity with v9. |
| `orchestrator/agentic_eo/models/unified_vlm_agent.py` | TODO | New `UnifiedVLMAgent` implementing `PolicyAgent`. Loads SFT'd VL + adapter, emits JSON. |
| `docs/UNIFIED_VLM_RESULTS.md` | TODO (post-A.4) | Head-to-head comparison table + analysis. |

---

## Reference: citation neighborhood

### Prior art
- [**Grace**](https://arxiv.org/abs/2510.24242): satellite-ground collaborative LVLM. Closest published prior art for on-orbit VL.
- [**Geo-OLM**](https://arxiv.org/abs/2504.04319): open SLMs + state-driven workflows for EO. Agentic but ground-side.
- [**Satellite-Ground Synergistic LVLM for EO**](https://arxiv.org/abs/2507.05731).
- [**VisionThink**](https://arxiv.org/abs/2507.13348): RL-trained VLM with single-tool image-conditioned tool calling.
- [**SpotAgent**](https://arxiv.org/abs/2602.09463): ReAct-style tool use in larger VLMs for visual geo-localization.
- [**SLMs for Agentic Systems**](https://arxiv.org/abs/2510.03847): survey arguing 1-12B SLMs match LLMs on schema-constrained tool use.

### Methodology
- [**Zhai et al.**, *Investigating Catastrophic Forgetting in MLLMs*](https://arxiv.org/abs/2402.18865).
- **SMoLoRA** (ICCV 2025): routing between task-specific LoRA blocks.
- **ColPro / Progressive LoRA** (ACL Findings 2025).
- [**Iskander et al.**, *Quality Matters: Synthetic Data for Tool-Using LLMs*](https://arxiv.org/abs/2409.16341).
- [**LFM2 Technical Report**](https://arxiv.org/abs/2511.23404).

### Operational context
NASA Dynamic Targeting; Satellogic edge AI; ESA Φsat-2 (operational/industrial precedents for autonomous downlink decisions in the EO domain).
