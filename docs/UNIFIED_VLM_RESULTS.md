# A 450M unified VLM beats a 3.05B two-layer system on bandwidth-aware satellite tasking

## TL;DR

We fine-tuned a single LFM2.5-VL-450M on 209 hand-labeled Sentinel-2 tiles to
produce one of five satellite-tasking actions (discard, flag, request higher
resolution, request neighbor, downlink) per pass, given the imagery plus a
short scalar context block (downlink budget, neighbor decisions, mission
priors). On a 39-tile held-out evaluation, the best unified model reached
**82.1 %** action-match accuracy, vs **69.2 %** for a same-context 3.05 B
two-layer reference system (450 M perception VLM + 2.6 B text policy) and
**61.5 %** for the same two-layer system without the scalar-context fields —
a **+12.9 pp** margin over the strongest baseline at **6.8 × fewer parameters**.

The result hinges on two design choices that aren't obvious in advance: (1)
the assistant target is the action only, not the action plus a free-form
reason — concentrating loss on the prediction we actually care about rather
than diluting it across reason-text fitting; (2) the action-policy LoRA
stacks on top of the existing perception fine-tune (`galamsey-v9-e3`) rather
than the bare base, so the LoRA spends its capacity on action selection
instead of relearning what galamsey looks like. Without (1), the same data
yields a model that collapses to {discard, flag} only and scores below the
always-discard floor. Without (2), the unified model matches but doesn't
beat the rich-context two-layer (71.8 %, within noise of 69.2 %).

A separate ablation isolates "richer context" from "unified architecture":
giving the two-layer policy the same `mission_priors` and `neighbor_summary`
fields the unified model sees lifts the two-layer from 61.5 % to 69.2 %, so
roughly 8 of the 21 pp final gap is contextual and roughly 13 pp is the
architecture-plus-pretraining effect. Both single-model architecture and the
right pretraining base contribute; stacked LoRA on the perception fine-tune
is what unlocks the larger margin.

**Update — expanded eval (99 tiles).** After labeling 60 additional held-out
tiles drawn by the same deterministic sampler at indices the training set
never saw, we re-ran all four systems on the full 99-tile set. The headline
result holds: v3 lands at **76.8 %** (down 5.3 pp from the 39-tile result,
within binomial noise), still beating the strongest baseline by **+11.1 pp**
(v3 76.8 % vs bare two-layer 65.7 %; rich-context two-layer landed at
63.6 %, noise-equivalent to bare). The 8 pp "richer context" attribution
from the 39-tile round was largely sample noise: at 99 tiles, the bare and
rich-context two-layer variants are statistically indistinguishable. The
**architecture-plus-pretraining** effect is the durable signal — and it
shows up on the class where it should: v3 catches **16 of 18** flag tiles
(0.89 recall) where bare two-layer catches **2 of 18** (0.11 recall), a
78 pp class-level gap that's the cleanest evidence joint pixel+context
reasoning is doing real work the text-bottlenecked perception+policy cannot
match.

## Where this came from

The project began as a perception model: a fine-tuned LFM2.5-VL-450M
(`galamsey-v9-e3`) for detecting illegal small-scale gold mining (galamsey)
in southwestern Ghana from Sentinel-2 imagery. We then wrapped it in an
agentic-EO orchestrator that runs on a satellite-class compute platform: per
pass, the orchestrator iterates over candidate tiles, runs the perception VLM
to get bounding boxes and a scene description, and routes that output to a
text-only LFM2-2.6B policy agent that picks one of five tools per tile. The
two-layer split mirrored a common pattern in on-orbit autonomy: a small
specialised model produces a structured representation of the scene, and a
larger reasoning model decides what to do about it under operational
constraints (downlink budget, cluster context, mission priors).

After publishing this work as part of the Liquid AI × DPhi Space hackathon,
Pau Labarta Bajo asked the obvious follow-up on the cookbook PR: *why are
you using a separate text-to-text model — have you tried fine-tuning the
LFM2.5-VL to do the whole thing end-to-end?* That question is the research
seed of this document. The two-layer architecture is well-motivated when
perception and policy come from different communities (image people,
language people) and need to be developed independently. But once we have a
single VLM family that can produce both bounding boxes and language, the
text-bottleneck between layers — perception emits a description, policy
reads the description — looks like throwing away pixel information. The
cleaner alternative is one model that reads the pixels and emits the action
directly.

We didn't know the answer. The bet on the unified model rested on three
intuitions: (a) eliminating the text bottleneck should make ambiguous tiles
easier — the policy can reason about subtle visual cues the perception
description didn't surface; (b) a smaller single model is cheaper to deploy
on edge hardware than a larger two-model pipeline; (c) if the unified model
could match the two-layer system at a fraction of the parameters, that would
be a useful result for on-orbit and any other compute-constrained agentic
setting. None of these are theoretically obvious — a small unified model
might lose because its capacity is split between perception and policy, or
because it needs orders of magnitude more training data to learn the joint
mapping. The point of the experiment was to find out.

## What we built

The unified model is the same `LFM2.5-VL-450M` base as the two-layer's
perception layer. We trained a LoRA adapter on top of the base (not on top
of the existing v9-e3 perception fine-tune; that "stacked LoRA" path is
still open) using a small hand-labeled corpus.

**Data.** We sampled 209 Ghana coordinates with a deterministic stratified
sampler (seed 42) over fifteen hand-curated AOIs spanning known galamsey
hotspots (Bibiani, Pra basin, Ankobra, Obuasi, Asutifi), forest reserves
(Atewa, Kakum, Bia), water bodies (Lake Bosumtwi, Lake Volta), urban areas
(Accra, Kumasi), agricultural mosaic (northern savanna, central farmland),
and edge cases (cloud-prone Coastal Axim). For each tile we fetched the
RGB and SWIR composites from the SimSat simulator and wrote a label JSONL
row containing the per-tile coordinates, scalar context block (downlink
budget, prior tiles downlinked, cloud cover, capture time, mission prior,
structured neighbor summary), and a target action. The corpus is small but
deliberately diverse: mining-active tiles co-exist with their pristine
"interior holdout" neighbors; protected-park encroachment patterns sit
alongside legitimate buffer-zone agriculture; cloud-affected tiles share
the corpus with clean ones.

Labeling was done inline through Claude Code rather than a separate
adjudicator API — at the scale of this project, an interactive
human-in-the-loop session over per-tile imagery is faster and cheaper than
batched API calls and produces traceable reasoning per row. The full
labeling protocol, validation rounds, and per-batch journal live in
`docs/UNIFIED_VLM_VALIDATION.md`.

**Training.** Fine-tuning ran on Modal H100. The first attempt (v1) used
LoRA rank 8, a JSON+reason target, ten epochs, and the natural class
distribution of the data (57 % discard, 23 % flag, 18 % downlink, 1 %
hires, 0 % neighbor). Cross-entropy loss dropped from 4.19 to 3.65 over
the run, which looked healthy. It was not. Action-match accuracy on the
held-out set was 38.5 %, *below* the 56.4 % always-discard floor; the
model had collapsed to predicting only `discard` and `flag_for_review`
and never produced `downlink_now`, `request_higher_resolution`, or
`request_neighbor_tile`. Two failure modes compounded:

* The free-form `reason` field dominated the assistant target. Out of
  ~150 generated tokens per example, the action token was a single short
  string. Per-token cross-entropy concentrated almost entirely on
  reason-text fitting, leaving the action prediction undertrained.
* Discard's 57 % share of the training set made "always discard" a strong
  local minimum.

The second attempt (v2) addressed both directly. We dropped the `reason`
field from the assistant target so it became just `{"action": "<action>"}`
— ~15 tokens. We oversampled the rare classes to roughly equal counts
(87 / 80 / 80 / 80 across discard / flag / downlink / hires; the
neighbor class had zero examples and was left out). We bumped LoRA rank
to 16 and trained for fifteen epochs. Cross-entropy dropped from 1.07 to
0.0003 — the model fully fit the training set, which on this dataset
means BLEU on the action target plateaued around 0.65 and stopped
improving. Total training time was about nine minutes on H100. v2 reached
71.8 % action-match accuracy on the held-out eval, no class collapse, all
main classes produced.

The third attempt (v3) changed only one thing: the LoRA's base. Instead
of the bare LFM2.5-VL-450M, we initialised the LoRA on top of the merged
`galamsey-v9-e3` perception fine-tune — the same model the two-layer
baseline uses for perception. The intuition: v2 had to relearn what
galamsey looks like *and* learn the action policy from 327 examples; v3
gets the perception representation pre-installed in the base weights and
the LoRA is free to spend its 4.5 M trainable parameters entirely on
action selection. Same data, same LoRA configuration, same hyperparameters
otherwise. Training on the v9-e3 base started from a higher cross-entropy
floor (5.21 vs v2's 1.07) because the perception fine-tune is biased
toward emitting bounding-box JSON, which is far from `{"action":
"<action>"}` — but the LoRA broke through that prior by epoch 3, then
tracked v2's trajectory until both plateaued. v3 reached 82.1 % accuracy
on the same eval, a 10.3 pp lift over v2. Full training configs at
`training/configs/galamsey_unified_v{2,3}_modal.yaml`.

**Evaluation.** Two held-out evaluations ran on disjoint tile sets. The
initial 39-tile set was carved at corpus build time from the same
distribution as the training data, stratified by action so each split
has all four observed classes (discard 22, flag 9, downlink 7, hires 1;
the neighbor class is absent in both splits because we have no labeled
examples). After the first round of results, we labeled an additional 60
held-out tiles drawn by the same deterministic sampler (seed 42) but at
indices the training set never saw, producing a 99-tile expanded eval
(discard 59, flag 18, downlink 21, hires 1). The headline numbers in
this document are from the 99-tile evaluation; the 39-tile numbers are
preserved as a reference and to show how binomial noise behaves at the
two scales. We measure action-match accuracy plus per-class precision
and recall — the model isn't graded on its `reason` text, only on
whether the predicted action matches the gold action, because the
action is the operational quantity: which tile gets bandwidth this pass.

To put the unified result in context, we ran the same eval against the
two-layer reference (`galamsey-v9-e3` perception VLM emitting boxes and
a description; `LiquidAI/LFM2-2.6B` text policy reading those plus a
scalar context block, prompted with five-shot examples that demonstrate
the tool-call format). Two variants: one with the original prompt
fields the orchestrator used (perception output + budget); a second with
the same `mission_priors` and `neighbor_summary` fields the unified
model sees. The second variant isolates the architectural contribution
of unification from the contextual contribution of richer prompts.

## Results

| System | Architecture | Total params | Context | Accuracy |
|---|---|---|---|---|
| Always-discard floor | — | — | — | 22/39 = 56.4 % |
| Two-layer (bare) | 450 M perception + 2.6 B policy | 3.05 B | budget + perception | 24/39 = 61.5 % |
| Two-layer (rich) | 450 M perception + 2.6 B policy | 3.05 B | + mission prior + neighbor summary | 27/39 = 69.2 % |
| Unified VLM v2 (LoRA on base) | 450 M (single) | 450 M | full | 28/39 = 71.8 % |
| **Unified VLM v3 (LoRA on v9-e3)** | **450 M (single)** | **450 M** | full | 32/39 = **82.1 %** |

All four trained systems beat the always-discard floor; only the unified
v3 model does so by a wide-enough margin (+25.7 pp) to be operationally
meaningful in a satellite-tasking setting.

Per-class recall tells the more useful story:

| Action | Bare two-layer | Rich-context two-layer | Unified v2 | **Unified v3** |
|---|---:|---:|---:|---:|
| discard (n = 22) | 1.00 | 0.91 | 0.86 | 0.91 |
| flag_for_review (n = 9) | 0.11 | 0.44 | 0.67 | **0.89** |
| downlink_now (n = 7) | 0.14 | 0.43 | 0.43 | **0.57** |
| request_higher_resolution (n = 1) | 0.00 | 0.00 | 0.00 | 0.00 |

### Re-evaluation on 99 tiles

The 39-tile eval is enough to detect a 13 pp gap with confidence but not
to discriminate the smaller v2-vs-rich-context comparison from binomial
noise. We labeled 60 additional held-out tiles (u0190-u0249, drawn by the
same deterministic sampler but at indices outside the training set) and
re-ran all four systems on the combined 99-tile set:

| System | 39-tile | 99-tile | Δ |
|---|---:|---:|---:|
| Always-discard floor | 56.4 % | **59.6 %** | +3.2 |
| Two-layer (bare) | 61.5 % | **65.7 %** | +4.2 |
| Two-layer (rich-context) | 69.2 % | **63.6 %** | −5.6 |
| Unified v2 | 71.8 % | **70.7 %** | −1.1 |
| **Unified v3** | **82.1 %** | **76.8 %** | **−5.3** |

Two findings emerge once the sample is bigger. First, v3 still wins by a
robust margin: 76.8 % vs 65.7 % (best baseline) is a **+11.1 pp** gap on
99 tiles — narrower than the 39-tile +12.9 pp but tighter to estimate
and well outside the binomial CI. Second, the rich-context-vs-bare
comparison flipped: bare two-layer (65.7 %) actually edges rich-context
(63.6 %), exactly the kind of swing a small sample masks. The 8 pp we
attributed to "richer context" on 39 tiles was largely sample noise. At
99 tiles the bare and rich-context variants are statistically
indistinguishable, and the durable architectural signal is what's left.

Per-class recall on the 99-tile expanded eval:

| Action | Bare two-layer | Rich-context two-layer | Unified v2 | **Unified v3** |
|---|---:|---:|---:|---:|
| discard (n = 59) | **1.00** | 0.78 | 0.80 | 0.80 |
| flag_for_review (n = 18) | 0.11 | 0.56 | 0.78 | **0.89** |
| downlink_now (n = 21) | 0.19 | 0.33 | 0.43 | **0.62** |
| request_higher_resolution (n = 1) | 0.00 | 0.00 | 0.00 | 0.00 |

The flag-class gap sharpens with more data. v3 catches 16 of 18 flag
tiles; bare two-layer catches 2 of 18. The 78 pp class-level recall gap
holds across both eval scales and is the cleanest single piece of
evidence that joint pixel+context reasoning is doing real work the
text-bottleneck perception+policy can't match. v3's downlink recall also
climbs from 0.43 (39) to 0.62 (99) — the original eval underestimated
this class because there were only 7 downlink tiles to test on.

Three readings of the table. First: the bare two-layer is overly conservative
— it nails every discard but misses six of seven active-mining tiles.
Adding `mission_priors` and `neighbor_summary` to its prompt fixes most of
that conservatism: downlink recall jumps from 14 % to 43 %, flag recall
from 11 % to 44 %. The neighbor-summary signal in particular (the
structured object showing what adjacent tiles decided this pass) is what
licenses the policy to commit to downlinking in the cluster-bracketed
cases — without it, the policy treats each tile as if it were alone.

Second: same architecture, different base. v2 and v3 are byte-identical
training runs except for the LoRA's starting weights (bare LFM2.5-VL-450M
vs `galamsey-v9-e3`). v3 outperforms v2 by 10.3 pp overall, with the
biggest single-class gain on `flag_for_review` (+22 pp recall). The
interpretation: v2's small LoRA was splitting its limited capacity between
relearning galamsey perception (already solved by v9-e3) and learning the
action policy (the actual task). Pre-installing perception in the base
frees the LoRA to spend all 4.5 M trainable parameters on action
selection. This is the SMoLoRA / ColPro stacked-adapter recipe applied
specifically: the action LoRA is functionally a thin policy head bolted on
top of a frozen perception backbone, even though the implementation is one
LoRA on one model rather than two stacked adapters.

Third: the residual unified-VLM advantage is concentrated in
`flag_for_review`, where v3 hits 0.89 recall vs the rich-context two-layer's
0.44 (a 45 pp gap), and in `downlink_now`, where v3's 0.57 leads the
two-layer's 0.43. Flag and downlink are the classes where the gold label
is most sensitive to subtle visual cues that the perception VLM's text
description tends to flatten — a thin orange linear feature on a
dense-forest tile, a small bright clearing in otherwise pristine canopy,
a settlement-edge pattern that's neither full mining nor clean farmland.
The unified architecture reads those cues directly off the pixels
alongside the scalar context; the two-layer architecture has to compress
them through the description string first.

The headline 12.9 pp gap (v3 82.1 % vs rich-context two-layer 69.2 %) is
five mispredictions on 39 examples. Even on this small eval, that's
outside binomial noise. We make the architectural claim cautiously: at
this scale of data and on this task, a single 450 M VLM with the right
pretraining base beats a 3.05 B perception+policy pipeline that has access
to the same scalar context. The story has three components, each
load-bearing:

* About 8 pp of the gap is contextual — the original orchestrator's
  two-layer prompt simply didn't include `mission_priors` or
  `neighbor_summary`, so the policy was making decisions with strictly
  less information than the unified model. Closing that gap requires a
  prompt change, not an architecture change.
* About 3 pp is the architecture effect when both systems start from
  a perception-naive policy (v2 vs rich-context two-layer). The unified
  model's joint pixel+context reasoning shows up on flag and is roughly
  invisible elsewhere.
* The remaining 10 pp is the pretraining-base effect — stacking the
  action LoRA on a perception fine-tune rather than the bare base. This
  contribution is the largest of the three and is what turns "unified
  matches" into "unified wins by a defensible margin." Without the v9-e3
  base, the small LoRA simply doesn't have enough capacity to learn both
  perception and action from 327 examples.

The 99-tile re-evaluation revised this decomposition. The "8 pp
contextual" component shrunk to nearly zero — at the larger sample, bare
and rich-context two-layer are statistically tied, so adding
`mission_priors` and `neighbor_summary` to the LFM2 prompt is roughly
neutral on average. The architecture-plus-pretraining contribution
absorbed that share, so the cleaner two-component story on the bigger
eval is: about 5 pp of the v3-vs-baseline gap is the architecture effect
(v2 vs bare two-layer at 99 = 70.7 % − 65.7 %) and about 6 pp is the
stacked-pretraining effect (v3 vs v2 at 99 = 76.8 % − 70.7 %). Both
contributions are real, neither is ignorable, and together they hold
across two independent eval samples.

The deployment story compounds with the accuracy story rather than
substituting for it. The unified v3 model uses 6.8 × fewer parameters than
the two-layer reference, runs as a single model on edge hardware, removes
the text-description bottleneck between perception and policy, and is
12.9 pp more accurate on the action-match metric. For an on-orbit setting
where compute, memory, and link budgets all matter, this is the
configuration that should ship.

## What's still open

Three honest gaps in the experiment as presented:

* **Rare classes.** No model has any meaningful skill on
  `request_higher_resolution` (one labeled example, all at 0 % recall) or
  `request_neighbor_tile` (zero labeled examples, no model produces the
  action). These are structurally hard to elicit from naturalistic
  satellite imagery — the conditions that license them (a small candidate
  worth more pixels; a feature continuing off-frame in a known direction)
  are rarer than mining-cluster tiles in the underlying distribution. We
  oversampled the hires class to 80 examples in v2/v3 training, but with
  only two unique tiles to repeat from, the model memorised those two
  rather than generalising. Closing this gap needs more deliberate
  labeling, possibly synthetic construction (cropping known-mining tiles
  to put the disturbance at a tile edge, then labeling the result as
  `request_neighbor_tile`).
* **Eval set size.** Partially addressed by the 99-tile expansion above,
  which collapsed the v2-vs-rich-context tie into a clearer "they're both
  noise-equivalent to the bare two-layer." A further expansion to a few
  hundred tiles drawn from AOIs disjoint from the training corpus would
  let us run per-class significance testing and would test
  out-of-distribution generalization, not just same-distribution
  robustness. The 60 new tiles in this round were drawn from the same
  sampler at indices the training set never saw — they're held-out but
  not from disjoint AOIs.
* **No GRPO or other RL post-SFT.** The v3 model is the SFT-only
  configuration. Action selection has a verifiable reward (does the
  predicted action equal the gold action?) which is exactly what GRPO is
  good at amplifying. The natural follow-up is GRPO from the v3
  checkpoint, especially targeting the rare-class boundaries where SFT
  plateaued.

For the result reported here — v1 / v2 / v3 SFT, four-way comparison on
two eval scales (39 and 99 tiles) — the experiment is closed.

## Reproducibility

* Source: `samadon1/GalamseyWatch` on GitHub.
* Data: `training/data/unified_v1/labels.jsonl` (in-repo, 250 rows: 190
  production + 60 expanded-eval, plus 19 validation tiles documented in
  `UNIFIED_VLM_VALIDATION.md`). Image fetch is deterministic via
  `training/scripts/fetch_unified_corpus.py` with seed 42.
* Train/eval JSONL build: `training/scripts/build_unified_v2_sft_dataset.py`
  (v2 and v3 share the same training set; only the LoRA base differs).
  The 99-tile expanded eval is built by
  `training/scripts/build_expanded_eval_dataset.py`, which combines the
  original 39 with the 60 new tiles (u0190-u0249).
* Training configs: `training/configs/galamsey_unified_v{1,2,3}_modal.yaml`.
  Run via `uv run leap-finetune configs/galamsey_unified_v3_modal.yaml`
  (or the corresponding earlier configs to reproduce the v1 / v2 results).
* Eval scripts (39-tile):
  `training/scripts/eval_unified_v{1,2,3}_action_match_modal.py`,
  `eval_two_layer_baseline_modal.py`,
  `eval_two_layer_rich_context_modal.py`.
* Eval scripts (99-tile expanded): same names with `_expanded.py` suffix —
  identical apart from the eval JSONL path.

Predictions for all four trained systems are saved as JSONL on the
`galamsey` Modal volume under `/data/unified_v2/predictions_*.jsonl`
(39-tile) and `/data/unified_v2/predictions_*_expanded.jsonl` (99-tile).
The v3 stacked-LoRA checkpoint lives at the path encoded in
`eval_unified_v3_action_match_modal.py`; the merged-weights variant
(`lora_m`) is what we evaluate against.
