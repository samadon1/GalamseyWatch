"use client";

import {
  AutoModelForImageTextToText,
  AutoProcessor,
  RawImage,
  env,
} from "@huggingface/transformers";

// Tell the browser to pick the discrete GPU on dual-GPU laptops (MacBook Pros
// with Apple Silicon use the unified GPU regardless; this matters on Windows
// laptops with Intel iGPU + NVIDIA dGPU, where the default is usually the
// iGPU). No-op if only one GPU is available.
if (typeof navigator !== "undefined" && (navigator as Navigator & { gpu?: unknown }).gpu) {
  // @ts-expect-error, webgpu config is dynamic on the onnxruntime env surface
  env.backends.onnx.webgpu = env.backends.onnx.webgpu ?? {};
  // @ts-expect-error, same
  env.backends.onnx.webgpu.powerPreference = "high-performance";
}

// Our v9-e3 fine-tuned model, exported to ONNX via Liquid4All/onnx-export,
// uploaded to HF. This is what runs in the browser via WebGPU in live mode.
// Trained on SmallMinesDS with 8× D4-group augmentation, pixel IoU 0.332 vs
// base model's 0.069 (see the Metrics modal on the landing page).
const MODEL_ID = "samwell/galamsey-v9-e3-onnx";

const GROUNDING_PROMPT =
  "You are viewing two images of the same Sentinel-2 patch: a natural-color RGB " +
  "composite and a SWIR false-color composite. Using both views, detect any " +
  "illegal small-scale gold mining pits. Include any exposed soil, excavation, " +
  "or sediment-laden water even if you are uncertain, err toward detection. " +
  'Provide result as a valid JSON: [{"label": str, "bbox": [x1,y1,x2,y2]}, ...]. ' +
  "Coordinates must be normalized to 0-1. Only return [] if the scene is entirely " +
  "pristine forest, clean water, or urban built-up area with no disturbance.";

// Matches the description prompt used during v9 training (scripts/prepare_v9_production_modal.py).
const DESCRIPTION_PROMPT =
  "You are analyzing two views of the same Sentinel-2 patch of southwestern Ghana: " +
  "the first image is a natural-color RGB composite, and the second is a SWIR " +
  "false-color composite (SWIR2, SWIR1, NIR) where bright areas indicate exposed " +
  "soil and mining disturbance. Using both views, describe any signs of illegal " +
  "small-scale gold mining (galamsey) activity: exposed soil, excavation pits, " +
  "sediment plumes, vegetation loss, and proximity to water bodies. " +
  "If no mining is visible, say so.";

export type Bbox = { label: string; bbox: [number, number, number, number] };

// transformers.js ships loose typings for multi-modal processors and generation
// output, so model/processor are untyped and call sites cast as needed.
type LoadedModel = { model: any; processor: any }; // eslint-disable-line @typescript-eslint/no-explicit-any
let modelPromise: Promise<LoadedModel> | null = null;

export function loadModel(onProgress?: (pct: number) => void): Promise<LoadedModel> {
  if (!modelPromise) {
    // Track bytes across ALL files so progress is smooth + monotonic instead of
    // resetting to 0 every time a new file starts downloading.
    const fileBytes = new Map<string, { loaded: number; total: number }>();

    modelPromise = (async () => {
      // transformers.js types progress as a discriminated union (ReadyProgressInfo
      // has no fields in common), so we take the callback as-any and duck-type
      // each event at runtime. The guards below only act on download events.
      const progressCallback = (info: unknown) => {
        if (!onProgress) return;
        const p = info as { file?: string; loaded?: number; total?: number; progress?: number };
        if (p.file && typeof p.loaded === "number" && typeof p.total === "number") {
          fileBytes.set(p.file, { loaded: p.loaded, total: p.total });
          let loaded = 0, total = 0;
          for (const f of fileBytes.values()) {
            loaded += f.loaded;
            total += f.total;
          }
          if (total > 0) onProgress(Math.min(1, loaded / total));
        } else if (typeof p.progress === "number") {
          onProgress(Math.min(1, p.progress / 100));
        }
      };

      // Prefer fp16 (80% recall on our eval); fall back to q4 (50%) if the user's
      // WebGPU adapter lacks the `shader-f16` feature. Some older Intel/AMD GPUs,
      // and some Chromium GPU-blocklist-override configurations, don't advertise it.
      let model;
      try {
        model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
          device: "webgpu",
          dtype: {
            vision_encoder: "fp16",
            embed_tokens: "fp16",
            decoder_model_merged: "fp16",
          },
          progress_callback: progressCallback,
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("does not support fp16")) {
          console.warn("[inference] fp16 unsupported on this GPU, falling back to q4 (lower recall)");
          fileBytes.clear();
          model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
            device: "webgpu",
            dtype: {
              vision_encoder: "q4",
              embed_tokens: "fp16",
              decoder_model_merged: "q4",
            },
            progress_callback: progressCallback,
          });
        } else {
          throw err;
        }
      }
      const processor = await AutoProcessor.from_pretrained(MODEL_ID);

      // CRITICAL: Liquid4All/onnx-export uploaded a preprocessor_config.json with
      // do_resize: true (wrong) and missing patch_size / max_num_patches / return_row_col_info.
      // With the broken config, input_ids balloon from ~285 to ~653 tokens, cross-attention
      // gets wrong vision positions, and the model confidently returns [] on real mining.
      // Patch to match the base LiquidAI/LFM2-VL-450M preprocessor the model was trained with.
      // See scripts/test_onnx_local.mjs for the diagnostic that proved this.
      // ImageProcessor's typed surface doesn't expose these dynamic fields,
      // but the runtime reads them. Cast is narrowly scoped.
      const ip = processor.image_processor as Record<string, unknown> | undefined;
      if (ip) {
        ip.do_resize = false;
        ip.patch_size = 16;
        ip.max_num_patches = 1024;
        ip.return_row_col_info = true;
        ip.default_to_square = true;
      }

      return { model, processor };
    })().catch((err) => {
      console.error("[inference] loadModel FAILED:", err);
      modelPromise = null; // allow retry on failure
      throw err;
    });
  }
  return modelPromise;
}

function parseBboxes(text: string): Bbox[] {
  const match = text.match(/\[[\s\S]*\]/);
  if (!match) return [];
  try {
    const parsed = JSON.parse(match[0]);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((it) => it && Array.isArray(it.bbox) && it.bbox.length === 4)
      .map((it) => {
        const c = it.bbox.map((x: number) => Math.max(0, Math.min(1, Number(x))));
        if (!(c[2] > c[0]) || !(c[3] > c[1])) return null;
        return { label: it.label ?? "mining_pit", bbox: c as [number, number, number, number] };
      })
      .filter((x): x is Bbox => x !== null);
  } catch {
    return [];
  }
}

function bboxIoU(a: [number, number, number, number], b: [number, number, number, number]): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  if (x2 <= x1 || y2 <= y1) return 0;
  const inter = (x2 - x1) * (y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

// Non-Maximum Suppression: when the model emits overlapping bboxes for the same pit,
// keep only the largest. Also drops sliver bboxes < MIN_AREA (relative units).
// No confidence scores from the VLM, so we rank by bbox area as a proxy for "stronger" detection.
const NMS_IOU_THRESHOLD = 0.5;
const MIN_BBOX_AREA = 0.0001; // ~1.6px² on a 128×128 patch, loosened from 0.0003 so we don't drop small/uncertain detections the model emitted at the edge of its confidence

function nmsBboxes(bboxes: Bbox[]): Bbox[] {
  const withArea = bboxes
    .map((b) => ({ b, area: (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]) }))
    .filter(({ area }) => area >= MIN_BBOX_AREA)
    .sort((x, y) => y.area - x.area);

  const kept: Bbox[] = [];
  for (const { b } of withArea) {
    const overlapsExisting = kept.some((k) => bboxIoU(b.bbox, k.bbox) >= NMS_IOU_THRESHOLD);
    if (!overlapsExisting) kept.push(b);
  }
  return kept;
}

// Runs a single prompt against the two input images and returns the decoded text.
async function generate(rgbUrl: string, swirUrl: string, prompt: string): Promise<string> {
  const t0 = performance.now();
  const { model, processor } = await loadModel();
  const tLoaded = performance.now();

  // Browser decodes PNGs as RGBA by default, we trained on RGB (3 channels).
  // Force 3-channel RGB to match training distribution; leaving the alpha channel
  // in silently corrupts the tensor the model sees and flips detections to [].
  const rgbRaw = await RawImage.fromURL(rgbUrl);
  const swirRaw = await RawImage.fromURL(swirUrl);
  const rgb = rgbRaw.rgb();
  const swir = swirRaw.rgb();
  const tImages = performance.now();

  const messages = [
    {
      role: "user",
      content: [
        { type: "image" },
        { type: "image" },
        { type: "text", text: prompt },
      ],
    },
  ];

  const chatPrompt = processor.apply_chat_template(messages, { add_generation_prompt: true });
  const inputs = await processor([rgb, swir], chatPrompt, { add_special_tokens: false });
  const tPreproc = performance.now();

  const outputs = await model.generate({
    ...inputs,
    do_sample: false,
    // Grounding JSON tops out around ~180 tokens in practice; description
    // around ~35. 256 is well above both; the old 384 was pure tail-latency
    // padding that cost a few seconds on runs that hit the ceiling.
    max_new_tokens: 256,
  });
  const tGenerate = performance.now();

  const inputLength = inputs.input_ids.dims.at(-1);
  const generated = outputs.slice(null, [inputLength, null]);
  const decoded = processor.batch_decode(generated, { skip_special_tokens: true })[0];

  if (process.env.NODE_ENV !== "production") {
    console.log(
      `[inference] load=${(tLoaded - t0).toFixed(0)}ms imgs=${(tImages - tLoaded).toFixed(0)}ms preproc=${(tPreproc - tImages).toFixed(0)}ms generate=${(tGenerate - tPreproc).toFixed(0)}ms tokens=${generated.dims.at(-1)}`,
    );
  }

  return decoded;
}

// Grounding pass: returns the raw JSON string (for the "Model output" panel)
// plus the parsed bboxes. UI calls describe() separately for the prose explanation.
export async function detect(
  rgbUrl: string,
  swirUrl: string,
): Promise<{ description: string; bboxes: Bbox[] }> {
  const text = await generate(rgbUrl, swirUrl, GROUNDING_PROMPT);
  const bboxes = parseBboxes(text);
  return { description: text, bboxes };
}

// Description pass: natural-language explanation (exposed soil, sediment plumes,
// vegetation loss, etc). Run AFTER detect() so bboxes appear first and prose
// streams in second. Batched inference was tested (scripts/test_onnx_batched.mjs)
// and produced corrupted descriptions + no speedup, so sequential it is.
export async function describe(rgbUrl: string, swirUrl: string): Promise<string> {
  return generate(rgbUrl, swirUrl, DESCRIPTION_PROMPT);
}

// Bitemporal change detection, marks each present bbox as "new" or "persistent"
// by matching against past-year bboxes. Matching is greedy by IoU, good enough
// for a demo; production would use Hungarian assignment.

export type ChangeBbox = Bbox & { status: "new" | "persistent" };

export type BitemporalResult = {
  pastYear: number;
  past: { description: string; bboxes: Bbox[]; rgbUrl: string; swirUrl: string };
  present: { description: string; bboxes: Bbox[]; rgbUrl: string; swirUrl: string };
  change: ChangeBbox[];
  presentProse?: string;
};

const MATCH_IOU = 0.3;

export function matchChange(pastBboxes: Bbox[], presentBboxes: Bbox[]): ChangeBbox[] {
  const remaining = [...pastBboxes];
  return presentBboxes.map((p) => {
    let bestIdx = -1;
    let bestIoU = 0;
    for (let i = 0; i < remaining.length; i++) {
      const iou = bboxIoU(p.bbox, remaining[i].bbox);
      if (iou > bestIoU) {
        bestIoU = iou;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0 && bestIoU >= MATCH_IOU) {
      remaining.splice(bestIdx, 1);
      return { ...p, status: "persistent" as const };
    }
    return { ...p, status: "new" as const };
  });
}

// Runs inference on just the past-year tile and returns the full bitemporal
// result. Takes the already-computed present-day result as input to avoid
// re-running inference on the current frame.
export async function detectPastOnly(
  lng: number,
  lat: number,
  present: { description: string; bboxes: Bbox[]; rgbUrl: string; swirUrl: string },
  pastYear = 2016,
): Promise<BitemporalResult> {
  const pastRgbUrl = `/api/simsat/sentinel?lon=${lng}&lat=${lat}&year=${pastYear}&bands=red,green,blue`;
  const pastSwirUrl = `/api/simsat/sentinel?lon=${lng}&lat=${lat}&year=${pastYear}&bands=swir22,swir16,nir`;
  const pastOut = await detect(pastRgbUrl, pastSwirUrl);
  return {
    pastYear,
    past: { ...pastOut, rgbUrl: pastRgbUrl, swirUrl: pastSwirUrl },
    present,
    change: matchChange(pastOut.bboxes, present.bboxes),
  };
}
