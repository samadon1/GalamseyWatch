"use client";

import { memo, useEffect, useRef, useState } from "react";
import type { LiveResult } from "@/app/dashboard/page";

const COMPARISON_YEARS = [2016, 2018, 2020, 2022];

// Multi-step progress messages for the two loading phases. Strings are flavor,
// not ground truth, actual fetch/inference runs independently. The ticker parks
// at the last step once reached, so if real work outlasts the scripted sequence
// the user just sees "Parsing bounding-box JSON…" until completion.
const FETCH_STEPS = [
  "Establishing uplink to SimSat…",
  "Locating latest Sentinel-2 overpass…",
  "Streaming RGB bands (B04 · B03 · B02)…",
  "Streaming SWIR bands (B12 · B11 · B8A)…",
  "Compositing dual-sensor view…",
];

const INFER_STEPS = [
  "Loading dual-sensor tensors into VRAM…",
  "Encoding image patches (224 × 224)…",
  "Running vision encoder on WebGPU…",
  "Fusing RGB + SWIR features…",
  "Cross-attending vision ↔ language…",
  "Decoding grounding tokens…",
  "Parsing bounding-box JSON…",
];

function useLoadingStepIndex(steps: readonly string[], intervalMs: number, active: boolean) {
  const [idx, setIdx] = useState(0);
  useEffect(() => {
    if (!active) {
      setIdx(0);
      return;
    }
    const id = setInterval(() => {
      setIdx((i) => (i < steps.length - 1 ? i + 1 : i));
    }, intervalMs);
    return () => clearInterval(id);
  }, [active, steps, intervalMs]);
  return idx;
}

interface DetectionPanelProps {
  liveResult: LiveResult | null;
  onRunBitemporal?: (year: number) => void;
  onExitBitemporal?: () => void;
  bitemporalLoadingYear?: number | null;
}

export default function DetectionPanel({
  liveResult,
  onRunBitemporal,
  onExitBitemporal,
  bitemporalLoadingYear,
}: DetectionPanelProps) {
  return (
    <LivePanel
      result={liveResult}
      onRunBitemporal={onRunBitemporal}
      onExitBitemporal={onExitBitemporal}
      bitemporalLoadingYear={bitemporalLoadingYear}
    />
  );
}

function LivePanel({
  result,
  onRunBitemporal,
  onExitBitemporal,
  bitemporalLoadingYear,
}: {
  result: LiveResult | null;
  onRunBitemporal?: (year: number) => void;
  onExitBitemporal?: () => void;
  bitemporalLoadingYear?: number | null;
}) {
  // Image modal state, shared across RGB/SWIR/past/present previews
  const [modal, setModal] = useState<{ src: string; title: string } | null>(null);
  const openModal = (src: string, title: string) => setModal({ src, title });
  useEffect(() => {
    if (!modal) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setModal(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [modal]);

  const fetchStepIdx = useLoadingStepIndex(FETCH_STEPS, 700, result?.status === "fetching");
  const inferStepIdx = useLoadingStepIndex(INFER_STEPS, 900, result?.status === "inferring");
  if (!result) {
    return (
      <div className="px-6 py-6 text-white/80">
        {/* Status header, mirrors the map-surface pattern */}
        <div className="mb-5 pb-4 border-b border-white/10">
          <div className="flex items-center justify-between mb-3 font-mono">
            <div className="flex items-center gap-2.5">
              <span className="h-1.5 w-1.5 bg-white" />
              <span className="text-[10px] uppercase tracking-[0.22em] text-white/70">
                Ready · Awaiting Input
              </span>
            </div>
            <span className="text-[10px] tracking-wider text-white/40">IDLE</span>
          </div>
          <h2 className="text-lg font-semibold text-white">
            Click anywhere on the map
          </h2>
          <p className="text-sm text-white/60 mt-1 leading-relaxed">
            We&apos;ll fetch a Sentinel-2 tile around that point, run our
            fine-tuned LFM2.5-VL-450M in your browser via WebGPU, and mark any
            detected galamsey pits.
          </p>
        </div>

        {/* Legend, rebuilt as a defense-console table */}
        <div className="mb-5 border border-white/10 font-mono">
          <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
            <span className="text-[10px] uppercase tracking-[0.22em] text-white/50">
              Legend · Symbols
            </span>
            <span className="text-[10px] tabular-nums text-white/40">05</span>
          </div>
          <ul className="divide-y divide-white/5">
            <LegendRow
              swatch={<span className="w-3.5 h-3.5 border-2 border-[#F5C518]" style={{ boxShadow: "0 0 6px rgba(245,197,24,0.45)" }} />}
              label="Detected mining pit"
              sub="Gold bbox on the current tile"
            />
            <LegendRow
              swatch={<span className="w-3.5 h-3.5 border-2 border-red-500" style={{ boxShadow: "0 0 6px rgba(239,68,68,0.45)" }} />}
              label="New since earlier imagery"
              sub="Pit exists today but not in the past tile"
            />
            <LegendRow
              swatch={<span className="w-3.5 h-3.5 border-2 border-white/80" />}
              label="Past-year detection"
              sub="Outline on the left side of the swipe"
            />
            <LegendRow
              swatch={<span className="w-2.5 h-2.5 rounded-full bg-[#F5C518]" />}
              label="Earthrise-verified site"
              sub="326 sites where v9 agrees with ground truth"
            />
            <LegendRow
              swatch={<span className="w-2.5 h-2.5 rounded-full bg-gray-400 opacity-50" />}
              label="Earthrise site · no detection"
              sub="Model didn't detect mining at a flagged site"
            />
          </ul>
        </div>

      </div>
    );
  }

  // Formatted coordinate readout: "5.11720°N, 2.24600°W" (padded, mono, N/S E/W)
  const latStr = `${Math.abs(result.lat).toFixed(5)}°${result.lat >= 0 ? "N" : "S"}`;
  const lngStr = `${Math.abs(result.lng).toFixed(5)}°${result.lng >= 0 ? "E" : "W"}`;

  const statusColor =
    result.status === "error" ? "bg-red-500" :
    result.status === "done" ? "bg-emerald-400" :
    "bg-amber-400";
  const statusLabel =
    result.status === "fetching" ? "ACQUIRING" :
    result.status === "inferring" ? "INFERENCE" :
    result.status === "error" ? "ERROR" :
    "LOCKED";

  return (
    <div className="pl-6 pr-7 py-6">
      {/* Header: status dot + label + coordinate readout.
          Hidden in comparison mode, the back button serves as the header there. */}
      {!result.bitemporal && (
        <div className="mb-5 pb-4 border-b border-gray-800">
          <div className="flex items-center gap-2 mb-3">
            <span className="relative flex h-2 w-2">
              {(result.status === "fetching" || result.status === "inferring") && (
                <span className={`absolute inline-flex h-full w-full rounded-full ${statusColor} opacity-75 animate-ping`} />
              )}
              <span className={`relative inline-flex rounded-full h-2 w-2 ${statusColor}`} />
            </span>
            <span className="text-[10px] tracking-[0.2em] text-gray-400 font-mono">
              LIVE INFERENCE · {statusLabel}
            </span>
          </div>
          <div className="font-mono text-lg text-gray-100 tabular-nums leading-tight">
            {latStr}
            <span className="text-gray-600">,</span> {lngStr}
          </div>

          {/* Telemetry chips */}
          <div className="mt-3 flex items-center gap-2 flex-wrap text-[10px] font-mono tracking-wider">
            {typeof result.cloudCover === "number" && (
              <span className="px-2 py-0.5 rounded-sm bg-gray-800 border border-gray-700 text-gray-300">
                CLOUD {result.cloudCover.toFixed(0)}%
              </span>
            )}
            <span className="px-2 py-0.5 rounded-sm bg-gray-800 border border-gray-700 text-gray-300">
              1.28 KM
            </span>
            <span className="px-2 py-0.5 rounded-sm bg-[#F5C518]/10 border border-[#F5C518]/30 text-[#F5C518]">
              V9-E3
            </span>
          </div>
        </div>
      )}

      {result.status === "fetching" && (
        <StepperStatus steps={FETCH_STEPS} currentIdx={fetchStepIdx} />
      )}
      {result.status === "inferring" && (
        <StepperStatus steps={INFER_STEPS} currentIdx={inferStepIdx} />
      )}
      {result.status === "error" && (
        <StatusBox color="red" label={`Error: ${result.error ?? "unknown"}`} />
      )}

      {/* Dual-sensor view: RGB + SWIR side-by-side, click to expand.
          Hidden in comparison mode, the swipe pane replaces it. */}
      {(result.status === "inferring" || result.status === "done") && !result.bitemporal && (
        <div className="mb-4">
          <p className="text-[10px] text-gray-500 uppercase tracking-[0.2em] mb-2 font-mono">
            Dual-sensor view · click to expand
          </p>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => openModal(result.rgbUrl, "RGB · natural color")}
              className="block text-left cursor-zoom-in"
            >
              <p className="text-[10px] text-gray-500 mb-1 font-mono tracking-wider">RGB</p>
              <RgbWithBboxes rgbSrc={result.rgbUrl} bboxes={result.bboxes} compact />
            </button>
            <button
              type="button"
              onClick={() => openModal(result.swirUrl, "SWIR · false color (SWIR2 · SWIR1 · NIR)")}
              className="block text-left cursor-zoom-in"
            >
              <p className="text-[10px] text-gray-500 mb-1 font-mono tracking-wider">SWIR</p>
              <SwirPreview src={result.swirUrl} compact />
            </button>
          </div>
        </div>
      )}

      {/* Image modal, 128x128 native source gets pixelated-upscaled to fill the viewport */}
      {modal && (
        <div
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-6"
          onClick={() => setModal(null)}
        >
          <div
            className="relative flex flex-col items-center"
            style={{ width: "min(85vh, 85vw, 768px)" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="w-full flex items-center justify-between mb-3 text-xs font-mono tracking-wider text-gray-300 gap-4">
              <span className="uppercase">{modal.title}</span>
              <button
                onClick={() => setModal(null)}
                className="text-gray-400 hover:text-white whitespace-nowrap"
                aria-label="Close"
              >
                ESC ✕
              </button>
            </div>
            <div className="w-full aspect-square rounded-sm border border-gray-700 shadow-2xl overflow-hidden bg-gray-800">
              <img
                src={modal.src}
                alt={modal.title}
                className="w-full h-full object-cover"
                style={{ imageRendering: "pixelated" }}
                draggable={false}
              />
            </div>
            <p className="mt-3 text-[10px] text-gray-500 font-mono tracking-[0.15em] uppercase self-start">
              Sentinel-2 · 128 × 128 PX · 10 m/px native resolution
            </p>
          </div>
        </div>
      )}

      {result.status === "done" && result.bitemporal && (
        <TimelineComparison
          bt={result.bitemporal}
          onYearSelect={onRunBitemporal}
          loadingYear={bitemporalLoadingYear ?? null}
          onExit={onExitBitemporal}
        />
      )}

      {result.status === "done" && !result.bitemporal && (
        <>
          {result.bboxes.length === 0 ? (
            <div className="mb-4 border border-emerald-500/30 font-mono">
              <div className="flex items-center justify-between px-3 py-2 border-b border-emerald-500/20">
                <div className="flex items-center gap-2.5">
                  <svg
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="text-emerald-400 flex-shrink-0"
                  >
                    <path d="M20 6 9 17l-5-5" />
                  </svg>
                  <span className="text-[10px] uppercase tracking-[0.22em] text-emerald-400">
                    Result · Clear
                  </span>
                </div>
                <span className="text-[10px] tabular-nums text-emerald-400/60">0 BBOX</span>
              </div>
              <div className="px-3 py-2.5">
                <p className="text-sm text-white font-medium">No mining detected</p>
                <p className="text-xs text-white/50 mt-1 leading-relaxed">
                  Model found no galamsey pits in this tile.
                </p>
              </div>
            </div>
          ) : (
            <>
              {/* Natural-language description (2nd inference, fills in after bboxes) */}
              <div className="mb-4 p-4 rounded-lg bg-[#F5C518]/5 border border-[#F5C518]/20">
                <p className="text-xs text-[#F5C518] uppercase tracking-wider mb-2">
                  Analysis
                </p>
                {result.proseStatus === "pending" || !result.prose ? (
                  <div className="flex items-center gap-2 text-sm text-gray-400">
                    <div className="animate-spin h-3 w-3 border-2 border-[#F5C518] border-t-transparent rounded-full" />
                    Generating description…
                  </div>
                ) : result.proseStatus === "error" ? (
                  <p className="text-sm text-gray-500 italic">
                    Description unavailable.
                  </p>
                ) : (
                  <p className="text-sm text-gray-100 leading-relaxed">
                    {result.prose}
                  </p>
                )}
              </div>

              {/* Raw JSON, collapsed by default */}
              <details className="mb-4 rounded-sm bg-gray-900 border border-gray-800 group">
                <summary className="px-4 py-2 text-[10px] text-gray-500 uppercase tracking-[0.2em] font-mono cursor-pointer hover:text-gray-300 transition-colors list-none flex items-center justify-between">
                  <span>Raw model output · {result.bboxes.length} bbox</span>
                  <span className="text-gray-600 group-open:rotate-180 transition-transform">⌄</span>
                </summary>
                <div className="px-4 pb-3 pt-1 border-t border-gray-800">
                  <p className="text-xs text-gray-400 leading-relaxed whitespace-pre-wrap break-all font-mono">
                    {result.description}
                  </p>
                </div>
              </details>
            </>
          )}

          {onRunBitemporal && !result.bitemporal && (
            <div className="mt-4 border border-white/10 font-mono">
              <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
                <span className="text-[10px] uppercase tracking-[0.22em] text-white/50">
                  Temporal · Compare
                </span>
                <span className="text-[10px] tabular-nums text-white/40">
                  {String(COMPARISON_YEARS.length).padStart(2, "0")} EPOCHS
                </span>
              </div>
              <div className="px-4 pt-5 pb-4 relative">
                {/* Horizontal rail spanning all ticks */}
                <div className="absolute left-6 right-6 top-[calc(1.25rem+5px)] h-px bg-white/15" />
                {/* Left/right endcaps, |  ──  | visual hint of a closed scale */}
                <div className="absolute left-4 top-[calc(1.25rem)] h-2.5 w-px bg-white/20" />
                <div className="absolute right-4 top-[calc(1.25rem)] h-2.5 w-px bg-white/20" />

                <div className="flex justify-between relative">
                  {COMPARISON_YEARS.map((y) => {
                    const isLoading = bitemporalLoadingYear === y;
                    const delta = new Date().getFullYear() - y;
                    return (
                      <button
                        key={y}
                        onClick={() => onRunBitemporal(y)}
                        disabled={bitemporalLoadingYear !== null}
                        className="group flex flex-col items-center gap-2 disabled:opacity-40 disabled:cursor-wait"
                      >
                        {/* Tick marker sitting on the rail */}
                        {isLoading ? (
                          <span className="animate-spin h-3 w-3 border-2 border-white border-t-transparent rounded-full flex-shrink-0" />
                        ) : (
                          <span className="w-2.5 h-2.5 bg-white/40 group-hover:bg-white transition-colors flex-shrink-0" />
                        )}
                        <span className="text-[11px] text-white tabular-nums tracking-wider">
                          {y}
                        </span>
                        <span className="text-[9px] text-white/40 uppercase tracking-wider">
                          Δ {delta}YR
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// Swipe/split comparison: vertical divider across the image. Left side shows the
// past tile + past bboxes (white outlines); right side shows the present tile +
// present bboxes (red for new, gold for persistent). Drag anywhere in the image
// to move the divider.
function TimelineComparison({
  bt,
  onYearSelect,
  loadingYear,
  onExit,
}: {
  bt: NonNullable<LiveResult["bitemporal"]>;
  onYearSelect?: (year: number) => void;
  loadingYear: number | null;
  onExit?: () => void;
}) {
  // Divider position as % from the left edge (0 = all present, 100 = all past).
  // Default 50 splits the image in half.
  const [split, setSplit] = useState(50);
  const [band, setBand] = useState<"rgb" | "swir">("rgb");
  const [dragging, setDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const pastSrc = band === "swir" ? bt.past.swirUrl : bt.past.rgbUrl;
  const presentSrc = band === "swir" ? bt.present.swirUrl : bt.present.rgbUrl;

  // Window-level drag listeners, prevent the map from grabbing the pointer when
  // the user's drag crosses out of the sidebar bounds during a swipe gesture.
  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: PointerEvent) => {
      e.preventDefault();
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      setSplit(Math.max(0, Math.min(100, x * 100)));
    };
    const onUp = () => setDragging(false);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [dragging]);
  const newCount = bt.change.filter((c) => c.status === "new").length;
  const persistCount = bt.change.filter((c) => c.status === "persistent").length;

  function handleDragStart(e: React.PointerEvent) {
    e.preventDefault();
    e.stopPropagation();
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    setSplit(Math.max(0, Math.min(100, x * 100)));
    setDragging(true);
  }

  return (
    <div className="space-y-4">
      {/* Back button, exits comparison and restores dual-sensor view */}
      {onExit && (
        <button
          onClick={onExit}
          className="flex items-center gap-2 text-[10px] font-mono tracking-[0.2em] text-gray-400 hover:text-white transition-colors uppercase"
        >
          <span className="text-sm">←</span> Back to detection
        </button>
      )}

      {/* Year picker, timeline strip matching the pre-compare view */}
      {onYearSelect && (
        <div className="border border-white/10 font-mono">
          <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
            <span className="text-[10px] uppercase tracking-[0.22em] text-white/50">
              Temporal · Compare
            </span>
            <span className="text-[10px] tabular-nums text-white/40">
              ACTIVE · {bt.pastYear}
            </span>
          </div>
          <div className="px-4 pt-5 pb-4 relative">
            <div className="absolute left-6 right-6 top-[calc(1.25rem+5px)] h-px bg-white/15" />
            <div className="absolute left-4 top-[calc(1.25rem)] h-2.5 w-px bg-white/20" />
            <div className="absolute right-4 top-[calc(1.25rem)] h-2.5 w-px bg-white/20" />
            <div className="flex justify-between relative">
              {COMPARISON_YEARS.map((y) => {
                const isActive = bt.pastYear === y;
                const isLoading = loadingYear === y;
                const delta = new Date().getFullYear() - y;
                return (
                  <button
                    key={y}
                    onClick={() => onYearSelect(y)}
                    disabled={loadingYear !== null}
                    className="group flex flex-col items-center gap-2 disabled:opacity-40 disabled:cursor-wait"
                  >
                    {isLoading ? (
                      <span className="animate-spin h-3 w-3 border-2 border-white border-t-transparent rounded-full flex-shrink-0" />
                    ) : (
                      <span
                        className={`w-2.5 h-2.5 transition-colors flex-shrink-0 ${
                          isActive ? "bg-white" : "bg-white/40 group-hover:bg-white"
                        }`}
                      />
                    )}
                    <span
                      className={`text-[11px] tabular-nums tracking-wider ${
                        isActive ? "text-white font-semibold" : "text-white/70"
                      }`}
                    >
                      {y}
                    </span>
                    <span className="text-[9px] text-white/40 uppercase tracking-wider">
                      Δ {delta}YR
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Change summary, mission-console delta readout */}
      <div className="border border-white/10 font-mono">
        <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
          <span className="text-[10px] uppercase tracking-[0.22em] text-white/50">
            Epoch · Delta
          </span>
          <span className="text-[10px] tabular-nums text-white/50">
            {bt.pastYear} → NOW
          </span>
        </div>
        <div className="px-3 py-2.5 flex items-center gap-6 text-sm tabular-nums">
          <div className="flex items-baseline gap-1.5">
            <span className="text-red-400 font-semibold">
              {String(newCount).padStart(2, "0")}
            </span>
            <span className="text-[10px] uppercase tracking-[0.22em] text-white/50">
              New
            </span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-[#F5C518] font-semibold">
              {String(persistCount).padStart(2, "0")}
            </span>
            <span className="text-[10px] uppercase tracking-[0.22em] text-white/50">
              Persistent
            </span>
          </div>
        </div>
      </div>

      {/* Swipe comparison pane, drag anywhere to move the vertical divider */}
      <div>
        {/* RGB ↔ SWIR band toggle */}
        <div className="mb-2 inline-flex border border-white/15 bg-gray-900 overflow-hidden text-[10px] font-mono tracking-[0.22em] uppercase">
          <button
            onClick={() => setBand("rgb")}
            className={`px-3 py-1 transition-colors ${
              band === "rgb" ? "bg-white text-black" : "text-white/60 hover:text-white"
            }`}
          >
            RGB
          </button>
          <button
            onClick={() => setBand("swir")}
            className={`px-3 py-1 transition-colors border-l border-white/15 ${
              band === "swir" ? "bg-white text-black" : "text-white/60 hover:text-white"
            }`}
          >
            SWIR
          </button>
        </div>
        <div
          ref={containerRef}
          onPointerDown={handleDragStart}
          className="relative overflow-hidden border border-white/15 bg-gray-800 aspect-square cursor-ew-resize select-none touch-none"
        >
          {/* Left side (past), clipped to show only left of the divider */}
          <div
            className="absolute inset-0"
            style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
          >
            <img
              src={pastSrc}
              alt={`${bt.pastYear} ${band.toUpperCase()}`}
              className="w-full h-full object-cover"
              draggable={false}
            />
            {bt.past.bboxes.map((b, i) => {
              const [x1, y1, x2, y2] = b.bbox;
              return (
                <div
                  key={`past-${i}`}
                  className="absolute border-2 border-white/80 pointer-events-none"
                  style={{
                    left: `${x1 * 100}%`,
                    top: `${y1 * 100}%`,
                    width: `${(x2 - x1) * 100}%`,
                    height: `${(y2 - y1) * 100}%`,
                    boxShadow: "0 0 6px rgba(255,255,255,0.4)",
                  }}
                />
              );
            })}
            <div className="absolute top-2 left-2 px-2 py-1 bg-black/80 border border-white/15 text-[10px] font-mono tracking-[0.22em] uppercase text-white">
              {bt.pastYear}
            </div>
          </div>

          {/* Right side (present), clipped to show only right of the divider */}
          <div
            className="absolute inset-0"
            style={{ clipPath: `inset(0 0 0 ${split}%)` }}
          >
            <img
              src={presentSrc}
              alt={`Present ${band.toUpperCase()}`}
              className="w-full h-full object-cover"
              draggable={false}
            />
            {bt.change.map((b, i) => {
              const [x1, y1, x2, y2] = b.bbox;
              const isNew = b.status === "new";
              return (
                <div
                  key={`pres-${i}`}
                  className={`absolute border-2 pointer-events-none ${
                    isNew ? "border-red-500" : "border-[#F5C518]"
                  }`}
                  style={{
                    left: `${x1 * 100}%`,
                    top: `${y1 * 100}%`,
                    width: `${(x2 - x1) * 100}%`,
                    height: `${(y2 - y1) * 100}%`,
                    boxShadow: isNew
                      ? "0 0 12px rgba(239, 68, 68, 0.6)"
                      : "0 0 8px rgba(245, 197, 24, 0.4)",
                  }}
                >
                  <span
                    className={`absolute -top-5 left-0 text-[10px] font-mono px-1 rounded-sm ${
                      isNew ? "text-red-400 bg-black/80" : "text-[#F5C518] bg-black/80"
                    }`}
                  >
                    {isNew ? "NEW" : "#" + (i + 1)}
                  </span>
                </div>
              );
            })}
            <div className="absolute top-2 right-2 px-2 py-1 bg-black/80 border border-white/15 text-[10px] font-mono tracking-[0.22em] uppercase text-white">
              Present
            </div>
          </div>

          {/* Vertical divider line + drag handle */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white/80 pointer-events-none shadow-[0_0_6px_rgba(0,0,0,0.6)]"
            style={{ left: `${split}%`, transform: "translateX(-50%)" }}
          >
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white/90 flex items-center justify-center shadow-lg">
              <span className="text-black text-xs font-bold select-none">⇔</span>
            </div>
          </div>
        </div>

        <p className="mt-2 text-[10px] text-white/40 text-center font-mono uppercase tracking-[0.22em]">
          Drag to swipe · {bt.pastYear} ↔ Present
        </p>
      </div>

      {/* Legend, single row, mono, matches the Earthrise Index row grammar */}
      <div className="border border-white/10 font-mono">
        <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
          <span className="text-[10px] uppercase tracking-[0.22em] text-white/50">
            Legend · Overlay
          </span>
        </div>
        <div className="px-3 py-2 flex items-center gap-5 flex-wrap text-[10px] uppercase tracking-wider">
          <div className="flex items-center gap-2 text-white/70">
            <span className="w-2.5 h-2.5 border-2 border-red-500 inline-block" />
            New Since {bt.pastYear}
          </div>
          <div className="flex items-center gap-2 text-white/70">
            <span className="w-2.5 h-2.5 border-2 border-[#F5C518] inline-block" />
            Persistent
          </div>
          <div className="flex items-center gap-2 text-white/70">
            <span className="w-2.5 h-2.5 border-2 border-white/70 inline-block" />
            {bt.pastYear} Detection
          </div>
        </div>
      </div>

      {/* Present analysis, titled card, white mono header (no gold tint) */}
      {bt.presentProse && (
        <div className="border border-white/10 font-mono">
          <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
            <span className="text-[10px] uppercase tracking-[0.22em] text-white/50">
              Present · Analysis
            </span>
            <span className="text-[10px] tabular-nums text-white/40">NLP</span>
          </div>
          <p className="px-3 py-2.5 text-sm text-white/85 leading-relaxed font-sans">
            {bt.presentProse}
          </p>
        </div>
      )}
    </div>
  );
}

function StatusBox({
  color,
  label,
}: {
  color: "amber" | "red";
  label: string;
}) {
  const cls =
    color === "amber"
      ? "bg-amber-900/20 border-amber-700/30 text-amber-400"
      : "bg-red-900/20 border-red-700/30 text-red-400";
  return (
    <div className={`mb-4 p-3 rounded-lg border flex items-center gap-3 ${cls}`}>
      <div
        className={`animate-spin h-4 w-4 border-2 border-t-transparent rounded-full ${
          color === "amber" ? "border-amber-500" : "border-red-500"
        }`}
      />
      <p className="text-xs">{label}</p>
    </div>
  );
}

// Palantir/defense-console stepper. Sharp rectangles, mono uppercase, numeric
// line prefixes, and a [XX/YY] sequence counter. Completed steps sit at 40%
// opacity with a check; the active step has a blinking hairline spinner.
function StepperStatus({
  steps,
  currentIdx,
}: {
  steps: readonly string[];
  currentIdx: number;
}) {
  const total = steps.length;
  const visible = steps.slice(0, currentIdx + 1);
  return (
    <div className="mb-4 border border-white/10 bg-white/[0.02] font-mono">
      {/* Header: label left, step counter right, hairline separator under */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
        <span className="text-[10px] uppercase tracking-[0.2em] text-white/50">
          Sequence
        </span>
        <span className="text-[10px] tabular-nums text-white/70">
          {String(currentIdx + 1).padStart(2, "0")} / {String(total).padStart(2, "0")}
        </span>
      </div>

      <ul className="divide-y divide-white/5">
        {visible.map((step, i) => {
          const isCurrent = i === currentIdx;
          return (
            <li
              key={step}
              className="flex items-center gap-3 px-3 py-2 text-[11px] uppercase tracking-wider"
            >
              {/* Numeric line prefix, 001, 002, …  */}
              <span className="tabular-nums text-white/30 w-8">
                {String(i + 1).padStart(3, "0")}
              </span>

              {/* Status glyph */}
              {isCurrent ? (
                <span className="animate-spin h-3 w-3 border border-white border-t-transparent rounded-full flex-shrink-0" />
              ) : (
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="text-white/50 flex-shrink-0"
                >
                  <path d="M20 6 9 17l-5-5" />
                </svg>
              )}

              {/* Step label */}
              <span className={`flex-1 ${isCurrent ? "text-white" : "text-white/40"}`}>
                {step.replace(/…$/, "")}
              </span>

              {/* Right-side status: OK on done rows, RUN on current */}
              <span
                className={`text-[9px] tabular-nums tracking-wider ${
                  isCurrent ? "text-white/70" : "text-white/30"
                }`}
              >
                {isCurrent ? "RUN" : "OK"}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function LegendRow({
  swatch,
  label,
  sub,
}: {
  swatch: React.ReactNode;
  label: string;
  sub: string;
}) {
  return (
    <li className="flex items-start gap-3 px-3 py-2">
      <div className="shrink-0 mt-0.5 flex items-center justify-center w-4 h-4">
        {swatch}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-[11px] uppercase tracking-wider text-white/80">{label}</p>
        <p className="text-[11px] text-white/45 leading-relaxed mt-0.5">{sub}</p>
      </div>
    </li>
  );
}

const RgbWithBboxes = memo(function RgbWithBboxes({
  rgbSrc,
  bboxes,
  compact,
}: {
  rgbSrc: string;
  bboxes: Array<{ bbox: [number, number, number, number] }>;
  compact?: boolean;
}) {
  const inner = (
    <div className="relative rounded-sm overflow-hidden border border-gray-700 bg-gray-800 aspect-square">
      <img src={rgbSrc} alt="RGB composite" className="w-full h-full object-cover" draggable={false} />
      {bboxes.map((b, i) => {
        const [x1, y1, x2, y2] = b.bbox;
        return (
          <div
            key={i}
            className="absolute border-2 border-[#F5C518] pointer-events-none"
            style={{
              left: `${x1 * 100}%`,
              top: `${y1 * 100}%`,
              width: `${(x2 - x1) * 100}%`,
              height: `${(y2 - y1) * 100}%`,
              boxShadow: "0 0 10px rgba(245, 197, 24, 0.5)",
            }}
          >
            {!compact && (
              <span className="absolute -top-5 left-0 text-[10px] font-mono text-[#F5C518] bg-black/80 px-1 rounded-sm">
                #{i + 1}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );

  if (compact) return inner;
  return (
    <div className="mb-4">
      <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
        RGB composite · model bboxes
      </p>
      {inner}
    </div>
  );
});

const SwirPreview = memo(function SwirPreview({ src, compact }: { src: string; compact?: boolean }) {
  const img = (
    <div className="relative rounded-sm overflow-hidden border border-gray-700 bg-gray-800 aspect-square">
      <img src={src} alt="SWIR composite" className="w-full h-full object-cover" draggable={false} />
    </div>
  );
  if (compact) return img;
  return (
    <div className="mb-4">
      <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
        SWIR false-color (SWIR2 · SWIR1 · NIR)
      </p>
      {img}
    </div>
  );
});

