"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import DetectionPanel from "@/components/DetectionPanel";
import AgentPanel from "@/components/AgentPanel";
import MapSearch from "@/components/MapSearch";
import { describe, detect, detectPastOnly, loadModel, type Bbox, type BitemporalResult } from "@/lib/inference";

const Map = dynamic(() => import("@/components/Map"), { ssr: false });

export type DemoPatch = {
  slug: string;
  rgb: string;
  swir: string;
  lat: number;
  lng: number;
  description: string;
  bboxes: Array<{ label: string; bbox: [number, number, number, number] }>;
};

export type LiveResult = {
  lat: number;
  lng: number;
  rgbUrl: string;
  swirUrl: string;
  description: string;         // raw JSON from grounding prompt (for Model output block)
  bboxes: Bbox[];
  prose?: string;              // natural-language description from DESCRIPTION_PROMPT
  proseStatus?: "pending" | "done" | "error";
  cloudCover?: number;
  capturedAt?: string;
  status: "fetching" | "inferring" | "done" | "error";
  error?: string;
  bitemporal?: BitemporalResult | null;
};

export default function DashboardPage() {
  const [liveResult, setLiveResult] = useState<LiveResult | null>(null);
  const [modelProgress, setModelProgress] = useState<number>(0);
  const [flyTo, setFlyTo] = useState<{ lng: number; lat: number } | null>(null);
  const [bitemporalLoadingYear, setBitemporalLoadingYear] = useState<number | null>(null);
  const [modelReady, setModelReady] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [showTip, setShowTip] = useState(false);
  const [miningSites, setMiningSites] = useState<
    Array<{ id: number; lng: number; lat: number; status?: string }>
  >([]);
  const [paneMode, setPaneMode] = useState<"explore" | "agent">("explore");

  // Auto-open the mobile results drawer when a new detection starts.
  // Desktop ignores sheetOpen (sidebar is always visible at md+).
  useEffect(() => {
    if (liveResult) setSheetOpen(true);
  }, [liveResult?.lat, liveResult?.lng]);

  // On mobile the sidebar is hidden until a detection runs, so users never see
  // the "click the map" instructions. Surface them as a dismissible tip once
  // the WebGPU model is ready. Dismiss once per session; a refresh re-shows it.
  useEffect(() => {
    if (modelReady && !liveResult) setShowTip(true);
  }, [modelReady, liveResult]);

  useEffect(() => {
    fetch("/mining_sites.json")
      .then((r) => r.json())
      .then(setMiningSites)
      .catch((e) => console.error("Failed to load mining sites", e));
  }, []);

  // Pre-warm the WebGPU model in the background after page load.
  // Starts downloading the ONNX weights while the user explores curated mode,
  // so by the time they switch to Live, the model is already warm.
  useEffect(() => {
    // Fail-fast pre-flight: no point downloading 1 GB of weights if the browser
    // can't run WebGPU at all (Firefox mobile, older Safari, corp-locked Chrome).
    if (typeof navigator !== "undefined" && !(navigator as Navigator & { gpu?: unknown }).gpu) {
      setModelError("no-webgpu");
      return;
    }

    const idle =
      typeof window !== "undefined" && "requestIdleCallback" in window
        ? window.requestIdleCallback
        : (cb: () => void) => setTimeout(cb, 800);
    const handle = idle(() => {
      loadModel((p) => setModelProgress(p))
        .then(() => setModelReady(true))
        .catch((err) => {
          console.error("WebGPU model load failed", err);
          setModelError("load-failed");
        });
    });
    return () => {
      if (typeof window !== "undefined" && "cancelIdleCallback" in window) {
        window.cancelIdleCallback(handle as number);
      }
    };
  }, []);

  // Tracks the most recent click. Detection takes 15-45s; if the user clicks
  // again during that window (or Mapbox fires a stray second event) we'd race
  // two inferences and the second's result could arrive first or overwrite
  // the first mid-flow, causing visually-changing bboxes. Each click bumps
  // this ID; inference paths check whether they're still the latest before
  // writing state.
  const clickIdRef = useRef(0);

  async function handleMapClick(lng: number, lat: number) {
    if (!modelReady) return; // block clicks until WebGPU model loaded
    // A click is an explicit "inspect this tile" intent, surface the result in
    // Explore even if the user was viewing Agent Mode.
    setPaneMode("explore");

    const myId = ++clickIdRef.current;
    const rgbUrl = `/api/simsat/sentinel?lon=${lng}&lat=${lat}&bands=red,green,blue`;
    const swirUrl = `/api/simsat/sentinel?lon=${lng}&lat=${lat}&bands=swir22,swir16,nir`;

    setLiveResult({
      lat,
      lng,
      rgbUrl,
      swirUrl,
      description: "",
      bboxes: [],
      status: "fetching",
      bitemporal: null,
    });

    try {
      // Fetch both tiles ONCE as blobs, then pass object URLs to detect().
      // Previously we fetched rgb for metadata here, and RawImage.fromURL re-fetched
      // both inside detect(), three total hits to a no-store proxy, risking divergent
      // tile selection on the second call. Now: two fetches total, decoded once.
      const [rgbRes, swirRes] = await Promise.all([fetch(rgbUrl), fetch(swirUrl)]);

      const meta = rgbRes.headers.get("sentinel-metadata");
      let cloudCover: number | undefined;
      let capturedAt: string | undefined;
      if (meta) {
        try {
          const m = JSON.parse(meta);
          cloudCover = m.cloud_cover;
          capturedAt = m.datetime;
        } catch {}
      }

      const [rgbBlob, swirBlob] = await Promise.all([rgbRes.blob(), swirRes.blob()]);
      const rgbObjectUrl = URL.createObjectURL(rgbBlob);
      const swirObjectUrl = URL.createObjectURL(swirBlob);

      // If a newer click happened while we were fetching tiles, drop this one.
      if (clickIdRef.current !== myId) {
        URL.revokeObjectURL(rgbObjectUrl);
        URL.revokeObjectURL(swirObjectUrl);
        return;
      }

      setLiveResult((prev) =>
        prev ? { ...prev, status: "inferring", cloudCover, capturedAt } : prev,
      );

      const out = await detect(rgbObjectUrl, swirObjectUrl);

      // Gate the write on still-latest-click. Without this, a stale detect()
      // from an earlier click can overwrite a newer click's bboxes.
      if (clickIdRef.current !== myId) {
        URL.revokeObjectURL(rgbObjectUrl);
        URL.revokeObjectURL(swirObjectUrl);
        return;
      }

      setLiveResult((prev) =>
        prev
          ? {
              ...prev,
              status: "done",
              description: out.description,
              bboxes: out.bboxes,
              proseStatus: "pending",
            }
          : prev,
      );

      // Second pass: natural-language description. Runs after bboxes already rendered
      // so the user sees detections immediately; prose fills in a few seconds later.
      try {
        const prose = await describe(rgbObjectUrl, swirObjectUrl);
        if (clickIdRef.current !== myId) return; // stale, newer click is in flight
        setLiveResult((prev) =>
          prev ? { ...prev, prose, proseStatus: "done" } : prev,
        );
      } catch (err) {
        console.error("describe() failed", err);
        if (clickIdRef.current !== myId) return;
        setLiveResult((prev) =>
          prev ? { ...prev, proseStatus: "error" } : prev,
        );
      } finally {
        URL.revokeObjectURL(rgbObjectUrl);
        URL.revokeObjectURL(swirObjectUrl);
      }
    } catch (err) {
      if (clickIdRef.current !== myId) return; // stale click error, suppress
      const message = err instanceof Error ? err.message : String(err);
      setLiveResult((prev) =>
        prev ? { ...prev, status: "error", error: message } : prev,
      );
    }
  }

  // User picks a year from the timeline, runs only the past-year inference
  // (present-day is already cached in liveResult). Adds just +1 inference per year.
  async function runBitemporal(pastYear: number) {
    if (!liveResult || liveResult.status !== "done" || bitemporalLoadingYear !== null) return;
    setBitemporalLoadingYear(pastYear);
    try {
      const bt = await detectPastOnly(
        liveResult.lng,
        liveResult.lat,
        {
          description: liveResult.description,
          bboxes: liveResult.bboxes,
          rgbUrl: liveResult.rgbUrl,
          swirUrl: liveResult.swirUrl,
        },
        pastYear,
      );
      // Reuse the present prose we already cached, no extra inference needed
      const btWithProse = { ...bt, presentProse: liveResult.prose };
      setLiveResult((prev) => (prev ? { ...prev, bitemporal: btWithProse } : prev));
    } catch (err) {
      console.error("Bitemporal detection failed", err);
    } finally {
      setBitemporalLoadingYear(null);
    }
  }

  return (
    <main className="md:flex h-screen bg-gray-950 relative">
      <div className="h-screen md:h-auto md:flex-1 relative">
        <Map
          patches={[]}
          selectedSlug={null}
          onSelect={() => {}}
          onMapClick={handleMapClick}
          liveMarker={
            liveResult ? { lat: liveResult.lat, lng: liveResult.lng } : null
          }
          flyTo={flyTo}
          miningSites={miningSites}
        />

        <div className="absolute top-4 left-4 z-10">
          <div className="pointer-events-none mb-3">
            <h1 className="text-2xl font-bold text-white drop-shadow-lg">
              GalamseyWatch
            </h1>
            <p className="text-sm text-gray-300 drop-shadow">
              Click anywhere to run WebGPU inference on a Sentinel-2 tile
            </p>
          </div>
          <MapSearch
            onSelect={(p) => {
              setFlyTo({ lng: p.lng, lat: p.lat });
              // Town search: just fly there, let the user pick a specific mining pit
              // by clicking on the map. Coordinate paste: run inference immediately
              // (explicit "inspect this point" intent).
              if (p.type === "coord") handleMapClick(p.lng, p.lat);
            }}
          />
        </div>

        {/* Mining sites legend, hidden on mobile, it collides with the bottom drawer */}
        {miningSites.length > 0 && (
          <div className="hidden md:block absolute bottom-4 left-4 z-10 bg-gray-900/90 backdrop-blur border border-white/15 font-mono text-xs text-white/70 min-w-[260px]">
            <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
              <span className="text-[10px] uppercase tracking-[0.22em] text-white/50">
                Earthrise Index
              </span>
              <span className="text-[10px] tabular-nums text-white/70">
                N = {miningSites.length}
              </span>
            </div>
            <ul className="divide-y divide-white/5">
              {(() => {
                const agree = miningSites.filter((s) => s.status === "confirmed").length;
                const miss = miningSites.filter((s) => s.status === "empty").length;
                const unverif = miningSites.filter(
                  (s) => !s.status || s.status === "unverified" || s.status === "fetch_error",
                ).length;
                const rows = [
                  { dot: "bg-[#F5C518]", label: "v9 Agrees", count: agree, code: "AGREE" },
                  { dot: "bg-gray-500 opacity-60", label: "No Detection", count: miss, code: "NO_DET" },
                  { dot: "bg-gray-400 opacity-45", label: "Unverified", count: unverif, code: "UNVERIF" },
                ];
                return rows.map((r) => (
                  <li
                    key={r.code}
                    className="flex items-center gap-3 px-3 py-2 uppercase tracking-wider"
                  >
                    <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${r.dot}`} />
                    <span className="flex-1 text-[11px] text-white/80">{r.label}</span>
                    <span className="tabular-nums text-white text-[11px]">
                      {String(r.count).padStart(3, "0")}
                    </span>
                  </li>
                ));
              })()}
            </ul>
          </div>
        )}

        {/* Mobile onboarding tip, shown once the WebGPU model is ready,
            since the sidebar (which carries the same instructions on desktop)
            is hidden until a detection runs. Dismissible; session-scoped. */}
        {showTip && modelReady && !liveResult && (
          <div className="md:hidden absolute bottom-6 left-4 right-4 z-20 bg-gray-900/95 backdrop-blur border border-white/15 shadow-2xl font-mono">
            {/* Header row: status code + dismiss */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/10">
              <div className="flex items-center gap-2.5">
                <span className="h-1.5 w-1.5 bg-white" />
                <span className="text-[10px] uppercase tracking-[0.22em] text-white/70">
                  Ready · Awaiting Input
                </span>
              </div>
              <button
                type="button"
                onClick={() => setShowTip(false)}
                aria-label="Dismiss tip"
                className="text-white/40 hover:text-white transition-colors"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M18 6 6 18" /><path d="m6 6 12 12" />
                </svg>
              </button>
            </div>
            {/* Body */}
            <div className="px-4 py-3">
              <p className="text-sm text-white font-medium mb-1">
                Tap anywhere on the map
              </p>
              <p className="text-xs text-white/60 leading-relaxed">
                We&apos;ll fetch a Sentinel-2 tile around that point and scan it
                for mining pits in your browser. No data leaves your device.
              </p>
              <button
                type="button"
                onClick={() => {
                  setShowTip(false);
                  setSheetOpen(true);
                }}
                className="mt-3 text-[10px] font-medium text-white/70 hover:text-white transition-colors inline-flex items-center gap-1.5 uppercase tracking-[0.22em] border-t border-white/10 pt-2 w-full"
              >
                View Detail
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="m9 18 6-6-6-6" />
                </svg>
              </button>
            </div>
          </div>
        )}

        {/* Model load overlay, blocks Live mode until WebGPU model is ready.
            Progress covers weights + shader compilation + GPU upload, not just
            network download; first visit downloads + compiles, repeat visits
            just compile shaders + allocate GPU memory (still 10-30s typically).
            If WebGPU is unavailable or the load crashed, flips to an error card. */}
        {!modelReady && (
          <div className="absolute inset-0 z-20 bg-black/60 backdrop-blur-sm flex items-center justify-center p-6">
            {modelError ? (
              <div className="bg-gray-900 border border-red-500/40 max-w-md w-full shadow-2xl font-mono">
                <div className="flex items-center justify-between px-4 py-2.5 border-b border-red-500/20">
                  <div className="flex items-center gap-2.5">
                    <span className="h-1.5 w-1.5 bg-red-500" />
                    <span className="text-[10px] uppercase tracking-[0.22em] text-red-400">
                      {modelError === "no-webgpu" ? "Fault · Capability" : "Fault · Runtime"}
                    </span>
                  </div>
                  <span className="text-[10px] tracking-wider text-red-400/70">
                    {modelError === "no-webgpu" ? "E_NO_GPU" : "E_INIT"}
                  </span>
                </div>
                <div className="px-4 py-4">
                  <p className="text-sm text-white/80 leading-relaxed">
                    {modelError === "no-webgpu" ? (
                      <>
                        Your browser doesn&apos;t expose WebGPU, so the 450M VLM
                        can&apos;t run locally. Open this page in{" "}
                        <strong className="text-white">desktop Chrome, Edge, or Safari 17+</strong>.
                      </>
                    ) : (
                      <>
                        WebGPU is available, but the model failed to initialize.
                        Usually insufficient GPU memory (need ~2 GB free) or a
                        driver that doesn&apos;t support required features. Try a
                        machine with a discrete or Apple Silicon GPU.
                      </>
                    )}
                  </p>
                </div>
              </div>
            ) : (
              <div className="bg-gray-900 border border-white/15 max-w-md w-full shadow-2xl font-mono">
                {/* Header row: status label · phase code */}
                <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/10">
                  <div className="flex items-center gap-2.5">
                    <div className="animate-spin h-3 w-3 border border-white border-t-transparent rounded-full" />
                    <span className="text-[10px] uppercase tracking-[0.22em] text-white/70">
                      Cold Start · v9-e3
                    </span>
                  </div>
                  <span className="text-[10px] tracking-wider text-white/50">
                    {modelProgress >= 0.99
                      ? "PHASE · COMPILE"
                      : modelProgress > 0
                        ? "PHASE · LOAD"
                        : "PHASE · INIT"}
                  </span>
                </div>
                {/* Body: description */}
                <div className="px-4 py-3 border-b border-white/10">
                  <p className="text-xs uppercase tracking-wider text-white/50 mb-2">
                    Runtime
                  </p>
                  <p className="text-sm text-white/80 leading-relaxed">
                    Fine-tuned LFM2.5-VL-450M (~1 GB, fp16). First visit
                    downloads the weights; subsequent visits load from cache and
                    recompile GPU shaders.
                  </p>
                </div>
                {/* Progress footer: numeric + bar */}
                <div className="px-4 py-3">
                  <div className="flex items-center justify-between mb-2 text-[10px] uppercase tracking-[0.22em]">
                    <span className="text-white/50">Progress</span>
                    <span className="tabular-nums text-white">
                      {Math.round(modelProgress * 100).toString().padStart(3, "0")} / 100
                    </span>
                  </div>
                  <div className="h-[3px] bg-white/10">
                    <div
                      className="h-full bg-white transition-[width] duration-300"
                      style={{ width: `${Math.max(modelProgress * 100, 2)}%` }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Results panel -
          Desktop (md+): flex child, 28rem wide, always visible.
          Mobile: fixed bottom sheet, slides up when sheetOpen, full-width, 75vh cap. */}
      <div
        className={`bg-gray-900 border-white/10 overflow-y-auto sidebar-scroll
          md:static md:w-[28rem] md:h-auto md:max-h-none md:border-l md:border-t-0 md:rounded-none md:translate-y-0
          fixed inset-x-0 bottom-0 z-30 max-h-[75vh] border-t rounded-t-xl
          transform transition-transform duration-300 ease-out
          ${sheetOpen ? "translate-y-0" : liveResult ? "translate-y-[calc(100%-3.5rem)]" : "translate-y-full md:translate-y-0"}`}
        style={{ scrollbarGutter: "stable both-edges" }}
      >
        {/* Mobile drawer handle, chevron flips to signal state.
            Chevron-down when open (tap to collapse), up when collapsed (tap to expand). */}
        <button
          type="button"
          onClick={() => setSheetOpen((o) => !o)}
          aria-label={sheetOpen ? "Collapse results" : "Expand results"}
          className="md:hidden sticky top-0 z-10 w-full flex items-center justify-center py-3 bg-gray-900/95 backdrop-blur border-b border-gray-800 text-gray-400 hover:text-white transition-colors"
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`transition-transform duration-300 ${sheetOpen ? "" : "rotate-180"}`}
          >
            <path d="m6 9 6 6 6-6" />
          </svg>
        </button>
        {/* Mode tabs, Explore (click-driven WebGPU inference) vs Agent
            (orchestrator-driven autonomous pass). Both share this pane. */}
        <div className="flex border-b border-white/10 sticky top-0 md:top-0 z-10 bg-gray-900">
          {(["explore", "agent"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setPaneMode(m)}
              className={`flex-1 py-2.5 font-mono text-[10px] uppercase tracking-[0.22em] transition-colors ${
                paneMode === m
                  ? "text-white border-b border-white"
                  : "text-white/40 hover:text-white/70 border-b border-transparent"
              }`}
            >
              {m === "explore" ? "Explore" : "Agent Mode"}
            </button>
          ))}
        </div>

        {paneMode === "explore" ? (
          <DetectionPanel
            liveResult={liveResult}
            onRunBitemporal={runBitemporal}
            onExitBitemporal={() =>
              setLiveResult((prev) => (prev ? { ...prev, bitemporal: null } : prev))
            }
            bitemporalLoadingYear={bitemporalLoadingYear}
          />
        ) : (
          <AgentPanel />
        )}
      </div>
    </main>
  );
}
