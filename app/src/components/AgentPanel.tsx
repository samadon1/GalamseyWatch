"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";

const ORCHESTRATOR_URL =
  process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || "http://localhost:8765";

// Tightly bounded AOI around the Bibiani galamsey cluster, four v9-confirmed
// sites at 6 bboxes each within this bbox. With a 6-tile grid, ≥5 of 6 tiles
// reliably fire positive detections, so Initiate Pass produces the agent's
// most interesting behaviour (downlink decisions on real galamsey, budget
// exhaustion mid-pass) rather than a sea of forest negatives.
const DEFAULT_AOI = {
  name: "Bibiani cluster",
  lon_min: -2.79,
  lat_min: 5.62,
  lon_max: -2.71,
  lat_max: 5.66,
};
const DEFAULT_BANDWIDTH_KB = 512;
const DEFAULT_TILE_COUNT = 6;
const DEFAULT_MODE: PassMode = "agent";

type PassMode = "threshold" | "rules" | "agent";

type Action =
  | "flag"
  | "request_neighbor"
  | "request_hires"
  | "downlink"
  | "discard";

type TileState = {
  tile_id: string;
  lon: number;
  lat: number;
  image_url?: string | null;
  image_available?: boolean;
  cloud_cover?: number | null;
  captured_at?: string | null;
  detection?: { boxes_count: number; description: string; confidence: number };
  action?: Action;
  reasoning?: string;
  scratchpad?: string;
  vlm_inference_ms?: number;
  decision_ms?: number;
};

type Stage =
  | "fetching"
  | "perceiving"
  | "thinking"
  | "decided";

type Status = "idle" | "starting" | "running" | "complete" | "error";

type PassSummary = {
  pass_id: string;
  tiles_processed: number;
  tiles_flagged: number;
  tiles_downlinked: number;
  tiles_discarded: number;
  bandwidth_used_kb: number;
  bandwidth_remaining_kb: number;
  elapsed_ms: number;
  flagged_summary: string[];
};

type FeedItem = { type: string; tile_id?: string; detail?: string; ts: number };

export default function AgentPanel() {
  const [status, setStatus] = useState<Status>("idle");
  const [tiles, setTiles] = useState<Record<string, TileState>>({});
  const [tileOrder, setTileOrder] = useState<string[]>([]);
  const [budget, setBudget] = useState({
    used: 0,
    remaining: 0,
    total: DEFAULT_BANDWIDTH_KB,
  });
  const [summary, setSummary] = useState<PassSummary | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [activeTileId, setActiveTileId] = useState<string | null>(null);
  const [activeStage, setActiveStage] = useState<Stage | null>(null);
  const [pinnedTileId, setPinnedTileId] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const statusRef = useRef<Status>("idle");

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(
    () => () => {
      esRef.current?.close();
    },
    [],
  );

  function pushFeed(type: string, tile_id?: string, detail?: string) {
    setFeed((prev) =>
      [{ type, tile_id, detail, ts: Date.now() }, ...prev].slice(0, 10),
    );
  }

  async function startPass() {
    if (status === "running" || status === "starting") return;
    setStatus("starting");
    setTiles({});
    setTileOrder([]);
    setBudget({ used: 0, remaining: 0, total: DEFAULT_BANDWIDTH_KB });
    setSummary(null);
    setErrorMsg(null);
    setFeed([]);
    setActiveTileId(null);
    setActiveStage(null);
    setPinnedTileId(null);

    try {
      const res = await fetch(`${ORCHESTRATOR_URL}/pass/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          aoi: DEFAULT_AOI,
          tile_count: DEFAULT_TILE_COUNT,
          mode: DEFAULT_MODE,
          bandwidth_kb: DEFAULT_BANDWIDTH_KB,
        }),
      });
      if (!res.ok) throw new Error(`POST /pass/start -> ${res.status}`);
      const { pass_id } = (await res.json()) as { pass_id: string };
      setStatus("running");
      subscribe(pass_id);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErrorMsg(`orchestrator unreachable at ${ORCHESTRATOR_URL}, ${msg}`);
      setStatus("error");
    }
  }

  function subscribe(pid: string) {
    esRef.current?.close();
    const es = new EventSource(`${ORCHESTRATOR_URL}/pass/${pid}/events`);
    esRef.current = es;

    es.addEventListener("pass_started", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data);
      setBudget({ used: 0, remaining: data.bandwidth_kb, total: data.bandwidth_kb });
      pushFeed("pass_started");
    });

    es.addEventListener("tile_arrived", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data);
      setTiles((prev) => ({
        ...prev,
        [data.tile_id]: {
          tile_id: data.tile_id,
          lon: data.lon,
          lat: data.lat,
          image_url: data.image_url ?? null,
          image_available: data.image_available ?? true,
          cloud_cover: data.cloud_cover ?? null,
          captured_at: data.captured_at ?? null,
        },
      }));
      setTileOrder((prev) =>
        prev.includes(data.tile_id) ? prev : [...prev, data.tile_id],
      );
      setActiveTileId(data.tile_id);
      setActiveStage("perceiving");
      pushFeed(
        "tile_arrived",
        data.tile_id,
        data.image_available === false ? "NO_IMG" : undefined,
      );
    });

    es.addEventListener("vlm_done", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data);
      setTiles((prev) => ({
        ...prev,
        [data.tile_id]: {
          ...(prev[data.tile_id] ?? {
            tile_id: data.tile_id,
            lon: 0,
            lat: 0,
          }),
          detection: {
            boxes_count: data.detection.boxes.length,
            description: data.detection.description,
            confidence: data.detection.overall_confidence,
          },
          vlm_inference_ms: data.inference_ms,
        },
      }));
      setActiveTileId(data.tile_id);
      setActiveStage("thinking");
      pushFeed(
        "vlm_done",
        data.tile_id,
        `${data.detection.boxes.length} box${data.detection.boxes.length === 1 ? "" : "es"}`,
      );
    });

    es.addEventListener("agent_thinking", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data);
      setTiles((prev) => ({
        ...prev,
        [data.tile_id]: {
          ...(prev[data.tile_id] ?? {
            tile_id: data.tile_id,
            lon: 0,
            lat: 0,
          }),
          scratchpad: data.scratchpad,
        },
      }));
    });

    es.addEventListener("agent_decided", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data);
      setTiles((prev) => ({
        ...prev,
        [data.tile_id]: {
          ...(prev[data.tile_id] ?? {
            tile_id: data.tile_id,
            lon: 0,
            lat: 0,
          }),
          action: data.action,
          reasoning: data.reasoning,
          decision_ms: data.decision_ms,
        },
      }));
      setActiveTileId(data.tile_id);
      setActiveStage("decided");
      pushFeed("decided", data.tile_id, String(data.action).toUpperCase());
    });

    es.addEventListener("budget_update", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data);
      setBudget((prev) => ({
        ...prev,
        used: data.bandwidth_used_kb,
        remaining: data.bandwidth_remaining_kb,
      }));
    });

    es.addEventListener("pass_complete", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data);
      setSummary(data.summary);
      setStatus("complete");
      pushFeed("pass_complete");
      es.close();
    });

    es.onerror = () => {
      // EventSource auto-reconnects by default. If the pass already completed,
      // close cleanly; otherwise let the browser retry.
      if (statusRef.current === "complete") es.close();
    };
  }

  const counts = {
    processed: tileOrder.length,
    flagged: Object.values(tiles).filter((t) => t.action === "flag").length,
    downlinked: Object.values(tiles).filter((t) => t.action === "downlink").length,
    discarded: Object.values(tiles).filter((t) => t.action === "discard").length,
  };

  const budgetPct =
    budget.total > 0 ? Math.min(100, (budget.used / budget.total) * 100) : 0;

  const focusTileId = pinnedTileId ?? activeTileId;
  const focusTile = focusTileId ? tiles[focusTileId] : null;
  const focusStage: Stage | null = pinnedTileId
    ? focusTile?.action
      ? "decided"
      : focusTile?.detection
        ? "thinking"
        : "perceiving"
    : activeStage;

  return (
    <div className="font-mono text-white/90 text-xs">
      {/* Header, status + progress count */}
      <div className="border-b border-white/10 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            {status === "running" || status === "starting" ? (
              <div className="animate-spin h-2.5 w-2.5 border border-white border-t-transparent rounded-full" />
            ) : (
              <span
                className={`h-1.5 w-1.5 ${
                  status === "complete"
                    ? "bg-emerald-400"
                    : status === "error"
                      ? "bg-red-500"
                      : "bg-white/60"
                }`}
              />
            )}
            <span className="text-[10px] uppercase tracking-[0.22em] text-white/70">
              {status === "idle" && "Agent · Idle"}
              {status === "starting" && "Agent · Starting"}
              {status === "running" && "Agent · Pass In Progress"}
              {status === "complete" && "Agent · Pass Complete"}
              {status === "error" && "Agent · Fault"}
            </span>
          </div>
          <span className="text-[10px] tabular-nums text-white/60">
            N = {String(counts.processed).padStart(3, "0")} /{" "}
            {String(DEFAULT_TILE_COUNT).padStart(3, "0")}
          </span>
        </div>
      </div>

      {/* Target / AOI */}
      <div className="border-b border-white/10 px-4 py-3">
        <div className="text-[10px] uppercase tracking-[0.22em] text-white/50 mb-2">
          Target
        </div>
        <div className="text-sm text-white">{DEFAULT_AOI.name}</div>
        <div className="mt-1 text-[10px] text-white/50 tabular-nums tracking-wider">
          {DEFAULT_AOI.lon_min.toFixed(2)},{DEFAULT_AOI.lat_min.toFixed(2)}
          <span className="mx-1.5">→</span>
          {DEFAULT_AOI.lon_max.toFixed(2)},{DEFAULT_AOI.lat_max.toFixed(2)}
        </div>
        <div className="mt-1 text-[10px] text-white/40 uppercase tracking-[0.22em]">
          {DEFAULT_TILE_COUNT} Tiles · {DEFAULT_BANDWIDTH_KB} KB Budget
        </div>
      </div>

      {/* Initiate */}
      <div className="border-b border-white/10 px-4 py-3">
        <button
          type="button"
          onClick={startPass}
          disabled={status === "running" || status === "starting"}
          className="w-full text-[11px] uppercase tracking-[0.22em] py-2.5 border border-white/15 hover:border-white/40 hover:bg-white/5 transition-colors disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent"
        >
          {status === "running"
            ? "Pass Running"
            : status === "starting"
              ? "Starting…"
              : status === "complete"
                ? "Restart Pass"
                : status === "error"
                  ? "Retry"
                  : "Initiate Pass"}
        </button>
        {errorMsg && (
          <div className="mt-3 border border-red-500/30 px-3 py-2 text-[10px] text-red-400/90 leading-relaxed">
            <div className="uppercase tracking-[0.22em] text-red-400/60 mb-1">
              Fault · Connect
            </div>
            {errorMsg}
            <div className="mt-1.5 text-red-400/40 normal-case">
              Start the orchestrator: <span className="text-red-400/70">cd orchestrator && uv run uvicorn agentic_eo.main:app --port 8765</span>
            </div>
          </div>
        )}
      </div>

      {/* Bandwidth */}
      {(status === "running" || status === "complete") && (
        <div className="border-b border-white/10 px-4 py-3">
          <div className="flex items-center justify-between mb-2 text-[10px] uppercase tracking-[0.22em]">
            <span className="text-white/50">Bandwidth</span>
            <span className="tabular-nums text-white/70">
              {String(budget.used).padStart(4, "0")} /{" "}
              {String(budget.total).padStart(4, "0")} KB
            </span>
          </div>
          <div className="h-[3px] bg-white/10">
            <motion.div
              className="h-full bg-white"
              animate={{ width: `${Math.max(2, budgetPct)}%` }}
              transition={{ duration: 0.4 }}
            />
          </div>
        </div>
      )}

      {/* Live agent thinking, pinned to most-recent tile, or click a cell to pin */}
      {focusTile && (
        <div className="border-b border-white/10 px-4 py-3">
          <div className="flex items-center justify-between mb-2 text-[10px] uppercase tracking-[0.22em]">
            <div className="flex items-center gap-2">
              {focusStage === "perceiving" || focusStage === "thinking" ? (
                <div className="animate-spin h-2 w-2 border border-white border-t-transparent rounded-full" />
              ) : focusStage === "decided" ? (
                <span
                  className={`h-1.5 w-1.5 ${
                    focusTile.action === "downlink"
                      ? "bg-emerald-400"
                      : focusTile.action === "flag"
                        ? "bg-amber-300"
                        : focusTile.action === "discard"
                          ? "bg-white/40"
                          : "bg-sky-300"
                  }`}
                />
              ) : (
                <span className="h-1.5 w-1.5 bg-white/40" />
              )}
              <span className="text-white/70">
                {focusStage === "perceiving" && "VLM Perceiving…"}
                {focusStage === "thinking" && "Agent Thinking…"}
                {focusStage === "decided" && "Agent · Decided"}
                {!focusStage && "Agent · Idle"}
              </span>
            </div>
            <div className="flex items-center gap-2 text-white/50">
              <span className="tabular-nums">{focusTile.tile_id}</span>
              {pinnedTileId && (
                <button
                  type="button"
                  onClick={() => setPinnedTileId(null)}
                  className="text-[9px] tracking-[0.22em] text-white/40 hover:text-white/70 underline-offset-2 hover:underline"
                >
                  unpin
                </button>
              )}
            </div>
          </div>

          {/* Tile thumbnail + meta */}
          <div className="flex gap-3 mb-3">
            <div className="w-20 h-20 border border-white/15 overflow-hidden flex-shrink-0 relative bg-white/5">
              {focusTile.image_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={`${ORCHESTRATOR_URL}${focusTile.image_url}`}
                  alt={focusTile.tile_id}
                  className="absolute inset-0 w-full h-full object-cover"
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center text-[9px] text-white/30 uppercase tracking-[0.22em]">
                  …
                </div>
              )}
            </div>
            <div className="flex-1 min-w-0 space-y-1">
              <div className="flex items-center gap-3 text-[10px] tabular-nums text-white/60">
                <span>
                  {focusTile.lon.toFixed(4)},{focusTile.lat.toFixed(4)}
                </span>
                {focusTile.cloud_cover != null && (
                  <span>cloud {(focusTile.cloud_cover * 100).toFixed(0)}%</span>
                )}
              </div>
              {focusTile.captured_at && (
                <div className="text-[9px] uppercase tracking-[0.22em] text-white/40">
                  {focusTile.captured_at.slice(0, 10)}
                </div>
              )}
              {focusTile.detection && (
                <div className="flex items-center gap-2 text-[10px] text-white/70 tabular-nums">
                  <span>
                    {focusTile.detection.boxes_count} box
                    {focusTile.detection.boxes_count === 1 ? "" : "es"}
                  </span>
                  <span className="text-white/30">·</span>
                  <span>conf {focusTile.detection.confidence.toFixed(2)}</span>
                  {focusTile.vlm_inference_ms != null && (
                    <>
                      <span className="text-white/30">·</span>
                      <span className="text-white/50">
                        {(focusTile.vlm_inference_ms / 1000).toFixed(1)}s
                      </span>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* VLM description */}
          {focusTile.detection && (
            <div className="mb-3 border-l border-white/15 pl-3">
              <div className="text-[9px] uppercase tracking-[0.22em] text-white/40 mb-1">
                VLM · LFM2.5-VL-450M
              </div>
              <p className="text-[11px] text-white/80 leading-relaxed italic">
                &ldquo;{focusTile.detection.description}&rdquo;
              </p>
            </div>
          )}

          {/* Agent action + reasoning */}
          {focusTile.action ? (
            <div
              className={`border-l-2 pl-3 ${
                focusTile.action === "downlink"
                  ? "border-emerald-400"
                  : focusTile.action === "flag"
                    ? "border-amber-300"
                    : focusTile.action === "discard"
                      ? "border-white/30"
                      : "border-sky-300"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-[9px] uppercase tracking-[0.22em] text-white/40">
                  Agent · LFM2-2.6B
                </span>
                {focusTile.decision_ms != null && (
                  <span className="text-[9px] tabular-nums text-white/40">
                    {(focusTile.decision_ms / 1000).toFixed(1)}s
                  </span>
                )}
              </div>
              <div
                className={`text-[11px] uppercase tracking-[0.22em] mb-1 ${
                  focusTile.action === "downlink"
                    ? "text-emerald-400"
                    : focusTile.action === "flag"
                      ? "text-amber-300"
                      : focusTile.action === "discard"
                        ? "text-white/60"
                        : "text-sky-300"
                }`}
              >
                ► {String(focusTile.action).replace("_", " ")}
              </div>
              {focusTile.reasoning && (
                <p className="text-[11px] text-white/75 leading-relaxed">
                  {focusTile.reasoning}
                </p>
              )}
            </div>
          ) : focusTile.detection ? (
            <div className="border-l-2 border-white/15 pl-3">
              <div className="text-[9px] uppercase tracking-[0.22em] text-white/40 mb-1">
                Agent · LFM2-2.6B
              </div>
              <div className="flex items-center gap-2 text-[10px] text-white/40 italic">
                <div className="animate-pulse">deliberating…</div>
              </div>
            </div>
          ) : null}
        </div>
      )}

      {/* Tile grid */}
      {tileOrder.length > 0 && (
        <div className="border-b border-white/10 px-4 py-3">
          <div className="flex items-center justify-between mb-2 text-[10px] uppercase tracking-[0.22em]">
            <span className="text-white/50">Tiles</span>
            <span className="tabular-nums text-white/70">{tileOrder.length}</span>
          </div>
          <div className="grid grid-cols-5 gap-1">
            {tileOrder.map((tid) => {
              const t = tiles[tid];
              const action = t?.action;
              const borderCls =
                action === "downlink"
                  ? "border-emerald-400"
                  : action === "flag"
                    ? "border-amber-300/80"
                    : action === "discard"
                      ? "border-white/15"
                      : t?.detection
                        ? "border-white/25"
                        : "border-white/15";
              const label =
                action === "downlink"
                  ? "D"
                  : action === "flag"
                    ? "F"
                    : action === "discard"
                      ? "·"
                      : t?.detection
                        ? "?"
                        : "·";
              const labelCls =
                action === "downlink"
                  ? "bg-emerald-400 text-black"
                  : action === "flag"
                    ? "bg-amber-300 text-black"
                    : "bg-black/60 text-white/80";
              const fallback =
                action === "downlink"
                  ? "bg-emerald-400/30"
                  : action === "flag"
                    ? "bg-amber-300/30"
                    : "bg-white/5";
              const tip =
                `${tid}` +
                (t?.cloud_cover != null
                  ? ` · cloud=${(t.cloud_cover * 100).toFixed(1)}%`
                  : "") +
                (t?.captured_at ? ` · ${t.captured_at.slice(0, 10)}` : "") +
                (t?.detection
                  ? ` · conf=${t.detection.confidence.toFixed(2)}`
                  : "") +
                (t?.action ? ` · ${t.action}` : "");
              const isFocus = focusTileId === tid;
              return (
                <button
                  key={tid}
                  type="button"
                  onClick={() => {
                    setPinnedTileId((prev) => (prev === tid ? null : tid));
                  }}
                  className={`relative h-14 border ${borderCls} overflow-hidden cursor-pointer transition-all ${
                    isFocus ? "ring-1 ring-white/60 ring-offset-1 ring-offset-gray-900" : "hover:brightness-125"
                  } ${
                    !t?.image_url ? `${fallback} ${!t?.detection ? "animate-pulse" : ""}` : ""
                  }`}
                  title={tip}
                >
                  {t?.image_url && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={`${ORCHESTRATOR_URL}${t.image_url}`}
                      alt={tid}
                      className="absolute inset-0 w-full h-full object-cover"
                      loading="lazy"
                    />
                  )}
                  <span
                    className={`absolute bottom-0 right-0 px-1 text-[9px] font-mono tabular-nums ${labelCls}`}
                  >
                    {label}
                  </span>
                  {t?.image_available === false && (
                    <span className="absolute top-0 left-0 px-1 text-[8px] uppercase tracking-wider bg-red-500/70 text-white">
                      NO_IMG
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[9px] uppercase tracking-[0.22em] text-white/40">
            <span>
              <span className="inline-block w-2 h-2 bg-emerald-400/80 mr-1.5 align-middle" />
              Downlink
            </span>
            <span>
              <span className="inline-block w-2 h-2 bg-amber-300/70 mr-1.5 align-middle" />
              Flag
            </span>
            <span>
              <span className="inline-block w-2 h-2 bg-white/15 mr-1.5 align-middle" />
              Discard
            </span>
          </div>
        </div>
      )}

      {/* Decisions tally */}
      {counts.processed > 0 && (
        <div className="border-b border-white/10 px-4 py-3 space-y-1.5">
          <div className="text-[10px] uppercase tracking-[0.22em] text-white/50 mb-2">
            Decisions
          </div>
          <CountRow label="Downlink" count={counts.downlinked} dot="bg-emerald-400" />
          <CountRow label="Flag" count={counts.flagged} dot="bg-amber-300" />
          <CountRow label="Discard" count={counts.discarded} dot="bg-white/30" />
        </div>
      )}

      {/* Event feed */}
      {feed.length > 0 && (
        <div className="border-b border-white/10 px-4 py-3">
          <div className="text-[10px] uppercase tracking-[0.22em] text-white/50 mb-2">
            Event Feed
          </div>
          <ul className="space-y-1">
            {feed.map((f, i) => (
              <li
                key={`${f.ts}-${i}`}
                className="flex items-center gap-2 text-[10px] text-white/70 tracking-wider"
              >
                <span className="tabular-nums text-white/40 w-8">
                  {String(feed.length - i).padStart(3, "0")}
                </span>
                <span className="flex-1 uppercase tracking-[0.18em] truncate">
                  {f.type.replace(/_/g, " ")}
                  {f.tile_id && (
                    <span className="text-white/40 ml-1.5 normal-case lowercase">
                      {f.tile_id}
                    </span>
                  )}
                </span>
                {f.detail && (
                  <span className="text-white/50 text-[9px] uppercase tracking-[0.18em]">
                    {f.detail}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* End-of-pass summary */}
      {summary && (
        <div className="border-b border-white/10 px-4 py-3">
          <div className="text-[10px] uppercase tracking-[0.22em] text-white/50 mb-2">
            Pass Summary
          </div>
          <div className="grid grid-cols-2 gap-2 mb-3">
            <Stat label="Processed" value={summary.tiles_processed} />
            <Stat
              label="Elapsed"
              value={`${(summary.elapsed_ms / 1000).toFixed(1)}s`}
            />
            <Stat label="Bandwidth Used" value={`${summary.bandwidth_used_kb} KB`} />
            <Stat
              label="Bandwidth Left"
              value={`${summary.bandwidth_remaining_kb} KB`}
            />
          </div>
          {summary.flagged_summary.length > 0 && (
            <>
              <div className="text-[9px] uppercase tracking-[0.22em] text-white/40 mb-1.5">
                Flagged · Description
              </div>
              <ul className="space-y-1.5">
                {summary.flagged_summary.map((d, i) => (
                  <li
                    key={i}
                    className="text-[11px] text-white/70 leading-relaxed border-l border-white/15 pl-2"
                  >
                    {d}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}

      {/* Empty state */}
      {status === "idle" && (
        <div className="px-4 py-6 text-[11px] text-white/40 leading-relaxed">
          Press{" "}
          <span className="text-white/70 uppercase tracking-[0.22em]">
            Initiate Pass
          </span>{" "}
          to start a simulated orbital pass over the Pra basin. Phase-1 backend
          emits mocked tile fetches, VLM perception, and agent decisions. Real
          models slot in next.
        </div>
      )}
    </div>
  );
}

function CountRow({
  label,
  count,
  dot,
}: {
  label: string;
  count: number;
  dot: string;
}) {
  return (
    <div className="flex items-center gap-3 text-[11px] uppercase tracking-[0.22em]">
      <span className={`h-1.5 w-1.5 ${dot}`} />
      <span className="flex-1 text-white/70">{label}</span>
      <span className="tabular-nums text-white">
        {String(count).padStart(3, "0")}
      </span>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="border border-white/10 px-2 py-1.5">
      <div className="text-[9px] uppercase tracking-[0.22em] text-white/40">
        {label}
      </div>
      <div className="text-sm text-white tabular-nums mt-0.5">{value}</div>
    </div>
  );
}
