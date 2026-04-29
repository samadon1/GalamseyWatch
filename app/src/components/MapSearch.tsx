"use client";

import { useEffect, useMemo, useRef, useState } from "react";

// Known galamsey-affected districts + major towns in Ghana's gold belt.
// Hardcoded, no external geocoding API needed, no rate limits, works offline.
const GHANA_PLACES: Array<{ name: string; lng: number; lat: number; note?: string }> = [
  { name: "Obuasi", lng: -1.6667, lat: 6.2, note: "AngloGold Ashanti mine · galamsey hotspot" },
  { name: "Tarkwa", lng: -1.9956, lat: 5.3, note: "historical gold-mining town" },
  { name: "Prestea", lng: -2.1433, lat: 5.4333, note: "Western region, active artisanal mining" },
  { name: "Dunkwa-on-Offin", lng: -1.7833, lat: 5.9667, note: "Offin river galamsey corridor" },
  { name: "Bibiani", lng: -2.3167, lat: 6.4583, note: "Western North gold zone" },
  { name: "Konongo", lng: -1.2167, lat: 6.6167, note: "Ashanti region" },
  { name: "Bogoso", lng: -2.0, lat: 5.5833, note: "Golden Star Resources · nearby galamsey" },
  { name: "Ntotoroso", lng: -2.6, lat: 7.0, note: "Newmont Ahafo · surrounding galamsey" },
  { name: "Akwatia", lng: -0.8167, lat: 6.05, note: "Eastern region diamond/gold" },
  { name: "Kibi", lng: -0.55, lat: 6.1667, note: "Atewa forest · illegal mining threat" },
  { name: "Kumasi", lng: -1.6167, lat: 6.6833, note: "Ashanti capital, regional hub" },
  { name: "Sefwi Wiawso", lng: -2.4833, lat: 6.2, note: "Western North, cocoa + mining" },
  { name: "Ashanti Akim", lng: -1.2, lat: 6.4, note: "illegal mining in cocoa areas" },
  { name: "Enchi", lng: -2.8167, lat: 5.8167, note: "Western region border districts" },
  { name: "Amansie West", lng: -2.05, lat: 6.35, note: "active galamsey district" },
];

interface MapSearchProps {
  onSelect: (place: {
    name: string;
    lng: number;
    lat: number;
    // "place" = town from the list (fly-to only, don't infer);
    // "coord" = user pasted a lat/lng (run inference immediately).
    type: "place" | "coord";
  }) => void;
}

export default function MapSearch({ onSelect }: MapSearchProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Parse "lat, lng" or "lat lng", accepts signed decimals. Returns null if not a coordinate pair.
  const parsedCoord = useMemo(() => {
    const q = query.trim();
    const m = q.match(/^(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)$/);
    if (!m) return null;
    const lat = parseFloat(m[1]);
    const lng = parseFloat(m[2]);
    if (Math.abs(lat) > 90 || Math.abs(lng) > 180) return null;
    return { lat, lng };
  }, [query]);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return GHANA_PLACES.slice(0, 6);
    return GHANA_PLACES.filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        p.note?.toLowerCase().includes(q),
    ).slice(0, 6);
  }, [query]);

  useEffect(() => {
    const onClickOutside = (e: MouseEvent) => {
      if (!containerRef.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", onClickOutside);
    return () => window.removeEventListener("mousedown", onClickOutside);
  }, []);

  function choose(p: (typeof GHANA_PLACES)[number]) {
    onSelect({ name: p.name, lng: p.lng, lat: p.lat, type: "place" });
    setQuery(p.name);
    setOpen(false);
  }

  function handleKey(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (parsedCoord) {
        onSelect({
          name: `${parsedCoord.lat.toFixed(4)}, ${parsedCoord.lng.toFixed(4)}`,
          ...parsedCoord,
          type: "coord",
        });
        setOpen(false);
      } else if (results[activeIndex]) {
        choose(results[activeIndex]);
      }
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div ref={containerRef} className="relative w-72 md:w-96 font-mono">
      <input
        type="text"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
          setActiveIndex(0);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={handleKey}
        placeholder="QUERY · TOWN OR LAT,LNG…"
        className="w-full bg-gray-900/90 backdrop-blur border border-white/15 px-3 py-2 text-sm text-white placeholder-white/30 tracking-wide focus:outline-none focus:border-white/40 transition-colors"
      />

      {open && parsedCoord && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-gray-900/95 backdrop-blur border border-white/15 overflow-hidden z-50">
          <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
            <span className="text-[10px] uppercase tracking-[0.22em] text-white/50">
              Coordinate · Parsed
            </span>
            <span className="text-[10px] text-white/50">⏎</span>
          </div>
          <button
            onClick={() => {
              onSelect({
                name: `${parsedCoord.lat.toFixed(4)}, ${parsedCoord.lng.toFixed(4)}`,
                ...parsedCoord,
                type: "coord",
              });
              setOpen(false);
            }}
            className="w-full text-left px-3 py-2.5 hover:bg-white/5 transition-colors"
          >
            <div className="text-sm text-white tabular-nums">
              {parsedCoord.lat.toFixed(4)}, {parsedCoord.lng.toFixed(4)}
            </div>
            <div className="text-[10px] uppercase tracking-[0.22em] text-white/40 mt-0.5">
              Goto · Run Inference
            </div>
          </button>
        </div>
      )}

      {open && !parsedCoord && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-gray-900/95 backdrop-blur border border-white/15 overflow-hidden z-50">
          <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
            <span className="text-[10px] uppercase tracking-[0.22em] text-white/50">
              Known Sites
            </span>
            <span className="text-[10px] tabular-nums text-white/50">
              {String(results.length).padStart(2, "0")} / {String(GHANA_PLACES.length).padStart(2, "0")}
            </span>
          </div>
          <ul className="divide-y divide-white/5">
            {results.map((p, i) => (
              <li key={p.name}>
                <button
                  onClick={() => choose(p)}
                  onMouseEnter={() => setActiveIndex(i)}
                  className={`w-full text-left px-3 py-2 transition-colors ${
                    i === activeIndex ? "bg-white/10" : "hover:bg-white/5"
                  }`}
                >
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-sm text-white font-medium tracking-wide">
                      {p.name}
                    </span>
                    <span className="text-[10px] tabular-nums text-white/40">
                      {p.lat.toFixed(2)}, {p.lng.toFixed(2)}
                    </span>
                  </div>
                  {p.note && (
                    <div className="text-[11px] text-white/50 truncate mt-0.5">
                      {p.note}
                    </div>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
