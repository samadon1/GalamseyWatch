"use client";

import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { DemoPatch } from "@/app/dashboard/page";

interface MapProps {
  patches: DemoPatch[];
  selectedSlug: string | null;
  onSelect: (slug: string) => void;
  onMapClick?: (lng: number, lat: number) => void;
  liveMarker?: { lat: number; lng: number } | null;
  flyTo?: { lat: number; lng: number } | null;
  miningSites?: Array<{ id: number; lng: number; lat: number; status?: string }>;
}

// Center on southwestern Ghana (Ashanti / Western region, galamsey hotspot)
const INITIAL_CENTER: [number, number] = [-1.8, 6.1];
const INITIAL_ZOOM = 7.5;

export default function Map({
  patches,
  selectedSlug,
  onSelect,
  onMapClick,
  liveMarker,
  flyTo,
  miningSites,
}: MapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const markers = useRef<Map<string, maplibregl.Marker>>(new window.Map());
  const liveMarkerRef = useRef<maplibregl.Marker | null>(null);
  const onMapClickRef = useRef(onMapClick);
  onMapClickRef.current = onMapClick;

  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    // Pre-check WebGL, MapLibre throws on ctor if WebGL is disabled at the
    // browser level (Chrome GPU blocklist, Brave fingerprinting, hw-accel off).
    // Probe silently and render a fallback panel instead of crashing the page.
    const probe = document.createElement("canvas");
    if (!probe.getContext("webgl2") && !probe.getContext("webgl")) {
      mapContainer.current.innerHTML = `
        <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#0a0e14;padding:2rem;">
          <div style="max-width:28rem;text-align:center;color:#9ca3af;font-family:ui-sans-serif,system-ui;">
            <p style="font-family:ui-monospace,monospace;font-size:10px;letter-spacing:0.25em;color:#F5C518;margin-bottom:1rem;">WEBGL DISABLED</p>
            <h2 style="font-size:1.25rem;color:#f3f4f6;margin-bottom:0.75rem;">Map unavailable</h2>
            <p style="font-size:0.875rem;line-height:1.5;margin-bottom:1rem;">Your browser has WebGL disabled, so the map can't render. Detection pipeline still works, use the coordinate paste in the top-left search box to run inference on specific lat,lng points.</p>
            <p style="font-size:0.75rem;color:#6b7280;">Fix: <code style="color:#F5C518;">chrome://flags/#ignore-gpu-blocklist</code> → enable → fully restart browser.</p>
          </div>
        </div>
      `;
      return;
    }

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          "esri-satellite": {
            type: "raster",
            tiles: [
              "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            ],
            tileSize: 256,
            attribution: "Tiles &copy; Esri",
            maxzoom: 18,
          },
        },
        layers: [
          {
            id: "satellite",
            type: "raster",
            source: "esri-satellite",
          },
        ],
      },
      center: INITIAL_CENTER,
      zoom: INITIAL_ZOOM,
      maxZoom: 15,
    });

    map.current.addControl(new maplibregl.NavigationControl(), "top-right");

    map.current.on("click", (e) => {
      onMapClickRef.current?.(e.lngLat.lng, e.lngLat.lat);
    });

    return () => {
      map.current?.remove();
      map.current = null;
      markers.current.clear();
    };
  }, []);

  // Sync markers with patches
  useEffect(() => {
    if (!map.current) return;
    // Remove existing
    markers.current.forEach((m) => m.remove());
    markers.current.clear();

    for (const p of patches) {
      // Outer wrapper, maplibre controls its translate; don't touch its transform.
      const el = document.createElement("div");
      // Inner circle, safe to animate transform here.
      const inner = document.createElement("div");
      inner.style.width = "20px";
      inner.style.height = "20px";
      inner.style.borderRadius = "9999px";
      inner.style.border = "2px solid white";
      inner.style.cursor = "pointer";
      inner.style.boxShadow = "0 2px 8px rgba(0,0,0,0.4)";
      inner.style.transition = "transform 150ms";
      const isMining = p.bboxes.length > 0;
      inner.style.background = isMining ? "rgba(239, 68, 68, 0.95)" : "rgba(34, 197, 94, 0.95)";
      inner.setAttribute("role", "button");
      inner.setAttribute("aria-label", `Detection at ${p.lat}, ${p.lng}`);
      inner.addEventListener("mouseenter", () => { inner.style.transform = "scale(1.25)"; });
      inner.addEventListener("mouseleave", () => { inner.style.transform = "scale(1)"; });
      inner.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        onSelect(p.slug);
      });
      el.appendChild(inner);

      const marker = new maplibregl.Marker({ element: el, anchor: "center" })
        .setLngLat([p.lng, p.lat])
        .addTo(map.current!);
      markers.current.set(p.slug, marker);
    }
  }, [patches, onSelect]);

  // Pan to selection, stay at a zoom level the tile source covers in rural Ghana
  useEffect(() => {
    if (!map.current || !selectedSlug) return;
    const p = patches.find((x) => x.slug === selectedSlug);
    if (!p) return;
    map.current.easeTo({ center: [p.lng, p.lat], zoom: 11, duration: 600 });

    // Highlight the selected marker (style the inner circle, not the wrapper).
    markers.current.forEach((marker, slug) => {
      const inner = marker.getElement().firstElementChild as HTMLElement | null;
      if (!inner) return;
      if (slug === selectedSlug) {
        inner.style.outline = "3px solid #F5C518";
        inner.style.outlineOffset = "2px";
      } else {
        inner.style.outline = "";
      }
    });
  }, [selectedSlug, patches]);

  // Live-mode "I clicked here" marker
  useEffect(() => {
    if (!map.current) return;
    if (liveMarkerRef.current) {
      liveMarkerRef.current.remove();
      liveMarkerRef.current = null;
    }
    if (!liveMarker) return;

    const outer = document.createElement("div");
    const inner = document.createElement("div");
    inner.style.width = "24px";
    inner.style.height = "24px";
    inner.style.borderRadius = "9999px";
    inner.style.background = "rgba(245, 197, 24, 0.9)";
    inner.style.border = "2px solid #000";
    inner.style.boxShadow = "0 0 0 6px rgba(245, 197, 24, 0.25)";
    outer.appendChild(inner);

    liveMarkerRef.current = new maplibregl.Marker({ element: outer, anchor: "center" })
      .setLngLat([liveMarker.lng, liveMarker.lat])
      .addTo(map.current);
    map.current.easeTo({ center: [liveMarker.lng, liveMarker.lat], zoom: 11, duration: 500 });
  }, [liveMarker]);

  // Fly to a location from the search box
  useEffect(() => {
    if (!map.current || !flyTo) return;
    map.current.flyTo({ center: [flyTo.lng, flyTo.lat], zoom: 11, duration: 1200 });
  }, [flyTo]);

  // Render the 664 verified mining sites as a circle layer (GPU-rendered, scales well).
  // Click a circle → same behavior as clicking the map in live mode.
  useEffect(() => {
    const m = map.current;
    if (!m || !miningSites || miningSites.length === 0) return;

    const setup = () => {
      if (m.getLayer("mining-sites-layer")) m.removeLayer("mining-sites-layer");
      if (m.getSource("mining-sites")) m.removeSource("mining-sites");

      m.addSource("mining-sites", {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: miningSites.map((s) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [s.lng, s.lat] },
            properties: { id: s.id, status: s.status ?? "unverified" },
          })),
        },
      });

      // Gold = v9 confirms the earthrise detection.
      // Grey = earthrise detected, v9 did not (out-of-distribution tile or off-center centroid).
      // Muted = not yet verified (the ~85 sites cut off by the network drop).
      m.addLayer({
        id: "mining-sites-layer",
        type: "circle",
        source: "mining-sites",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 2.5, 12, 6],
          "circle-color": [
            "match",
            ["get", "status"],
            "confirmed", "#F5C518",
            "empty", "#6b7280",
            "fetch_error", "#6b7280",
            /* unverified */ "#9ca3af",
          ],
          "circle-opacity": [
            "match",
            ["get", "status"],
            "confirmed", 0.75,
            /* otherwise */ 0.45,
          ],
          "circle-stroke-color": "#000",
          "circle-stroke-width": 0.5,
        },
      });

      m.on("click", "mining-sites-layer", (e) => {
        if (!e.features || e.features.length === 0) return;
        const feat = e.features[0];
        if (feat.geometry.type !== "Point") return;
        const [lng, lat] = feat.geometry.coordinates;
        onMapClickRef.current?.(lng, lat);
      });

      m.on("mouseenter", "mining-sites-layer", () => {
        m.getCanvas().style.cursor = "pointer";
      });
      m.on("mouseleave", "mining-sites-layer", () => {
        m.getCanvas().style.cursor = "";
      });
    };

    if (m.isStyleLoaded()) setup();
    else m.once("load", setup);
  }, [miningSites]);

  return <div ref={mapContainer} className="w-full h-full" />;
}
