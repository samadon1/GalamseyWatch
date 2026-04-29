"""LFM2-1.2B tool-calling policy.

This is the contribution of the project — replacing the threshold-on-
confidence baseline with a small (1.2B) LLM that picks one of five tools
per tile (``downlink_now``, ``flag_for_review``, ``request_neighbor_tile``,
``request_higher_resolution``, ``discard``).

LFM2 emits Pythonic function calls wrapped in ``<|tool_call_start|>`` /
``<|tool_call_end|>`` markers. We parse the call with the AST, extract the
name + kwargs, and map to the orchestrator's existing ``Action`` literal.

Tool design follows Liquid's own guidance for small models:
- bounded action space (5 tools, finite state)
- maximally distinct names (avoid sibling-tool confusion)
- crisp one-line descriptions, no overlap
- single-step decision per tile (the bounded "guided dispatcher" regime
  Liquid recommends; their published 26% multi-step number is *not* what
  this loop is asking the model to do)
"""
from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
from dataclasses import dataclass
from time import perf_counter

logger = logging.getLogger(__name__)

DEFAULT_AGENT_MODEL = os.environ.get("LFM2_AGENT_MODEL", "LiquidAI/LFM2-2.6B")
AGENT_MAX_NEW_TOKENS = 192


# --- Tool definitions -----------------------------------------------------

TOOLS: list[dict] = [
    # Order matters: small models exhibit position bias on tool lists. The
    # default-most action (discard) is listed first deliberately — most tiles
    # over the AOI are forest/water with zero signal and should be skipped.
    {
        "type": "function",
        "function": {
            "name": "discard",
            "description": (
                "Skip this tile entirely. THIS IS THE DEFAULT FOR MOST TILES — forest, water, "
                "undisturbed land, heavy cloud, or ocean. Use whenever the VLM found 0 boxes or "
                "clearly described undisturbed terrain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_for_review",
            "description": (
                "Add this tile to the end-of-pass TEXT summary — no image downlink, just a brief "
                "log entry. Use for moderate-confidence detections (1-2 small boxes, ambiguous "
                "description) worth recording but not worth bandwidth."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_higher_resolution",
            "description": (
                "Request a higher-resolution recapture of this same tile next pass. Use when "
                "the VLM found a SMALL candidate (1 tiny box) that needs more pixels to confirm."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_neighbor_tile",
            "description": (
                "Fetch a tile in a given compass direction. Use only when the VLM described a "
                "feature that likely continues into the adjacent tile (e.g., sediment plume "
                "extending off-frame to the east)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["north", "south", "east", "west"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["direction", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "downlink_now",
            "description": (
                "Use the precious downlink budget to send THIS tile's image to ground during "
                "the current pass. RESERVE FOR HIGH-CONFIDENCE DETECTIONS ONLY: 2+ clear "
                "bounding boxes, confidence ≥ 0.85, AND the description explicitly mentions "
                "active pits, exposed soil, or sediment plumes. If in doubt, prefer "
                "flag_for_review — it's text-only and far cheaper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why this tile is worth the bandwidth.",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]

# Map LFM2 tool name → orchestrator Action literal
_TOOL_TO_ACTION: dict[str, str] = {
    "downlink_now": "downlink",
    "flag_for_review": "flag",
    "request_neighbor_tile": "request_neighbor",
    "request_higher_resolution": "request_hires",
    "discard": "discard",
}

# 3-tool minimal space — same names as the full toolset, just no hedge options.
_MINIMAL_TOOLS = [t for t in TOOLS if t["function"]["name"] in {"discard", "flag_for_review", "downlink_now"}]

_MINIMAL_FEW_SHOT: list[tuple[str, str]] = [
    (
        "Tile e001 at lon=-1.5000, lat=5.0000.\n"
        "Cloud cover: 5%. Captured: 2024-01-15T10:39:00Z.\n\n"
        "VLM detection:\n"
        "- 0 mining-pit candidate bounding box(es)\n"
        '- Description: "Continuous tropical forest canopy. No visible disturbance."\n'
        "- Derived confidence: 0.00\n\n"
        "Pass budget: 480 KB of 512 KB remaining.\n\n"
        "Choose the appropriate tool.",
        '<|tool_call_start|>[discard(reason="forest canopy, no detection")]<|tool_call_end|>',
    ),
    (
        "Tile e002 at lon=-1.9900, lat=5.3000.\n"
        "Cloud cover: 5%. Captured: 2024-01-15T10:39:00Z.\n\n"
        "VLM detection:\n"
        "- 4 mining-pit candidate bounding box(es) (largest area = 0.180)\n"
        '- Description: "Multiple active excavation pits with sediment plumes and exposed lateritic soil."\n'
        "- Derived confidence: 0.95\n\n"
        "Pass budget: 400 KB of 512 KB remaining.\n\n"
        "Choose the appropriate tool.",
        '<|tool_call_start|>[downlink_now(reason="4 high-confidence pits with sediment plumes — clear active galamsey")]<|tool_call_end|>',
    ),
    (
        "Tile e003 at lon=-1.7000, lat=5.4000.\n"
        "Cloud cover: 20%. Captured: 2024-01-15T10:39:00Z.\n\n"
        "VLM detection:\n"
        "- 1 mining-pit candidate bounding box(es) (largest area = 0.012)\n"
        '- Description: "A small bright patch near a stream bend; could be exposed soil or a sandbar."\n'
        "- Derived confidence: 0.70\n\n"
        "Pass budget: 200 KB of 512 KB remaining.\n\n"
        "Choose the appropriate tool.",
        '<|tool_call_start|>[flag_for_review(reason="single ambiguous small candidate; not worth bandwidth")]<|tool_call_end|>',
    ),
    (
        "Tile e004 at lon=-2.0500, lat=5.4500.\n"
        "Cloud cover: 8%. Captured: 2024-01-15T10:39:00Z.\n\n"
        "VLM detection:\n"
        "- 2 mining-pit candidate bounding box(es) (largest area = 0.090)\n"
        '- Description: "Two excavation pits with sediment-laden water; exposed soil clearly visible."\n'
        "- Derived confidence: 0.92\n\n"
        "Pass budget: 350 KB of 512 KB remaining.\n\n"
        "Choose the appropriate tool.",
        '<|tool_call_start|>[downlink_now(reason="2 high-confidence pits with sediment-laden water — active galamsey")]<|tool_call_end|>',
    ),
]


SYSTEM_PROMPT = (
    "You are an on-orbit Earth-observation agent on a satellite-class compute platform during a "
    "Sentinel-2 pass over southwestern Ghana. Mission: detect illegal small-scale gold mining "
    "(galamsey).\n\n"
    "For each tile, the on-board VLM has given you bounding boxes + a description. Reply with "
    "EXACTLY ONE tool call — no preamble, no alternatives, no questions.\n\n"
    "Three actions, each with a clear trigger:\n"
    "- **discard**: 0 boxes, OR description says forest/water/undisturbed/cloud-obscured.\n"
    "- **flag_for_review**: 1-2 boxes with low/moderate confidence, ambiguous descriptions, OR "
    "boundary cases. Cheap (text-only).\n"
    "- **downlink_now**: confidence ≥ 0.85 AND (≥2 boxes OR a single large box) AND the "
    "description names active galamsey features (pits, sediment plumes, exposed soil, turbid "
    "water). When these conditions are met you MUST call downlink_now — hedging to flag on "
    "clear positives defeats the satellite's purpose. Do not use flag_for_review as a default "
    "fallback; commit to downlink when the evidence is unambiguous."
)

# Few-shot examples teach the model (1) the output format and (2) calibration —
# discard for no-signal, downlink for clear positives, flag for ambiguous.
# Without these, LFM2-1.2B reliably falls back to chatty assistant mode and
# emits plain text instead of tool calls.
_FEW_SHOT: list[tuple[str, str]] = [
    (
        "Tile e001 at lon=-1.5000, lat=5.0000.\n"
        "Cloud cover: 5%. Captured: 2024-01-15T10:39:00Z.\n\n"
        "VLM detection:\n"
        "- 0 mining-pit candidate bounding box(es)\n"
        '- Description: "Continuous tropical forest canopy. No visible disturbance."\n'
        "- Derived confidence: 0.00\n\n"
        "Pass budget: 480 KB of 512 KB remaining.\n\n"
        "Choose the appropriate tool.",
        '<|tool_call_start|>[discard(reason="forest canopy, no detection")]<|tool_call_end|>',
    ),
    (
        "Tile e002 at lon=-1.9900, lat=5.3000.\n"
        "Cloud cover: 5%. Captured: 2024-01-15T10:39:00Z.\n\n"
        "VLM detection:\n"
        "- 4 mining-pit candidate bounding box(es) (largest area = 0.180)\n"
        '- Description: "Multiple active excavation pits with sediment plumes and exposed lateritic soil."\n'
        "- Derived confidence: 0.95\n\n"
        "Pass budget: 400 KB of 512 KB remaining.\n\n"
        "Choose the appropriate tool.",
        '<|tool_call_start|>[downlink_now(reason="4 high-confidence pits with sediment plumes — clear active galamsey")]<|tool_call_end|>',
    ),
    (
        "Tile e003 at lon=-1.7000, lat=5.4000.\n"
        "Cloud cover: 20%. Captured: 2024-01-15T10:39:00Z.\n\n"
        "VLM detection:\n"
        "- 1 mining-pit candidate bounding box(es) (largest area = 0.012)\n"
        '- Description: "A small bright patch near a stream bend; could be exposed soil or a sandbar."\n'
        "- Derived confidence: 0.70\n\n"
        "Pass budget: 200 KB of 512 KB remaining.\n\n"
        "Choose the appropriate tool.",
        '<|tool_call_start|>[flag_for_review(reason="single ambiguous small candidate; not worth bandwidth")]<|tool_call_end|>',
    ),
    (
        "Tile e004 at lon=-2.0500, lat=5.4500.\n"
        "Cloud cover: 8%. Captured: 2024-01-15T10:39:00Z.\n\n"
        "VLM detection:\n"
        "- 2 mining-pit candidate bounding box(es) (largest area = 0.090)\n"
        '- Description: "Two excavation pits with sediment-laden water; exposed soil clearly visible."\n'
        "- Derived confidence: 0.92\n\n"
        "Pass budget: 350 KB of 512 KB remaining.\n\n"
        "Choose the appropriate tool.",
        '<|tool_call_start|>[downlink_now(reason="2 high-confidence pits with sediment-laden water — active galamsey")]<|tool_call_end|>',
    ),
]


@dataclass
class AgentDecision:
    action: str  # one of the orchestrator Action literals
    tool_name: str
    reason: str
    raw_text: str
    decision_ms: int


# --- LFM2 agent ----------------------------------------------------------

class LFM2Agent:
    """Process-level singleton; weights load on first call."""

    def __init__(self, model_id: str = DEFAULT_AGENT_MODEL) -> None:
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._device: str | None = None
        self._lock = asyncio.Lock()

    def _ensure_loaded_sync(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        logger.info(
            "loading agent %s on %s (dtype=%s)", self.model_id, device, dtype
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=dtype
        ).to(device)
        self._model.eval()
        self._device = device

    def _generate_sync(self, user_prompt: str) -> str:
        import torch

        self._ensure_loaded_sync()
        # AGENT_TOOLSET=minimal collapses to {discard, flag_for_review, downlink_now}.
        # The full toolset (default) keeps the active-observation actions
        # (request_higher_resolution, request_neighbor_tile) so the agent can
        # demonstrate non-binary behaviour on edge-case tiles.
        toolset = os.environ.get("AGENT_TOOLSET", "full")
        active_tools = TOOLS if toolset == "full" else _MINIMAL_TOOLS
        active_fewshot = _FEW_SHOT if toolset == "full" else _MINIMAL_FEW_SHOT

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u, a in active_fewshot:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_prompt})
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tools=active_tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self._device
        )
        with torch.inference_mode():
            # Low-temperature sampling rather than greedy. Greedy on a 1.2B with
            # 5-way tool choice consistently converged on whatever single tool
            # the model viewed as safest (flag_for_review or request_higher_resolution),
            # ignoring the per-tile signal. T=0.4 + top_p=0.9 lets the model
            # commit to differentiated actions without hallucinating noise.
            out = self._model.generate(
                **inputs,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                max_new_tokens=AGENT_MAX_NEW_TOKENS,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[-1]:]
        return self._tokenizer.batch_decode(gen, skip_special_tokens=False)[0]

    async def decide(self, tile_context: str) -> AgentDecision:
        async with self._lock:
            started = perf_counter()
            raw = await asyncio.to_thread(self._generate_sync, tile_context)
            elapsed_ms = int((perf_counter() - started) * 1000)
        return _parse_response(raw, elapsed_ms)


AGENT = LFM2Agent()


# --- Parsing -------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>",
    re.DOTALL,
)


def _parse_response(raw: str, elapsed_ms: int) -> AgentDecision:
    """Extract the tool call + free-form reasoning from a generation.

    Falls back to ``discard`` with a self-describing reason if the model
    didn't emit a parseable tool call. We surface this rather than silently
    falling through, so eval/ablation can count parser failures honestly.
    """
    match = _TOOL_CALL_RE.search(raw)
    if match is None:
        return AgentDecision(
            action="discard",
            tool_name="(no_tool_call)",
            reason="agent emitted no parseable tool call; default discard",
            raw_text=raw,
            decision_ms=elapsed_ms,
        )

    inner = match.group(1).strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1].strip()

    name, kwargs = _parse_pythonic_call(inner)
    if name is None:
        return AgentDecision(
            action="discard",
            tool_name="(unparseable)",
            reason=f"agent tool-call body did not parse: {inner[:120]}",
            raw_text=raw,
            decision_ms=elapsed_ms,
        )

    action = _TOOL_TO_ACTION.get(name)
    if action is None:
        return AgentDecision(
            action="discard",
            tool_name=name,
            reason=f"agent called unknown tool {name!r}; default discard",
            raw_text=raw,
            decision_ms=elapsed_ms,
        )

    primary_reason = str(kwargs.get("reason", "")).strip()
    direction = kwargs.get("direction")
    if direction:
        primary_reason = (
            f"[{direction}] {primary_reason}" if primary_reason else f"direction={direction}"
        )
    # Surface any free-form text the model wrote *after* the tool call —
    # LFM2 sometimes adds a one-line rationale.
    trailing = raw.split("<|tool_call_end|>", 1)[-1]
    trailing = trailing.split("<|im_end|>", 1)[0].strip()
    if trailing and trailing not in primary_reason:
        primary_reason = (
            f"{primary_reason} — {trailing}" if primary_reason else trailing
        )

    return AgentDecision(
        action=action,
        tool_name=name,
        reason=primary_reason or "(no reason given)",
        raw_text=raw,
        decision_ms=elapsed_ms,
    )


def _parse_pythonic_call(text: str) -> tuple[str | None, dict]:
    """Parse a Pythonic call like ``downlink_now(reason="hi")`` into name + kwargs."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None, {}
    if not isinstance(tree.body, ast.Call) or not isinstance(tree.body.func, ast.Name):
        return None, {}
    call = tree.body
    name = call.func.id  # type: ignore[attr-defined]
    kwargs: dict = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            kwargs[kw.arg] = None
    return name, kwargs


def build_tile_prompt(
    *,
    tile_id: str,
    lon: float,
    lat: float,
    cloud_cover: float | None,
    captured_at: str | None,
    boxes_count: int,
    max_area: float,
    description: str,
    overall_confidence: float,
    bandwidth_remaining_kb: int,
    bandwidth_total_kb: int,
) -> str:
    cc = f"{cloud_cover * 100:.0f}%" if cloud_cover is not None else "n/a"
    return (
        f"Tile {tile_id} at lon={lon:.4f}, lat={lat:.4f}.\n"
        f"Cloud cover: {cc}. Captured: {captured_at or 'unknown'}.\n\n"
        f"VLM detection:\n"
        f"- {boxes_count} mining-pit candidate bounding box(es)"
        + (f" (largest area = {max_area:.3f})" if boxes_count > 0 else "")
        + "\n"
        f'- Description: "{description.strip()}"\n'
        f"- Derived confidence: {overall_confidence:.2f}\n\n"
        f"Pass budget: {bandwidth_remaining_kb} KB of {bandwidth_total_kb} KB remaining.\n\n"
        "Choose the appropriate tool."
    )
