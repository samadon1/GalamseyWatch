"""JSONL conversation format, matches leap-finetune's `vlm_sft` schema.

This is the schema consumed by Liquid AI's training framework. The
`make_vlm_message` function is intentionally a line-for-line mirror of
`prepare_vrsbench.py` in Liquid's cookbook so we stay schema-compatible.
Any schema drift between our data and Liquid's expectations shows up as
silent training failures, so we do not invent our own format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def make_vlm_message(
    image_filename: str,
    user_text: str,
    assistant_text: str,
) -> dict[str, Any]:
    """Create a single VLM SFT training sample.

    The `image_filename` is a *relative* path, leap-finetune joins it with
    the `image_root` field in the YAML config at load time.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_filename},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]
    }


def write_jsonl(samples: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Serialize a list of samples to a JSONL file, creating parents as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    return output_path
