"""GalamseyWatch training primitives, shared by notebooks and Modal launchers."""

from galamseywatch.composites import (
    compose_swir_false_color,
    encode_png,
    percentile_stretch,
)
from galamseywatch.constants import (
    BAND_NIR,
    BAND_SWIR1,
    BAND_SWIR2,
    DESCRIPTION_PROMPT,
    GROUNDING_PROMPT,
    HECTARES_PER_PIXEL,
    HF_DATASET_ID,
    MODAL_DATA_DIR,
    MODAL_MOUNT_POINT,
    MODAL_VOLUME_NAME,
)
from galamseywatch.masks import analyze_mask, generate_description
from galamseywatch.vlm_format import make_vlm_message, write_jsonl

__all__ = [
    "BAND_NIR",
    "BAND_SWIR1",
    "BAND_SWIR2",
    "DESCRIPTION_PROMPT",
    "GROUNDING_PROMPT",
    "HECTARES_PER_PIXEL",
    "HF_DATASET_ID",
    "MODAL_DATA_DIR",
    "MODAL_MOUNT_POINT",
    "MODAL_VOLUME_NAME",
    "analyze_mask",
    "compose_swir_false_color",
    "encode_png",
    "generate_description",
    "make_vlm_message",
    "percentile_stretch",
    "write_jsonl",
]
