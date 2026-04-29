"""Constants, Hugging Face IDs, Modal names, Sentinel-2 band indices, prompts.

Band indices are deliberately left as None. They must be resolved empirically
by loading one SmallMinesDS sample and inspecting the `.features` schema
(see `notebooks/01_dataset_schema.ipynb`). The dataset card documents a 10-band
Sentinel-2 L2A stack but the in-memory channel order is dataset-specific.
"""

# Hugging Face source dataset
HF_DATASET_ID = "ellaampy/SmallMinesDS"

# Modal configuration, our own app/volume names, distinct from Liquid's
# shared `satellite-vlm` volume
MODAL_VOLUME_NAME = "galamsey"
MODAL_MOUNT_POINT = "/galamsey"
MODAL_DATA_DIR = f"{MODAL_MOUNT_POINT}/data/smallminesds"

# Sentinel-2 band indices within SmallMinesDS's 13-channel stack.
# Resolved empirically from the dataset README, NOT from a notebook inspection.
# Canonical band order per SmallMinesDS docs:
#   Index 0-9:   S2 L2A [blue, green, red, rededge1, rededge2, rededge3,
#                        nir, rededge4, swir1, swir2]
#   Index 10-11: S1 RTC [vv, vh]
#   Index 12:    Copernicus DEM
#
# Our SWIR2-SWIR1-NIR false-color composite uses indices (9, 8, 6).
BAND_SWIR2: int = 9   # swir2, ~2190 nm
BAND_SWIR1: int = 8   # swir1, ~1610 nm
BAND_NIR: int = 6     # near infrared, ~842 nm (B8)

# Other useful indices, unused in v1 but kept for reference:
BAND_BLUE: int = 0
BAND_GREEN: int = 1
BAND_RED: int = 2
BAND_VV: int = 10
BAND_VH: int = 11
BAND_DEM: int = 12

# Ground sampling distance: Sentinel-2 is 10 m/pixel. SmallMinesDS patches are
# 128 x 128 pixels → each pixel covers 100 m² = 0.01 hectares.
HECTARES_PER_PIXEL = 0.01

# Prompts, channel-agnostic per ARCHITECTURE.md §2 prompt discipline.
# Red/green/blue vocabulary is forbidden since the red channel in our composite
# is SWIR2 reflectance, not visible red.
DESCRIPTION_PROMPT = (
    "You are analyzing a Sentinel-2 SWIR false-color composite (SWIR2, SWIR1, NIR) "
    "of southwestern Ghana. Describe any signs of illegal small-scale gold mining "
    "(galamsey) activity: exposed subsurface soil, excavation pits, sediment plumes, "
    "vegetation loss, and proximity to water bodies. If no mining is visible, say so."
)

GROUNDING_PROMPT = (
    "Inspect the image and detect any illegal small-scale gold mining pits. "
    'Provide result as a valid JSON: [{"label": str, "bbox": [x1,y1,x2,y2]}, ...]. '
    "Coordinates must be normalized to 0-1. If no pits are visible, return []."
)
