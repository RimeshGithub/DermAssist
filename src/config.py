"""Central configuration. Every value can be overridden with an environment variable
(handy for Hugging Face Space "Variables" without editing code)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- Model artifacts (produced by Section 13 of the notebook) -----------------
# MODEL_TAG = BACKBONE.split(".")[0] in the notebook -> "convnext_small"
MODEL_PATH = Path(os.getenv(
    "MODEL_PATH", ROOT / "models" / "dermassist_multimodal_convnext_small_final.pt"))
ENCODERS_PATH = Path(os.getenv(
    "ENCODERS_PATH", ROOT / "models" / "metadata_encoders.pkl"))
DISEASE_INFO_PATH = ROOT / "data" / "disease_info.json"
STATIC_DIR = ROOT / "static"

# ---- Network ------------------------------------------------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))

# ---- Upload limits ------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "15")) * 1024 * 1024
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(64_000_000)))      # reject absurd images (decompression bombs)
MIN_SIDE = 64                                                   # reject tiny images
WORKING_MAX_SIDE = int(os.getenv("WORKING_MAX_SIDE", "2048"))   # big images are down-scaled once on upload
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# ---- Session cache (images live in RAM only) ----------------------------------
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "1800"))
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "24"))

# ---- Metadata validation ------------------------------------------------------
AGE_MIN, AGE_MAX_INPUT = 0, 120
SEX_VALUES = ["female", "male", "unknown"]
SITE_VALUES = ["head_neck", "lower_extremity", "oral_genital", "palms_soles",
               "trunk", "unknown", "upper_extremity"]

# ---- Result presentation ------------------------------------------------------
TOP_K = 3
# Simple UI heuristic (NOT validated): below this top-1 score, show a "low score" notice.
LOW_SCORE_THRESHOLD = float(os.getenv("LOW_SCORE_THRESHOLD", "0.50"))
