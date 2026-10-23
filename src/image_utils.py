"""Image validation, in-memory storage, and the optional crop / adjustment step.

The SAME `apply_edits` function produces the preview shown to the user and the image sent to the model,
so what you see is exactly what is analysed (before the model's own 224x224 resize).
"""
from __future__ import annotations

import io
import threading
import time
import uuid
import warnings
from dataclasses import dataclass

from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from . import config


class ImageError(ValueError):
    """A problem with the uploaded image or edit parameters, phrased for end users."""


# --------------------------------------------------------------------------- upload
def load_upload(data: bytes) -> Image.Image:
    """Validate raw bytes and return a clean RGB working image."""
    if not data:
        raise ImageError("The uploaded file is empty.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ImageError(f"The file is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
                         "Please upload a smaller image.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(data))
            fmt = probe.format
            width, height = probe.size
            probe.verify()  # detects many truncated / corrupted files
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageError("The image has too many pixels to process safely. "
                         "Please resize it below ~64 megapixels and try again.") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ImageError("This file could not be read as an image. It may be corrupted or not an image.") from None

    if fmt not in config.ALLOWED_FORMATS:
        raise ImageError(f"Unsupported format{f' ({fmt})' if fmt else ''}. Please upload a JPG, PNG or WEBP image.")
    if width * height > config.MAX_PIXELS:
        raise ImageError(f"The image is too large ({width}x{height} pixels). "
                         "Please resize it below ~64 megapixels and try again.")
    if min(width, height) < config.MIN_SIDE:
        raise ImageError(f"The image is too small ({width}x{height}). Each side must be at least {config.MIN_SIDE} px.")

    try:
        img = Image.open(io.BytesIO(data))      # verify() invalidates the first handle; reopen
        img.seek(0)                              # first frame only (animated WEBP/PNG)
        img = ImageOps.exif_transpose(img)       # phones store rotation in EXIF; show what the user sees
        img = _flatten_to_rgb(img)
    except Exception:  # noqa: BLE001
        raise ImageError("This image appears to be corrupted and could not be decoded.") from None

    longest = max(img.size)
    if longest > config.WORKING_MAX_SIDE:        # keep memory/latency bounded; done once, on upload
        scale = config.WORKING_MAX_SIDE / longest
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


def _flatten_to_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB; transparent pixels are composited on white instead of turning black."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


# --------------------------------------------------------------------------- edits
@dataclass(frozen=True)
class Adjustments:
    brightness: float = 1.0   # 1.0 = unchanged (PIL ImageEnhance semantics)
    contrast: float = 1.0
    saturation: float = 1.0
    sharpness: float = 1.0

    def validate(self) -> "Adjustments":
        limits = {"brightness": (0.5, 1.5), "contrast": (0.5, 1.5),
                  "saturation": (0.0, 2.0), "sharpness": (0.0, 3.0)}
        for name, (lo, hi) in limits.items():
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value != value or not lo <= value <= hi:
                raise ImageError(f"{name.capitalize()} must be a number between {lo} and {hi}.")
        return self

    @property
    def is_identity(self) -> bool:
        return (self.brightness, self.contrast, self.saturation, self.sharpness) == (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class CropBox:
    """Fractions (0-1) of the working image, so it is independent of the display resolution."""
    x: float
    y: float
    w: float
    h: float


def parse_crop(raw: dict | None) -> CropBox | None:
    if not raw:
        return None
    try:
        box = CropBox(float(raw["x"]), float(raw["y"]), float(raw["w"]), float(raw["h"]))
    except (KeyError, TypeError, ValueError):
        raise ImageError("The crop area is invalid. Please select the crop again.") from None
    vals = (box.x, box.y, box.w, box.h)
    if any(v != v for v in vals) or box.w <= 0 or box.h <= 0 or box.x < -1e-6 or box.y < -1e-6 \
            or box.x + box.w > 1 + 1e-6 or box.y + box.h > 1 + 1e-6:
        raise ImageError("The crop area is outside the image. Please select the crop again.")
    return box


def parse_adjustments(raw: dict | None) -> Adjustments:
    raw = raw or {}
    try:
        adj = Adjustments(**{k: float(raw.get(k, 1.0)) for k in
                             ("brightness", "contrast", "saturation", "sharpness")})
    except (TypeError, ValueError):
        raise ImageError("Adjustment values must be numbers.") from None
    return adj.validate()


def apply_edits(img: Image.Image, crop: CropBox | None, adj: Adjustments) -> Image.Image:
    """Optional crop, then optional adjustments in a fixed order. With no crop and identity
    adjustments the original pixels are returned untouched."""
    out = img
    if crop is not None:
        left = max(0, round(crop.x * img.width))
        top = max(0, round(crop.y * img.height))
        right = min(img.width, round((crop.x + crop.w) * img.width))
        bottom = min(img.height, round((crop.y + crop.h) * img.height))
        if right - left < config.MIN_SIDE or bottom - top < config.MIN_SIDE:
            raise ImageError(f"The crop is too small. Keep at least {config.MIN_SIDE} px on each side.")
        out = img.crop((left, top, right, bottom))
    if adj.brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(adj.brightness)
    if adj.contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(adj.contrast)
    if adj.saturation != 1.0:
        out = ImageEnhance.Color(out).enhance(adj.saturation)
    if adj.sharpness != 1.0:
        out = ImageEnhance.Sharpness(out).enhance(adj.sharpness)
    return out


def to_jpeg(img: Image.Image, max_side: int | None = None, quality: int = 90) -> bytes:
    if max_side and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# --------------------------------------------------------------------------- in-memory store
class ImageStore:
    """Tiny TTL + LRU cache. Uploaded images are kept in RAM only and never written to disk."""

    def __init__(self, ttl: int = config.CACHE_TTL_SECONDS, max_items: int = config.CACHE_MAX_ITEMS):
        self._ttl, self._max = ttl, max_items
        self._items: dict[str, tuple[float, Image.Image]] = {}
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        for key in [k for k, (t, _) in self._items.items() if now - t > self._ttl]:
            del self._items[key]
        while len(self._items) > self._max:
            oldest = min(self._items, key=lambda k: self._items[k][0])
            del self._items[oldest]

    def put(self, img: Image.Image) -> str:
        key = uuid.uuid4().hex
        with self._lock:
            now = time.time()
            self._items[key] = (now, img)
            self._purge(now)
        return key

    def get(self, key: str) -> Image.Image:
        with self._lock:
            now = time.time()
            self._purge(now)
            item = self._items.get(key)
            if item is None:
                raise ImageError("Your image session expired. Please upload the image again.")
            self._items[key] = (now, item[1])   # touch
            return item[1]

    def delete(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)
