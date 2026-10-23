"""Model loading and inference.

Everything here mirrors the notebook's final cell ("Predict function"):

    image  -> RGB -> Resize((img_size, img_size)) -> ToTensor -> Normalize(mean, std)
    meta   -> [age / age_max] + sex one-hot + site one-hot      (encoders fitted on the TRAIN split)
    logits -> softmax(logits / temperature)                     (temperature fitted on the VAL split)

img_size, mean, std, class_names, metadata_dim, backbone and temperature are all read from the
checkpoint, so nothing is hard-coded here.
"""
from __future__ import annotations

import logging
import pickle
import threading
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from . import config
from .model_def import MultimodalClassifier

import io, base64

log = logging.getLogger("dermassist.inference")


class ModelLoadError(RuntimeError):
    """Raised when the model artifacts are missing or incompatible."""


@dataclass
class Assets:
    model: torch.nn.Module
    class_names: list[str]
    temperature: float
    encoders: dict[str, Any]
    resize: T.Resize
    to_tensor_norm: T.Compose
    img_size: int
    backbone: str
    device: torch.device


_ASSETS: Assets | None = None
_LOAD_ERROR: str | None = None
_INFER_LOCK = threading.Lock()  # one forward pass at a time keeps CPU/RAM use predictable


# --------------------------------------------------------------------------- loading
def _select_device() -> torch.device:
    """Use a GPU when one is genuinely usable, otherwise fall back to CPU."""
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")  # fails early on broken driver/runtime combinations
            return torch.device("cuda")
    except Exception as exc:  # noqa: BLE001
        log.warning("CUDA reported available but is unusable (%s); using CPU.", exc)
    return torch.device("cpu")


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        # The notebook only stores tensors, floats, ints, strings and lists, so weights_only=True should
        # succeed. If a different torch version objects, fall back (the file is your own artifact).
        log.warning("weights_only load failed (%s); retrying with weights_only=False.", exc)
        return torch.load(path, map_location="cpu", weights_only=False)


def load_assets() -> Assets:
    """Load model + encoders once. Raises ModelLoadError with a readable message on failure."""
    if not config.MODEL_PATH.exists():
        raise ModelLoadError(
            f"Model checkpoint not found at '{config.MODEL_PATH}'. Copy "
            "'dermassist_multimodal_convnext_small_final.pt' from the notebook into the models/ folder.")
    if not config.ENCODERS_PATH.exists():
        raise ModelLoadError(
            f"Metadata encoders not found at '{config.ENCODERS_PATH}'. Copy 'metadata_encoders.pkl' "
            "from the notebook into the models/ folder.")

    device = _select_device()
    ckpt = _torch_load(config.MODEL_PATH)

    required = ["model_state_dict", "temperature", "class_names", "metadata_dim",
                "backbone", "img_size", "mean", "std"]
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ModelLoadError(f"Checkpoint is missing keys written by notebook Section 13: {missing}")

    class_names = list(ckpt["class_names"])
    with open(config.ENCODERS_PATH, "rb") as fh:
        encoders = pickle.load(fh)  # noqa: S301 - your own artifact from the notebook
    for key in ("sex_encoder", "site_encoder", "age_median", "age_max"):
        if key not in encoders:
            raise ModelLoadError(f"metadata_encoders.pkl is missing '{key}'.")

    model = MultimodalClassifier(
        num_classes=len(class_names),
        metadata_dim=int(ckpt["metadata_dim"]),
        meta_embed_dim=64,
        backbone=ckpt["backbone"],
        pretrained=False,   # weights come from the checkpoint; nothing is downloaded
        dropout=0.4,        # irrelevant in eval mode; matches the notebook's loader
    )
    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(
            "Checkpoint weights do not match the architecture in src/model_def.py "
            f"(wrong backbone/timm version?). Details: {str(exc)[:400]}") from exc
    model.to(device).eval()

    # Sanity check: encoders must produce the vector length the model was trained with.
    probe = encode_metadata(None, "unknown", "unknown", encoders)
    if probe.shape[1] != int(ckpt["metadata_dim"]):
        raise ModelLoadError(
            f"Encoder output length {probe.shape[1]} != checkpoint metadata_dim {ckpt['metadata_dim']}. "
            "The encoders and checkpoint come from different runs.")

    _warn_on_category_mismatch(encoders)

    img_size = int(ckpt["img_size"])
    resize = T.Resize((img_size, img_size))
    to_tensor_norm = T.Compose([T.ToTensor(), T.Normalize(mean=list(ckpt["mean"]), std=list(ckpt["std"]))])

    log.info("Loaded %s | classes=%s | T=%.4f | img=%d | device=%s",
             ckpt["backbone"], class_names, float(ckpt["temperature"]), img_size, device)
    return Assets(model, class_names, float(ckpt["temperature"]), encoders, resize,
                  to_tensor_norm, img_size, ckpt["backbone"], device)


def _warn_on_category_mismatch(encoders) -> None:
    """The encoders use handle_unknown='ignore': a category not seen in training silently becomes an
    all-zero block. Log loudly if a value offered in the UI would hit that path."""
    sex_cats = set(map(str, encoders["sex_encoder"].categories_[0]))
    site_cats = set(map(str, encoders["site_encoder"].categories_[0]))
    log.info("Encoder sex categories: %s", sorted(sex_cats))
    log.info("Encoder site categories: %s", sorted(site_cats))
    for v in config.SEX_VALUES:
        if v not in sex_cats:
            log.warning("UI sex value '%s' is NOT in the trained encoder; it will be encoded as zeros.", v)
    for v in config.SITE_VALUES:
        if v not in site_cats:
            log.warning("UI site value '%s' is NOT in the trained encoder; it will be encoded as zeros.", v)


def init_assets() -> None:
    """Called once at startup. A failure is stored (not raised) so the UI can show a clear message."""
    global _ASSETS, _LOAD_ERROR
    try:
        _ASSETS = load_assets()
        _LOAD_ERROR = None
    except Exception as exc:  # noqa: BLE001
        _ASSETS = None
        _LOAD_ERROR = str(exc)
        log.exception("Model loading failed")


def get_assets() -> Assets:
    if _ASSETS is None:
        raise ModelLoadError(_LOAD_ERROR or "Model is not loaded.")
    return _ASSETS


def status() -> dict:
    if _ASSETS is None:
        return {"ready": False, "error": _LOAD_ERROR}
    return {"ready": True, "error": None, "classes": _ASSETS.class_names,
            "backbone": _ASSETS.backbone, "device": str(_ASSETS.device), "img_size": _ASSETS.img_size}


# --------------------------------------------------------------------------- preprocessing
def encode_metadata(age: float | None, sex: str | None, site: str | None, enc: dict) -> np.ndarray:
    """Same as `encode_single_metadata` in the notebook. Returns shape (1, metadata_dim), float32.

    Missing age -> training-set median (as in training); missing sex/site -> the 'unknown' category."""
    age_val = enc["age_median"] if age is None else age
    age_scaled = np.array([[age_val / enc["age_max"]]], dtype=np.float32)   # NOT clipped, as in training
    sex_val = "unknown" if sex is None else sex
    site_val = "unknown" if site is None else site
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # encoders were fitted on DataFrames; we pass plain lists
        sex_oh = enc["sex_encoder"].transform([[sex_val]]).astype(np.float32)
        site_oh = enc["site_encoder"].transform([[site_val]]).astype(np.float32)
    return np.concatenate([age_scaled, sex_oh, site_oh], axis=1)


def prepare_image(image: Image.Image, assets: Assets) -> tuple[torch.Tensor, Image.Image]:
    """RGB PIL image -> normalized (1, 3, H, W) tensor. Also returns the resized PIL image (the exact
    pixels the model sees, before normalization) for display.

    Normalization is applied exactly once, here."""
    resized = assets.resize(image.convert("RGB"))
    tensor = assets.to_tensor_norm(resized).unsqueeze(0)
    return tensor, resized


# --------------------------------------------------------------------------- prediction
def predict(image: Image.Image, age: float | None, sex: str | None, site: str | None,
            top_k: int = config.TOP_K) -> dict:
    assets = get_assets()
    img_tensor, model_input = prepare_image(image, assets)
    meta_tensor = torch.from_numpy(encode_metadata(age, sex, site, assets.encoders))

    with _INFER_LOCK, torch.inference_mode():
        logits = assets.model(img_tensor.to(assets.device), meta_tensor.to(assets.device))
        logits = logits.float().cpu().squeeze(0)

    if not torch.isfinite(logits).all():
        raise RuntimeError("The model returned non-finite values for this input.")

    calibrated = F.softmax(logits / assets.temperature, dim=0)
    raw = F.softmax(logits, dim=0)
    k = min(top_k, len(assets.class_names))
    top_scores, top_idx = torch.topk(calibrated, k=k)

    return {
        "top_k": [
            {
                "rank": r + 1,
                "class_code": assets.class_names[i.item()],
                "class_index": i.item(),
                "score": float(s.item()),                        # temperature-scaled softmax
                "raw_score": float(raw[i.item()].item()),        # plain softmax (for transparency)
            }
            for r, (s, i) in enumerate(zip(top_scores, top_idx))
        ],
        "temperature": assets.temperature,
        "supported_classes": assets.class_names,
        "model_input": model_input,
    }

def _colormap(x):  # x in [0,1] -> RGB uint8 (blue -> green -> yellow -> red), no matplotlib needed
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def gradcam_overlays(image, age, sex, site, class_indices, alpha=0.45) -> dict:
    """Grad-CAM on the last ConvNeXt stage (7x7 grid at 224 px) for each class index.
    Returns {class_index: PNG data-URL of the heatmap blended over the 224x224 model input}."""
    a = get_assets()
    img_t, resized = prepare_image(image, a)
    meta = torch.from_numpy(encode_metadata(age, sex, site, a.encoders))
    layer = a.model.image_encoder.stages[-1]
    store = {}
    hook = layer.register_forward_hook(lambda m, i, o: store.setdefault("act", o))
    out = {}
    try:
        with _INFER_LOCK, torch.enable_grad():   # predict() uses inference_mode, which blocks gradients
            logits = a.model(img_t.to(a.device), meta.to(a.device))
            act = store["act"]
            for n, ci in enumerate(class_indices):
                grad = torch.autograd.grad(logits[0, ci], act,
                                           retain_graph=n < len(class_indices) - 1)[0]
                w = grad.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((w * act).sum(1, keepdim=True)).detach()
                cam = F.interpolate(cam, size=resized.size[::-1], mode="bilinear",
                                    align_corners=False)[0, 0].cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                base = np.asarray(resized.convert("RGB"), dtype=np.float32)
                blend = ((1 - alpha) * base + alpha * _colormap(cam).astype(np.float32)).astype(np.uint8)
                buf = io.BytesIO()
                Image.fromarray(blend).save(buf, "PNG")
                out[ci] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    finally:
        hook.remove()
    return out