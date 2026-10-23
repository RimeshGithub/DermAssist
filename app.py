"""DermAssist - web app entry point (FastAPI + static frontend).

Run locally:   python app.py        (then open http://localhost:7860)
Docker/Spaces: see Dockerfile (uvicorn on 0.0.0.0:7860)
"""
from __future__ import annotations

import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config, disease_info, inference
from src.image_utils import (ImageError, ImageStore, apply_edits, load_upload,
                             parse_adjustments, parse_crop, to_jpeg)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dermassist")

store = ImageStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    torch.set_num_threads(max(1, torch.get_num_threads()))
    inference.init_assets()   # load the model ONCE at startup
    yield


app = FastAPI(title="DermAssist", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(ImageError)
async def image_error_handler(_: Request, exc: ImageError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


# --------------------------------------------------------------------------- request models
class EditRequest(BaseModel):
    image_id: str
    crop: Optional[dict] = None
    adjustments: Optional[dict] = None


class AnalyzeRequest(EditRequest):
    age: Optional[float | str] = None
    sex: Optional[str] = None
    site: Optional[str] = None


# --------------------------------------------------------------------------- helpers
def validate_metadata(age, sex, site) -> tuple[Optional[float], str, str]:
    """Validate user metadata. Blank age -> None (the model then uses the training median, exactly as
    missing ages were handled in training). Blank sex/site -> 'unknown'."""
    clean_age: Optional[float] = None
    if age is not None and str(age).strip() != "":
        try:
            clean_age = float(age)
        except (TypeError, ValueError):
            raise ImageError("Age must be a number, for example 54.") from None
        if clean_age != clean_age or not config.AGE_MIN <= clean_age <= config.AGE_MAX_INPUT:
            raise ImageError(f"Age must be between {config.AGE_MIN} and {config.AGE_MAX_INPUT}.")
    sex = (sex or "unknown").strip().lower()
    site = (site or "unknown").strip().lower()
    if sex not in config.SEX_VALUES:
        raise ImageError("Please choose a valid option for sex.")
    if site not in config.SITE_VALUES:
        raise ImageError("Please choose a valid option for the affected area.")
    return clean_age, sex, site


def _png_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return {
        **inference.status(),
        "options": {"sex": config.SEX_VALUES, "site": config.SITE_VALUES},
        "limits": {"max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024),
                   "age_min": config.AGE_MIN, "age_max": config.AGE_MAX_INPUT},
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read(config.MAX_UPLOAD_BYTES + 1)   # never buffer more than the limit + 1 byte
    img = load_upload(data)
    image_id = store.put(img)
    return {"image_id": image_id, "width": img.width, "height": img.height}


@app.get("/api/image/{image_id}")
def get_working_image(image_id: str):
    """The (unedited) working image used for display and for the crop tool."""
    img = store.get(image_id)
    return Response(to_jpeg(img, max_side=1600, quality=92), media_type="image/jpeg")


@app.delete("/api/image/{image_id}")
def delete_image(image_id: str):
    store.delete(image_id)
    return {"deleted": True}


@app.post("/api/image/{image_id}/discard")
def discard_image(image_id: str):
    """Target for navigator.sendBeacon when the tab closes (beacons can only POST)."""
    store.delete(image_id)
    return Response(status_code=204)


@app.post("/api/preview")
def preview(req: EditRequest):
    """Applies the same crop + adjustments as /api/analyze and returns a JPEG preview."""
    img = store.get(req.image_id)
    edited = apply_edits(img, parse_crop(req.crop), parse_adjustments(req.adjustments))
    return Response(to_jpeg(edited, max_side=900, quality=88), media_type="image/jpeg")


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    status = inference.status()
    if not status["ready"]:
        return JSONResponse(status_code=503, content={
            "error": "The model is not available right now, so no prediction can be made.",
            "detail": status["error"]})

    age, sex, site = validate_metadata(req.age, req.sex, req.site)
    img = store.get(req.image_id)
    edited = apply_edits(img, parse_crop(req.crop), parse_adjustments(req.adjustments))

    try:
        result = inference.predict(edited, age, sex, site)
        try:
            heatmaps = inference.gradcam_overlays(edited, age, sex, site,
                                                [p["class_index"] for p in result["top_k"]])
        except Exception:
            log.exception("Grad-CAM failed")
            heatmaps = {}
    except Exception:  # noqa: BLE001 - never fabricate output; report failure
        log.exception("Inference failed")
        return JSONResponse(status_code=500, content={
            "error": "The analysis failed unexpectedly. Please try again with a different image."})

    classes = result["supported_classes"]
    predictions = []
    for item in result["top_k"]:
        info = disease_info.get_info(item["class_code"])
        predictions.append({**item, "score_pct": round(item["score"] * 100, 1),
                    "info": info, "heatmap": heatmaps.get(item["class_index"])})

    top = predictions[0]
    notices = []
    if top["class_code"] == "unknown":
        notices.append({"level": "warn", "text": "The model's top category is 'unknown'. The image may not "
                        "match any supported category or may be unsuitable for classification."})
    if "unknown" not in classes:
        notices.append({"level": "info", "text": "This model was trained on 8 lesion categories and has no "
                        "'unknown' output, so it cannot say 'none of these'. It will always pick from the "
                        "supported categories, even for images that are not skin lesions."})
    if top["score"] < config.LOW_SCORE_THRESHOLD:
        notices.append({"level": "warn", "text": "The model's top score is low, so it is not clearly "
                        "favouring any one category. Treat this result with extra caution."})

    return {
        "predictions": predictions,
        "notices": notices,
        "supported_classes": classes,
        "temperature": result["temperature"],
        "metadata_used": {"age": age, "age_supplied": age is not None, "sex": sex, "site": site},
        "model_input": _png_data_url(result["model_input"]),
        "model_input_size": status["img_size"],
        "general_sources": disease_info.general_sources(),
    }


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, log_level="info")
