"""End-to-end tests of the API using RANDOM dummy artifacts (tests plumbing, not accuracy).

    pip install pytest httpx
    pytest -q tests
"""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    out = tmp_path_factory.mktemp("dummy")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_dummy_artifacts.py"),
                    "--out", str(out), "--backbone", "convnext_atto"], check=True)
    os.environ["MODEL_PATH"] = str(out / "model.pt")
    os.environ["ENCODERS_PATH"] = str(out / "encoders.pkl")
    from fastapi.testclient import TestClient
    import app as app_module
    with TestClient(app_module.app) as c:    # runs lifespan -> loads the model once
        yield c


def make_image(fmt="PNG", size=(300, 240), mode="RGB"):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr).convert(mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def upload(client, data, name="x.png", ctype="image/png"):
    return client.post("/api/upload", files={"file": (name, data, ctype)})


def test_health_ready(client):
    h = client.get("/api/health").json()
    assert h["ready"] is True and len(h["classes"]) == 8
    assert h["options"]["site"] == ["head_neck", "lower_extremity", "oral_genital", "palms_soles",
                                    "trunk", "unknown", "upper_extremity"]


@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_upload_formats(client, fmt, mime):
    r = upload(client, make_image(fmt), f"x.{fmt.lower()}", mime)
    assert r.status_code == 200, r.text
    assert r.json()["width"] == 300


def test_rejects_non_image_and_unsupported(client):
    assert upload(client, b"hello world", "a.png").status_code == 400
    assert upload(client, b"", "a.png").status_code == 400
    gif = make_image("GIF")
    r = upload(client, gif, "a.gif", "image/gif")
    assert r.status_code == 400 and "Unsupported" in r.json()["error"]


def test_rejects_truncated_and_tiny(client):
    data = make_image("JPEG")
    assert upload(client, data[: len(data) // 3], "a.jpg", "image/jpeg").status_code == 400
    assert upload(client, make_image("PNG", (20, 20))).status_code == 400


def test_rgba_png_is_flattened(client):
    r = upload(client, make_image("PNG", mode="RGBA"))
    assert r.status_code == 200


def test_preview_identity_matches_original(client):
    iid = upload(client, make_image()).json()["image_id"]
    orig = Image.open(io.BytesIO(client.get(f"/api/image/{iid}").content))
    prev = Image.open(io.BytesIO(client.post("/api/preview", json={"image_id": iid}).content))
    assert orig.size == prev.size


def test_preview_crop_and_adjust(client):
    iid = upload(client, make_image()).json()["image_id"]
    r = client.post("/api/preview", json={"image_id": iid, "crop": {"x": .1, "y": .1, "w": .5, "h": .5},
                                          "adjustments": {"brightness": 1.3, "sharpness": 2}})
    assert r.status_code == 200
    assert Image.open(io.BytesIO(r.content)).size == (150, 120)


def test_bad_crop_and_adjustments(client):
    iid = upload(client, make_image()).json()["image_id"]
    assert client.post("/api/preview", json={"image_id": iid, "crop": {"x": .9, "y": .9, "w": .5, "h": .5}}).status_code == 400
    assert client.post("/api/preview", json={"image_id": iid, "crop": {"x": 0, "y": 0, "w": .01, "h": .01}}).status_code == 400
    assert client.post("/api/preview", json={"image_id": iid, "adjustments": {"brightness": 9}}).status_code == 400


def test_analyze_top3(client):
    iid = upload(client, make_image()).json()["image_id"]
    r = client.post("/api/analyze", json={"image_id": iid, "age": 54, "sex": "female", "site": "trunk"})
    assert r.status_code == 200, r.text
    body = r.json()
    preds = body["predictions"]
    assert len(preds) == 3
    scores = [p["score"] for p in preds]
    assert scores == sorted(scores, reverse=True)
    assert all(p["class_code"] in body["supported_classes"] for p in preds)
    assert all(p["info"]["display_name"] for p in preds)
    assert body["model_input"].startswith("data:image/png;base64,")
    # 8-class model => must warn that it cannot reject unsupported images
    assert any("no 'unknown' output" in n["text"] for n in body["notices"])


def test_metadata_changes_output(client):
    """The model is multimodal: different metadata should be able to change the scores."""
    iid = upload(client, make_image()).json()["image_id"]
    a = client.post("/api/analyze", json={"image_id": iid, "age": 20, "sex": "male", "site": "head_neck"}).json()
    b = client.post("/api/analyze", json={"image_id": iid, "age": 80, "sex": "female", "site": "palms_soles"}).json()
    assert [p["score"] for p in a["predictions"]] != [p["score"] for p in b["predictions"]]


@pytest.mark.parametrize("payload", [
    {"age": -3}, {"age": 200}, {"age": "abc"}, {"sex": "robot"}, {"site": "elbow"}])
def test_invalid_metadata_rejected(client, payload):
    iid = upload(client, make_image()).json()["image_id"]
    r = client.post("/api/analyze", json={"image_id": iid, **payload})
    assert r.status_code == 400 and "error" in r.json()


def test_blank_metadata_ok(client):
    iid = upload(client, make_image()).json()["image_id"]
    r = client.post("/api/analyze", json={"image_id": iid, "age": "", "sex": None, "site": None})
    assert r.status_code == 200
    assert r.json()["metadata_used"]["age_supplied"] is False


def test_unknown_image_id(client):
    r = client.post("/api/analyze", json={"image_id": "deadbeef"})
    assert r.status_code == 400 and "expired" in r.json()["error"]


def test_encoder_matches_notebook_formula():
    """encode_metadata must equal [age/age_max] + sex one-hot + site one-hot, as in the notebook."""
    import pickle
    from src.inference import encode_metadata
    with open(os.environ["ENCODERS_PATH"], "rb") as fh:
        enc = pickle.load(fh)
    v = encode_metadata(42.0, "male", "trunk", enc)
    assert v.shape == (1, 1 + 3 + 7) and v.dtype == np.float32
    assert v[0, 0] == pytest.approx(42.0 / 85.0)
    assert v[0, 1:4].tolist() == [0, 1, 0]                       # female, male, unknown
    assert v[0, 4:].tolist() == [0, 0, 0, 0, 1, 0, 0]            # sorted site categories: trunk is index 4
    assert encode_metadata(None, None, None, enc)[0, 0] == pytest.approx(50.0 / 85.0)   # median imputation


def test_static_files(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/vendor/cropper.min.js").status_code == 200


def test_disease_info_complete():
    info = json.load(open(ROOT / "data" / "disease_info.json"))["classes"]
    for code in ["nv", "mel", "bcc", "bkl", "akiec", "scc", "df", "vasc", "unknown"]:
        assert code in info and info[code]["display_name"] and info[code]["description"]
