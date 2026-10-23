---
title: DermAssist
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Experimental skin-lesion image classifier (not a diagnosis)
---

# DermAssist - skin lesion classifier web app

An experimental, AI-assisted information tool built around the multimodal ConvNeXt-Small model trained in
`multimodal_aggregated_skin_regularized_convnext_small_stratified.ipynb`. It ranks the top three lesion
categories for a photo plus optional age / sex / body-site details. **It does not provide a medical diagnosis.**

Stack: Python, PyTorch + timm (inference), FastAPI (backend), plain HTML/CSS/JS with Cropper.js (frontend),
Docker (Hugging Face Space). The frontend is custom rather than Gradio because the brief needs a
crop tool, live server-rendered adjustment previews, a stepped workflow and custom result cards.

---

## 1. What the notebook actually contains (inspected, not assumed)

| Item | Finding in the notebook | How the app handles it |
|---|---|---|
| Architecture | `MultimodalClassifier`: timm `convnext_small.fb_in22k_ft_in1k` (768-d, head removed) + metadata MLP (`dim -> 64 -> 64`, BatchNorm/ReLU/Dropout) -> concat (832) -> `Linear(832,256)` -> ReLU -> Dropout -> `Linear(256, C)` | `src/model_def.py` is a verbatim copy; weights load with `strict=True`. |
| Uses metadata? | **Yes.** Age, sex and anatomical site are fused into the prediction. | Metadata is encoded exactly as in training. The UI tells users these fields can influence the ranking. |
| Metadata encoding | `[age / age_max]` + sex one-hot + site one-hot. Missing age -> train median; missing sex/site -> `"unknown"`. `OneHotEncoder(handle_unknown="ignore")` fitted on the **train split**. | `encode_metadata()` in `src/inference.py` mirrors `encode_single_metadata`. Encoders, `age_median`, `age_max` come from `metadata_encoders.pkl`. |
| Image preprocessing | `Resize((IMG_SIZE, IMG_SIZE))` -> `ToTensor()` -> `Normalize(mean, std)`; no center-crop, no aspect-ratio preservation. `IMG_SIZE`/mean/std come from timm's config (224 px, ImageNet stats). | Read from the checkpoint (`img_size`, `mean`, `std`), applied once in `prepare_image()`. |
| Class order | `CLASS_NAMES = ["nv","mel","bcc","bkl","akiec","scc","df","vasc"]` (+ `"unknown"` only if `INCLUDE_UNKNOWN=True`). | Read from the checkpoint's `class_names`; index order is never assumed. |
| **8 vs 9 classes** | **`INCLUDE_UNKNOWN = False` in the notebook as provided, so the trained model has 8 outputs and no `unknown` class.** The brief lists 9. | The app adapts to whatever the checkpoint contains. With 8 classes the UI warns that the model **cannot reject unsupported images** (it always picks a supported category). If you retrain with `INCLUDE_UNKNOWN = True`, the `unknown` card and its information section work with no code changes. |
| Calibration | Temperature scaling fitted on validation logits; `temperature` saved in the checkpoint; the notebook's `predict()` reports softmax(logits / T). | Scores shown are temperature-scaled and labelled as such, with a plain-language note that they are not the probability that a person has the condition. |
| Saved artifacts | `dermassist_multimodal_convnext_small_final.pt` (keys: `model_state_dict, temperature, class_names, metadata_dim, backbone, img_size, mean, std`) and `metadata_encoders.pkl` (`sex_encoder, site_encoder, age_median, age_max`). | Both are required; see section 2. Nothing needs re-exporting. |
| Colab dependencies | Drive mount + `/content` paths exist only in the *training* cells. | None of that is used here; the app only needs the two artifact files. |

The notebook as uploaded contains **no cell outputs**, so I could not read the encoder category lists, temperature or metrics from it. The app logs the encoder categories at startup and warns if any dropdown value (for example `oral_genital`) is missing from the trained encoder, because `handle_unknown="ignore"` would silently turn it into an all-zero block.

## 2. Files you must copy from the notebook

Put both in `models/` (they are produced by notebook Section 13):

```
models/dermassist_multimodal_convnext_small_final.pt   # ~200 MB, tracked with Git LFS
models/metadata_encoders.pkl
```

Different filenames? Set the `MODEL_PATH` / `ENCODERS_PATH` environment variables (Space -> Settings -> Variables).

**Pin scikit-learn and numpy to your Colab versions.** `metadata_encoders.pkl` is a pickle of fitted sklearn objects. Run this in the notebook and copy the versions into `requirements.txt`:

```python
import sklearn, numpy, torch, timm
print(f"scikit-learn=={sklearn.__version__}  numpy=={numpy.__version__}  torch=={torch.__version__}  timm=={timm.__version__}")
```

## 3. Project structure

```
app.py                      FastAPI entry point and API routes
src/config.py               paths, limits, allowed metadata values (env-overridable)
src/model_def.py            model architecture (verbatim from the notebook)
src/inference.py            model/encoder loading, preprocessing, prediction
src/image_utils.py          upload validation, crop + adjustments, in-memory image store
src/disease_info.py         loads data/disease_info.json
data/disease_info.json      all disease text and source links (edit without touching code)
static/index.html|style.css|app.js    frontend
static/vendor/              Cropper.js 1.6.2 (vendored, MIT)
models/                     put the notebook artifacts here
scripts/make_dummy_artifacts.py   random-weight artifacts for testing plumbing
scripts/check_links.py      verifies the source URLs in disease_info.json
tests/test_app.py           API tests (22)
Dockerfile, requirements.txt, .dockerignore, .gitattributes
```

## 4. How the interface becomes the model's input tensor

1. **Upload** (`POST /api/upload`): bytes are validated (size, real decodable image, JPG/PNG/WEBP only, pixel count, minimum side). EXIF rotation is applied so the image looks the way it does on the user's phone; transparent pixels are flattened onto white; images larger than 2048 px on the long side are downscaled once. The result is held in memory only.
2. **Crop (optional)**: the browser sends a crop box as fractions of the working image. Nothing is cropped unless the user presses *Apply crop*.
3. **Adjustments (optional)**: brightness, contrast, saturation, sharpness via PIL `ImageEnhance` (1.0 = unchanged), applied in that fixed order after the crop. Sliders at 1.0 leave the pixels untouched.
4. **One function for preview and analysis**: `apply_edits()` produces both the on-screen preview and the image sent to the model, so what you see is what is analysed.
5. **Model preprocessing** (`prepare_image`): `Resize((224, 224))` -> `ToTensor()` (RGB, CHW, 0-1) -> `Normalize(mean, std)` -> add batch dimension. Normalization happens exactly once, here. The results page shows the resized 224x224 image the model received.
6. **Metadata**: age (blank -> training median), sex, site -> the notebook's vector, `float32`, shape `(1, metadata_dim)`.
7. **Inference**: model in `eval()`, `torch.inference_mode()`, `softmax(logits / temperature)`, `topk(3)`.

Small deviation worth knowing: the notebook did not apply EXIF rotation or alpha flattening when loading images. Both only make user phone photos match what the user sees; they do not change the model's pipeline.

## 5. Run locally

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# copy the two artifacts into models/ first (section 2)
python app.py                                             # http://localhost:7860
```

No artifacts yet? Check the plumbing with random weights (predictions are meaningless):

```bash
python scripts/make_dummy_artifacts.py --out models/_dummy --backbone convnext_atto
MODEL_PATH=models/_dummy/model.pt ENCODERS_PATH=models/_dummy/encoders.pkl python app.py
pip install pytest httpx && pytest -q tests               # 22 API tests
```

Docker locally:

```bash
docker build -t dermassist .
docker run --rm -p 7860:7860 dermassist
```

## 6. Deploy to Hugging Face Spaces (Docker)

1. Create a Space at huggingface.co/new-space: **SDK = Docker**, hardware = CPU basic (16 GB RAM is plenty; ConvNeXt-Small inference takes a couple of seconds on CPU).
2. Clone it and copy in all project files (this README's front matter already sets `sdk: docker` and `app_port: 7860`):
   ```bash
   git lfs install
   git clone https://huggingface.co/spaces/<you>/<space-name> && cd <space-name>
   # copy the project files here, then add models/*.pt and models/*.pkl
   git lfs track "*.pt"      # already in .gitattributes
   git add . && git commit -m "DermAssist" && git push
   ```
   Files over 10 MB are rejected unless tracked by LFS, so confirm `git lfs ls-files` lists the `.pt` file before pushing.
3. Watch **Logs** until you see `Loaded convnext_small.fb_in22k_ft_in1k | classes=[...]`. The header of the page also shows "Model ready".
4. Alternative for private weights: keep the checkpoint in a (private) Hub model repo, add an `HF_TOKEN` secret, and download it into `models/` at build time with `huggingface_hub`. This is not wired in by default, so the build never depends on network access beyond PyPI and the PyTorch wheel index.
5. Memory/CPU: the model is loaded once at startup, one worker, one forward pass at a time. The free CPU tier works; expect a short cold start after the Space sleeps.

## 7. Testing checklist

**Upload**
- [ ] JPG, PNG and WEBP upload by click and by drag-and-drop.
- [ ] A `.gif`, a `.txt` renamed to `.jpg`, an empty file, a truncated JPEG, and an image over the size limit each show a clear message and do not break the page.
- [ ] A phone photo that is rotated in EXIF appears upright.

**Crop and adjustments** (the crop tool itself should be checked in a real browser, including on a phone)
- [ ] *Crop* shows a draggable box; *Apply crop* updates the preview; *Remove crop* restores the full image.
- [ ] Sliders update the preview within about a second; *Reset adjustments* and *Restore original* work.
- [ ] With no crop and all sliders at 1.00, the image sent to the model is the unedited upload.

**Metadata**
- [ ] Age blank is accepted; `-5`, `200`, `abc` are rejected with a message under the field.
- [ ] Sex shows `female / male / unknown`; area shows the seven categories with readable labels.
- [ ] Changing age, sex or area and re-analyzing the same image can change the scores (the model is multimodal).

**Prediction and information**
- [ ] Exactly three cards, ranked high to low, each with name, class code, percentage, bar and short description.
- [ ] *More information* shows all seven sections plus sources; the "when to seek care" box is highlighted.
- [ ] The 8-class notice appears; the low-score notice appears when the top score is under 50%.
- [ ] Double-clicking *Analyze Image* sends one request; the button shows a spinner while running.
- [ ] Stop the model (rename the `.pt`) and reload: the app shows "Model unavailable" and no predictions are fabricated.

**Deployment**
- [ ] Space builds, logs show the model loaded, page loads over HTTPS, a full run-through works on desktop and mobile.
- [ ] `python scripts/check_links.py` reports no failing source links.

## 8. Assumptions, limits and things to review before going public

- **Disease text and links need a clinical review.** The content is written for a general audience from standard dermatology references. Only the AAD skin-cancer-types page was confirmed live while building; the other URLs follow each site's usual patterns, so run `scripts/check_links.py` and fix anything that fails.
- **`unknown` cannot be relied on for rejection.** The provided notebook trains no `unknown` class at all. Even with one, out-of-distribution photos (non-skin images, other conditions) can still be classified confidently.
- **Domain shift.** The training set mixes dermoscopic images (HAM10000, BCN20000) and smartphone clinical photos (PAD-UFES-20). The split was image-level, not lesion-level, so reported metrics may be optimistic. Performance on arbitrary user photos, and across skin tones, may be lower than on the test split.
- **The 50% low-score notice is a simple UI heuristic**, not a validated rejection threshold (`LOW_SCORE_THRESHOLD`).
- **Privacy.** Images live in server memory only (30-minute TTL, deleted on replace/close) and are never written to disk by this app. Hugging Face infrastructure logs are outside the app's control; don't log request bodies if you extend it.
- The crop tool relies on Cropper.js and could only be verified here through code review and a simulated-DOM run, not in a real browser. Please test it on desktop and mobile before launch.

## 9. Disclaimer

DermAssist is an experimental AI-assisted information tool for education and research. It does not provide medical advice, diagnosis or treatment. A visual classification cannot establish a diagnosis. If you are worried about a skin spot, see a doctor or dermatologist.
