/* DermAssist frontend. No framework; server strings are always inserted with textContent. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SITE_LABELS = {
    head_neck: "Head or neck",
    trunk: "Trunk (chest, back, abdomen)",
    upper_extremity: "Upper extremity (arm, hand)",
    lower_extremity: "Lower extremity (leg, foot)",
    palms_soles: "Palms or soles",
    oral_genital: "Oral or genital area",
    unknown: "Unknown / not sure",
  };
  const SITE_ORDER = ["head_neck", "trunk", "upper_extremity", "lower_extremity", "palms_soles", "oral_genital", "unknown"];
  const SEX_LABELS = { female: "Female", male: "Male", unknown: "Unknown / prefer not to say" };
  const ADJ_KEYS = ["brightness", "contrast", "saturation", "sharpness"];
  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

  const state = {
    imageId: null,
    crop: null,                       // {x, y, w, h} as fractions of the working image, or null
    adj: { brightness: 1, contrast: 1, saturation: 1, sharpness: 1 },
    cropper: null,
    modelReady: false,
    maxMb: 15,
    ageMin: 0, ageMax: 120,
    analyzing: false,
    previewAbort: null,
    previewTimer: null,
    stageUrl: null,                   // current object URL (revoked when replaced)
  };

  // ------------------------------------------------------------------ small helpers
  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    for (const c of [].concat(children)) if (c != null) node.append(c.nodeType ? c : document.createTextNode(c));
    return node;
  }
  function showError(node, message) { node.textContent = message; node.hidden = !message; }
  const hasAdjustments = () => ADJ_KEYS.some((k) => state.adj[k] !== 1);
  const hasEdits = () => !!state.crop || hasAdjustments();

  async function api(path, options) {
    let res;
    try { res = await fetch(path, options); }
    catch { throw new Error("Could not reach the server. Check your connection and try again."); }
    if (!res.ok) {
      let msg = "Something went wrong. Please try again.";
      try { const j = await res.json(); msg = j.error || msg; } catch { /* non-JSON error */ }
      const err = new Error(msg); err.status = res.status; throw err;
    }
    return res;
  }

  // ------------------------------------------------------------------ stepper chips
  function updateChips() {
    const done = (id, on) => $(id).classList.toggle("done", on);
    const active = (id, on) => $(id).classList.toggle("active", on);
    const hasImg = !!state.imageId;
    done("chip-1", hasImg); active("chip-1", !hasImg);
    done("chip-2", hasImg && hasEdits()); active("chip-2", hasImg && !hasEdits());
    const ageVal = $("f-age").value.trim();
    const detailsTouched = ageVal !== "" || $("f-sex").value !== "unknown" || $("f-site").value !== "unknown";
    done("chip-3", hasImg && detailsTouched); active("chip-3", hasImg && !detailsTouched);
    active("chip-4", hasImg);
  }

  // ------------------------------------------------------------------ startup
  async function init() {
    for (const v of SITE_ORDER) $("f-site").append(el("option", { value: v, text: SITE_LABELS[v] }));
    for (const v of ["female", "male", "unknown"]) $("f-sex").append(el("option", { value: v, text: SEX_LABELS[v] }));
    $("f-sex").value = "unknown"; $("f-site").value = "unknown";

    try {
      const h = await (await api("/api/health")).json();
      state.modelReady = !!h.ready;
      state.maxMb = h.limits.max_upload_mb; state.ageMin = h.limits.age_min; state.ageMax = h.limits.age_max;
      $("max-mb").textContent = state.maxMb;
      // Rebuild dropdowns from the server's authoritative values (labels fall back to the raw value)
      $("f-site").replaceChildren(...[...h.options.site].sort((a, b) => SITE_ORDER.indexOf(a) - SITE_ORDER.indexOf(b))
        .map((v) => el("option", { value: v, text: SITE_LABELS[v] || v })));
      $("f-sex").replaceChildren(...h.options.sex.map((v) => el("option", { value: v, text: SEX_LABELS[v] || v })));
      $("f-sex").value = "unknown"; $("f-site").value = "unknown";
      const st = $("model-status");
      if (h.ready) { st.textContent = `Model ready (${h.classes.length} categories, ${h.device.toUpperCase()})`; st.className = "model-status ok"; }
      else {
        st.textContent = "Model unavailable"; st.className = "model-status bad";
        showError($("load-error"), "The model could not be loaded, so analysis is disabled. " + (h.error || ""));
      }
    } catch (e) {
      $("model-status").textContent = "Server unreachable"; $("model-status").className = "model-status bad";
      showError($("load-error"), e.message);
    }
    refreshControls(); updateChips();
  }

  // ------------------------------------------------------------------ upload
  const dz = $("dropzone"), fileInput = $("file-input");
  dz.addEventListener("click", () => fileInput.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer?.files?.[0]; if (f) handleFile(f); });
  fileInput.addEventListener("change", () => { const f = fileInput.files[0]; if (f) handleFile(f); fileInput.value = ""; });
  $("btn-replace").addEventListener("click", () => fileInput.click());

  async function handleFile(file) {
    showError($("upload-error"), "");
    if (!ALLOWED_TYPES.includes(file.type)) {
      return showError($("upload-error"), "Unsupported file type. Please choose a JPG, PNG or WEBP image.");
    }
    if (file.size > state.maxMb * 1024 * 1024) {
      return showError($("upload-error"), `That file is larger than ${state.maxMb} MB. Please choose a smaller image.`);
    }
    if (file.size === 0) return showError($("upload-error"), "That file is empty.");

    setStageBusy(true);
    try {
      const body = new FormData(); body.append("file", file);
      const data = await (await api("/api/upload", { method: "POST", body })).json();
      const previous = state.imageId;
      destroyCropper();
      state.imageId = data.image_id; state.crop = null; resetAdjustmentsState();
      if (previous) api(`/api/image/${previous}`, { method: "DELETE" }).catch(() => {});
      clearResults();
      dz.hidden = true; $("stage").hidden = false;
      await refreshStage();
    } catch (e) {
      showError($("upload-error"), e.message);
    } finally {
      setStageBusy(false);
      refreshControls(); updateChips();
    }
  }

  // ------------------------------------------------------------------ stage / preview
  function setStageBusy(on) {
    dz.style.opacity = on && !state.imageId ? ".6" : "";
    $("stage-busy").hidden = !on;
  }

  function editPayload() {
    return { image_id: state.imageId, crop: state.crop, adjustments: { ...state.adj } };
  }

  /** Fetch the server-rendered image (crop + adjustments applied by the same code used for analysis). */
  async function refreshStage() {
    if (!state.imageId) return;
    if (state.previewAbort) state.previewAbort.abort();
    const ctrl = new AbortController(); state.previewAbort = ctrl;
    $("stage-busy").hidden = false;
    try {
      const res = hasEdits()
        ? await api("/api/preview", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(editPayload()), signal: ctrl.signal })
        : await api(`/api/image/${state.imageId}`, { signal: ctrl.signal });
      const url = URL.createObjectURL(await res.blob());
      const old = state.stageUrl; state.stageUrl = url;
      $("stage-img").src = url;
      if (old) URL.revokeObjectURL(old);
      showError($("upload-error"), "");
    } catch (e) {
      if (e.name !== "AbortError") showError($("upload-error"), e.message);
    } finally {
      if (state.previewAbort === ctrl) { $("stage-busy").hidden = true; state.previewAbort = null; }
    }
  }

  function schedulePreview() {
    clearTimeout(state.previewTimer);
    state.previewTimer = setTimeout(refreshStage, 250);   // debounce slider drags
  }

  // ------------------------------------------------------------------ cropping
  function destroyCropper() {
    if (state.cropper) { state.cropper.destroy(); state.cropper = null; }
    $("crop-controls-active").hidden = true; $("crop-controls-idle").hidden = false;
  }

  $("btn-crop").addEventListener("click", async () => {
    if (!state.imageId) return;
    // Cropping always happens on the UNEDITED working image so crop fractions map to the original pixels.
    if (state.previewAbort) state.previewAbort.abort();
    clearTimeout(state.previewTimer);
    try {
      const blob = await (await api(`/api/image/${state.imageId}`)).blob();
      const url = URL.createObjectURL(blob);
      const old = state.stageUrl; state.stageUrl = url;
      const img = $("stage-img");
      await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = reject; img.src = url; });
      if (old) URL.revokeObjectURL(old);
      const natural = { w: img.naturalWidth, h: img.naturalHeight };
      state.cropper = new Cropper(img, {
        viewMode: 1, autoCropArea: state.crop ? 0.8 : 0.85, background: false, responsive: true, rotatable: false,
        scalable: false, zoomable: false, movable: false, dragMode: "crop", guides: true, center: true,
        data: state.crop ? { x: state.crop.x * natural.w, y: state.crop.y * natural.h,
                             width: state.crop.w * natural.w, height: state.crop.h * natural.h } : undefined,
      });
      $("crop-controls-idle").hidden = true; $("crop-controls-active").hidden = false;
      $("stage-hint").textContent = "Drag the corners or the box to select the affected skin, then choose Apply crop.";
    } catch (e) {
      showError($("upload-error"), e.message || "Could not start the crop tool.");
    }
  });

  $("btn-apply-crop").addEventListener("click", async () => {
    if (!state.cropper) return;
    const d = state.cropper.getData(true);
    const nat = state.cropper.getImageData();
    const crop = { x: d.x / nat.naturalWidth, y: d.y / nat.naturalHeight, w: d.width / nat.naturalWidth, h: d.height / nat.naturalHeight };
    const clamp = (v) => Math.min(1, Math.max(0, v));
    crop.x = clamp(crop.x); crop.y = clamp(crop.y);
    crop.w = Math.min(crop.w, 1 - crop.x); crop.h = Math.min(crop.h, 1 - crop.y);
    const nearlyFull = crop.x < 0.005 && crop.y < 0.005 && crop.w > 0.99 && crop.h > 0.99;
    state.crop = nearlyFull ? null : crop;       // selecting everything == no crop
    destroyCropper();
    $("stage-hint").textContent = "Cropping is optional. Drag the box to cover just the affected skin, with a little healthy skin around it.";
    await refreshStage();
    refreshControls(); updateChips();
  });

  $("btn-cancel-crop").addEventListener("click", async () => {
    destroyCropper();
    await refreshStage();
  });

  $("btn-reset-crop").addEventListener("click", async () => {
    state.crop = null; await refreshStage(); refreshControls(); updateChips();
  });

  // ------------------------------------------------------------------ adjustments
  function resetAdjustmentsState() {
    for (const k of ADJ_KEYS) { state.adj[k] = 1; $("s-" + k).value = 1; $("o-" + k).textContent = "1.00"; }
  }
  for (const k of ADJ_KEYS) {
    $("s-" + k).addEventListener("input", (e) => {
      state.adj[k] = parseFloat(e.target.value);
      $("o-" + k).textContent = state.adj[k].toFixed(2);
      refreshControls(); updateChips();
      if (!state.cropper) schedulePreview();
    });
  }
  $("btn-reset-adj").addEventListener("click", async () => {
    resetAdjustmentsState(); refreshControls(); updateChips(); await refreshStage();
  });
  $("btn-reset-all").addEventListener("click", async () => {
    destroyCropper(); state.crop = null; resetAdjustmentsState(); refreshControls(); updateChips(); await refreshStage();
  });

  // ------------------------------------------------------------------ form + controls state
  function refreshControls() {
    const hasImg = !!state.imageId;
    for (const k of ADJ_KEYS) $("s-" + k).disabled = !hasImg;
    $("btn-crop").disabled = !hasImg;
    $("btn-reset-crop").hidden = !state.crop;
    $("btn-reset-adj").disabled = !hasAdjustments();
    $("btn-reset-all").disabled = !hasEdits();
    const canAnalyze = hasImg && state.modelReady && !state.analyzing;
    $("btn-analyze").disabled = !canAnalyze;
    $("analyze-hint").textContent = !hasImg ? "Upload a photo to enable analysis."
      : !state.modelReady ? "The model is unavailable, so analysis is disabled."
      : "Ready. The photo is analyzed with any crop and adjustments you applied.";
  }

  function validateAge() {
    const input = $("f-age"), help = $("age-help"), raw = input.value.trim();
    const reset = () => { input.classList.remove("invalid"); help.classList.remove("error");
      help.textContent = `${state.ageMin} to ${state.ageMax}. If left blank, the typical age from the training data is used.`; };
    if (raw === "") { reset(); return { ok: true, value: "" }; }
    const n = Number(raw);
    if (!Number.isFinite(n) || n < state.ageMin || n > state.ageMax) {
      input.classList.add("invalid"); help.classList.add("error");
      help.textContent = `Enter an age between ${state.ageMin} and ${state.ageMax}, or leave it blank.`;
      return { ok: false };
    }
    reset(); return { ok: true, value: n };
  }
  $("f-age").addEventListener("input", () => { validateAge(); updateChips(); });
  $("f-sex").addEventListener("change", updateChips);
  $("f-site").addEventListener("change", updateChips);

  // ------------------------------------------------------------------ analyze
  $("btn-analyze").addEventListener("click", analyze);

  async function analyze() {
    if (state.analyzing) return;                  // blocks double submissions
    showError($("form-error"), "");
    if (!state.imageId) return showError($("form-error"), "Please upload a photo first.");
    if (state.cropper) return showError($("form-error"), "Finish cropping first: choose Apply crop or Cancel.");
    const age = validateAge();
    if (!age.ok) { $("f-age").focus(); return showError($("form-error"), "Please fix the age field before analyzing."); }

    state.analyzing = true; refreshControls();
    $("btn-analyze").querySelector(".btn-label").hidden = true;
    $("btn-analyze").querySelector(".btn-loading").hidden = false;
    $("results").hidden = false; $("results-body").hidden = true; showError($("results-error"), "");
    $("results-loading").hidden = false;
    $("results").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });

    try {
      const res = await api("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...editPayload(), age: age.value === "" ? null : age.value,
                               sex: $("f-sex").value, site: $("f-site").value }) });
      renderResults(await res.json());
    } catch (e) {
      $("results-loading").hidden = true;
      showError($("results-error"), e.message);          // no predictions are shown on failure
    } finally {
      state.analyzing = false;
      $("btn-analyze").querySelector(".btn-label").hidden = false;
      $("btn-analyze").querySelector(".btn-loading").hidden = true;
      refreshControls();
    }
  }

  function clearResults() { $("results").hidden = true; $("results-body").hidden = true; $("prediction-list").replaceChildren(); }

  // ------------------------------------------------------------------ results rendering
  function listSection(title, items, cls = "") {
    if (!items || !items.length) return null;
    return el("div", { class: "info-sec " + cls }, [el("h4", { text: title }), el("ul", {}, items.map((t) => el("li", { text: t })))]);
  }
  function textSection(title, text) {
    if (!text) return null;
    return el("div", { class: "info-sec" }, [el("h4", { text: title }), el("p", { text })]);
  }

  function renderPrediction(p, i) {
    const info = p.info;
    const sections = [
      textSection("What it is", info.description),
      listSection("Common causes or associated factors", info.causes),
      listSection("Risk factors", info.risk_factors),
      listSection("Common characteristics", info.characteristics),
      listSection("Precautions and prevention", info.prevention),
      listSection("When to seek medical attention", info.seek_care, "care"),
      textSection("Additional information", info.additional),
    ];
    if (info.sources && info.sources.length) {
      sections.push(el("div", { class: "info-sec" }, [el("h4", { text: "Sources" }),
        el("ul", { class: "src-list" }, info.sources.map((s) =>
          el("li", {}, [el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer", text: s.name })])))]));
    }
    const bar = el("div", { class: "bar", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100",
      "aria-valuenow": String(p.score_pct), "aria-label": `Model score for ${info.display_name}` }, [el("span")]);
    requestAnimationFrame(() => { bar.firstChild.style.width = Math.max(1, p.score_pct) + "%"; });

    return el("li", { class: "pred" + (i === 0 ? " top" : "") }, [
      el("div", { class: "pred-head" }, [
        el("span", { class: "rank", "aria-label": `Rank ${p.rank}`, text: String(p.rank) }),
        el("div", {}, [
          el("h3", { class: "pred-name", text: info.display_name }),
          el("div", { class: "pred-sub" }, [
            el("span", { class: "badge code", text: p.class_code }),
            info.category ? el("span", { class: "badge", text: info.category }) : null,
          ]),
        ]),
      ]),
      el("div", { class: "meter" }, [
        el("div", { class: "meter-row" }, [el("span", { text: "Model score" }), el("strong", { text: p.score_pct.toFixed(1) + "%" })]),
        bar,
      ]),
      el("p", { class: "pred-desc", text: info.short_description }),
      
      p.heatmap
      ? (() => {
          const bodyId = `heat-body-${p.id ?? Math.random().toString(36).slice(2, 8)}`;

          const chevron = el("span", { class: "heat-chevron", text: "▾", "aria-hidden": "true" });

          const toggleBtn = el(
            "button",
            {
              type: "button",
              class: "heat-toggle",
              "aria-expanded": "false",
              "aria-controls": bodyId
            },
            [
              el("div", { class: "heat-title-wrap" }, [
                el("span", { class: "heat-icon", text: "◎" }),
                el("div", {}, [
                  el("h4", { text: "AI Attention Heatmap" }),
                  el("p", {
                    class: "heat-subtitle",
                    text: "Regions influencing the prediction"
                  })
                ])
              ]),
              el("div", { class: "heat-header-right" }, [
                el("span", { class: "heat-badge", text: "Grad-CAM" }),
                chevron
              ])
            ]
          );

          const body = el("div", { class: "heat-body", id: bodyId }, [
            el("div", { class: "heat-body-inner" }, [
              el("div", { class: "heat-image-wrap" }, [
                el("img", {
                  src: p.heatmap,
                  alt: `Grad-CAM visualization of image regions influencing the prediction for ${info.display_name}`,
                  class: "heat-image",
                  loading: "lazy",
                  width: "320",
                  height: "320"
                })
              ]),

              el("div", { class: "heat-legend" }, [
                el("span", { class: "heat-dot heat-hot" }),
                el("span", { text: "Higher influence" }),
                el("span", { class: "heat-dot heat-cool" }),
                el("span", { text: "Lower influence" })
              ]),

              el("figcaption", {
                class: "heat-caption",
                text: "Warmer colors indicate image regions that contributed more strongly to this prediction."
              })
            ])
          ]);

          const card = el("figure", { class: "heat-card" }, [toggleBtn, body]);

          toggleBtn.addEventListener("click", () => {
            const open = card.classList.toggle("is-open");
            toggleBtn.setAttribute("aria-expanded", String(open));
          });

          return card;
        })()
      : null,

      el("details", { class: "more" }, [
        el("summary", {}, [
          el("span", { class: "more-icon", text: "ⓘ", "aria-hidden": "true" }),
          el("span", { class: "more-label", text: "More Information" }),
          el("span", { class: "more-chevron", text: "▾", "aria-hidden": "true" })
        ]),
        el("div", { class: "more-body" }, [
          el("div", { class: "info-grid" }, sections.filter(Boolean))
        ])
      ]),
    ]);
  }

  function renderResults(r) {
    $("results-loading").hidden = true;
    const notices = $("notices"); notices.replaceChildren();
    for (const n of r.notices) notices.append(el("div", { class: "alert alert-" + (n.level === "warn" ? "warn" : "info"), text: n.text }));

    $("prediction-list").replaceChildren(...r.predictions.map(renderPrediction));
    $("model-input").src = r.model_input;
    $("model-input-caption").textContent = `The photo (with your crop and adjustments) resized to ${r.model_input_size}\u00d7${r.model_input_size} pixels, the size the model reads.`;
    const m = r.metadata_used;
    $("meta-used").textContent = `Details used: age ${m.age_supplied ? m.age : "not given (typical training age used)"}, sex ${SEX_LABELS[m.sex] || m.sex}, area ${SITE_LABELS[m.site] || m.site}.`;
    $("supported").textContent = "Categories this model can predict: " + r.supported_classes.join(", ") + ".";

    const gs = $("general-sources"); gs.replaceChildren();
    if (r.general_sources?.length) {
      gs.append(el("p", { text: "Learn more from trusted sources:" }),
        el("ul", {}, r.general_sources.map((s) => el("li", {}, [el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer", text: s.name })]))));
    }
    $("results-body").hidden = false;
    $("results-body").querySelector(".pred.top details")?.setAttribute("open", "");
  }

  window.addEventListener("beforeunload", () => {
    if (state.imageId) navigator.sendBeacon?.(`/api/image/${state.imageId}/discard`);
  });

  init();
})();
