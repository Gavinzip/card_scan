const state = {
  file: null,
  previewUrl: null,
  clientCropPreviewUrl: null,
  scanning: false,
};

const TCGP_OBB_MODE = "tcgp_obb";
const TCGP_OBB_MODEL_URL = "/assets/tcgp-obb-model/model.json";
const TCGP_OBB_TFJS_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
const TCGP_OBB_TFJS_WASM_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js";
const TCGP_OBB_TFJS_WASM_PATH = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/";
const TCGP_OBB_INPUT_SIZE = 640;
const TCGP_OBB_SCORE_THRESHOLD = 0.1;
const TCGP_OBB_IOU_THRESHOLD = 0.1;

let tfjsLoadPromise = null;
let tfjsBackendPromise = null;
let tcgpObbModelPromise = null;
let tcgpObbModel = null;
let tcgpObbWarmupPromise = null;

const els = {
  apiBaseInput: document.querySelector("#apiBaseInput"),
  cardCodeOcrToggle: document.querySelector("#cardCodeOcrToggle"),
  confidenceInput: document.querySelector("#confidenceInput"),
  confidenceOutput: document.querySelector("#confidenceOutput"),
  cropPreview: document.querySelector("#cropPreview"),
  cropModeInput: document.querySelector("#cropModeInput"),
  cropStatus: document.querySelector("#cropStatus"),
  cropToggle: document.querySelector("#cropToggle"),
  dropZone: document.querySelector("#dropZone"),
  fallbackToggle: document.querySelector("#fallbackToggle"),
  fileInput: document.querySelector("#fileInput"),
  fileMeta: document.querySelector("#fileMeta"),
  healthPill: document.querySelector("#healthPill"),
  healthText: document.querySelector("#healthText"),
  hintTextInput: document.querySelector("#hintTextInput"),
  languageOcrToggle: document.querySelector("#languageOcrToggle"),
  languageRerankToggle: document.querySelector("#languageRerankToggle"),
  previewFrame: document.querySelector("#previewFrame"),
  previewImage: document.querySelector("#previewImage"),
  recognizeButton: document.querySelector("#recognizeButton"),
  resultCount: document.querySelector("#resultCount"),
  resultsList: document.querySelector("#resultsList"),
  statusText: document.querySelector("#statusText"),
  slabLookupToggle: document.querySelector("#slabLookupToggle"),
  timingGrid: document.querySelector("#timingGrid"),
  topKInput: document.querySelector("#topKInput"),
  visualRerankToggle: document.querySelector("#visualRerankToggle"),
  warmupButton: document.querySelector("#warmupButton"),
};

function endpoint(path) {
  const base = els.apiBaseInput.value.trim().replace(/\/+$/, "");
  return base ? `${base}${path}` : path;
}

function seconds(value) {
  return typeof value === "number" ? `${value.toFixed(3)}s` : "-";
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      existing.addEventListener("load", resolve, { once: true });
      existing.addEventListener("error", reject, { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(script);
  });
}

async function loadTensorFlowJs() {
  if (window.tf) return configureTensorFlowBackend(window.tf);
  if (!tfjsLoadPromise) {
    tfjsLoadPromise = loadScript(TCGP_OBB_TFJS_URL).then(() => {
      if (!window.tf) throw new Error("TensorFlow.js did not initialize.");
      return window.tf;
    });
  }
  return configureTensorFlowBackend(await tfjsLoadPromise);
}

async function configureTensorFlowBackend(tf) {
  if (!tfjsBackendPromise) {
    tfjsBackendPromise = (async () => {
      let usingWebgl = false;
      try {
        if (tf.getBackend?.() !== "webgl") await tf.setBackend("webgl");
        await tf.ready();
        usingWebgl = tf.getBackend?.() === "webgl";
      } catch (error) {
        console.warn("TCGP OBB WebGL backend unavailable, using TensorFlow.js fallback.", error);
      }
      if (!usingWebgl) {
        try {
          await loadScript(TCGP_OBB_TFJS_WASM_URL);
          if (typeof tf.setWasmPaths === "function") tf.setWasmPaths(TCGP_OBB_TFJS_WASM_PATH);
          if (typeof tf.wasm?.setWasmPaths === "function") tf.wasm.setWasmPaths(TCGP_OBB_TFJS_WASM_PATH);
          await tf.setBackend("wasm");
          await tf.ready();
        } catch (error) {
          console.warn("TCGP OBB WASM backend unavailable, using TensorFlow.js CPU fallback.", error);
        }
      }
      await tf.ready();
      return tf;
    })();
  }
  return tfjsBackendPromise;
}

async function loadTcgpObbModel() {
  const tf = await loadTensorFlowJs();
  if (tcgpObbModel) return tcgpObbModel;
  if (!tcgpObbModelPromise) {
    tcgpObbModelPromise = tf.loadGraphModel(TCGP_OBB_MODEL_URL).then((model) => {
      tcgpObbModel = model;
      return model;
    });
  }
  return tcgpObbModelPromise;
}

async function warmupTcgpObbModel() {
  if (!tcgpObbWarmupPromise) {
    tcgpObbWarmupPromise = (async () => {
      const tf = await loadTensorFlowJs();
      const model = await loadTcgpObbModel();
      const input = tf.zeros([1, TCGP_OBB_INPUT_SIZE, TCGP_OBB_INPUT_SIZE, 3]);
      const predictions = model.predict(input);
      const outputs = Array.isArray(predictions) ? predictions : [predictions];
      try {
        await Promise.all(outputs.map((tensor) => tensor.data()));
      } finally {
        tfDispose(tf, [input, outputs]);
      }
      return { backend: tf.getBackend?.() || "unknown" };
    })().catch((error) => {
      tcgpObbWarmupPromise = null;
      throw error;
    });
  }
  return tcgpObbWarmupPromise;
}

function maybePreloadTcgpObb() {
  if (els.cropModeInput.value !== TCGP_OBB_MODE) return;
  if (!state.scanning) setStatus("TCGP OBB crop will run on the server.");
}

function fileToImage(file) {
  if (window.createImageBitmap) {
    return createImageBitmap(file, { imageOrientation: "from-image" })
      .catch(() => fileToHtmlImage(file));
  }
  return fileToHtmlImage(file);
}

function fileToHtmlImage(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not read image for TCGP OBB crop."));
    };
    image.src = url;
  });
}

function canvasToBlob(canvas, type = "image/jpeg", quality = 0.92) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("Could not encode TCGP OBB crop."));
      }
    }, type, quality);
  });
}

function tfDispose(tf, tensors) {
  tensors.flat().forEach((tensor) => {
    if (tensor && typeof tensor.dispose === "function") tensor.dispose();
  });
}

async function detectTcgpObbBox(image) {
  const tf = await loadTensorFlowJs();
  const model = await loadTcgpObbModel();
  const originalWidth = image.naturalWidth || image.width;
  const originalHeight = image.naturalHeight || image.height;
  const paddedSize = Math.max(originalWidth, originalHeight);
  const scale = TCGP_OBB_INPUT_SIZE / paddedSize;
  const inputCanvas = document.createElement("canvas");
  inputCanvas.width = TCGP_OBB_INPUT_SIZE;
  inputCanvas.height = TCGP_OBB_INPUT_SIZE;
  const inputContext = inputCanvas.getContext("2d");
  inputContext.fillStyle = "black";
  inputContext.fillRect(0, 0, TCGP_OBB_INPUT_SIZE, TCGP_OBB_INPUT_SIZE);
  inputContext.drawImage(
    image,
    0,
    0,
    Math.max(1, Math.round(originalWidth * scale)),
    Math.max(1, Math.round(originalHeight * scale)),
  );

  const input = tf.tidy(() => tf.browser
    .fromPixels(inputCanvas)
    .div(255)
    .expandDims(0));
  const predictions = model.predict(input);
  const output = Array.isArray(predictions) ? predictions[0] : predictions;
  const [boxes, scores, classes, angles, boxesTransposed] = tf.tidy(() => {
    const transposed = output.shape.length === 3 && output.shape[0] === 1
      ? output.squeeze([0])
      : output;
    const rawBoxes = transposed.slice([0, 0], [4, -1]).transpose();
    const x = rawBoxes.slice([0, 0], [-1, 1]);
    const y = rawBoxes.slice([0, 1], [-1, 1]);
    const width = rawBoxes.slice([0, 2], [-1, 1]);
    const height = rawBoxes.slice([0, 3], [-1, 1]);
    const x1 = tf.sub(x, tf.div(width, 2));
    const y1 = tf.sub(y, tf.div(height, 2));
    const x2 = tf.add(x1, width);
    const y2 = tf.add(y1, height);
    const nmsBoxes = tf.concat([y1, x1, y2, x2], 1);
    const classScores = transposed.slice([4, 0], [1, -1]);
    const maxScores = tf.max(classScores, 0);
    const maxClasses = tf.argMax(classScores, 0);
    const angleValues = transposed.shape[0] > 5
      ? transposed.slice([5, 0], [1, -1]).squeeze()
      : tf.zerosLike(maxScores);
    return [nmsBoxes, maxScores, maxClasses, angleValues, rawBoxes];
  });

  try {
    const selectedIndexes = await tf.image.nonMaxSuppressionAsync(
      boxes,
      scores,
      20,
      TCGP_OBB_IOU_THRESHOLD,
      TCGP_OBB_SCORE_THRESHOLD,
    );
    const detections = tf.tidy(() => tf.concat([
      boxesTransposed.gather(selectedIndexes, 0),
      scores.gather(selectedIndexes, 0).expandDims(1),
      classes.gather(selectedIndexes, 0).expandDims(1),
      angles.gather(selectedIndexes, 0).expandDims(1),
    ], 1));
    const values = await detections.data();
    const count = detections.shape[0];
    const candidates = [];
    for (let index = 0; index < count; index += 1) {
      const offset = index * 7;
      const centerX = (values[offset] * paddedSize) / TCGP_OBB_INPUT_SIZE;
      const centerY = (values[offset + 1] * paddedSize) / TCGP_OBB_INPUT_SIZE;
      let width = (values[offset + 2] * paddedSize) / TCGP_OBB_INPUT_SIZE;
      let height = (values[offset + 3] * paddedSize) / TCGP_OBB_INPUT_SIZE;
      const confidence = values[offset + 4];
      let angle = values[offset + 6] || 0;
      if (centerX < 0 || centerX > originalWidth || centerY < 0 || centerY > originalHeight) continue;
      if (width < 10 || height < 10) continue;
      if (width > height) {
        [width, height] = [height, width];
        angle += Math.PI / 2;
      }
      candidates.push({ centerX, centerY, width, height, confidence, angle });
    }
    tfDispose(tf, [selectedIndexes, detections]);
    candidates.sort((a, b) => b.confidence - a.confidence);
    return candidates[0] || null;
  } finally {
    tfDispose(tf, [input, predictions, boxes, scores, classes, angles, boxesTransposed]);
  }
}

async function cropWithTcgpObb(file) {
  const started = performance.now();
  const image = await fileToImage(file);
  const box = await detectTcgpObbBox(image);
  if (!box) throw new Error("TCGP OBB did not detect a card.");

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(box.width));
  canvas.height = Math.max(1, Math.round(box.height));
  const context = canvas.getContext("2d");
  context.save();
  context.translate(canvas.width / 2, canvas.height / 2);
  context.rotate(-box.angle);
  context.drawImage(image, -box.centerX, -box.centerY);
  context.restore();

  const blob = await canvasToBlob(canvas);
  if (typeof image.close === "function") image.close();
  const secondsElapsed = (performance.now() - started) / 1000;
  const croppedFile = new File([blob], file.name.replace(/\.[^.]+$/, "") + "-tcgp-obb.jpg", {
    type: "image/jpeg",
  });
  return {
    file: croppedFile,
    previewUrl: URL.createObjectURL(blob),
    seconds: secondsElapsed,
    status: "tcgp_obb_browser",
    confidence: box.confidence,
    box,
  };
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setStatus(message) {
  els.statusText.textContent = message;
}

function setHealth(stateName, text) {
  els.healthPill.dataset.state = stateName;
  els.healthText.textContent = text;
}

function setTimings(timings = {}) {
  const values = [
    ["Crop", timings.crop_seconds],
    ["Embed", timings.embedding_seconds],
    ["Search", timings.search_seconds],
    ["RapidOCR", timings.language_rerank_seconds],
    ["Slab", timings.slab_barcode_seconds],
    ["Rerank", timings.rerank_seconds ?? timings.visual_rerank_seconds],
    ["OCR", timings.ocr_seconds],
    ["Total", timings.total_seconds],
  ];
  els.timingGrid.innerHTML = values
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${seconds(value)}</dd></div>`)
    .join("");
}

function updateButtonState() {
  els.recognizeButton.disabled = !state.file || state.scanning;
  els.warmupButton.disabled = state.scanning;
}

function useFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Please choose an image file.");
    return;
  }
  state.file = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  if (state.clientCropPreviewUrl) {
    URL.revokeObjectURL(state.clientCropPreviewUrl);
    state.clientCropPreviewUrl = null;
  }
  state.previewUrl = URL.createObjectURL(file);
  els.previewImage.src = state.previewUrl;
  els.previewFrame.hidden = false;
  els.fileMeta.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  setStatus("Image ready.");
  updateButtonState();
  maybePreloadTcgpObb();
}

function renderCrop(payload) {
  const status = payload?.status || "unknown";
  const fallback = payload?.fallback_used ? " · fallback used" : "";
  els.cropStatus.textContent = `${status}${fallback}`;
  if (payload?.debug_crop_jpeg_base64) {
    els.cropPreview.innerHTML = `<img alt="Detected card crop" src="data:image/jpeg;base64,${payload.debug_crop_jpeg_base64}">`;
    return;
  }
  els.cropPreview.innerHTML = `<span>${escapeHtml(status)}</span>`;
}

function renderClientCrop(payload) {
  const confidence = typeof payload.confidence === "number" ? ` · ${(payload.confidence * 100).toFixed(1)}%` : "";
  if (state.clientCropPreviewUrl && state.clientCropPreviewUrl !== payload.previewUrl) {
    URL.revokeObjectURL(state.clientCropPreviewUrl);
  }
  state.clientCropPreviewUrl = payload.previewUrl || null;
  els.cropStatus.textContent = `${payload.status}${confidence}`;
  els.cropPreview.innerHTML = `<img alt="TCGP OBB browser crop" src="${payload.previewUrl || payload.dataUrl}">`;
}

function resultTitle(result) {
  return result.name_ja || result.name_en || result.name || result.card_id || "Unknown card";
}

function resultCode(result) {
  return [result.set_id, result.card_code].filter(Boolean).join("-");
}

function renderCandidateSelection(selection) {
  const candidates = selection?.candidates || [];
  if (!candidates.length) return "";

  const recommendedId = selection.recommended_card_id;
  const statusText = selection.needs_user_choice ? "請確認版本" : "已用卡號推薦";
  const choices = candidates
    .map((candidate) => {
      const title = resultTitle(candidate);
      const code = resultCode(candidate);
      const score = Number(candidate.score || 0).toFixed(3);
      const imageUrl = candidate.display_image_url || candidate.reference_image_url || candidate.image_url;
      const image = imageUrl
        ? `<img alt="${escapeHtml(title)}" src="${escapeHtml(imageUrl)}" loading="lazy">`
        : "<span>No image</span>";
      const isRecommended = candidate.card_id === recommendedId;
      const exactCode = candidate.ocr_card_code_match || candidate.hint_card_code_match;
      const tags = [
        candidate.rank ? `#${candidate.rank}` : null,
        candidate.language || null,
        isRecommended ? "Recommended" : null,
        exactCode ? "Code match" : null,
      ].filter(Boolean);
      return `
        <article class="candidate-choice${isRecommended ? " is-recommended" : ""}" data-card-id="${escapeHtml(candidate.card_id || "")}">
          <div class="candidate-image">${image}</div>
          <div class="candidate-body">
            <div class="candidate-tags">${tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("")}</div>
            <strong>${escapeHtml(code || candidate.card_id || "Unknown code")}</strong>
            <p>${escapeHtml(title)}</p>
            <small>Score ${score}</small>
            <button type="button" data-select-card-id="${escapeHtml(candidate.card_id || "")}">選擇這版</button>
          </div>
        </article>
      `;
    })
    .join("");

  return `
    <section class="candidate-selection" aria-label="Ambiguous card versions">
      <div class="candidate-selection-header">
        <div>
          <strong>版本確認</strong>
          <span>${escapeHtml(statusText)}</span>
        </div>
        <small>${escapeHtml((selection.reasons || []).join(" · "))}</small>
      </div>
      <div class="candidate-choice-grid">${choices}</div>
    </section>
  `;
}

function renderResults(results = [], selection = null) {
  els.resultCount.textContent = `${results.length} result${results.length === 1 ? "" : "s"}`;
  if (!results.length) {
    els.resultsList.innerHTML = '<p class="empty-state">沒有符合結果，請確認照片中卡片清楚且裁切有成功。</p>';
    return;
  }

  const selectionHtml = renderCandidateSelection(selection);
  const resultHtml = results
    .map((result, index) => {
      const title = resultTitle(result);
      const subtitle = [result.name_en, result.name_ja].filter(Boolean).filter((item, idx, arr) => arr.indexOf(item) === idx).join(" / ");
      const code = resultCode(result);
      const score = Number(result.score || 0).toFixed(3);
      const embeddingScore = typeof result.embedding_score === "number" ? Number(result.embedding_score).toFixed(3) : null;
      const siglipScore = typeof result.siglip_score === "number" ? Number(result.siglip_score).toFixed(3) : null;
      const siglipNorm = typeof result.siglip_norm === "number" ? Number(result.siglip_norm).toFixed(3) : null;
      const visualScore = typeof result.visual_score === "number" ? Number(result.visual_score).toFixed(3) : null;
      const ocrBoost = typeof result.ocr_card_code_boost === "number" && result.ocr_card_code_boost > 0
        ? `OCR +${Number(result.ocr_card_code_boost).toFixed(3)}`
        : null;
      const hintCodeBoost = typeof result.hint_card_code_boost === "number" && result.hint_card_code_boost > 0
        ? `Hint code +${Number(result.hint_card_code_boost).toFixed(3)}`
        : null;
      const languageBoost = typeof result.language_hint_boost === "number" && result.language_hint_boost > 0
        ? `Lang +${Number(result.language_hint_boost).toFixed(3)}`
        : null;
      const scoreParts = [
        embeddingScore ? `Emb ${embeddingScore}` : null,
        siglipScore ? `SigLIP ${siglipScore}` : null,
        siglipNorm ? `S-norm ${siglipNorm}` : null,
        hintCodeBoost,
        languageBoost,
        ocrBoost,
        visualScore ? `Visual ${visualScore}` : null,
        typeof result.visual_color_score === "number" ? `Color ${Number(result.visual_color_score).toFixed(3)}` : null,
        typeof result.visual_structure_score === "number" ? `Structure ${Number(result.visual_structure_score).toFixed(3)}` : null,
        result.original_rank ? `Old #${result.original_rank}` : null,
      ].filter(Boolean);
      const displayImageUrl = result.display_image_url || result.reference_image_url || result.image_url;
      const variantText = result.edition || result.variant || "";
      const codeText = [code || "No set/card code", variantText].filter(Boolean).join(" · ");
      const image = displayImageUrl
        ? `<img alt="${escapeHtml(title)}" src="${escapeHtml(displayImageUrl)}" loading="lazy">`
        : "No image URL";
      const snkr = result.snkr || {};
      const productId = snkr.product_id ? String(snkr.product_id) : "";
      const productIdLine = productId
        ? snkr.url
          ? `<a href="${escapeHtml(snkr.url)}" target="_blank" rel="noreferrer">${escapeHtml(productId)}</a>`
          : escapeHtml(productId)
        : "No SNKR ID";
      const matchStatus = snkr.match_status || (productId ? "matched" : "not_mapped");
      const candidateText = snkr.verified_candidate_count !== null && snkr.verified_candidate_count !== undefined
        ? ` · Candidates ${escapeHtml(String(snkr.verified_candidate_count))}`
        : "";
      const mappingFlag = snkr.mapping_flag || null;
      const mappingClass = mappingFlag ? "snkr-warning" : productId ? "snkr-ok" : "snkr-missing";
      const mappingText = mappingFlag
        ? `Mapping issue: ${mappingFlag.severity || "suspicious"}`
        : productId
          ? "Mapping: OK"
          : "Mapping: missing product id";
      const mappingDetail = mappingFlag && mappingFlag.catalog_name && mappingFlag.snkr_product_name
        ? ` · ${mappingFlag.catalog_name} -> ${mappingFlag.snkr_product_name}`
        : "";
      const price = snkr.min_price_format || snkr.min_price;
      const priceLine = snkr.url
        ? `<a href="${escapeHtml(snkr.url)}" target="_blank" rel="noreferrer">${escapeHtml(price || "SNKRDUNK")}</a>`
        : escapeHtml(price || "No SNKR price");

      return `
        <article class="result-card">
          <div class="result-image">${image}</div>
          <div class="result-main">
            <div class="result-topline">
              <span class="badge">#${index + 1}</span>
              <span class="badge">${escapeHtml(result.index || "index")}</span>
              <span class="badge">${escapeHtml(result.language || "unknown")}</span>
              <span class="badge score-badge">${score}</span>
            </div>
            <div class="result-title">${escapeHtml(title)}</div>
            <div class="result-subtitle">${escapeHtml(subtitle || "No alternate name")}</div>
            <div class="result-meta">${escapeHtml(codeText)} · ${escapeHtml(result.rarity || "No rarity")} · ${escapeHtml(result.canonical_source || result.source || "No source")}</div>
            ${result.canonical_key ? `<div class="result-meta">${escapeHtml(result.canonical_key)}</div>` : ""}
            ${scoreParts.length ? `<div class="result-meta">${escapeHtml(scoreParts.join(" · "))}</div>` : ""}
            <div class="snkr-line">SNKR ID: ${productIdLine} · ${escapeHtml(matchStatus)}${candidateText}</div>
            ${snkr.product_name ? `<div class="snkr-line">SNKR product: ${escapeHtml(snkr.product_name)}</div>` : ""}
            <div class="snkr-line ${mappingClass}">${escapeHtml(mappingText + mappingDetail)}</div>
            <div class="price-line">Price: ${priceLine}</div>
          </div>
        </article>
      `;
    })
    .join("");
  els.resultsList.innerHTML = `${selectionHtml}${resultHtml}`;
}

function renderScanningResults() {
  els.resultCount.textContent = "Scanning";
  els.resultsList.innerHTML = '<p class="empty-state">正在分析圖片...</p>';
}

async function checkHealth() {
  try {
    const response = await fetch(endpoint("/health"));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const indexes = Object.keys(data.available_indexes || {}).join(", ") || "no indexes";
    setHealth("ok", `API ready · ${indexes}`);
  } catch (error) {
    setHealth("error", "API offline");
  }
}

async function warmup() {
  setStatus("Warming up models and indexes...");
  els.warmupButton.disabled = true;
  try {
    const warmups = [fetch(endpoint("/warmup"), { method: "POST" })];
    const [response] = await Promise.all(warmups);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    setStatus(`Warmup finished in ${seconds(data.seconds)}.`);
    await checkHealth();
  } catch (error) {
    setStatus(`Warmup failed: ${error.message}`);
  } finally {
    els.warmupButton.disabled = false;
  }
}

async function recognize() {
  if (!state.file) return;
  state.scanning = true;
  updateButtonState();
  setStatus("Scanning card...");
  setTimings();
  renderScanningResults();
  els.cropStatus.textContent = "Scanning";
  els.cropPreview.innerHTML = "<span>Processing...</span>";

  let requestFile = state.file;
  let clientCrop = null;
  const selectedCropMode = els.cropModeInput.value;
  const useTcgpObbCrop = false;

  try {
    if (useTcgpObbCrop) {
      setStatus("Running TCGP OBB crop in browser...");
      els.cropStatus.textContent = "Loading TCGP OBB";
      clientCrop = await cropWithTcgpObb(state.file);
      requestFile = clientCrop.file;
      renderClientCrop(clientCrop);
      setStatus("Searching with TCGP OBB crop...");
    }
  } catch (error) {
    setStatus(`TCGP OBB crop failed: ${error.message}`);
    els.cropStatus.textContent = "Error";
    els.cropPreview.innerHTML = "<span>沒有裁切結果</span>";
    state.scanning = false;
    updateButtonState();
    return;
  }

  const params = new URLSearchParams({
    crop: String(els.cropToggle.checked && selectedCropMode !== "none" && !useTcgpObbCrop),
    crop_mode: useTcgpObbCrop ? "none" : selectedCropMode,
    fallback_to_original: String(els.fallbackToggle.checked),
    top_k: String(Number(els.topKInput.value || 5)),
    per_index_top_k: String(Math.max(Number(els.topKInput.value || 5), 5)),
    visual_rerank: String(els.visualRerankToggle.checked),
    visual_rerank_candidates: "5",
    visual_rerank_weight: "0.40",
    rerank_model: "siglip",
    card_code_ocr: String(els.cardCodeOcrToggle.checked),
    language_rerank: String(els.languageRerankToggle.checked),
    language_rerank_boost: "0.08",
    language_rerank_candidates: "25",
    language_hint_text: els.hintTextInput.value.trim(),
    language_ocr: String(els.languageOcrToggle.checked),
    language_ocr_engine: "rapidocr",
    hint_card_code: "true",
    hint_card_code_boost: "0.10",
    slab_barcode_lookup: String(els.slabLookupToggle.checked),
    confidence: els.confidenceInput.value,
    include_debug_crop_base64: "true",
  });
  const body = new FormData();
  body.append("file", requestFile);

  try {
    const response = await fetch(`${endpoint("/recognize")}?${params.toString()}`, {
      method: "POST",
      body,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    const timings = { ...(data.timings || {}) };
    if (clientCrop) {
      timings.crop_seconds = clientCrop.seconds;
      timings.total_seconds = (timings.total_seconds || 0) + clientCrop.seconds;
    }
    setTimings(timings);
    if (clientCrop) {
      renderClientCrop(clientCrop);
    } else {
      renderCrop(data.crop);
    }
    renderResults(data.results, data.candidate_selection);
    const parsed = data.card_code_ocr?.best?.parsed;
    const hintParsed = data.hint_card_code?.best?.parsed;
    const language = data.language_rerank?.language;
    const hintStatus = hintParsed ? ` Hint ${hintParsed.set_id}-${hintParsed.card_number}` : "";
    const languageStatus = language ? ` Lang ${language}` : "";
    const ocrStatus = parsed ? ` OCR ${parsed.set_id}-${parsed.card_number}` : "";
    const cropStatus = clientCrop ? " TCGP OBB" : "";
    setStatus(data.status === "ok" ? `Scan complete.${cropStatus}${hintStatus}${languageStatus}${ocrStatus}` : data.status);
  } catch (error) {
    setStatus(`Scan failed: ${error.message}`);
    els.cropStatus.textContent = "Error";
    els.cropPreview.innerHTML = "<span>沒有裁切結果</span>";
  } finally {
    state.scanning = false;
    updateButtonState();
  }
}

els.confidenceInput.addEventListener("input", () => {
  els.confidenceOutput.textContent = els.confidenceInput.value;
});

els.fileInput.addEventListener("change", (event) => {
  useFile(event.target.files?.[0]);
});

els.cropModeInput.addEventListener("change", maybePreloadTcgpObb);

for (const eventName of ["dragenter", "dragover"]) {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.dataset.dragging = "true";
  });
}

for (const eventName of ["dragleave", "drop"]) {
  els.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropZone.dataset.dragging = "false";
  });
}

els.dropZone.addEventListener("drop", (event) => {
  useFile(event.dataTransfer?.files?.[0]);
});

els.resultsList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-select-card-id]");
  if (!button) return;
  const cardId = button.dataset.selectCardId;
  els.resultsList.querySelectorAll(".candidate-choice").forEach((item) => {
    item.classList.toggle("is-selected", item.dataset.cardId === cardId);
  });
  setStatus(cardId ? `Selected ${cardId}.` : "Selected candidate.");
});

els.recognizeButton.addEventListener("click", recognize);
els.warmupButton.addEventListener("click", warmup);
els.apiBaseInput.addEventListener("change", checkHealth);

setTimings();
checkHealth();
