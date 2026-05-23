const state = {
  file: null,
  previewUrl: null,
  scanning: false,
};

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
  previewFrame: document.querySelector("#previewFrame"),
  previewImage: document.querySelector("#previewImage"),
  recognizeButton: document.querySelector("#recognizeButton"),
  resultCount: document.querySelector("#resultCount"),
  resultsList: document.querySelector("#resultsList"),
  statusText: document.querySelector("#statusText"),
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
  state.previewUrl = URL.createObjectURL(file);
  els.previewImage.src = state.previewUrl;
  els.previewFrame.hidden = false;
  els.fileMeta.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  setStatus("Image ready.");
  updateButtonState();
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

function resultTitle(result) {
  return result.name_ja || result.name_en || result.name || result.card_id || "Unknown card";
}

function renderResults(results = []) {
  els.resultCount.textContent = `${results.length} result${results.length === 1 ? "" : "s"}`;
  if (!results.length) {
    els.resultsList.innerHTML = '<p class="empty-state">沒有符合結果，請確認照片中卡片清楚且裁切有成功。</p>';
    return;
  }

  els.resultsList.innerHTML = results
    .map((result, index) => {
      const title = resultTitle(result);
      const subtitle = [result.name_en, result.name_ja].filter(Boolean).filter((item, idx, arr) => arr.indexOf(item) === idx).join(" / ");
      const code = [result.set_id, result.card_code].filter(Boolean).join("-");
      const score = Number(result.score || 0).toFixed(3);
      const embeddingScore = typeof result.embedding_score === "number" ? Number(result.embedding_score).toFixed(3) : null;
      const siglipScore = typeof result.siglip_score === "number" ? Number(result.siglip_score).toFixed(3) : null;
      const siglipNorm = typeof result.siglip_norm === "number" ? Number(result.siglip_norm).toFixed(3) : null;
      const visualScore = typeof result.visual_score === "number" ? Number(result.visual_score).toFixed(3) : null;
      const ocrBoost = typeof result.ocr_card_code_boost === "number" && result.ocr_card_code_boost > 0
        ? `OCR +${Number(result.ocr_card_code_boost).toFixed(3)}`
        : null;
      const scoreParts = [
        embeddingScore ? `Emb ${embeddingScore}` : null,
        siglipScore ? `SigLIP ${siglipScore}` : null,
        siglipNorm ? `S-norm ${siglipNorm}` : null,
        ocrBoost,
        visualScore ? `Visual ${visualScore}` : null,
        typeof result.visual_color_score === "number" ? `Color ${Number(result.visual_color_score).toFixed(3)}` : null,
        typeof result.visual_structure_score === "number" ? `Structure ${Number(result.visual_structure_score).toFixed(3)}` : null,
        result.original_rank ? `Old #${result.original_rank}` : null,
      ].filter(Boolean);
      const displayImageUrl = result.display_image_url || result.reference_image_url || result.image_url;
      const image = displayImageUrl
        ? `<img alt="${escapeHtml(title)}" src="${escapeHtml(displayImageUrl)}" loading="lazy">`
        : "No image URL";
      const snkr = result.snkr || {};
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
            <div class="result-meta">${escapeHtml(code || "No set/card code")} · ${escapeHtml(result.rarity || "No rarity")} · ${escapeHtml(result.canonical_source || result.source || "No source")}</div>
            ${scoreParts.length ? `<div class="result-meta">${escapeHtml(scoreParts.join(" · "))}</div>` : ""}
            <div class="price-line">Price: ${priceLine}</div>
          </div>
        </article>
      `;
    })
    .join("");
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
    const response = await fetch(endpoint("/warmup"), { method: "POST" });
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
  renderResults([]);
  els.cropStatus.textContent = "Scanning";
  els.cropPreview.innerHTML = "<span>Processing...</span>";

  const params = new URLSearchParams({
    crop: String(els.cropToggle.checked && els.cropModeInput.value !== "none"),
    crop_mode: els.cropModeInput.value,
    fallback_to_original: String(els.fallbackToggle.checked),
    top_k: String(Number(els.topKInput.value || 5)),
    per_index_top_k: String(els.visualRerankToggle.checked ? 100 : Math.max(Number(els.topKInput.value || 5), 5)),
    visual_rerank: String(els.visualRerankToggle.checked),
    visual_rerank_candidates: "100",
    visual_rerank_weight: "0.40",
    rerank_model: "siglip",
    card_code_ocr: String(els.cardCodeOcrToggle.checked),
    confidence: els.confidenceInput.value,
    include_debug_crop_base64: "true",
  });
  const body = new FormData();
  body.append("file", state.file);

  try {
    const response = await fetch(`${endpoint("/recognize")}?${params.toString()}`, {
      method: "POST",
      body,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    setTimings(data.timings);
    renderCrop(data.crop);
    renderResults(data.results);
    const parsed = data.card_code_ocr?.best?.parsed;
    const ocrStatus = parsed ? ` OCR ${parsed.set_id}-${parsed.card_number}` : "";
    setStatus(data.status === "ok" ? `Scan complete.${ocrStatus}` : data.status);
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

els.recognizeButton.addEventListener("click", recognize);
els.warmupButton.addEventListener("click", warmup);
els.apiBaseInput.addEventListener("change", checkHealth);

setTimings();
checkHealth();
