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
  setTimings();
  renderScanningResults();
  els.cropStatus.textContent = "Scanning";
  els.cropPreview.innerHTML = "<span>Processing...</span>";

  const params = new URLSearchParams({
    crop: String(els.cropToggle.checked && els.cropModeInput.value !== "none"),
    crop_mode: els.cropModeInput.value,
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
    renderResults(data.results, data.candidate_selection);
    const parsed = data.card_code_ocr?.best?.parsed;
    const hintParsed = data.hint_card_code?.best?.parsed;
    const language = data.language_rerank?.language;
    const hintStatus = hintParsed ? ` Hint ${hintParsed.set_id}-${hintParsed.card_number}` : "";
    const languageStatus = language ? ` Lang ${language}` : "";
    const ocrStatus = parsed ? ` OCR ${parsed.set_id}-${parsed.card_number}` : "";
    setStatus(data.status === "ok" ? `Scan complete.${hintStatus}${languageStatus}${ocrStatus}` : data.status);
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
