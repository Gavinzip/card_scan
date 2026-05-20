FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    WEB_CONCURRENCY=1 \
    OMP_NUM_THREADS=2 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    CARD_SCAN_DEVICE=cpu \
    CARD_SCAN_PRELOAD=true \
    CARD_SCAN_INDEXES="pokemon_en=data/processed/image_index,pokemon_ja=data/processed/pokemon_ja_canonical_image_index"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    python -m pip install -r requirements-deploy.txt

COPY scripts ./scripts
COPY data/processed/image_index ./data/processed/image_index
COPY data/processed/pokemon_ja_canonical_image_index ./data/processed/pokemon_ja_canonical_image_index
COPY data/processed/pokemon_ja_canonical_catalog.jsonl ./data/processed/pokemon_ja_canonical_catalog.jsonl
COPY data/processed/pokemon_ja_canonical_summary.json ./data/processed/pokemon_ja_canonical_summary.json
COPY data/processed/pokemon_ja_canonical_image_manifest.jsonl ./data/processed/pokemon_ja_canonical_image_manifest.jsonl

RUN mkdir -p data/models && \
    python - <<'PY'
from pathlib import Path
from scripts.cropping.auto_crop_cards import DEFAULT_MODEL_FILE, DEFAULT_REPO_ID, ensure_model

ensure_model(Path("data/models/cardcaptor_v3_best.pt"), DEFAULT_REPO_ID, DEFAULT_MODEL_FILE)
PY

EXPOSE 8080

CMD ["sh", "-c", "uvicorn scripts.server.recognition_api:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WEB_CONCURRENCY:-1}"]
