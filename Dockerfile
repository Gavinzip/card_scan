FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    WEB_CONCURRENCY=1 \
    OMP_NUM_THREADS=2 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    CARD_SCAN_DEVICE=cpu \
    CARD_SCAN_PRELOAD=false \
    CARD_SCAN_INDEXES="pokemon_en=data/processed/image_index_base,pokemon_ja=data/processed/pokemon_ja_canonical_image_index_base,onepiece=data/processed/onepiece_image_index_base"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libice6 \
    libjpeg62-turbo \
    libsm6 \
    libx11-6 \
    libxcb1 \
    libxext6 \
    libxrender1 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    python -m pip install -r requirements-deploy.txt

COPY web ./web
COPY scripts ./scripts
COPY data/processed/image_index_base ./data/processed/image_index_base
COPY data/processed/pokemon_ja_canonical_image_index_base ./data/processed/pokemon_ja_canonical_image_index_base
COPY data/processed/onepiece_image_index_base ./data/processed/onepiece_image_index_base
COPY data/processed/pokemon_ja_canonical_catalog.jsonl ./data/processed/pokemon_ja_canonical_catalog.jsonl
COPY data/processed/pokemon_ja_canonical_summary.json ./data/processed/pokemon_ja_canonical_summary.json
COPY data/processed/pokemon_ja_canonical_image_manifest.jsonl ./data/processed/pokemon_ja_canonical_image_manifest.jsonl

EXPOSE 8080

CMD ["sh", "-c", "echo \"starting card_scan on port ${PORT:-8080}\" && exec uvicorn scripts.server.recognition_api:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WEB_CONCURRENCY:-1}"]
