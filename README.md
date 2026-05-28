# TCG Card Scan Data Pipeline

This repo builds the first-pass data pipeline for TCG card recognition. It focuses on ingestion, cleaning, unified manifests, quality checks, and an optional embedding index. It does not train a large classifier.

## Scope

- Pokemon: TCGdex metadata plus locally downloaded Kaggle image files.
- One Piece: Kaggle English catalog plus Hugging Face OPTCG English metadata.
- Yu-Gi-Oh: intentionally out of scope for this first pass.

Large datasets stay out of git. `data/raw/`, `data/images/`, and heavyweight processed files are ignored.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Kaggle access should be configured outside the repo:

```bash
chmod 600 ~/.kaggle/access_token
kaggle competitions list
```

## Minimal Pipeline

Fetch Pokemon metadata:

```bash
python scripts/ingest/fetch_tcgdex_pokemon.py
```

Import small One Piece sources:

```bash
python scripts/ingest/import_onepiece_kaggle.py --download
python scripts/ingest/import_optcg_hf.py
python scripts/quality/check_onepiece_urls.py
python scripts/ingest/download_onepiece_official_images.py
```

Normalize everything currently available:

```bash
python scripts/ingest/normalize_catalog.py
```

## Pokemon Images

Download the large Kaggle Pokemon image package outside git:

```bash
mkdir -p /data/cardsearch/raw/pokemon
kaggle datasets download -d ellimaaac/pokemon-tcg-all-image-cards \
  -p /data/cardsearch/raw/pokemon --unzip
```

Build and validate the image manifest:

```bash
python scripts/ingest/import_kaggle_pokemon_images.py \
  --image-root /data/cardsearch/raw/pokemon

python scripts/quality/check_images.py \
  --input-manifest data/manifests/pokemon_kaggle_images.jsonl

python scripts/quality/filter_clean_images.py \
  --input data/manifests/pokemon_kaggle_images.jsonl
```

Download Japanese Pokemon reference images from TCGdex:

```bash
python scripts/ingest/download_tcgdex_pokemon_images.py \
  --language ja \
  --quality high \
  --extension webp \
  --workers 16

python scripts/quality/check_images.py \
  --input-manifest data/manifests/pokemon_tcgdex_ja_images.jsonl \
  --report-output data/manifests/pokemon_tcgdex_ja_image_quality_report.json \
  --bad-output data/manifests/pokemon_tcgdex_ja_bad_images.jsonl \
  --duplicates-output data/manifests/pokemon_tcgdex_ja_duplicate_images.jsonl

python scripts/quality/filter_clean_images.py \
  --input data/manifests/pokemon_tcgdex_ja_images.jsonl \
  --output data/manifests/pokemon_tcgdex_ja_clean_images.jsonl \
  --summary-output data/manifests/pokemon_tcgdex_ja_clean_images_summary.json
```

These TCGdex images are for local reference search first. Do not treat them as training data until image rights are reviewed.

Download official Japanese standard-regulation Pokemon reference images:

```bash
python scripts/ingest/fetch_pokemon_card_official_ja.py \
  --download-images \
  --detail-workers 2 \
  --timeout 20
```

If official detail pages return temporary 403 responses, repair only the missing detail rows:

```bash
python scripts/ingest/repair_pokemon_card_official_ja.py \
  --download-images \
  --detail-workers 2 \
  --detail-retries 5 \
  --detail-retry-sleep 3 \
  --timeout 20
```

Then run image QA:

```bash
python scripts/quality/check_images.py \
  --input-manifest data/manifests/pokemon_card_official_ja_catalog.jsonl \
  --report-output data/manifests/pokemon_card_official_ja_image_quality_report.json \
  --bad-output data/manifests/pokemon_card_official_ja_bad_images.jsonl \
  --duplicates-output data/manifests/pokemon_card_official_ja_duplicate_images.jsonl

python scripts/quality/filter_clean_images.py \
  --input data/manifests/pokemon_card_official_ja_catalog.jsonl \
  --output data/manifests/pokemon_card_official_ja_clean_images.jsonl \
  --summary-output data/manifests/pokemon_card_official_ja_clean_images_summary.json
```

Map Japanese Pokemon records to SNKRDUNK products:

```bash
python scripts/ingest/map_snkr_pokemon_ja_products.py \
  --input data/manifests/pokemon_card_official_ja_clean_images.jsonl \
  --output data/manifests/snkr_pokemon_official_ja_product_map.jsonl \
  --summary-output data/manifests/snkr_pokemon_official_ja_product_map_summary.json \
  --max-pages-per-set 40 \
  --card-search-set-threshold 30
```

Official Japanese images are also reference-search assets first. Keep English and Japanese cards separate, and expose Japanese prices on English scans as related Japanese reference prices unless the scanned card itself matched the Japanese catalog.

Build the preferred Japanese Pokemon canonical dataset:

```bash
python scripts/ingest/build_pokemon_ja_canonical.py
```

This writes:

```text
data/processed/pokemon_ja_canonical_catalog.jsonl
data/processed/pokemon_ja_canonical_image_manifest.jsonl
data/processed/pokemon_ja_canonical_summary.json
```

The canonical catalog keeps official Japanese rows first, uses TCGdex to fill missing `set_id + card_code` keys, stores overlaps in `duplicate_sources`, and excludes exact duplicate image hashes from the index manifest.

## Optional Embedding Index

```bash
python -m pip install -r requirements-embedding.txt
python scripts/indexing/build_image_embeddings.py \
  --image-manifest data/manifests/pokemon_clean_images.jsonl
```

This creates CLIP image embeddings and a FAISS index. It skips records marked as sample or watermarked.
By default the script uses a DINOv2 model through `timm`; pass `--backend open_clip --model ViT-B-32` only if OpenCLIP is stable in your local Python environment.
On this machine, OpenCLIP segfaulted in Python 3.12/3.14, so the working index uses DINOv2/timm.

Build a separate Japanese Pokemon reference index:

```bash
python scripts/indexing/build_image_embeddings.py \
  --image-manifest data/manifests/pokemon_tcgdex_ja_clean_images.jsonl \
  --output-dir data/processed/pokemon_ja_image_index \
  --batch-size 64
```

Build a separate official Japanese Pokemon reference index:

```bash
python scripts/indexing/build_image_embeddings.py \
  --image-manifest data/manifests/pokemon_card_official_ja_clean_images.jsonl \
  --output-dir data/processed/pokemon_ja_official_image_index \
  --batch-size 64
```

Build the preferred canonical Japanese Pokemon reference index:

```bash
python scripts/indexing/build_image_embeddings.py \
  --image-manifest data/processed/pokemon_ja_canonical_image_manifest.jsonl \
  --output-dir data/processed/pokemon_ja_canonical_image_index \
  --batch-size 64
```

Build a separate One Piece official-image reference index:

```bash
python scripts/indexing/build_image_embeddings.py \
  --image-manifest data/manifests/onepiece_official_images.jsonl \
  --output-dir data/processed/onepiece_official_image_index \
  --allow-watermarked \
  --batch-size 64
```

Query the index with a phone/photo image:

```bash
python scripts/indexing/query_image_index.py \
  --index-dir data/processed/image_index \
  --image /path/to/phone_photo.jpg \
  --top-k 10
```

## Server-Side Auto Crop

Install the optional crop dependencies in the same environment used for indexing:

```bash
python -m pip install -r requirements-cropping.txt
python -m pip install -r requirements-server.txt
```

Run card detection and perspective crop on a photo or folder:

```bash
python scripts/cropping/auto_crop_cards.py \
  --input /path/to/card_photos \
  --output-dir data/processed/crops \
  --save-debug
```

The first run downloads the CardCaptor YOLO OBB weights from Hugging Face into `data/models/`. If a card is not detected, the crop manifest records `no_detection`; it does not fall back to a fake center crop.

Run the crop API:

```bash
python -m uvicorn scripts.server.crop_api:app --host 127.0.0.1 --port 8000
```

Upload a photo:

```bash
curl -F "file=@/path/to/photo.jpg" "http://127.0.0.1:8000/crop?confidence=0.2"
```

## Recognition API / Zeabur

Run the full recognition API locally:

```bash
python -m uvicorn scripts.server.recognition_api:app \
  --host 0.0.0.0 \
  --port 8080
```

Recognize one photo:

```bash
curl -F "file=@/path/to/photo.jpg" \
  "http://127.0.0.1:8080/recognize?top_k=5&per_index_top_k=5"
```

Generate the full card market report from the same API server:

```bash
curl -F "file=@/path/to/photo.jpg" \
  "http://127.0.0.1:8080/market-report?crop=true&crop_mode=tcgp_obb&top_k=5&include_posters=true"
```

`/market-report` runs recognition first, reads the top candidate's SNKRDUNK
`product_id`, fetches SNKRDUNK trading histories, applies a per-bucket IQR
outlier filter, and returns Raw/A plus PSA 10 prices. The response includes
`recognition`, `snkr`, `prices.raw_A`, `prices.psa_10`, `markdown`, and
downloadable files under `/reports/{report_id}/...`, including the text report
and TCGPro poster PNGs when `include_posters=true`. Set
`include_poster_base64=true` when a client needs inline PNG payloads instead of
URLs. Generated files are written to `CARD_SCAN_REPORT_OUTPUT_DIR`, defaulting
to `/tmp/card_scan_reports`.

Open the built-in frontend:

```text
http://127.0.0.1:8080/
```

The frontend is a static console served by FastAPI. It can upload a card photo,
call `/recognize`, display the debug crop, show timings, and list top matches
with SNKRDUNK price fields when available.

Deploy to Zeabur with the included `Dockerfile`. The Docker image serves:

```text
scripts.server.recognition_api:app
```

The included `zbpack.json` pins Zeabur to the root Dockerfile. The crop model is
downloaded lazily on `/warmup` or the first cropped `/recognize` request, so the
first request after deploy can take longer.

See `docs/deploy_zeabur.md` for environment variables and deployment notes.

## Docs

- `docs/data_sources.md`: source purpose, license, size/count notes, and training eligibility.
- `docs/pipeline.md`: end-to-end runbook.
- `docs/schema.md`: unified catalog columns.
- `docs/deploy_zeabur.md`: Docker/Zeabur deployment guide.
- `docs/reference_images.md`: mount local image packs on a private server and display them in the frontend.

## No Silent Fallbacks

- TCGdex defaults to `brief` metadata. Use `--detail-level full` when you want full per-card detail requests.
- Hugging Face OPTCG defaults to the dataset rows API. Use `--input-parquet` when you want local parquet import.
- One Piece image URLs are conservatively flagged for SAMPLE/watermark risk. To test official preview images anyway, download them with `download_onepiece_official_images.py` and build a separate reference index with `--allow-watermarked`.
