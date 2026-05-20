# Pipeline

The first version prepares a clean, repeatable catalog and image manifest. It does not train a classifier.

## Directory Layout

```text
data/raw/          downloaded source files and raw API responses
data/interim/      temporary transforms
data/processed/    normalized catalog outputs
data/images/       optional local image staging
data/manifests/    JSONL manifests and QA reports
scripts/ingest/    source import scripts
scripts/quality/   validation scripts
scripts/indexing/  optional embedding index scripts
scripts/cropping/  optional server-side card detection and crop scripts
docs/              source and pipeline documentation
```

Large images and raw downloads are ignored by `.gitignore`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Kaggle downloads require a configured Kaggle client. The token should live outside the repo, for example `~/.kaggle/access_token` with `chmod 600`.

## 1. Fetch Pokemon Metadata from TCGdex

Fast manifest mode:

```bash
python scripts/ingest/fetch_tcgdex_pokemon.py
```

This fetches English and Japanese card/set brief metadata and writes:

```text
data/raw/tcgdex/en_cards_brief.json
data/raw/tcgdex/ja_cards_brief.json
data/raw/tcgdex/en_sets.json
data/raw/tcgdex/ja_sets.json
data/manifests/tcgdex_pokemon_cards.jsonl
data/manifests/tcgdex_pokemon_sets.jsonl
```

Full detail mode:

```bash
python scripts/ingest/fetch_tcgdex_pokemon.py --detail-level full --sleep 0.05
```

Full mode performs one detail request per card. It is slower but fills fields such as `rarity` and `variant`.

## 2. Import Pokemon Kaggle Images

Download the image package outside git or into ignored data folders:

```bash
mkdir -p /data/cardsearch/raw/pokemon
kaggle datasets download -d ellimaaac/pokemon-tcg-all-image-cards \
  -p /data/cardsearch/raw/pokemon --unzip
```

Then build the image manifest:

```bash
python scripts/ingest/import_kaggle_pokemon_images.py \
  --image-root /data/cardsearch/raw/pokemon
```

Output:

```text
data/manifests/pokemon_kaggle_images.jsonl
data/manifests/pokemon_kaggle_images_summary.json
```

## 3. Import One Piece Kaggle Catalog

The dataset is small enough to download through the script:

```bash
python scripts/ingest/import_onepiece_kaggle.py --download
```

If you already have the CSV:

```bash
python scripts/ingest/import_onepiece_kaggle.py --input /path/to/onepiece.csv
```

The importer marks `is_sample` and `is_watermarked` using URL heuristics. These flags are conservative and should gate training use.

## 4. Import Hugging Face OPTCG Metadata

Default mode uses the Hugging Face dataset rows API:

```bash
python scripts/ingest/import_optcg_hf.py
```

If you have a local parquet:

```bash
python -m pip install pandas pyarrow
python scripts/ingest/import_optcg_hf.py --input-parquet /path/to/cards.parquet
```

The rows API mode is explicit; it is not a silent fallback from parquet.

## 5. Quality Checks

Check local images:

```bash
python scripts/quality/check_images.py \
  --input-manifest data/manifests/pokemon_kaggle_images.jsonl \
  --min-width 200 \
  --min-height 280

python scripts/quality/filter_clean_images.py \
  --input data/manifests/pokemon_kaggle_images.jsonl
```

Check One Piece URL flags:

```bash
python scripts/quality/check_onepiece_urls.py
```

Quality outputs:

```text
data/manifests/image_quality_report.json
data/manifests/bad_images.jsonl
data/manifests/duplicate_images.jsonl
data/manifests/pokemon_clean_images.jsonl
data/manifests/onepiece_url_flags.jsonl
data/manifests/onepiece_url_flags_report.json
```

Some files are only written when there are matching findings.

## 6. Download One Piece Official Reference Images

For a first One Piece recognition pass, download the official preview images referenced by the Kaggle catalog into a separate local reference manifest. These records keep their `is_watermarked` flag; they are not treated as clean training data.

```bash
python scripts/ingest/download_onepiece_official_images.py \
  --workers 16
```

Outputs:

```text
data/manifests/onepiece_official_images.jsonl
data/manifests/onepiece_official_images_summary.json
data/manifests/onepiece_official_image_errors.jsonl
```

The errors file is only written when there are download failures.

## 7. Download Japanese Pokemon Reference Images

TCGdex exposes Japanese image URLs in its card metadata. These images are downloaded for local reference search only, not for training by default.

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

## 8. Download Official Japanese Pokemon Reference Images

The Pokemon Card Game official search API and detail pages can be used to build a separate Japanese standard-regulation reference catalog. These official images are for local matching and price lookup, not training by default.

```bash
python scripts/ingest/fetch_pokemon_card_official_ja.py \
  --download-images \
  --detail-workers 2 \
  --timeout 20
```

If the official detail pages return transient 403 responses, repair only the missing records instead of refetching the entire catalog:

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

SNKRDUNK mapping is optional but useful for Japanese market prices:

```bash
python scripts/ingest/map_snkr_pokemon_ja_products.py \
  --input data/manifests/pokemon_card_official_ja_clean_images.jsonl \
  --output data/manifests/snkr_pokemon_official_ja_product_map.jsonl \
  --summary-output data/manifests/snkr_pokemon_official_ja_product_map_summary.json \
  --max-pages-per-set 40 \
  --card-search-set-threshold 30
```

Records with multiple verified SNKRDUNK candidates keep `multiple_verified_matches` instead of guessing one product ID.
Records without both `set_id` and `card_code` are skipped by the SNKRDUNK mapper.

## 9. Build Japanese Pokemon Canonical Catalog

Build the preferred Japanese Pokemon lookup dataset by merging official Japanese rows with TCGdex Japanese metadata/images:

```bash
python scripts/ingest/build_pokemon_ja_canonical.py
```

Outputs:

```text
data/processed/pokemon_ja_canonical_catalog.jsonl
data/processed/pokemon_ja_canonical_image_manifest.jsonl
data/processed/pokemon_ja_canonical_summary.json
```

Rules:

- Official Japanese rows win when `set_id + card_code` overlaps.
- TCGdex rows become canonical only when the official source does not have the same key.
- Overlapped rows are preserved in `duplicate_sources`.
- Exact duplicate image hashes are excluded from the image manifest and represented by `duplicate_image_records`.

## 10. Normalize Catalog

```bash
python scripts/ingest/normalize_catalog.py
```

Outputs:

```text
data/processed/catalog.jsonl
data/processed/catalog.sqlite
data/processed/catalog_summary.json
```

The normalized catalog uses the shared schema in `scripts/lib/schema.py`.

## 11. Optional Image Embedding Index

This is intentionally separate from ingestion and does not train a classifier.

```bash
python -m pip install -r requirements-embedding.txt
python scripts/indexing/build_image_embeddings.py \
  --image-manifest data/manifests/pokemon_clean_images.jsonl \
  --output-dir data/processed/image_index
```

The embedding builder defaults to DINOv2 through `timm`, because it is usually stable for feature extraction. It skips records marked `is_sample` or `is_watermarked`, then writes:

On this machine, OpenCLIP segfaulted during model construction in both Python 3.12 and Python 3.14, so the generated index uses DINOv2/timm. This is an explicit model choice, not a silent fallback.

```text
data/processed/image_index/image_embeddings.npy
data/processed/image_index/image_embedding_manifest.jsonl
data/processed/image_index/faiss.index
data/processed/image_index/summary.json
```

Query it with a phone/photo image:

```bash
python scripts/indexing/query_image_index.py \
  --index-dir data/processed/image_index \
  --image /path/to/phone_photo.jpg \
  --top-k 10
```

To build the Japanese Pokemon reference index:

```bash
python scripts/indexing/build_image_embeddings.py \
  --image-manifest data/manifests/pokemon_tcgdex_ja_clean_images.jsonl \
  --output-dir data/processed/pokemon_ja_image_index \
  --batch-size 64
```

To build the official Japanese Pokemon reference index:

```bash
python scripts/indexing/build_image_embeddings.py \
  --image-manifest data/manifests/pokemon_card_official_ja_clean_images.jsonl \
  --output-dir data/processed/pokemon_ja_official_image_index \
  --batch-size 64
```

To build the preferred canonical Japanese Pokemon index:

```bash
python scripts/indexing/build_image_embeddings.py \
  --image-manifest data/processed/pokemon_ja_canonical_image_manifest.jsonl \
  --output-dir data/processed/pokemon_ja_canonical_image_index \
  --batch-size 64
```

To build the One Piece official-image reference index, explicitly allow the watermarked official preview assets:

```bash
python scripts/indexing/build_image_embeddings.py \
  --image-manifest data/manifests/onepiece_official_images.jsonl \
  --output-dir data/processed/onepiece_official_image_index \
  --allow-watermarked \
  --batch-size 64
```

## 12. Optional Server-Side Card Crop

The server-side crop step detects the card region before querying the embedding index. It uses a YOLO OBB model and perspective warps the detected card to a rectangular image.

```bash
python -m pip install -r requirements-cropping.txt
python scripts/cropping/auto_crop_cards.py \
  --input /path/to/card_photos \
  --output-dir data/processed/crops \
  --save-debug
```

Outputs:

```text
data/processed/crops/*_cardcrop.png
data/processed/crops/*_cardcrop_debug.png
data/processed/crops/crop_manifest.jsonl
data/processed/crops/crop_summary.json
```

If no card is detected, the manifest records `no_detection`; there is no automatic fallback crop.

The same cropper is exposed as a small server endpoint:

```bash
python -m pip install -r requirements-server.txt
python -m uvicorn scripts.server.crop_api:app --host 127.0.0.1 --port 8000
curl -F "file=@/path/to/photo.jpg" "http://127.0.0.1:8000/crop?confidence=0.2"
```
