# Data Sources

Last reviewed: 2026-05-20.

This project keeps large downloads out of git. Source manifests and small processed catalogs can be regenerated from the scripts.

## Pokemon

### TCGdex

- URL: https://tcgdex.dev/
- API base: `https://api.tcgdex.net/v2`
- Main use: multilingual Pokemon catalog metadata.
- Current endpoints used:
  - `GET /v2/en/cards`
  - `GET /v2/ja/cards`
  - `GET /v2/en/sets`
  - `GET /v2/ja/sets`
- License note: treat as public API metadata. Image asset reuse needs separate review before training.
- Image asset status: Japanese `high.webp` assets are downloaded for local reference indexing.
- Training image use: no, not by default; use as local reference/search images until rights are reviewed.
- Local outputs:
  - `data/raw/tcgdex/*_cards_brief.json`
  - `data/raw/tcgdex/*_sets.json`
  - `/Users/xiecongfeng/card_data/raw/tcgdex/pokemon/ja/**/*.webp`
  - `data/manifests/tcgdex_pokemon_cards.jsonl`
  - `data/manifests/tcgdex_pokemon_sets.jsonl`
  - `data/manifests/pokemon_tcgdex_ja_images.jsonl`
  - `data/manifests/pokemon_tcgdex_ja_clean_images.jsonl`

TCGdex brief card list is fast and gives `id`, `localId`, `name`, and `image`. Full card details add fields such as rarity and variants, but require one request per card.

### Kaggle Pokemon TCG - All Image Cards

- URL: https://www.kaggle.com/datasets/ellimaaac/pokemon-tcg-all-image-cards
- Main use: primary Pokemon image corpus.
- Kaggle metadata fetched: 2026-05-19.
- License: `CC0-1.0`.
- Published dataset details from Kaggle metadata:
  - 193 sets
  - 20,741 cards
  - expected update frequency: monthly
- Training image use: yes, subject to normal dataset QA.
- Local outputs after import:
  - `data/manifests/pokemon_kaggle_images.jsonl`
  - `data/manifests/pokemon_kaggle_images_summary.json`

The importer expects the user to download/extract the image package first. It computes local path, SHA-256, width, height, and image validity.

### Official Pokemon Card Game Trainer's Website

- URL: https://www.pokemon-card.com/card-search/
- Current query: `regulation_sidebar_form=XY`, which the site labels as standard regulation.
- Main use: Japanese Pokemon catalog metadata and Japanese official card images for local reference matching.
- Current site count for this query: 5,366 cards.
- Local result after QA: 5,366 catalog rows, 5,185 clean reference images, 181 images below the default 200x280 threshold.
- Metadata gap after repair: 8 basic-energy rows still have no `set_id` because the official image path has no set folder; 506 rows have no `card_code`, mostly promo or energy-style records where the official detail page does not expose a card number.
- License note: official site assets. Keep as local reference/search images unless reuse rights are explicitly reviewed.
- Training image use: no, not by default.
- Local outputs:
  - `data/raw/pokemon_card_official/ja/standard_result_pages.jsonl`
  - `/Users/xiecongfeng/card_data/raw/pokemon_card_official/ja/standard/**/*.jpg`
  - `data/manifests/pokemon_card_official_ja_catalog.jsonl`
  - `data/manifests/pokemon_card_official_ja_clean_images.jsonl`

The official source is kept separate from English cards and from TCGdex. If an English scan needs a Japanese market price, that should be exposed as a related Japanese reference price, not as an exact English-card price.

## Japanese Pokemon Canonical Outputs

- Catalog: `data/processed/pokemon_ja_canonical_catalog.jsonl`
- Image manifest: `data/processed/pokemon_ja_canonical_image_manifest.jsonl`
- Image index: `data/processed/pokemon_ja_canonical_image_index/`
- Current canonical result:
  - 9,613 canonical Japanese Pokemon catalog rows
  - 7,215 selected index images after exact image-hash dedupe
  - 5,272 canonical rows use official metadata
  - 4,341 canonical rows use TCGdex metadata to fill official gaps

The canonical output is the recommended Japanese recognition surface. It does not delete raw TCGdex or official manifests; it records overlaps in `duplicate_sources` and exact duplicate image rows in `duplicate_image_records`.

### pokemontcg.io

- URL: https://pokemontcg.io/
- Main use: optional English metadata and price supplement.
- Training image use: no, not part of the first pipeline.
- Status: not implemented in this first pass.

## One Piece

### Kaggle One Piece TCG Card Database

- URL: https://www.kaggle.com/datasets/jbowski/one-piece-tcg-card-database
- Main use: English One Piece card catalog and image URLs.
- Kaggle metadata fetched: 2026-05-19.
- License: `Apache-2.0`.
- Published dataset details from Kaggle metadata:
  - title: One Piece TCG Card Database - July 2025
  - English cards
  - OP01 to OP12
  - ST01 to ST22
  - promo cards included
  - expected update frequency: quarterly
- Training image use: not directly. The source is primarily tabular metadata plus image URLs. URLs are flagged when they look like SAMPLE or watermarked official preview assets.
- Local outputs:
  - `data/raw/kaggle/onepiece/`
  - `data/manifests/onepiece_kaggle_catalog.jsonl`
  - `data/manifests/onepiece_kaggle_summary.json`

### Official One Piece Card Game image assets

- URL source: `image_url` values imported from the Kaggle One Piece catalog.
- Main use: local reference image index for first-pass One Piece recognition.
- Training image use: separate explicit opt-in only. Records keep `is_watermarked` when the source URL is an official preview image.
- Local outputs:
  - `/Users/xiecongfeng/card_data/raw/onepiece/official/`
  - `data/manifests/onepiece_official_images.jsonl`
  - `data/manifests/onepiece_official_images_summary.json`

### Hugging Face t22000t/optcg-en-cards

- URL: https://huggingface.co/datasets/t22000t/optcg-en-cards
- Main use: English One Piece metadata supplement.
- License: `CC-BY-4.0`.
- Dataset viewer at review time: 4.37k rows, parquet format, English.
- Training image use: no images in this dataset.
- Local outputs:
  - `data/raw/huggingface/optcg_en_cards_rows.jsonl`
  - `data/manifests/optcg_hf_catalog.jsonl`
  - `data/manifests/optcg_hf_summary.json`

The default importer uses the Hugging Face dataset rows API. If you have a local parquet file, pass `--input-parquet`; that requires `pandas` and `pyarrow`.

### Hugging Face t22000t/optcg-en-card-embeddings

- URL: https://huggingface.co/datasets/t22000t/optcg-en-card-embeddings
- Main use: text embedding reference.
- License: `CC-BY-4.0`.
- Dataset viewer at review time: 4.37k rows, parquet format.
- Training image use: no.
- Status: documented but not imported by the first-pass image pipeline.

### OPTCG API

- URL: https://optcgapi.com/
- Main use: optional latest English One Piece card table supplement.
- Site notes at review time:
  - free to use API
  - English release data
  - site states OP-01 through OP-15 plus starter decks
- Training image use: no, not by default.
- Status: not implemented in this first pass.

## First-Pass Training Eligibility

| Source | Metadata | Image source | Training image status |
| --- | --- | --- | --- |
| TCGdex | Yes | Local downloaded API assets for Japanese reference index | Do not train until image rights are reviewed |
| Pokemon official Japanese site | Yes | Local downloaded official assets for Japanese reference index | Do not train until image rights are reviewed |
| Kaggle Pokemon images | Minimal inferred metadata | Local image files | Eligible after quality checks |
| One Piece Kaggle | Yes | Image URLs | Do not train by default; SAMPLE/watermark risk |
| One Piece official images | Yes | Local downloaded official preview assets | Separate reference index only; requires explicit `--allow-watermarked` |
| OPTCG HF cards | Yes | None | No image training use |
| OPTCG HF embeddings | Text embeddings | None | Reference only |
| OPTCG API | Optional metadata | API data | Not implemented yet |
