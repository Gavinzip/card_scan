# Unified Catalog Schema

The normalized catalog keeps these columns in a stable order:

| Column | Meaning |
| --- | --- |
| `game` | `pokemon` or `onepiece` |
| `source` | Source adapter name |
| `source_license` | Source license or review note |
| `card_id` | Source-specific stable card id |
| `card_code` | Printed or source card code |
| `set_id` | Set, pack, or expansion id |
| `language` | Metadata language, such as `en` or `ja` |
| `name` | Name in the row language |
| `name_en` | English name when available |
| `name_ja` | Japanese name when available |
| `rarity` | Source rarity string |
| `variant` | Variant or printing marker |
| `image_url` | Remote image URL when available |
| `local_image_path` | Local image path when imported |
| `image_sha256` | SHA-256 for local image files |
| `width` | Local image width |
| `height` | Local image height |
| `is_watermarked` | Training exclusion flag |
| `is_sample` | Training exclusion flag |
| `snkr_match_status` | SNKRDUNK mapping status: `matched`, `multiple_verified_matches`, or `no_verified_match` |
| `snkr_product_id` | SNKRDUNK product id when there is exactly one verified match |
| `snkr_product_name` | SNKRDUNK product title for the selected product |
| `snkr_url` | SNKRDUNK product URL for the selected product |
| `snkr_min_price` | SNKRDUNK current minimum listing price from search results |
| `snkr_min_price_format` | SNKRDUNK price string and currency from search results |
| `snkr_verified_candidate_count` | Count of products that passed set/card-number verification |
| `snkr_matched_at` | Time the SNKRDUNK mapping row was generated |
| `created_at` | Pipeline record creation time |
| `updated_at` | Source update time or pipeline update time |

Source manifests may carry extra source-specific fields. The full SNKRDUNK candidate list stays in `data/manifests/snkr_pokemon_ja_product_map.jsonl`; the normalized catalog only stores the selected product when the match is unique.

## Japanese Pokemon Canonical Schema

`data/processed/pokemon_ja_canonical_catalog.jsonl` extends the unified columns. It is the preferred Japanese Pokemon lookup surface for recognition because it removes duplicate source rows while keeping provenance.

Additional fields:

| Column | Meaning |
| --- | --- |
| `canonical_id` | Stable canonical id, such as `pokemon-ja:M5-001` |
| `canonical_key` | Merge key; usually `set_id-card_code` |
| `canonical_source` | Metadata source chosen for the canonical row |
| `canonical_source_license` | License/review note for the chosen metadata source |
| `canonical_priority` | Numeric source priority; lower is preferred |
| `image_source` | Source chosen for the canonical image |
| `image_source_license` | License/review note for the chosen image |
| `image_format` | Local image format, such as `JPEG` or `WEBP` |
| `is_valid_image` | Image integrity flag |
| `has_index_image` | The row has a clean local image candidate |
| `index_image_selected` | The row is included in the canonical FAISS image index |
| `excluded_from_index_reason` | Why a clean image row was excluded from the index, currently `duplicate_image_sha256` |
| `duplicate_image_count` | Number of exact duplicate image rows represented by this index image |
| `duplicate_image_records` | Minimal records for exact duplicate images omitted from the index |
| `matchable_by_set_card` | Whether `set_id + card_code` exists and can be used for source merging / SNKRDUNK verification |
| `source_record_count` | Number of input rows collapsed into this canonical row |
| `duplicate_source_count` | Number of non-primary source rows preserved in `duplicate_sources` |
| `duplicate_sources` | Minimal source-row provenance for overlapped official / TCGdex records |
| `official_card_id` | Pokemon-card.com detail id when available |
| `official_detail_url` | Pokemon-card.com detail URL |
| `official_regulation` | Official search regulation used, such as `XY` |
| `number_total` | Official card number denominator |
| `rarity_image_url` | Official rarity icon URL |
| `illustrator` | Illustrator from the official detail page |
| `pokemon_no` | Pokemon species number when present |
| `pokemon_species` | Japanese Pokemon species text |
| `hp` | HP string when present |
| `set_id_source` | Indicates when `set_id` was derived from image path rather than detail text |
| `tcgdex_detail_level` | TCGdex fetch mode for metadata rows |

Canonical merge rules:

- Prefer `pokemon_card_official_ja` for duplicate `set_id + card_code` keys.
- Keep overlapping TCGdex rows in `duplicate_sources`.
- Promote TCGdex rows to canonical only when there is no official row with the same key.
- Keep source-specific rows that do not have both `set_id` and `card_code`, but mark `matchable_by_set_card=false`.
- Exclude exact duplicate image hashes from the FAISS image manifest while preserving their metadata in `duplicate_image_records`.
