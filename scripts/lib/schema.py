from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


UNIFIED_COLUMNS = [
    "game",
    "source",
    "source_license",
    "card_id",
    "card_code",
    "set_id",
    "language",
    "name",
    "name_en",
    "name_ja",
    "rarity",
    "variant",
    "image_url",
    "local_image_path",
    "image_sha256",
    "width",
    "height",
    "is_watermarked",
    "is_sample",
    "snkr_match_status",
    "snkr_product_id",
    "snkr_product_name",
    "snkr_url",
    "snkr_min_price",
    "snkr_min_price_format",
    "snkr_verified_candidate_count",
    "snkr_matched_at",
    "created_at",
    "updated_at",
]


SOURCE_LICENSES = {
    "tcgdex": "TCGdex public API metadata; image asset reuse requires separate review",
    "tcgdex_pokemon_images": "TCGdex image assets; use as local reference index only until rights are reviewed",
    "pokemon_card_official_ja": "Official Pokemon Card Game Trainer's Website assets; local reference index only",
    "kaggle_pokemon_all_image_cards": "CC0-1.0",
    "onepiece_kaggle": "Apache-2.0",
    "onepiece_official_ja_catalog": "Official One Piece Card Game Japan card list metadata; local reference index only",
    "onepiece_official_images": "Official One Piece Card Game image assets; local reference index only",
    "optcg_hf_cards": "CC-BY-4.0",
    "optcg_hf_card_embeddings": "CC-BY-4.0",
    "optcgapi": "Public API; verify terms before redistribution",
}


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def empty_catalog_record(**overrides: Any) -> dict[str, Any]:
    now = utc_now_iso()
    record = {column: None for column in UNIFIED_COLUMNS}
    record.update(
        {
            "is_watermarked": False,
            "is_sample": False,
            "created_at": now,
            "updated_at": now,
        }
    )
    record.update(overrides)
    return coerce_catalog_record(record)


def coerce_catalog_record(record: dict[str, Any]) -> dict[str, Any]:
    coerced = {column: record.get(column) for column in UNIFIED_COLUMNS}
    for key in ("is_watermarked", "is_sample"):
        coerced[key] = bool(coerced.get(key))
    for key in ("width", "height", "snkr_verified_candidate_count"):
        coerced[key] = coerce_int(coerced.get(key))
    return coerced


def coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def truthy_variant_string(variants: Any) -> str | None:
    if not variants:
        return None
    if isinstance(variants, str):
        return variants
    if isinstance(variants, dict):
        enabled = sorted(str(key) for key, value in variants.items() if value)
        return ",".join(enabled) or None
    if isinstance(variants, list):
        return ",".join(str(item) for item in variants) or None
    return str(variants)


def language_name_fields(language: str | None, name: str | None) -> dict[str, str | None]:
    if language == "en":
        return {"name_en": name, "name_ja": None}
    if language == "ja":
        return {"name_en": None, "name_ja": name}
    return {"name_en": None, "name_ja": None}


def derive_set_id_from_card_id(card_id: str | None) -> str | None:
    if not card_id or "-" not in card_id:
        return None
    return card_id.rsplit("-", 1)[0]
