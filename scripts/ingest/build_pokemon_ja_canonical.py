#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import iter_jsonl, write_json, write_jsonl
from scripts.lib.schema import UNIFIED_COLUMNS, utc_now_iso


CANONICAL_SOURCE = "pokemon_ja_canonical"
CANONICAL_SOURCE_LICENSE = "Mixed Japanese Pokemon sources; see canonical_source, image_source, and duplicate_sources"

CANONICAL_EXTRA_COLUMNS = [
    "canonical_id",
    "canonical_key",
    "canonical_source",
    "canonical_source_license",
    "canonical_priority",
    "image_source",
    "image_source_license",
    "image_format",
    "is_valid_image",
    "has_index_image",
    "index_image_selected",
    "excluded_from_index_reason",
    "duplicate_image_count",
    "duplicate_image_records",
    "matchable_by_set_card",
    "source_record_count",
    "duplicate_source_count",
    "duplicate_sources",
    "official_card_id",
    "official_detail_url",
    "official_regulation",
    "number_total",
    "rarity_image_url",
    "illustrator",
    "pokemon_no",
    "pokemon_species",
    "hp",
    "set_id_source",
    "tcgdex_detail_level",
]

CANONICAL_COLUMNS = UNIFIED_COLUMNS + [
    column for column in CANONICAL_EXTRA_COLUMNS if column not in UNIFIED_COLUMNS
]

SOURCE_PRIORITY = {
    "pokemon_card_official_ja": 10,
    "tcgdex": 20,
    "tcgdex_pokemon_images": 30,
}

METADATA_FIELDS = [
    "official_card_id",
    "official_detail_url",
    "official_regulation",
    "number_total",
    "rarity",
    "rarity_image_url",
    "illustrator",
    "pokemon_no",
    "pokemon_species",
    "hp",
    "set_id_source",
    "tcgdex_detail_level",
]

IMAGE_FIELDS = [
    "image_url",
    "local_image_path",
    "image_sha256",
    "width",
    "height",
    "image_format",
    "is_valid_image",
]

SNKR_FIELDS = [
    "snkr_match_status",
    "snkr_product_id",
    "snkr_product_name",
    "snkr_url",
    "snkr_min_price",
    "snkr_min_price_format",
    "snkr_verified_candidate_count",
    "snkr_matched_at",
]


def has_value(value: Any) -> bool:
    return value not in (None, "")


def source_priority(record: dict[str, Any]) -> int:
    return SOURCE_PRIORITY.get(str(record.get("source") or ""), 99)


def canonical_key(record: dict[str, Any]) -> str:
    set_id = record.get("set_id")
    card_code = record.get("card_code")
    if set_id and card_code:
        return f"{set_id}-{card_code}"
    if record.get("official_card_id"):
        return f"official:{record['official_card_id']}"
    return f"{record.get('source', 'unknown')}:{record.get('card_id', 'unknown')}"


def matchable_by_set_card(record: dict[str, Any]) -> bool:
    return bool(record.get("set_id") and record.get("card_code"))


def image_is_index_eligible(record: dict[str, Any], min_width: int, min_height: int) -> bool:
    if not record.get("local_image_path") or not record.get("image_sha256"):
        return False
    if record.get("is_sample") or record.get("is_watermarked"):
        return False
    if record.get("is_valid_image") is False:
        return False
    width = record.get("width")
    height = record.get("height")
    if width is None or height is None:
        return False
    return int(width) >= min_width and int(height) >= min_height


def image_area(record: dict[str, Any]) -> int:
    try:
        return int(record.get("width") or 0) * int(record.get("height") or 0)
    except (TypeError, ValueError):
        return 0


def metadata_score(record: dict[str, Any]) -> tuple[int, int, int, str]:
    filled = sum(1 for field in METADATA_FIELDS if has_value(record.get(field)))
    has_key = int(matchable_by_set_card(record))
    return (source_priority(record), -filled, -has_key, str(record.get("official_card_id") or record.get("card_id") or ""))


def image_score(record: dict[str, Any], min_width: int, min_height: int) -> tuple[int, int, int, str]:
    eligible = int(not image_is_index_eligible(record, min_width, min_height))
    has_image = int(not (record.get("local_image_path") and record.get("image_sha256")))
    return (
        eligible,
        source_priority(record),
        -image_area(record),
        str(record.get("official_card_id") or record.get("card_id") or ""),
    )


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for record in iter_jsonl(path):
            if record.get("game") == "pokemon" and record.get("language") == "ja":
                records.append(record)
    return records


def load_snkr_maps(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[str], int]:
    mapping: dict[str, dict[str, Any]] = {}
    loaded = []
    rows = 0
    for path in paths:
        if not path.exists():
            continue
        loaded.append(str(path))
        for record in iter_jsonl(path):
            card_id = record.get("card_id")
            if not card_id:
                continue
            mapping[str(card_id)] = record
            rows += 1
    return mapping, loaded, rows


def apply_snkr(record: dict[str, Any], snkr_map: dict[str, dict[str, Any]]) -> None:
    snkr = snkr_map.get(str(record.get("card_id") or ""))
    if not snkr:
        return
    record.update(
        {
            "snkr_match_status": snkr.get("match_status"),
            "snkr_product_id": snkr.get("snkr_product_id"),
            "snkr_product_name": snkr.get("snkr_product_name"),
            "snkr_url": snkr.get("snkr_url"),
            "snkr_min_price": snkr.get("snkr_min_price"),
            "snkr_min_price_format": snkr.get("snkr_min_price_format"),
            "snkr_verified_candidate_count": snkr.get("verified_candidate_count"),
            "snkr_matched_at": snkr.get("matched_at"),
        }
    )


def source_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": record.get("source"),
        "source_license": record.get("source_license"),
        "card_id": record.get("card_id"),
        "card_code": record.get("card_code"),
        "set_id": record.get("set_id"),
        "name_ja": record.get("name_ja") or record.get("name"),
        "official_card_id": record.get("official_card_id"),
        "image_url": record.get("image_url"),
        "local_image_path": record.get("local_image_path"),
        "image_sha256": record.get("image_sha256"),
        "width": record.get("width"),
        "height": record.get("height"),
        "image_format": record.get("image_format"),
        "is_valid_image": record.get("is_valid_image"),
        "rarity": record.get("rarity"),
        "variant": record.get("variant"),
    }


def index_duplicate_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_id": record.get("canonical_id"),
        "canonical_key": record.get("canonical_key"),
        "canonical_source": record.get("canonical_source"),
        "card_id": record.get("card_id"),
        "card_code": record.get("card_code"),
        "set_id": record.get("set_id"),
        "name_ja": record.get("name_ja") or record.get("name"),
        "image_source": record.get("image_source"),
        "local_image_path": record.get("local_image_path"),
        "image_sha256": record.get("image_sha256"),
    }


def canonical_record(
    key: str,
    group: list[dict[str, Any]],
    snkr_map: dict[str, dict[str, Any]],
    min_width: int,
    min_height: int,
) -> dict[str, Any]:
    primary = sorted(group, key=metadata_score)[0]
    image_candidates = [record for record in group if record.get("local_image_path") and record.get("image_sha256")]
    image_record = sorted(image_candidates, key=lambda item: image_score(item, min_width, min_height))[0] if image_candidates else None

    record = {column: None for column in CANONICAL_COLUMNS}
    for column in UNIFIED_COLUMNS:
        record[column] = primary.get(column)

    record["source"] = CANONICAL_SOURCE
    record["source_license"] = CANONICAL_SOURCE_LICENSE
    record["canonical_id"] = f"pokemon-ja:{key}"
    record["canonical_key"] = key
    record["canonical_source"] = primary.get("source")
    record["canonical_source_license"] = primary.get("source_license")
    record["canonical_priority"] = source_priority(primary)
    record["image_source"] = image_record.get("source") if image_record else None
    record["image_source_license"] = image_record.get("source_license") if image_record else None
    record["matchable_by_set_card"] = matchable_by_set_card(primary)
    record["source_record_count"] = len(group)

    if matchable_by_set_card(primary):
        record["card_id"] = f"{primary.get('set_id')}-{primary.get('card_code')}"

    for field in METADATA_FIELDS:
        record[field] = primary.get(field)
    for field in IMAGE_FIELDS:
        record[field] = image_record.get(field) if image_record else None

    record["has_index_image"] = image_is_index_eligible(record, min_width, min_height)
    record["index_image_selected"] = False
    record["excluded_from_index_reason"] = None
    record["duplicate_image_count"] = 0
    record["duplicate_image_records"] = []
    record["duplicate_sources"] = [
        source_summary(item)
        for item in group
        if item is not primary and (image_record is None or item is not image_record)
    ]
    record["duplicate_source_count"] = len(record["duplicate_sources"])

    for field in SNKR_FIELDS:
        if field not in record:
            record[field] = None
    apply_snkr(record, snkr_map)
    if record.get("snkr_match_status") is None and primary is not record:
        apply_snkr(primary, snkr_map)
        for field in SNKR_FIELDS:
            record[field] = primary.get(field)

    now = utc_now_iso()
    record["created_at"] = primary.get("created_at") or now
    record["updated_at"] = now
    return record


def index_selection_score(record: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(not bool(record.get("set_id") and record.get("card_code"))),
        int(not bool(record.get("set_id"))),
        SOURCE_PRIORITY.get(str(record.get("canonical_source") or ""), 99),
        SOURCE_PRIORITY.get(str(record.get("image_source") or ""), 99),
        str(record.get("canonical_id") or ""),
    )


def select_index_images(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("has_index_image") and record.get("image_sha256"):
            by_hash[str(record["image_sha256"])].append(record)

    selected = []
    for group in by_hash.values():
        winner = sorted(group, key=index_selection_score)[0]
        duplicates = [record for record in group if record is not winner]
        winner["index_image_selected"] = True
        winner["duplicate_image_count"] = len(duplicates)
        winner["duplicate_image_records"] = [index_duplicate_summary(record) for record in duplicates]
        selected.append(winner)
        for duplicate in duplicates:
            duplicate["index_image_selected"] = False
            duplicate["excluded_from_index_reason"] = "duplicate_image_sha256"
            duplicate["duplicate_image_count"] = 0
            duplicate["duplicate_image_records"] = []
    return sorted(selected, key=sort_key)


def sort_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("set_id") or ""),
        str(record.get("card_code") or ""),
        str(record.get("canonical_source") or ""),
        str(record.get("canonical_id") or ""),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a canonical Japanese Pokemon catalog from official and TCGdex sources.")
    parser.add_argument("--official", type=Path, default=ROOT / "data/manifests/pokemon_card_official_ja_catalog.jsonl")
    parser.add_argument("--tcgdex-metadata", type=Path, default=ROOT / "data/manifests/tcgdex_pokemon_cards.jsonl")
    parser.add_argument("--tcgdex-images", type=Path, default=ROOT / "data/manifests/pokemon_tcgdex_ja_clean_images.jsonl")
    parser.add_argument("--snkr-map", action="append", type=Path, default=None)
    parser.add_argument("--output-catalog", type=Path, default=ROOT / "data/processed/pokemon_ja_canonical_catalog.jsonl")
    parser.add_argument("--output-image-manifest", type=Path, default=ROOT / "data/processed/pokemon_ja_canonical_image_manifest.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/processed/pokemon_ja_canonical_summary.json")
    parser.add_argument("--min-width", type=int, default=200)
    parser.add_argument("--min-height", type=int, default=280)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now_iso()
    snkr_paths = args.snkr_map or [
        ROOT / "data/manifests/snkr_pokemon_ja_product_map.jsonl",
        ROOT / "data/manifests/snkr_pokemon_official_ja_product_map.jsonl",
    ]
    snkr_map, loaded_snkr_maps, snkr_rows = load_snkr_maps(snkr_paths)

    input_paths = [args.official, args.tcgdex_metadata, args.tcgdex_images]
    records = load_records(input_paths)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[canonical_key(record)].append(record)

    canonical = [
        canonical_record(key, group, snkr_map, args.min_width, args.min_height)
        for key, group in grouped.items()
    ]
    canonical.sort(key=sort_key)
    image_manifest = select_index_images(canonical)

    catalog_count = write_jsonl(args.output_catalog, canonical)
    image_count = write_jsonl(args.output_image_manifest, image_manifest)

    by_canonical_source = Counter(str(record.get("canonical_source")) for record in canonical)
    by_image_source = Counter(str(record.get("image_source")) for record in canonical)
    by_snkr_status = Counter(str(record.get("snkr_match_status") or "not_mapped") for record in canonical)
    input_by_source = Counter(str(record.get("source")) for record in records)
    duplicate_source_records = sum(int(record.get("duplicate_source_count") or 0) for record in canonical)
    duplicate_image_records = sum(int(record.get("duplicate_image_count") or 0) for record in image_manifest)
    excluded_duplicate_images = sum(1 for record in canonical if record.get("excluded_from_index_reason") == "duplicate_image_sha256")

    summary = {
        "inputs": [str(path) for path in input_paths if path.exists()],
        "records_read": len(records),
        "input_by_source": dict(sorted(input_by_source.items())),
        "canonical_records": catalog_count,
        "image_manifest_records": image_count,
        "duplicate_source_records": duplicate_source_records,
        "duplicate_image_records": duplicate_image_records,
        "excluded_duplicate_images": excluded_duplicate_images,
        "by_canonical_source": dict(sorted(by_canonical_source.items())),
        "by_image_source": dict(sorted(by_image_source.items())),
        "by_snkr_status": dict(sorted(by_snkr_status.items())),
        "snkr_maps": loaded_snkr_maps,
        "snkr_map_rows_loaded": snkr_rows,
        "min_width": args.min_width,
        "min_height": args.min_height,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "catalog": str(args.output_catalog),
            "image_manifest": str(args.output_image_manifest),
            "summary": str(args.summary_output),
        },
        "notes": [
            "Official Japanese Pokemon records are preferred for duplicate set_id/card_code keys.",
            "TCGdex records are retained as duplicate_sources for overlapping keys and become canonical only when the official source has no matching set_id/card_code.",
            "Rows without both set_id and card_code are kept with source-specific canonical keys.",
        ],
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
