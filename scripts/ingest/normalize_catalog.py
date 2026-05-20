#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import existing_paths, iter_jsonl, write_json, write_jsonl
from scripts.lib.schema import UNIFIED_COLUMNS, coerce_catalog_record, utc_now_iso


DEFAULT_INPUTS = [
    ROOT / "data/manifests/tcgdex_pokemon_cards.jsonl",
    ROOT / "data/manifests/pokemon_clean_images.jsonl",
    ROOT / "data/manifests/pokemon_tcgdex_ja_clean_images.jsonl",
    ROOT / "data/manifests/pokemon_card_official_ja_catalog.jsonl",
    ROOT / "data/manifests/onepiece_kaggle_catalog.jsonl",
    ROOT / "data/manifests/onepiece_official_images.jsonl",
    ROOT / "data/manifests/optcg_hf_catalog.jsonl",
]

DEFAULT_SNKR_MAPS = [
    ROOT / "data/manifests/snkr_pokemon_ja_product_map.jsonl",
    ROOT / "data/manifests/snkr_pokemon_official_ja_product_map.jsonl",
]

SQLITE_TYPES = {
    "width": "INTEGER",
    "height": "INTEGER",
    "is_watermarked": "INTEGER",
    "is_sample": "INTEGER",
    "snkr_min_price": "REAL",
    "snkr_verified_candidate_count": "INTEGER",
}


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    if record.get("local_image_path"):
        return (
            record.get("game"),
            record.get("source"),
            record.get("local_image_path"),
        )
    return (
        record.get("game"),
        record.get("source"),
        record.get("language"),
        record.get("card_id"),
        record.get("card_code"),
        record.get("variant"),
    )


def load_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for path in paths:
        for raw in iter_jsonl(path):
            record = coerce_catalog_record(raw)
            key = record_key(record)
            if key in seen:
                duplicates.append({"input": str(path), "key": key, "record": record})
                continue
            seen.add(key)
            records.append(record)

    records.sort(
        key=lambda item: (
            item.get("game") or "",
            item.get("source") or "",
            item.get("language") or "",
            item.get("set_id") or "",
            item.get("card_code") or "",
            item.get("card_id") or "",
        )
    )
    return records, duplicates


def load_snkr_product_maps(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[str], int, int]:
    mapping: dict[str, dict[str, Any]] = {}
    loaded_paths = []
    rows_loaded = 0
    overwritten = 0
    for path in existing_paths(paths):
        loaded_paths.append(str(path))
        for record in iter_jsonl(path):
            card_id = record.get("card_id")
            if not card_id:
                continue
            key = str(card_id)
            if key in mapping:
                overwritten += 1
            mapping[key] = record
            rows_loaded += 1
    return mapping, loaded_paths, rows_loaded, overwritten


def enrich_with_snkr_map(records: list[dict[str, Any]], snkr_map: dict[str, dict[str, Any]]) -> int:
    if not snkr_map:
        return 0
    count = 0
    for record in records:
        if record.get("game") != "pokemon" or record.get("language") != "ja":
            continue
        card_id = record.get("card_id")
        if not card_id or str(card_id) not in snkr_map:
            continue
        snkr = snkr_map[str(card_id)]
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
        count += 1
    return count


def write_sqlite(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    columns_sql = ", ".join(
        f"{column} {SQLITE_TYPES.get(column, 'TEXT')}" for column in UNIFIED_COLUMNS
    )
    placeholders = ", ".join("?" for _ in UNIFIED_COLUMNS)

    with sqlite3.connect(path) as connection:
        connection.execute(f"CREATE TABLE cards ({columns_sql})")
        rows = []
        for record in records:
            row = []
            for column in UNIFIED_COLUMNS:
                value = record.get(column)
                if column in ("is_watermarked", "is_sample"):
                    value = 1 if value else 0
                row.append(value)
            rows.append(row)
        connection.executemany(
            f"INSERT INTO cards ({', '.join(UNIFIED_COLUMNS)}) VALUES ({placeholders})",
            rows,
        )
        connection.execute("CREATE INDEX idx_cards_game_source ON cards (game, source)")
        connection.execute("CREATE INDEX idx_cards_card_id ON cards (card_id)")
        connection.execute("CREATE INDEX idx_cards_card_code ON cards (card_code)")
        connection.execute("CREATE INDEX idx_cards_language ON cards (language)")
        connection.execute("CREATE INDEX idx_cards_image_sha256 ON cards (image_sha256)")
        connection.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize source manifests into unified JSONL and SQLite catalogs.")
    parser.add_argument("--inputs", nargs="*", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=ROOT / "data/processed/catalog.jsonl")
    parser.add_argument("--output-sqlite", type=Path, default=ROOT / "data/processed/catalog.sqlite")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/processed/catalog_summary.json")
    parser.add_argument("--duplicates-output", type=Path, default=ROOT / "data/manifests/catalog_duplicates.jsonl")
    parser.add_argument(
        "--snkr-map",
        action="append",
        type=Path,
        default=None,
        help="SNKRDUNK mapping JSONL. Repeat to merge multiple maps. Defaults to known Pokemon JA maps.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now_iso()
    inputs = existing_paths(args.inputs if args.inputs is not None else DEFAULT_INPUTS)
    if not inputs:
        raise SystemExit("No input manifests found. Run ingestion scripts first or pass --inputs.")

    records, duplicates = load_records(inputs)
    snkr_map_paths = args.snkr_map if args.snkr_map is not None else DEFAULT_SNKR_MAPS
    snkr_map, loaded_snkr_paths, snkr_rows_loaded, snkr_overwritten = load_snkr_product_maps(snkr_map_paths)
    snkr_enriched_records = enrich_with_snkr_map(records, snkr_map)
    jsonl_count = write_jsonl(args.output_jsonl, records)
    write_sqlite(args.output_sqlite, records)
    duplicate_count = write_jsonl(args.duplicates_output, duplicates) if duplicates else 0

    by_game: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for record in records:
        by_game[record["game"]] = by_game.get(record["game"], 0) + 1
        by_source[record["source"]] = by_source.get(record["source"], 0) + 1

    summary = {
        "inputs": [str(path) for path in inputs],
        "records_written": jsonl_count,
        "duplicates_skipped": duplicate_count,
        "snkr_maps": loaded_snkr_paths,
        "snkr_map_rows_loaded": snkr_rows_loaded,
        "snkr_map_records": len(snkr_map),
        "snkr_map_overwritten_card_ids": snkr_overwritten,
        "snkr_enriched_records": snkr_enriched_records,
        "by_game": by_game,
        "by_source": by_source,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "jsonl": str(args.output_jsonl),
            "sqlite": str(args.output_sqlite),
            "duplicates": str(args.duplicates_output) if duplicates else None,
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
