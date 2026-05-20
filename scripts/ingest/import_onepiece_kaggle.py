#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import write_json, write_jsonl
from scripts.lib.onepiece_flags import flag_onepiece_image_url
from scripts.lib.schema import SOURCE_LICENSES, empty_catalog_record, utc_now_iso


DEFAULT_DATASET = "jbowski/one-piece-tcg-card-database"


def normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip().lower(): value for key, value in row.items()}


def first(row: dict[str, Any], *names: str) -> Any:
    normalized = normalized_row(row)
    for name in names:
        value = normalized.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def infer_set_id(card_code: str | None, expansion: str | None) -> str | None:
    if card_code and "-" in card_code:
        return card_code.split("-", 1)[0]
    return expansion


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        return list(csv.DictReader(handle, dialect=dialect))


def download_dataset(dataset: str, download_dir: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(download_dir), "--unzip"]
    subprocess.run(command, check=True)


def find_single_csv(download_dir: Path) -> Path:
    csv_files = sorted(download_dir.rglob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found under {download_dir}")
    if len(csv_files) > 1:
        joined = "\n".join(str(path) for path in csv_files)
        raise SystemExit(f"Multiple CSV files found; pass --input explicitly:\n{joined}")
    return csv_files[0]


def row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    card_id = first(row, "card_id", "id")
    card_code = first(row, "card_code", "code")
    image_url = first(row, "card_image", "image", "image_url")
    expansion = first(row, "card_expansion", "expansion", "set_name")
    flags = flag_onepiece_image_url(image_url)
    variant = first(row, "card_art_variant", "art_variant", "variant")
    name = first(row, "card_name", "name")
    if not card_id:
        card_id = f"{card_code}-{variant}" if card_code and variant is not None else card_code
    return empty_catalog_record(
        game="onepiece",
        source="onepiece_kaggle",
        source_license=SOURCE_LICENSES["onepiece_kaggle"],
        card_id=card_id,
        card_code=card_code,
        set_id=infer_set_id(card_code, expansion),
        language="en",
        name=name,
        name_en=name,
        rarity=first(row, "card_rarity", "rarity"),
        variant=str(variant) if variant is not None else None,
        image_url=image_url,
        is_watermarked=flags["is_watermarked"],
        is_sample=flags["is_sample"],
    ) | {
        "onepiece_url_flag_reasons": flags["reasons"],
        "card_type": first(row, "card_type", "type"),
        "card_color": first(row, "card_color", "color"),
        "card_effect": first(row, "card_effect", "effect"),
        "card_trigger": first(row, "card_trigger", "trigger"),
        "card_banned": first(row, "card_banned", "banned"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the One Piece Kaggle CSV into a catalog manifest.")
    parser.add_argument("--input", type=Path, default=None, help="CSV path. If omitted, use --download.")
    parser.add_argument("--download", action="store_true", help="Download the Kaggle dataset before importing.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--download-dir", type=Path, default=ROOT / "data/raw/kaggle/onepiece")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/onepiece_kaggle_catalog.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/manifests/onepiece_kaggle_summary.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now_iso()

    if args.download:
        download_dataset(args.dataset, args.download_dir)

    input_path = args.input or find_single_csv(args.download_dir)
    if not input_path.exists():
        raise SystemExit(f"CSV input does not exist: {input_path}")

    rows = read_csv_rows(input_path)
    if args.limit:
        rows = rows[: args.limit]
    records = [row_to_record(row) for row in rows]
    count = write_jsonl(args.output, records)
    flagged_sample = sum(1 for record in records if record["is_sample"])
    flagged_watermarked = sum(1 for record in records if record["is_watermarked"])

    summary = {
        "source": "onepiece_kaggle",
        "source_license": SOURCE_LICENSES["onepiece_kaggle"],
        "dataset": args.dataset,
        "input": str(input_path),
        "rows_read": len(rows),
        "records_written": count,
        "flagged_sample": flagged_sample,
        "flagged_watermarked": flagged_watermarked,
        "limit": args.limit,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "output": str(args.output),
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
