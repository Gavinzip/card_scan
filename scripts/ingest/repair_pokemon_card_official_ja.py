#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ingest.fetch_pokemon_card_official_ja import (
    download_images,
    fetch_detail_record,
    record_from_detail,
)
from scripts.lib.io_utils import iter_jsonl, write_json, write_jsonl
from scripts.lib.schema import utc_now_iso


def selected_for_repair(record: dict[str, Any], repair_missing_code: bool) -> bool:
    if record.get("source") != "pokemon_card_official_ja":
        return False
    if not record.get("official_card_id"):
        return False
    if not record.get("set_id"):
        return True
    return bool(repair_missing_code and not record.get("card_code"))


def row_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "cardID": record.get("official_card_id"),
        "cardThumbFile": record.get("image_url"),
        "cardNameViewText": record.get("name_ja") or record.get("name"),
        "cardNameAltText": record.get("name_ja") or record.get("name"),
    }


def fetch_repairs(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    repaired: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    rows = [row_from_record(record) for record in records]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.detail_workers)) as executor:
        futures = [
            executor.submit(
                fetch_detail_record,
                row,
                args.regulation,
                args.timeout,
                args.detail_retries,
                args.detail_retry_sleep,
            )
            for row in rows
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            detail, error = future.result()
            official_id = str(detail.get("official_card_id") or "")
            if error:
                errors.append(error)
            else:
                repaired[official_id] = detail
            if index % 100 == 0:
                print(f"Repaired official Pokemon card details {index}/{len(rows)}", file=sys.stderr)
    return repaired, errors


def count_missing(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "records": len(records),
        "missing_set_id": sum(1 for record in records if not record.get("set_id")),
        "missing_card_code": sum(1 for record in records if not record.get("card_code")),
        "missing_set_or_code": sum(1 for record in records if not record.get("set_id") or not record.get("card_code")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair missing official Japanese Pokemon card details without a full refetch.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/manifests/pokemon_card_official_ja_catalog.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/pokemon_card_official_ja_catalog.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/manifests/pokemon_card_official_ja_repair_summary.json")
    parser.add_argument("--errors-output", type=Path, default=ROOT / "data/manifests/pokemon_card_official_ja_repair_errors.jsonl")
    parser.add_argument("--image-root", type=Path, default=Path("/Users/xiecongfeng/card_data/raw/pokemon_card_official/ja"))
    parser.add_argument("--regulation-label", default="standard")
    parser.add_argument("--regulation", default="XY")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--detail-workers", type=int, default=2)
    parser.add_argument("--detail-retries", type=int, default=4)
    parser.add_argument("--detail-retry-sleep", type=float, default=2.0)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--download-sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--repair-missing-code",
        action="store_true",
        help="Also refetch records that have a set_id but no card_code. Some official promo/energy cards truly have no number.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.regulation_sidebar_form = args.regulation
    started_at = utc_now_iso()
    records = list(iter_jsonl(args.input))
    before = count_missing(records)
    targets = [record for record in records if selected_for_repair(record, args.repair_missing_code)]
    if not targets:
        summary = {
            "input": str(args.input),
            "output": str(args.output),
            "selected_for_repair": 0,
            "before": before,
            "after": before,
            "errors": 0,
            "started_at": started_at,
            "completed_at": utc_now_iso(),
        }
        write_json(args.summary_output, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    repaired_details, detail_errors = fetch_repairs(targets, args)
    if args.download_images and repaired_details:
        repaired_detail_list = list(repaired_details.values())
        _, _, download_errors = download_images(repaired_detail_list, args)
        detail_errors.extend(download_errors)

    repaired_records: dict[str, dict[str, Any]] = {}
    for official_id, detail in repaired_details.items():
        repaired_records[official_id] = record_from_detail(detail, args)

    output_records = []
    for record in records:
        official_id = str(record.get("official_card_id") or "")
        output_records.append(repaired_records.get(official_id, record))

    records_written = write_jsonl(args.output, output_records)
    if detail_errors:
        write_jsonl(args.errors_output, detail_errors)
    elif args.errors_output.exists():
        args.errors_output.unlink()

    after = count_missing(output_records)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "records_written": records_written,
        "selected_for_repair": len(targets),
        "repaired_records": len(repaired_records),
        "download_images": bool(args.download_images),
        "before": before,
        "after": after,
        "errors": len(detail_errors),
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "summary": str(args.summary_output),
            "errors": str(args.errors_output) if detail_errors else None,
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
