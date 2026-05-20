#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import iter_jsonl, write_json, write_jsonl
from scripts.lib.schema import utc_now_iso


def is_clean(
    record: dict[str, Any],
    min_width: int,
    min_height: int,
    allow_watermarked: bool,
    allow_sample: bool,
) -> bool:
    if not record.get("is_valid_image", True):
        return False
    if record.get("is_sample") and not allow_sample:
        return False
    if record.get("is_watermarked") and not allow_watermarked:
        return False
    if not record.get("local_image_path") or not record.get("image_sha256"):
        return False
    width = record.get("width")
    height = record.get("height")
    if width is None or height is None:
        return False
    return int(width) >= min_width and int(height) >= min_height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter an image manifest down to clean training/index candidates.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/manifests/pokemon_kaggle_images.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/pokemon_clean_images.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/manifests/pokemon_clean_images_summary.json")
    parser.add_argument("--min-width", type=int, default=200)
    parser.add_argument("--min-height", type=int, default=280)
    parser.add_argument(
        "--allow-watermarked",
        action="store_true",
        help="Include records flagged as watermarked, for explicit local reference indexes.",
    )
    parser.add_argument(
        "--allow-sample",
        action="store_true",
        help="Include records flagged as sample assets, for explicit local reference indexes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now_iso()
    records = list(iter_jsonl(args.input))
    clean = [
        record
        for record in records
        if is_clean(record, args.min_width, args.min_height, args.allow_watermarked, args.allow_sample)
    ]
    count = write_jsonl(args.output, clean)
    summary = {
        "input": str(args.input),
        "records_read": len(records),
        "records_written": count,
        "excluded": len(records) - count,
        "min_width": args.min_width,
        "min_height": args.min_height,
        "allow_watermarked": args.allow_watermarked,
        "allow_sample": args.allow_sample,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "output": str(args.output),
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
