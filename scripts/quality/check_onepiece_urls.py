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
from scripts.lib.onepiece_flags import flag_onepiece_image_url
from scripts.lib.schema import utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag One Piece image URLs that look like SAMPLE or watermarked assets.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/manifests/onepiece_kaggle_catalog.jsonl")
    parser.add_argument("--flagged-output", type=Path, default=ROOT / "data/manifests/onepiece_url_flags.jsonl")
    parser.add_argument("--report-output", type=Path, default=ROOT / "data/manifests/onepiece_url_flags_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input manifest does not exist: {args.input}")
    started_at = utc_now_iso()

    records: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for record in iter_jsonl(args.input):
        flags = flag_onepiece_image_url(record.get("image_url"))
        checked = {
            "card_id": record.get("card_id"),
            "card_code": record.get("card_code"),
            "image_url": record.get("image_url"),
            "is_sample": flags["is_sample"],
            "is_watermarked": flags["is_watermarked"],
            "reasons": flags["reasons"],
        }
        records.append(checked)
        if checked["is_sample"] or checked["is_watermarked"]:
            flagged.append(checked)

    flagged_count = write_jsonl(args.flagged_output, flagged) if flagged else 0
    report = {
        "records_checked": len(records),
        "flagged": flagged_count,
        "sample": sum(1 for item in records if item["is_sample"]),
        "watermarked": sum(1 for item in records if item["is_watermarked"]),
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "output": str(args.flagged_output) if flagged else None,
    }
    write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
