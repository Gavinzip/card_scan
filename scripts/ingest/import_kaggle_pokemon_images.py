#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.image_utils import infer_pokemon_kaggle_fields, inspect_image, iter_image_files, sha256_file
from scripts.lib.io_utils import write_json, write_jsonl
from scripts.lib.schema import SOURCE_LICENSES, empty_catalog_record, utc_now_iso


def build_record(path: Path, image_root: Path) -> dict[str, Any]:
    inspection = inspect_image(path)
    digest = sha256_file(path) if inspection["is_valid_image"] else None
    inferred = infer_pokemon_kaggle_fields(path, image_root)
    return empty_catalog_record(
        game="pokemon",
        source="kaggle_pokemon_all_image_cards",
        source_license=SOURCE_LICENSES["kaggle_pokemon_all_image_cards"],
        card_id=inferred["card_id"],
        card_code=inferred["card_code"],
        set_id=inferred["set_id"],
        language=inferred["language"],
        variant=inferred["variant"],
        local_image_path=str(path.resolve()),
        image_sha256=digest,
        width=inspection["width"],
        height=inspection["height"],
    ) | {
        "is_valid_image": inspection["is_valid_image"],
        "image_format": inspection["format"],
        "image_error": inspection["image_error"],
    } | {key: value for key, value in inferred.items() if key not in {"card_id", "card_code", "set_id", "language", "variant"}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a locally downloaded Kaggle Pokemon image folder.")
    parser.add_argument("--image-root", required=True, type=Path, help="Root folder of extracted Pokemon card images.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/pokemon_kaggle_images.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/manifests/pokemon_kaggle_images_summary.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit images for smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_root.exists():
        raise SystemExit(f"Image root does not exist: {args.image_root}")

    started_at = utc_now_iso()
    paths = list(iter_image_files(args.image_root))
    if args.limit:
        paths = paths[: args.limit]
    records = [build_record(path, args.image_root) for path in paths]
    count = write_jsonl(args.output, records)

    invalid = sum(1 for record in records if not record["is_valid_image"])
    summary = {
        "source": "kaggle_pokemon_all_image_cards",
        "source_license": SOURCE_LICENSES["kaggle_pokemon_all_image_cards"],
        "image_root": str(args.image_root.resolve()),
        "images_seen": len(paths),
        "records_written": count,
        "invalid_images": invalid,
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
