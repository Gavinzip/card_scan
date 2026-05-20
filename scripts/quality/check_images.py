#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.image_utils import inspect_image, iter_image_files, sha256_file
from scripts.lib.io_utils import iter_jsonl, write_json, write_jsonl
from scripts.lib.schema import utc_now_iso


def collect_paths(image_root: Path | None, input_manifest: Path | None) -> list[Path]:
    paths: list[Path] = []
    if image_root:
        paths.extend(iter_image_files(image_root))
    if input_manifest:
        for record in iter_jsonl(input_manifest):
            local_path = record.get("local_image_path")
            if local_path:
                paths.append(Path(local_path))
    unique = sorted({path.resolve() for path in paths})
    return unique


def inspect_path(path: Path, min_width: int, min_height: int) -> dict[str, Any]:
    result = inspect_image(path)
    digest = sha256_file(path) if result["is_valid_image"] else None
    width = result["width"]
    height = result["height"]
    too_small = (
        result["is_valid_image"]
        and width is not None
        and height is not None
        and (width < min_width or height < min_height)
    )
    return {
        "local_image_path": str(path),
        "exists": path.exists(),
        "image_sha256": digest,
        "width": width,
        "height": height,
        "image_format": result["format"],
        "is_valid_image": result["is_valid_image"],
        "image_error": result["image_error"],
        "too_small": too_small,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check image integrity, dimensions, and duplicate hashes.")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--input-manifest", type=Path, default=None)
    parser.add_argument("--min-width", type=int, default=200)
    parser.add_argument("--min-height", type=int, default=280)
    parser.add_argument("--report-output", type=Path, default=ROOT / "data/manifests/image_quality_report.json")
    parser.add_argument("--bad-output", type=Path, default=ROOT / "data/manifests/bad_images.jsonl")
    parser.add_argument("--duplicates-output", type=Path, default=ROOT / "data/manifests/duplicate_images.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_root and not args.input_manifest:
        raise SystemExit("Pass --image-root, --input-manifest, or both.")
    started_at = utc_now_iso()

    paths = collect_paths(args.image_root, args.input_manifest)
    checks = [inspect_path(path, args.min_width, args.min_height) for path in paths]
    bad = [item for item in checks if not item["is_valid_image"] or item["too_small"]]

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in checks:
        digest = item.get("image_sha256")
        if digest:
            by_hash[str(digest)].append(item)
    duplicates = [
        {"image_sha256": digest, "files": [entry["local_image_path"] for entry in entries]}
        for digest, entries in sorted(by_hash.items())
        if len(entries) > 1
    ]

    bad_count = write_jsonl(args.bad_output, bad) if bad else 0
    duplicate_count = write_jsonl(args.duplicates_output, duplicates) if duplicates else 0
    report = {
        "images_checked": len(checks),
        "invalid_or_too_small": bad_count,
        "duplicate_hashes": duplicate_count,
        "min_width": args.min_width,
        "min_height": args.min_height,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "bad": str(args.bad_output) if bad else None,
            "duplicates": str(args.duplicates_output) if duplicates else None,
        },
    }
    write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
