#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.image_utils import inspect_image, sha256_file
from scripts.lib.io_utils import iter_jsonl, write_json, write_jsonl
from scripts.lib.schema import SOURCE_LICENSES, empty_catalog_record, utc_now_iso


def tcgdex_asset_url(base_url: str, quality: str, extension: str) -> str:
    return f"{base_url.rstrip('/')}/{quality}.{extension.lstrip('.')}"


def safe_part(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def download(url: str, output: Path, timeout: int, retries: int, sleep_seconds: float) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return False

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "card-scan-pipeline/0.1 (+https://tcgdex.dev/)"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                output.write_bytes(response.read())
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def image_record(card: dict[str, Any], local_path: Path, image_url: str) -> dict[str, Any]:
    inspection = inspect_image(local_path)
    digest = sha256_file(local_path) if inspection["is_valid_image"] else None
    name = card.get("name")
    language = card.get("language")
    return empty_catalog_record(
        game="pokemon",
        source="tcgdex_pokemon_images",
        source_license=SOURCE_LICENSES["tcgdex_pokemon_images"],
        card_id=card.get("card_id"),
        card_code=card.get("card_code"),
        set_id=card.get("set_id"),
        language=language,
        name=name,
        name_en=name if language == "en" else None,
        name_ja=name if language == "ja" else None,
        rarity=card.get("rarity"),
        variant=card.get("variant"),
        image_url=image_url,
        local_image_path=str(local_path.resolve()),
        image_sha256=digest,
        width=inspection["width"],
        height=inspection["height"],
    ) | {
        "is_valid_image": inspection["is_valid_image"],
        "image_format": inspection["format"],
        "image_error": inspection["image_error"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Pokemon card images from TCGdex image URLs.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/manifests/tcgdex_pokemon_cards.jsonl")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--quality", choices=["high", "low"], default="high")
    parser.add_argument("--extension", choices=["webp", "png", "jpg"], default="webp")
    parser.add_argument("--image-root", type=Path, default=Path("/Users/xiecongfeng/card_data/raw/tcgdex/pokemon"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def process_card(
    index: int,
    card: dict[str, Any],
    args: argparse.Namespace,
    image_root: Path,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, bool]:
    set_id = safe_part(card.get("set_id"), "unknown_set")
    card_id = safe_part(card.get("card_id"), f"card_{index}")
    local_path = image_root / set_id / f"{card_id}_{args.quality}.{args.extension}"
    image_url = tcgdex_asset_url(card["image_url"], args.quality, args.extension)
    try:
        did_download = download(image_url, local_path, args.timeout, args.retries, args.sleep)
        return index, image_record(card, local_path, image_url), None, did_download
    except RuntimeError as exc:
        return (
            index,
            None,
            {
                "card_id": card.get("card_id"),
                "set_id": card.get("set_id"),
                "language": args.language,
                "image_url": image_url,
                "error": str(exc),
                "created_at": utc_now_iso(),
            },
            False,
        )


def main() -> int:
    args = parse_args()
    started_at = utc_now_iso()
    output = args.output or ROOT / f"data/manifests/pokemon_tcgdex_{args.language}_images.jsonl"
    summary_output = args.summary_output or ROOT / f"data/manifests/pokemon_tcgdex_{args.language}_images_summary.json"
    image_root = args.image_root / args.language

    candidates = [
        record
        for record in iter_jsonl(args.input)
        if record.get("source") == "tcgdex"
        and record.get("language") == args.language
        and record.get("image_url")
    ]
    if args.limit:
        candidates = candidates[: args.limit]

    record_items: list[tuple[int, dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []
    downloaded = 0
    skipped_existing = 0
    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(process_card, index, card, args, image_root)
            for index, card in enumerate(candidates, start=1)
        ]
        for future in concurrent.futures.as_completed(futures):
            index, record, error, did_download = future.result()
            completed_count += 1
            if record:
                record_items.append((index, record))
                downloaded += int(did_download)
                skipped_existing += int(not did_download)
            if error:
                errors.append(error)
            if completed_count % 250 == 0:
                print(
                    f"Processed {completed_count}/{len(candidates)} TCGdex {args.language} images",
                    file=sys.stderr,
                )

    records = [record for _, record in sorted(record_items, key=lambda item: item[0])]
    count = write_jsonl(output, records)
    error_output = ROOT / f"data/manifests/pokemon_tcgdex_{args.language}_image_errors.jsonl"
    if errors:
        write_jsonl(error_output, errors)
    invalid = sum(1 for record in records if not record["is_valid_image"])
    summary = {
        "source": "tcgdex_pokemon_images",
        "source_license": SOURCE_LICENSES["tcgdex_pokemon_images"],
        "language": args.language,
        "quality": args.quality,
        "extension": args.extension,
        "candidates": len(candidates),
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "records_written": count,
        "invalid_images": invalid,
        "errors": len(errors),
        "image_root": str(image_root),
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "manifest": str(output),
            "errors": str(error_output) if errors else None,
        },
    }
    write_json(summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
