#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.image_utils import inspect_image, sha256_file
from scripts.lib.io_utils import iter_jsonl, write_json, write_jsonl
from scripts.lib.schema import SOURCE_LICENSES, empty_catalog_record, utc_now_iso


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def safe_part(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def extension_from_url(url: str | None) -> str:
    if not url:
        return ".png"
    path = urllib.parse.urlparse(url).path
    suffix = Path(path).suffix.lower()
    return suffix if suffix in IMAGE_EXTENSIONS else ".png"


def referer_for_url(url: str) -> str:
    hostname = urllib.parse.urlparse(url).hostname or ""
    if hostname == "www.onepiece-cardgame.com":
        return "https://www.onepiece-cardgame.com/cardlist/"
    return "https://en.onepiece-cardgame.com/cardlist/"


def valid_existing_image(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and bool(inspect_image(path)["is_valid_image"])


def download(
    url: str,
    output: Path,
    timeout: int,
    retries: int,
    sleep_seconds: float,
    delay: float,
    delay_jitter: float,
) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    if valid_existing_image(output):
        return False
    if output.exists():
        output.unlink()

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        if delay > 0:
            time.sleep(delay + random.uniform(0, max(0.0, delay_jitter)))
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/123.0.0.0 Safari/537.36"
                    ),
                    "Referer": referer_for_url(url),
                    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                    "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                output.write_bytes(response.read())
            if not valid_existing_image(output):
                raise RuntimeError(f"downloaded file is not a valid image: {output}")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * (attempt + 1) + random.uniform(0, max(0.0, delay_jitter)))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def image_record(card: dict[str, Any], local_path: Path) -> dict[str, Any]:
    inspection = inspect_image(local_path)
    digest = sha256_file(local_path) if inspection["is_valid_image"] else None
    return empty_catalog_record(
        game="onepiece",
        source="onepiece_official_images",
        source_license=SOURCE_LICENSES["onepiece_official_images"],
        card_id=card.get("card_id"),
        card_code=card.get("card_code"),
        set_id=card.get("set_id"),
        language=card.get("language") or "en",
        name=card.get("name"),
        name_en=card.get("name_en") or card.get("name"),
        name_ja=card.get("name_ja"),
        rarity=card.get("rarity"),
        variant=card.get("variant"),
        image_url=card.get("image_url"),
        local_image_path=str(local_path.resolve()),
        image_sha256=digest,
        width=inspection["width"],
        height=inspection["height"],
        is_watermarked=card.get("is_watermarked"),
        is_sample=card.get("is_sample"),
    ) | {
        "is_valid_image": inspection["is_valid_image"],
        "image_format": inspection["format"],
        "image_error": inspection["image_error"],
        "source_catalog": card.get("source"),
        "source_catalog_license": card.get("source_license"),
        "onepiece_url_flag_reasons": card.get("onepiece_url_flag_reasons"),
    }


def process_card(
    index: int,
    card: dict[str, Any],
    args: argparse.Namespace,
    image_root: Path,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, bool]:
    image_url = card.get("image_url")
    if not image_url:
        return (
            index,
            None,
            {
                "card_id": card.get("card_id"),
                "card_code": card.get("card_code"),
                "error": "missing_image_url",
                "created_at": utc_now_iso(),
            },
            False,
        )

    set_id = safe_part(card.get("set_id"), "unknown_set")
    card_id = safe_part(card.get("card_id"), f"card_{index}")
    local_path = image_root / set_id / f"{card_id}{extension_from_url(image_url)}"
    try:
        did_download = download(
            image_url,
            local_path,
            args.timeout,
            args.retries,
            args.sleep,
            args.delay,
            args.delay_jitter,
        )
        return index, image_record(card, local_path), None, did_download
    except RuntimeError as exc:
        return (
            index,
            None,
            {
                "card_id": card.get("card_id"),
                "card_code": card.get("card_code"),
                "set_id": card.get("set_id"),
                "image_url": image_url,
                "error": str(exc),
                "created_at": utc_now_iso(),
            },
            False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download One Piece official card images from catalog image URLs.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/manifests/onepiece_kaggle_catalog.jsonl")
    parser.add_argument("--image-root", type=Path, default=Path("/Users/xiecongfeng/card_data/raw/onepiece/official"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/onepiece_official_images.jsonl")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "data/manifests/onepiece_official_images_summary.json",
    )
    parser.add_argument("--errors-output", type=Path, default=ROOT / "data/manifests/onepiece_official_image_errors.jsonl")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--delay", type=float, default=0.0, help="Per-request delay before each download attempt.")
    parser.add_argument("--delay-jitter", type=float, default=0.0, help="Random extra delay added to request and retry sleeps.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input manifest does not exist: {args.input}")
    started_at = utc_now_iso()

    candidates = [record for record in iter_jsonl(args.input) if record.get("image_url")]
    if args.limit:
        candidates = candidates[: args.limit]

    record_items: list[tuple[int, dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []
    downloaded = 0
    skipped_existing = 0
    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(process_card, index, card, args, args.image_root)
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
                print(f"Processed {completed_count}/{len(candidates)} One Piece images", file=sys.stderr)

    records = [record for _, record in sorted(record_items, key=lambda item: item[0])]
    count = write_jsonl(args.output, records)
    if errors:
        write_jsonl(args.errors_output, errors)
    invalid = sum(1 for record in records if not record["is_valid_image"])
    summary = {
        "source": "onepiece_official_images",
        "source_license": SOURCE_LICENSES["onepiece_official_images"],
        "candidates": len(candidates),
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "records_written": count,
        "invalid_images": invalid,
        "errors": len(errors),
        "flagged_sample": sum(1 for record in records if record["is_sample"]),
        "flagged_watermarked": sum(1 for record in records if record["is_watermarked"]),
        "image_root": str(args.image_root),
        "limit": args.limit,
        "workers": args.workers,
        "delay": args.delay,
        "delay_jitter": args.delay_jitter,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "manifest": str(args.output),
            "errors": str(args.errors_output) if errors else None,
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
