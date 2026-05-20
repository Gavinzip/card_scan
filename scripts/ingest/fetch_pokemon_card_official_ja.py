#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.image_utils import inspect_image, sha256_file
from scripts.lib.io_utils import write_json, write_jsonl
from scripts.lib.schema import SOURCE_LICENSES, empty_catalog_record, utc_now_iso

BASE_URL = "https://www.pokemon-card.com"
RESULT_API = f"{BASE_URL}/card-search/resultAPI.php"


def request_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": f"{BASE_URL}/card-search/index.php",
    }


def fetch_json(url: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers=request_headers() | {"Accept": "application/json, text/plain, */*", "X-Requested-With": "XMLHttpRequest"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch_text(url: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers=request_headers())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def absolute_url(value: str | None) -> str | None:
    if not value:
        return None
    return urllib.parse.urljoin(BASE_URL, value)


def strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return " ".join(text.split())


def safe_part(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def rarity_from_path(path: str | None) -> str | None:
    if not path:
        return None
    stem = Path(path).stem
    if stem.startswith("ic_rare_"):
        return stem.removeprefix("ic_rare_")
    return stem or None


def set_id_from_image_path(path: str | None) -> str | None:
    if not path:
        return None
    match = re.search(r"/card_images/large/([^/]+)/", path)
    if not match:
        return None
    return html.unescape(match.group(1)).strip() or None


def parse_detail_html(detail_html: str) -> dict[str, Any]:
    h1_match = re.search(r'<h1[^>]*class="[^"]*Heading1[^"]*"[^>]*>(.*?)</h1>', detail_html, flags=re.S)
    image_match = re.search(r'<img[^>]*class="[^"]*fit[^"]*"[^>]*src="([^"]+)"[^>]*alt="([^"]*)"', detail_html, flags=re.S)
    subtext_match = re.search(r'<div[^>]*class="[^"]*subtext[^"]*"[^>]*>(.*?)</div>', detail_html, flags=re.S)
    subtext_html = subtext_match.group(1) if subtext_match else ""
    subtext = strip_tags(subtext_html)

    set_match = re.search(r'class="[^"]*img-regulation[^"]*"[^>]*alt="([^"]+)"', subtext_html, flags=re.S)
    number_match = re.search(r"([A-Za-z0-9-]+)\s*/\s*([A-Za-z0-9-]+)", subtext)
    rarity_match = re.search(r'(/assets/images/card/rarity/[^"]+)', subtext_html)
    illustrator_match = re.search(r"<h4>イラストレーター</h4>\s*<a[^>]*>(.*?)</a>", detail_html, flags=re.S)
    pokemon_no_match = re.search(r"<h4>No\.([0-9]+)\s*([^<]+)</h4>", detail_html)
    hp_match = re.search(r'<span class="hp">HP</span>\s*<span class="hp-num">([^<]+)</span>', detail_html, flags=re.S)

    image_src = image_match.group(1) if image_match else None
    image_alt = html.unescape(image_match.group(2)) if image_match else None
    name = strip_tags(h1_match.group(1)) if h1_match else image_alt
    set_id_from_detail = html.unescape(set_match.group(1)).strip() if set_match else None
    derived_set_id = set_id_from_detail or set_id_from_image_path(image_src)

    return {
        "name_ja": name,
        "image_url": absolute_url(image_src),
        "image_path": image_src,
        "image_alt": image_alt,
        "set_id": derived_set_id,
        "set_id_source": "detail_subtext" if set_id_from_detail else ("image_path" if derived_set_id else None),
        "card_code": number_match.group(1) if number_match else None,
        "number_total": number_match.group(2) if number_match else None,
        "rarity": rarity_from_path(rarity_match.group(1) if rarity_match else None),
        "rarity_image_url": absolute_url(rarity_match.group(1) if rarity_match else None),
        "illustrator": strip_tags(illustrator_match.group(1)) if illustrator_match else None,
        "pokemon_no": pokemon_no_match.group(1) if pokemon_no_match else None,
        "pokemon_species": strip_tags(pokemon_no_match.group(2)) if pokemon_no_match else None,
        "hp": strip_tags(hp_match.group(1)) if hp_match else None,
    }


def result_api_url(regulation_sidebar_form: str, page: int) -> str:
    params = {
        "keyword": "",
        "se_ta": "",
        "regulation_sidebar_form": regulation_sidebar_form,
        "pg": "",
        "illust": "",
        "sm_and_keyword": "true",
        "page": str(page),
    }
    return f"{RESULT_API}?{urllib.parse.urlencode(params)}"


def fetch_result_pages(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    rows_by_card_id: dict[str, dict[str, Any]] = {}

    first = fetch_json(result_api_url(args.regulation_sidebar_form, 1), args.timeout)
    pages.append(first)
    max_page = int(first.get("maxPage") or 1)
    if args.max_pages:
        max_page = min(max_page, args.max_pages)

    for item in first.get("cardList") or []:
        card_id = str(item.get("cardID") or "").strip()
        if card_id:
            rows_by_card_id[card_id] = item

    for page in range(2, max_page + 1):
        if args.limit and len(rows_by_card_id) >= args.limit:
            break
        if args.page_delay:
            time.sleep(args.page_delay)
        data = fetch_json(result_api_url(args.regulation_sidebar_form, page), args.timeout)
        pages.append(data)
        for item in data.get("cardList") or []:
            card_id = str(item.get("cardID") or "").strip()
            if card_id:
                rows_by_card_id[card_id] = item
        if page % 25 == 0:
            print(f"Fetched official Pokemon card search page {page}/{max_page}", file=sys.stderr)

    rows = list(rows_by_card_id.values())
    if args.limit:
        rows = rows[: args.limit]
    summary = {
        "hitCnt": first.get("hitCnt"),
        "maxPage": first.get("maxPage"),
        "fetched_pages": len(pages),
        "unique_card_ids": len(rows_by_card_id),
        "selected_rows": len(rows),
        "regulation": first.get("regulation"),
        "searchCondition": first.get("searchCondition"),
    }
    return pages, rows, summary


def fetch_detail_record(
    row: dict[str, Any],
    regulation: str,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    official_card_id = str(row.get("cardID") or "").strip()
    detail_url = f"{BASE_URL}/card-search/details.php/card/{official_card_id}/regu/{regulation}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if retry_sleep and attempt > 0:
                time.sleep(retry_sleep * attempt)
            detail = parse_detail_html(fetch_text(detail_url, timeout))
            detail["official_card_id"] = official_card_id
            detail["official_detail_url"] = detail_url
            detail["list_image_url"] = absolute_url(row.get("cardThumbFile"))
            detail["list_name_ja"] = row.get("cardNameViewText") or row.get("cardNameAltText")
            return detail, None
        except Exception as exc:
            last_error = exc
    return (
        {
            "official_card_id": official_card_id,
            "official_detail_url": detail_url,
            "list_image_url": absolute_url(row.get("cardThumbFile")),
            "list_name_ja": row.get("cardNameViewText") or row.get("cardNameAltText"),
        },
        {
            "official_card_id": official_card_id,
            "official_detail_url": detail_url,
            "error": str(last_error),
            "created_at": utc_now_iso(),
        },
    )


def fetch_details(rows: list[dict[str, Any]], regulation: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    details: list[tuple[int, dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []
    order_by_id = {str(row.get("cardID") or ""): index for index, row in enumerate(rows)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.detail_workers)) as executor:
        futures = [
            executor.submit(fetch_detail_record, row, regulation, args.timeout, args.detail_retries, args.detail_retry_sleep)
            for row in rows
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            detail, error = future.result()
            official_id = detail.get("official_card_id")
            order = order_by_id.get(str(official_id), index)
            details.append((order, detail))
            if error:
                errors.append(error)
            if index % 250 == 0:
                print(f"Fetched official Pokemon card details {index}/{len(rows)}", file=sys.stderr)
    return [detail for _, detail in sorted(details, key=lambda item: item[0])], errors


def download(url: str, output: Path, timeout: int, retries: int, sleep_seconds: float) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return False
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=request_headers())
            with urllib.request.urlopen(request, timeout=timeout) as response:
                output.write_bytes(response.read())
            return True
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def local_image_path(detail: dict[str, Any], image_root: Path) -> Path:
    official_id = safe_part(str(detail.get("official_card_id") or ""), "unknown_id")
    set_id = safe_part(detail.get("set_id"), "unknown_set")
    card_code = safe_part(detail.get("card_code"), "unknown_code")
    image_url = detail.get("image_url") or detail.get("list_image_url") or ""
    suffix = Path(urllib.parse.urlparse(image_url).path).suffix or ".jpg"
    return image_root / set_id / f"official_{official_id}_{set_id}_{card_code}{suffix}"


def record_from_detail(detail: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    image_url = detail.get("image_url") or detail.get("list_image_url")
    local_path = detail.get("local_image_path")
    inspection = inspect_image(local_path) if local_path else {"width": None, "height": None, "is_valid_image": None, "format": None, "image_error": None}
    digest = sha256_file(local_path) if local_path and inspection["is_valid_image"] else None
    set_id = detail.get("set_id")
    card_code = detail.get("card_code")
    card_id = f"{set_id}-{card_code}" if set_id and card_code else f"official_{detail.get('official_card_id')}"
    name = detail.get("name_ja") or detail.get("list_name_ja")
    return empty_catalog_record(
        game="pokemon",
        source="pokemon_card_official_ja",
        source_license=SOURCE_LICENSES["pokemon_card_official_ja"],
        card_id=card_id,
        card_code=card_code,
        set_id=set_id,
        language="ja",
        name=name,
        name_ja=name,
        rarity=detail.get("rarity"),
        image_url=image_url,
        local_image_path=str(Path(local_path).resolve()) if local_path else None,
        image_sha256=digest,
        width=inspection["width"],
        height=inspection["height"],
    ) | {
        "official_card_id": detail.get("official_card_id"),
        "official_detail_url": detail.get("official_detail_url"),
        "official_regulation": args.regulation_sidebar_form,
        "set_id_source": detail.get("set_id_source"),
        "number_total": detail.get("number_total"),
        "rarity_image_url": detail.get("rarity_image_url"),
        "illustrator": detail.get("illustrator"),
        "pokemon_no": detail.get("pokemon_no"),
        "pokemon_species": detail.get("pokemon_species"),
        "hp": detail.get("hp"),
        "is_valid_image": inspection["is_valid_image"],
        "image_format": inspection["format"],
        "image_error": inspection["image_error"],
    }


def download_images(details: list[dict[str, Any]], args: argparse.Namespace) -> tuple[int, int, list[dict[str, Any]]]:
    image_root = args.image_root / args.regulation_label
    errors: list[dict[str, Any]] = []
    downloaded = 0
    skipped = 0

    def process(detail: dict[str, Any]) -> tuple[dict[str, Any], bool | None, dict[str, Any] | None]:
        image_url = detail.get("image_url") or detail.get("list_image_url")
        if not image_url:
            return detail, None, {"official_card_id": detail.get("official_card_id"), "error": "missing image_url", "created_at": utc_now_iso()}
        path = local_image_path(detail, image_root)
        try:
            did_download = download(image_url, path, args.timeout, args.retries, args.download_sleep)
            detail["local_image_path"] = str(path.resolve())
            return detail, did_download, None
        except RuntimeError as exc:
            return detail, None, {"official_card_id": detail.get("official_card_id"), "image_url": image_url, "error": str(exc), "created_at": utc_now_iso()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as executor:
        futures = [executor.submit(process, detail) for detail in details]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            detail, did_download, error = future.result()
            if did_download is True:
                downloaded += 1
            elif did_download is False:
                skipped += 1
            if error:
                errors.append(error)
            if index % 250 == 0:
                print(f"Downloaded/checked official Pokemon card images {index}/{len(details)}", file=sys.stderr)
    return downloaded, skipped, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Japanese Pokemon card catalog and images from pokemon-card.com.")
    parser.add_argument("--regulation-sidebar-form", default="XY", help="Official site filter. XY currently means standard.")
    parser.add_argument("--regulation-label", default="standard")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/pokemon_card_official/ja")
    parser.add_argument("--image-root", type=Path, default=Path("/Users/xiecongfeng/card_data/raw/pokemon_card_official/ja"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/pokemon_card_official_ja_catalog.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/manifests/pokemon_card_official_ja_summary.json")
    parser.add_argument("--errors-output", type=Path, default=ROOT / "data/manifests/pokemon_card_official_ja_errors.jsonl")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--page-delay", type=float, default=0.05)
    parser.add_argument("--detail-workers", type=int, default=8)
    parser.add_argument("--detail-retries", type=int, default=4)
    parser.add_argument("--detail-retry-sleep", type=float, default=2.0)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--download-sleep", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now_iso()
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    pages, rows, list_summary = fetch_result_pages(args)
    write_jsonl(args.raw_dir / f"{args.regulation_label}_result_pages.jsonl", pages)

    regulation = str(list_summary.get("regulation") or args.regulation_sidebar_form)
    details, detail_errors = fetch_details(rows, regulation, args)

    downloaded = 0
    skipped_existing = 0
    download_errors: list[dict[str, Any]] = []
    if args.download_images:
        downloaded, skipped_existing, download_errors = download_images(details, args)

    records = [record_from_detail(detail, args) for detail in details]
    records_written = write_jsonl(args.output, records)
    errors = detail_errors + download_errors
    if errors:
        write_jsonl(args.errors_output, errors)
    elif args.errors_output.exists():
        args.errors_output.unlink()

    missing_set_or_code = sum(1 for record in records if not record.get("set_id") or not record.get("card_code"))
    invalid_images = sum(1 for record in records if record.get("is_valid_image") is False)
    summary = {
        "source": "pokemon_card_official_ja",
        "source_url": f"{BASE_URL}/card-search/",
        "regulation_sidebar_form": args.regulation_sidebar_form,
        "regulation_label": args.regulation_label,
        "list_summary": list_summary,
        "records_written": records_written,
        "download_images": bool(args.download_images),
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "missing_set_or_code": missing_set_or_code,
        "invalid_images": invalid_images,
        "errors": len(errors),
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "catalog": str(args.output),
            "summary": str(args.summary_output),
            "raw_pages": str(args.raw_dir / f"{args.regulation_label}_result_pages.jsonl"),
            "errors": str(args.errors_output) if errors else None,
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
