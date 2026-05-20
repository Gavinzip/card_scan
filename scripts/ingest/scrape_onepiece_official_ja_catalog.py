#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import write_json, write_jsonl
from scripts.lib.schema import SOURCE_LICENSES, empty_catalog_record, utc_now_iso


BASE_URL = "https://www.onepiece-cardgame.com/cardlist/"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SET_ID_PATTERN = re.compile(r"^[A-Z]{1,4}\d{0,2}")


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def image_suffix(image_url: str | None) -> str:
    if not image_url:
        return "base"
    stem = Path(urllib.parse.urlparse(image_url).path).stem
    if "_" not in stem:
        return "base"
    return stem.rsplit("_", 1)[1].lower() or "base"


def set_id_from_code(card_code: str | None) -> str | None:
    if not card_code:
        return None
    match = SET_ID_PATTERN.match(card_code)
    return match.group(0) if match else None


def fetch_url(url: str, cache_path: Path, refresh: bool, timeout: int, retries: int, sleep_seconds: float) -> str:
    if not refresh and cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_text(encoding="utf-8")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "card-scan-pipeline/0.1 (+https://www.onepiece-cardgame.com/)",
                    "Referer": BASE_URL,
                    "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
            cache_path.write_text(text, encoding="utf-8")
            return text
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def parse_series_options(html_text: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html_text, "html.parser")
    select = soup.find("select", {"name": "series"})
    if not select:
        raise ValueError("Could not find series select on Japanese One Piece cardlist.")
    series: list[dict[str, str]] = []
    for option in select.find_all("option"):
        value = clean_text(option.get("value"))
        if not value or value == "ALL":
            continue
        label_html = "".join(str(child) for child in option.contents)
        label_html = label_html.replace("<br class=\"spInline\"/>", " ").replace("<br class=\"spInline\">", " ")
        label = clean_text(BeautifulSoup(html.unescape(label_html), "html.parser").get_text(" "))
        if not label:
            continue
        series.append({"series_id": value, "series_name": label})
    return series


def field_text(back_col: Any, class_name: str) -> str | None:
    node = back_col.find(class_=class_name) if back_col else None
    if not node:
        return None
    heading = node.find("h3")
    if heading:
        heading.extract()
    return clean_text(node.get_text(" "))


def attribute_text(back_col: Any) -> str | None:
    node = back_col.find(class_="attribute") if back_col else None
    if not node:
        return None
    image = node.find("img")
    if image and image.get("alt"):
        return clean_text(image.get("alt"))
    return field_text(back_col, "attribute")


def parse_card_modal(dl: Any, page_url: str, series: dict[str, str]) -> dict[str, Any] | None:
    official_card_id = clean_text(dl.get("id"))
    if not official_card_id:
        return None

    info_spans = [clean_text(span.get_text(" ")) for span in dl.select("dt .infoCol span")]
    if len(info_spans) < 3:
        return None
    card_code, rarity, card_category = info_spans[:3]
    name = clean_text(dl.select_one("dt .cardName").get_text(" ") if dl.select_one("dt .cardName") else None)
    image = dl.select_one(".frontCol img")
    image_src = image.get("data-src") or image.get("src") if image else None
    image_url = urllib.parse.urljoin(page_url, image_src) if image_src else None
    suffix = image_suffix(image_url)
    card_id = f"{official_card_id}-ja"
    back_col = dl.select_one(".backCol")
    get_info = field_text(back_col, "getInfo")

    record = empty_catalog_record(
        game="onepiece",
        source="onepiece_official_ja_catalog",
        source_license=SOURCE_LICENSES["onepiece_official_ja_catalog"],
        card_id=card_id,
        card_code=card_code,
        set_id=set_id_from_code(card_code),
        language="ja",
        name=name,
        name_en=None,
        name_ja=name,
        rarity=rarity,
        variant=suffix,
        image_url=image_url,
        local_image_path=None,
        image_sha256=None,
        width=None,
        height=None,
        is_watermarked=True,
        is_sample=False,
    )
    record.update(
        {
            "official_card_id": official_card_id,
            "image_suffix": suffix,
            "card_category": card_category,
            "card_cost": field_text(back_col, "cost"),
            "card_attribute": attribute_text(back_col),
            "card_power": field_text(back_col, "power"),
            "card_counter": field_text(back_col, "counter"),
            "card_color": field_text(back_col, "color"),
            "card_block_icon": field_text(back_col, "block"),
            "card_feature": field_text(back_col, "feature"),
            "card_effect": field_text(back_col, "text"),
            "card_get_info": get_info,
            "series_id": series["series_id"],
            "series_name": series["series_name"],
            "source_url": page_url,
            "onepiece_url_flag_reasons": ["official_cardlist_image"],
        }
    )
    return record


def parse_series_page(html_text: str, page_url: str, series: dict[str, str]) -> tuple[list[dict[str, Any]], int | None]:
    soup = BeautifulSoup(html_text, "html.parser")
    records = [
        record
        for record in (parse_card_modal(dl, page_url, series) for dl in soup.select("dl.modalCol"))
        if record is not None
    ]
    count_node = soup.select_one(".countCol")
    expected_count = None
    if count_node:
        match = re.search(r"(\d+)", count_node.get_text())
        if match:
            expected_count = int(match.group(1))
    return records, expected_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape the Japanese One Piece official card list metadata.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/onepiece_official_ja_catalog.jsonl")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "data/manifests/onepiece_official_ja_catalog_summary.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/interim/onepiece_official_ja_cardlist_cache",
    )
    parser.add_argument("--limit-series", type=int, default=None)
    parser.add_argument("--series", nargs="*", default=None, help="Optional official series ids to scrape.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now_iso()
    index_html = fetch_url(
        BASE_URL,
        args.cache_dir / "index.html",
        refresh=args.refresh,
        timeout=args.timeout,
        retries=args.retries,
        sleep_seconds=args.sleep,
    )
    series_items = parse_series_options(index_html)
    if args.series:
        requested = set(args.series)
        series_items = [item for item in series_items if item["series_id"] in requested]
    if args.limit_series:
        series_items = series_items[: args.limit_series]
    if not series_items:
        raise SystemExit("No Japanese One Piece series selected.")

    all_records: list[dict[str, Any]] = []
    series_summaries: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str | None]] = set()
    duplicates = 0
    for index, series in enumerate(series_items, start=1):
        params = urllib.parse.urlencode({"series": series["series_id"]})
        page_url = f"{BASE_URL}?{params}"
        print(f"[{index}/{len(series_items)}] Fetching {series['series_id']} {series['series_name']}", file=sys.stderr)
        page_html = fetch_url(
            page_url,
            args.cache_dir / f"series_{safe_filename(series['series_id'])}.html",
            refresh=args.refresh,
            timeout=args.timeout,
            retries=args.retries,
            sleep_seconds=args.sleep,
        )
        records, expected_count = parse_series_page(page_html, page_url, series)
        for record in records:
            key = (str(record.get("official_card_id")), str(record.get("series_id")), record.get("card_get_info"))
            if key in seen_keys:
                duplicates += 1
                continue
            seen_keys.add(key)
            all_records.append(record)
        series_summaries.append(
            {
                "series_id": series["series_id"],
                "series_name": series["series_name"],
                "expected_count": expected_count,
                "records_parsed": len(records),
                "count_matches": expected_count == len(records) if expected_count is not None else None,
            }
        )
        if args.sleep > 0 and index < len(series_items):
            time.sleep(args.sleep)

    count = write_jsonl(args.output, all_records)
    mismatched = [item for item in series_summaries if item["count_matches"] is False]
    summary = {
        "source": "onepiece_official_ja_catalog",
        "source_license": SOURCE_LICENSES["onepiece_official_ja_catalog"],
        "language": "ja",
        "series_seen": len(series_items),
        "records_written": count,
        "unique_card_codes": len({record.get("card_code") for record in all_records}),
        "duplicates_skipped": duplicates,
        "series_count_mismatches": mismatched,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "manifest": str(args.output),
            "summary": str(args.summary_output),
            "cache_dir": str(args.cache_dir),
        },
        "series": series_summaries,
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
