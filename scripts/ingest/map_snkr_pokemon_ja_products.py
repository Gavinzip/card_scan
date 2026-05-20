#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import iter_jsonl, write_json, write_jsonl
from scripts.lib.schema import utc_now_iso

SNKR_SEARCH_URL = "https://snkrdunk.com/en/v1/search"


def card_number_parts(card_code: str | None) -> tuple[str, str]:
    match = re.search(r"\d+", str(card_code or ""))
    clean = (match.group(0).lstrip("0") if match else "") or "0"
    return clean, clean.zfill(3)


def exact_set_match(title: str, set_id: str) -> bool:
    if not set_id:
        return False
    return re.search(
        rf"(?<![A-Za-z0-9]){re.escape(set_id)}(?![A-Za-z0-9])",
        str(title or ""),
        flags=re.IGNORECASE,
    ) is not None


def title_number_match(title: str, card_code: str | None) -> tuple[bool, str]:
    number_clean, number_padded = card_number_parts(card_code)
    if not number_clean or number_clean == "0":
        return True, "no_number_constraint"

    text = str(title or "").lower()
    fractions = re.findall(r"(\d{1,4})\s*/\s*(\d{1,4})", text)
    for numerator_raw, denominator_raw in fractions:
        numerator = numerator_raw.lstrip("0") or "0"
        if numerator == number_clean:
            return True, f"fraction_numerator:{numerator_raw}/{denominator_raw}"

    text_without_fractions = re.sub(r"\d{1,4}\s*/\s*\d{1,4}", " ", text)
    if number_padded and re.search(rf"(?<!\d){re.escape(number_padded)}(?!\d)", text_without_fractions):
        return True, "standalone_padded"
    if number_clean and re.search(rf"(?<!\d){re.escape(number_clean)}(?!\d)", text_without_fractions):
        return True, "standalone_clean"
    return False, "number_mismatch"


def title_card_ref_match(title: str, set_id: str, card_code: str | None) -> tuple[bool, str]:
    if not set_id:
        return False, "missing_set_id"
    number_clean, number_padded = card_number_parts(card_code)
    if not number_clean or number_clean == "0":
        return False, "missing_card_code"

    # SNKRDUNK Pokemon titles usually encode the canonical reference as:
    #   [SV8a 200/187]
    #   C[S9 001/100]
    # Require set_id and card number to appear together in that order. This
    # prevents short set ids such as S12 from matching unrelated denominators
    # like "S9/S12", and prevents card 001 from matching prose like "1st".
    pattern = (
        rf"(?<![A-Za-z0-9]){re.escape(set_id)}(?![A-Za-z0-9])"
        rf"\s+0*{re.escape(number_clean)}\s*/\s*[A-Za-z0-9-]+"
    )
    if re.search(pattern, str(title or ""), flags=re.IGNORECASE):
        return True, f"set_card_ref:{set_id} {number_padded}"
    return False, "set_card_ref_mismatch"


def title_has_language_exclusion(title: str) -> bool:
    title_l = str(title or "").lower()
    markers = [
        "[en]",
        "【en】",
        " english",
        "english version",
        "英語版",
        "英文版",
        "[kr]",
        "【kr】",
        " korean",
        "korean version",
        "韓文版",
        "韓語版",
        "韓国語",
        "韓國語",
    ]
    return any(marker in title_l for marker in markers)


def build_session() -> Any:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise SystemExit("SNKRDUNK mapping requires requests. Install requirements.txt.") from exc

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://snkrdunk.com/",
            "Origin": "https://snkrdunk.com",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        }
    )
    try:
        session.get("https://snkrdunk.com/", timeout=20)
    except Exception as exc:
        print(f"SNKRDUNK warmup failed; continuing: {exc}", file=sys.stderr)
    return session


def cache_path(cache_dir: Path, keyword: str, page: int, per_page: int) -> Path:
    digest = hashlib.sha256(f"{keyword}|{page}|{per_page}".encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def read_cached_json(path: Path, refresh: bool) -> dict[str, Any] | None:
    if refresh or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("status_code") == 200 and isinstance(cached.get("data"), dict):
            return cached
    except Exception:
        return None
    return None


def fetch_search_page(
    session: Any,
    keyword: str,
    page: int,
    per_page: int,
    cache_dir: Path,
    refresh: bool,
    delay: float,
    retries: int,
    rate_limit_sleep: float,
) -> dict[str, Any]:
    path = cache_path(cache_dir, keyword, page, per_page)
    cached = read_cached_json(path, refresh)
    if cached is not None:
        return cached | {"from_cache": True}

    cache_dir.mkdir(parents=True, exist_ok=True)
    params = {"keyword": keyword, "perPage": str(per_page), "page": str(page)}
    url = f"{SNKR_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    last_error = None
    for attempt in range(retries + 1):
        if delay > 0:
            time.sleep(delay)
        try:
            response = session.get(url, timeout=30)
            status_code = response.status_code
            if status_code == 429:
                last_error = "429 Too Many Requests"
                if attempt < retries:
                    print(
                        f"SNKRDUNK rate limited for {keyword!r} page {page}; "
                        f"sleeping {rate_limit_sleep:.0f}s before retry",
                        file=sys.stderr,
                    )
                    time.sleep(rate_limit_sleep)
                    continue
            if status_code == 403 and attempt < retries:
                try:
                    session.get("https://snkrdunk.com/", timeout=20)
                except Exception:
                    pass
                time.sleep(max(1.0, delay))
                continue
            response.raise_for_status()
            payload = {
                "keyword": keyword,
                "page": page,
                "per_page": per_page,
                "url": url,
                "status_code": status_code,
                "fetched_at": utc_now_iso(),
                "data": response.json(),
            }
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            return payload | {"from_cache": False}
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(max(1.0, delay) * (attempt + 1))

    return {
        "keyword": keyword,
        "page": page,
        "per_page": per_page,
        "url": url,
        "status_code": None,
        "fetched_at": utc_now_iso(),
        "data": {},
        "error": last_error,
        "from_cache": False,
    }


def page_items(data: dict[str, Any]) -> tuple[list[dict[str, Any]], int | None]:
    items: list[dict[str, Any]] = []
    total_count = None
    if not isinstance(data, dict):
        return items, total_count
    for key in ("streetwears", "products"):
        arr = data.get(key, [])
        if isinstance(arr, list):
            items.extend(item for item in arr if isinstance(item, dict))
    count = data.get("streetwearCount")
    if count is None:
        count = data.get("productCount")
    try:
        total_count = int(count) if count is not None else None
    except (TypeError, ValueError):
        total_count = None
    return items, total_count


def fetch_set_products(
    session: Any,
    set_id: str,
    args: argparse.Namespace,
    max_pages: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    products_by_id: dict[str, dict[str, Any]] = {}
    pages_seen = 0
    cache_hits = 0
    errors: list[dict[str, Any]] = []
    total_count = None
    page_limit = max_pages if max_pages is not None else args.max_pages_per_set

    for page in range(1, page_limit + 1):
        payload = fetch_search_page(
            session=session,
            keyword=set_id,
            page=page,
            per_page=args.per_page,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            delay=args.delay,
            retries=args.retries,
            rate_limit_sleep=args.rate_limit_sleep,
        )
        pages_seen += 1
        cache_hits += int(bool(payload.get("from_cache")))
        if payload.get("error"):
            errors.append(
                {
                    "set_id": set_id,
                    "page": page,
                    "error": payload.get("error"),
                    "status_code": payload.get("status_code"),
                }
            )
            break

        items, page_total_count = page_items(payload.get("data") or {})
        if page_total_count is not None:
            total_count = page_total_count
        for item in items:
            if item.get("isTradingCard") is False:
                continue
            pid = str(item.get("id") or "").strip()
            title = str(item.get("name") or "").strip()
            if not pid or not title:
                continue
            products_by_id.setdefault(
                pid,
                {
                    "snkr_product_id": pid,
                    "snkr_product_name": title,
                    "thumbnail_url": item.get("thumbnailUrl") or item.get("imageUrl") or item.get("image") or "",
                    "min_price": item.get("minPrice"),
                    "min_price_format": item.get("minPriceFormat"),
                    "listing_count": item.get("listingCount"),
                    "all_listing_count": item.get("allListingCount"),
                    "offer_count": item.get("offerCount"),
                },
            )

        if not items:
            break
        if total_count is not None:
            needed_pages = max(1, math.ceil(total_count / args.per_page))
            if page >= needed_pages:
                break
        if len(items) < args.per_page:
            break

    meta = {
        "set_id": set_id,
        "keyword": set_id,
        "search_mode": "set",
        "pages_seen": pages_seen,
        "cache_hits": cache_hits,
        "total_count": total_count,
        "unique_products": len(products_by_id),
        "errors": errors,
    }
    return list(products_by_id.values()), meta


def fetch_card_products(
    session: Any,
    card: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_id = str(card.get("set_id") or "")
    _, number_padded = card_number_parts(card.get("card_code"))
    keyword = f"{set_id} {number_padded}".strip()
    products_by_id: dict[str, dict[str, Any]] = {}
    pages_seen = 0
    cache_hits = 0
    errors: list[dict[str, Any]] = []
    total_count = None

    for page in range(1, args.max_pages_per_card + 1):
        payload = fetch_search_page(
            session=session,
            keyword=keyword,
            page=page,
            per_page=args.per_page,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            delay=args.delay,
            retries=args.retries,
            rate_limit_sleep=args.rate_limit_sleep,
        )
        pages_seen += 1
        cache_hits += int(bool(payload.get("from_cache")))
        if payload.get("error"):
            errors.append(
                {
                    "set_id": set_id,
                    "card_id": card.get("card_id"),
                    "keyword": keyword,
                    "page": page,
                    "error": payload.get("error"),
                    "status_code": payload.get("status_code"),
                }
            )
            break

        items, page_total_count = page_items(payload.get("data") or {})
        if page_total_count is not None:
            total_count = page_total_count
        for item in items:
            if item.get("isTradingCard") is False:
                continue
            pid = str(item.get("id") or "").strip()
            title = str(item.get("name") or "").strip()
            if not pid or not title:
                continue
            products_by_id.setdefault(
                pid,
                {
                    "snkr_product_id": pid,
                    "snkr_product_name": title,
                    "thumbnail_url": item.get("thumbnailUrl") or item.get("imageUrl") or item.get("image") or "",
                    "min_price": item.get("minPrice"),
                    "min_price_format": item.get("minPriceFormat"),
                    "listing_count": item.get("listingCount"),
                    "all_listing_count": item.get("allListingCount"),
                    "offer_count": item.get("offerCount"),
                },
            )

        if not items:
            break
        if total_count is not None:
            needed_pages = max(1, math.ceil(total_count / args.per_page))
            if page >= needed_pages:
                break
        if len(items) < args.per_page:
            break

    meta = {
        "set_id": set_id,
        "card_id": card.get("card_id"),
        "keyword": keyword,
        "search_mode": "card",
        "pages_seen": pages_seen,
        "cache_hits": cache_hits,
        "total_count": total_count,
        "unique_products": len(products_by_id),
        "errors": errors,
    }
    return list(products_by_id.values()), meta


def should_use_card_search(cards: list[dict[str, Any]], probe_products: list[dict[str, Any]], probe_meta: dict[str, Any], args: argparse.Namespace) -> bool:
    if len(cards) > args.card_search_set_threshold:
        return False
    total_count = probe_meta.get("total_count")
    if isinstance(total_count, int) and total_count > args.card_search_product_threshold:
        return True
    return total_count is None and len(probe_products) >= args.per_page


def verified_candidates_for_card(card: dict[str, Any], products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    set_id = str(card.get("set_id") or "")
    card_code = str(card.get("card_code") or "")
    verified: list[dict[str, Any]] = []
    for product in products:
        title = str(product.get("snkr_product_name") or "")
        reference_ok, reference_reason = title_card_ref_match(title, set_id, card_code)
        language_excluded = title_has_language_exclusion(title)
        reasons = []
        if reference_ok:
            reasons.append(reference_reason)
        if language_excluded:
            reasons.append("language_marker_excluded")
        if reference_ok and not language_excluded:
            candidate = dict(product)
            candidate["match_reasons"] = reasons
            candidate["snkr_url"] = f"https://snkrdunk.com/apparels/{product['snkr_product_id']}"
            verified.append(candidate)
    verified.sort(key=lambda item: (str(item.get("snkr_product_name") or "").lower(), item["snkr_product_id"]))
    return verified


def mapping_record(card: dict[str, Any], verified: list[dict[str, Any]], set_fetch_meta: dict[str, Any]) -> dict[str, Any]:
    if len(verified) == 1:
        status = "matched"
        selected = verified[0]
    elif len(verified) > 1:
        status = "multiple_verified_matches"
        selected = None
    else:
        status = "no_verified_match"
        selected = None

    return {
        "game": "pokemon",
        "language": "ja",
        "card_id": card.get("card_id"),
        "set_id": card.get("set_id"),
        "card_code": card.get("card_code"),
        "name_ja": card.get("name_ja") or card.get("name"),
        "match_status": status,
        "snkr_product_id": selected.get("snkr_product_id") if selected else None,
        "snkr_product_name": selected.get("snkr_product_name") if selected else None,
        "snkr_url": selected.get("snkr_url") if selected else None,
        "snkr_min_price": selected.get("min_price") if selected else None,
        "snkr_min_price_format": selected.get("min_price_format") if selected else None,
        "verified_candidate_count": len(verified),
        "verified_candidates": verified,
        "set_fetch_pages_seen": set_fetch_meta.get("pages_seen"),
        "set_fetch_unique_products": set_fetch_meta.get("unique_products"),
        "set_fetch_errors": set_fetch_meta.get("errors") or [],
        "matched_at": utc_now_iso(),
    }


def load_cards(path: Path, limit_cards: int | None) -> list[dict[str, Any]]:
    cards = []
    for record in iter_jsonl(path):
        if record.get("language") != "ja":
            continue
        if not record.get("set_id") or not record.get("card_code"):
            continue
        cards.append(record)
        if limit_cards and len(cards) >= limit_cards:
            break
    return cards


def summarize(records: list[dict[str, Any]], set_metas: list[dict[str, Any]], args: argparse.Namespace, started_at: str) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    for record in records:
        status = str(record.get("match_status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
    failed_sets = [meta for meta in set_metas if meta.get("errors")]
    by_search_mode: dict[str, int] = {}
    for meta in set_metas:
        mode = str(meta.get("search_mode") or "unknown")
        by_search_mode[mode] = by_search_mode.get(mode, 0) + 1
    return {
        "source": "snkrdunk",
        "input_manifest": str(args.input),
        "records_written": len(records),
        "sets_seen": len({str(meta.get("set_id")) for meta in set_metas if meta.get("set_id")}),
        "searches_seen": len(set_metas),
        "by_search_mode": by_search_mode,
        "by_status": by_status,
        "failed_sets": failed_sets,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "mapping": str(args.output),
            "summary": str(args.summary_output),
            "cache_dir": str(args.cache_dir),
        },
        "notes": [
            "SNKRDUNK products are verified by exact set_id boundary and card_code number match.",
            "Multiple verified products are not collapsed into a single snkr_product_id.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map Japanese Pokemon cards to SNKRDUNK product IDs.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/manifests/pokemon_tcgdex_ja_clean_images.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/snkr_pokemon_ja_product_map.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/manifests/snkr_pokemon_ja_product_map_summary.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/interim/snkr_search_cache")
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--max-pages-per-set", type=int, default=20)
    parser.add_argument("--max-pages-per-card", type=int, default=2)
    parser.add_argument("--card-search-set-threshold", type=int, default=20)
    parser.add_argument("--card-search-product-threshold", type=int, default=500)
    parser.add_argument("--delay", type=float, default=4.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--rate-limit-sleep", type=float, default=300.0)
    parser.add_argument("--limit-cards", type=int, default=None)
    parser.add_argument("--limit-sets", type=int, default=None)
    parser.add_argument("--sets", nargs="*", default=None)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now_iso()
    cards = load_cards(args.input, args.limit_cards)
    if not cards:
        raise SystemExit("No Japanese Pokemon cards found to map.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        grouped.setdefault(str(card["set_id"]), []).append(card)
    set_ids = sorted(grouped)
    if args.sets:
        requested = {str(item) for item in args.sets}
        set_ids = [set_id for set_id in set_ids if set_id in requested]
    if args.limit_sets:
        set_ids = set_ids[: args.limit_sets]
    if not set_ids:
        raise SystemExit("No set_ids selected.")

    session = build_session()
    records: list[dict[str, Any]] = []
    set_metas: list[dict[str, Any]] = []
    for index, set_id in enumerate(set_ids, start=1):
        print(f"[{index}/{len(set_ids)}] Fetching SNKRDUNK products for set {set_id}", file=sys.stderr)
        cards_in_set = grouped[set_id]
        products, meta = fetch_set_products(session, set_id, args, max_pages=1 if len(cards_in_set) <= args.card_search_set_threshold else None)
        if should_use_card_search(cards_in_set, products, meta, args):
            meta["search_mode"] = "set_probe_broad"
            set_metas.append(meta)
            print(
                f"  set={set_id} probe_products={len(products)} pages={meta['pages_seen']} -> card search",
                file=sys.stderr,
            )
            for card_index, card in enumerate(cards_in_set, start=1):
                card_products, card_meta = fetch_card_products(session, card, args)
                set_metas.append(card_meta)
                verified = verified_candidates_for_card(card, card_products)
                records.append(mapping_record(card, verified, card_meta))
                if card_index % 25 == 0:
                    print(f"  set={set_id} card searches {card_index}/{len(cards_in_set)}", file=sys.stderr)
            write_jsonl(args.output, records)
            write_json(args.summary_output, summarize(records, set_metas, args, started_at))
            continue
        total_count = meta.get("total_count")
        needs_full_set_fetch = len(products) >= args.per_page
        if isinstance(total_count, int):
            needs_full_set_fetch = total_count > len(products)
        if len(cards_in_set) <= args.card_search_set_threshold and needs_full_set_fetch:
            products, meta = fetch_set_products(session, set_id, args)
        set_metas.append(meta)
        print(
            f"  set={set_id} products={len(products)} pages={meta['pages_seen']} errors={len(meta['errors'])}",
            file=sys.stderr,
        )
        for card in cards_in_set:
            verified = verified_candidates_for_card(card, products)
            records.append(mapping_record(card, verified, meta))
        write_jsonl(args.output, records)
        write_json(args.summary_output, summarize(records, set_metas, args, started_at))

    summary = summarize(records, set_metas, args, started_at)
    write_jsonl(args.output, records)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
