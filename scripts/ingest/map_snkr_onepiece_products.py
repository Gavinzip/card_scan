#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import iter_jsonl, write_json, write_jsonl
from scripts.lib.schema import utc_now_iso


SNKR_SEARCH_URL = "https://snkrdunk.com/en/v1/search"
RARITY_ALIASES = {
    "COMMON": "C",
    "UNCOMMON": "UC",
    "RARE": "R",
    "SUPERRARE": "SR",
    "SUPER RARE": "SR",
    "SECRETRARE": "SEC",
    "SECRET RARE": "SEC",
    "LEADER": "L",
    "SPECIAL": "SP CARD",
}
RARITY_PATTERN = re.compile(r"\b(?P<base>SEC|SR|UC|R|C|L|P)(?:-(?P<modifier>SPC|SP|PL|P))?\b", re.IGNORECASE)


def normalize_rarity(value: str | None) -> str | None:
    if not value:
        return None
    clean = re.sub(r"\s+", " ", str(value).strip().upper())
    return RARITY_ALIASES.get(clean, clean)


def normalize_name(value: str | None) -> str:
    text = str(value or "")
    text = re.sub(r"\([^)]*parallel[^)]*\)", "", text, flags=re.IGNORECASE)
    text = text.replace("&amp;", "&")
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def is_parallel_name(value: str | None) -> bool:
    return "parallel" in str(value or "").lower()


def image_suffix(url: str | None) -> str | None:
    if not url:
        return None
    path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
    stem = Path(path).stem
    if "_" not in stem:
        return "base"
    suffix = stem.rsplit("_", 1)[1].lower()
    return suffix or None


def thumbnail_suffix(url: str | None) -> str | None:
    if not url:
        return None
    path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
    stem = Path(path).stem
    if "-of" in stem:
        stem = stem.split("-of", 1)[0]
    match = re.search(r"_(p\d+|r\d+)$", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "base"


def local_variant_kind(card: dict[str, Any]) -> str:
    rarity = normalize_rarity(card.get("rarity"))
    suffix = image_suffix(card.get("image_url"))
    if rarity == "SP CARD":
        return "special"
    if is_parallel_name(card.get("name")):
        return "parallel"
    if suffix and suffix.startswith("p"):
        return "printed_variant"
    return "regular"


def infer_base_rarity_by_code(cards: list[dict[str, Any]]) -> dict[str, str | None]:
    by_code: dict[str, Counter[str]] = {}
    for card in cards:
        code = str(card.get("card_code") or "")
        rarity = normalize_rarity(card.get("rarity"))
        if not code or not rarity or rarity == "SP CARD":
            continue
        by_code.setdefault(code, Counter())[rarity] += 1
    return {
        code: counter.most_common(1)[0][0] if counter else None
        for code, counter in by_code.items()
    }


def expected_base_rarity(card: dict[str, Any], base_rarity_by_code: dict[str, str | None]) -> str | None:
    rarity = normalize_rarity(card.get("rarity"))
    if rarity == "SP CARD":
        return base_rarity_by_code.get(str(card.get("card_code") or ""))
    return rarity


def parse_snkr_title(title: str, card_code: str) -> dict[str, Any]:
    code_marker = f"[{card_code}]"
    before_code = title.split(code_marker, 1)[0] if code_marker in title else title
    matches = list(RARITY_PATTERN.finditer(before_code))
    parsed: dict[str, Any] = {
        "code_exact": code_marker in title,
        "name_prefix": "",
        "name_prefix_normalized": "",
        "base_rarity": None,
        "modifier": None,
        "descriptor": None,
        "is_parallel": "parallel" in before_code.lower(),
        "language": parse_snkr_language(title),
    }
    if not matches:
        return parsed
    match = matches[-1]
    parsed["name_prefix"] = before_code[: match.start()].strip(" -:()[]")
    parsed["name_prefix_normalized"] = normalize_name(parsed["name_prefix"])
    parsed["base_rarity"] = match.group("base").upper()
    modifier = match.group("modifier")
    parsed["modifier"] = modifier.upper() if modifier else None
    trail = before_code[match.end() :].strip()
    trail = re.sub(r"^\s*[:：]\s*", "", trail).strip()
    parsed["descriptor"] = trail or None
    return parsed


def parse_snkr_language(title: str) -> str | None:
    lower = title.lower()
    if "[en]" in lower or "【en】" in lower:
        return "en"
    if "[chn]" in lower or "[cn]" in lower or "【chn】" in lower:
        return "zh"
    if "[kr]" in lower or "【kr】" in lower:
        return "ko"
    return None


def thumbnail_language(url: str | None) -> str | None:
    upper = urllib.parse.unquote(str(url or "")).upper()
    if "OPC-EN-TCG" in upper or "/EN/" in upper:
        return "en"
    return None


def snkr_language_ok(card: dict[str, Any], product: dict[str, Any], parsed: dict[str, Any]) -> tuple[bool, str]:
    language = str(card.get("language") or "").lower()
    if language != "en":
        return True, "language_not_constrained"
    title_language = parsed.get("language")
    image_language = thumbnail_language(product.get("thumbnail_url"))
    if title_language == "en":
        return True, "title_language_en"
    if image_language == "en":
        return True, "thumbnail_language_en"
    return False, "missing_en_marker"


def names_match(card: dict[str, Any], parsed: dict[str, Any]) -> bool:
    local = normalize_name(card.get("name") or card.get("name_en"))
    remote = str(parsed.get("name_prefix_normalized") or "")
    return bool(local and remote and local == remote)


def variant_match(
    card: dict[str, Any],
    product: dict[str, Any],
    parsed: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    kind = local_variant_kind(card)
    local_suffix = image_suffix(card.get("image_url"))
    remote_suffix = thumbnail_suffix(product.get("thumbnail_url"))
    modifier = parsed.get("modifier")
    descriptor = str(parsed.get("descriptor") or "").lower()
    is_snkr_special = modifier in {"SP", "SPC"} or "comic parallel" in descriptor
    is_snkr_parallel = modifier in {"P", "PL"} or (parsed.get("is_parallel") and not is_snkr_special)
    exact_suffix = bool(local_suffix and remote_suffix == local_suffix)

    # If the source catalog gives a concrete image variant, do not collapse it
    # to a broad class such as SR-P. Multiple local variants can share a class.
    if local_suffix and local_suffix != "base":
        if not exact_suffix:
            return False, [f"thumbnail_suffix_mismatch:{local_suffix}!={remote_suffix or 'unknown'}"]

    if kind == "regular":
        if modifier is None and not parsed.get("descriptor") and not parsed.get("is_parallel"):
            if exact_suffix:
                reasons.append(f"thumbnail_suffix:{remote_suffix}")
            reasons.append("regular_class")
            return True, reasons
        return False, ["regular_variant_mismatch"]

    if kind == "printed_variant":
        if exact_suffix:
            reasons.append(f"thumbnail_suffix:{remote_suffix}")
            return True, reasons
        return False, [f"missing_thumbnail_suffix:{local_suffix or 'unknown'}"]

    if kind == "parallel":
        if exact_suffix and is_snkr_parallel:
            reasons.append(f"thumbnail_suffix:{remote_suffix}")
            reasons.append(f"parallel_class:{modifier or 'parallel'}")
            return True, reasons
        if exact_suffix:
            return False, ["parallel_class_mismatch"]
        return False, ["parallel_variant_mismatch"]

    if kind == "special":
        if exact_suffix and is_snkr_special:
            reasons.append(f"thumbnail_suffix:{remote_suffix}")
            reasons.append(f"special_class:{modifier or 'comic_parallel'}")
            return True, reasons
        if exact_suffix:
            return False, ["special_class_mismatch"]
        return False, ["special_variant_mismatch"]

    return False, ["unknown_variant_kind"]


def exact_code_match(title: str, card_code: str) -> bool:
    return f"[{card_code}]" in str(title or "")


def search_name(value: str | None) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s*\([^)]*parallel[^)]*\)\s*", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


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
    delay_jitter: float,
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
            time.sleep(delay + random.uniform(0, max(0.0, delay_jitter)))
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
                    time.sleep(rate_limit_sleep + random.uniform(0, max(0.0, delay_jitter)))
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
                time.sleep(max(1.0, delay) * (attempt + 1) + random.uniform(0, max(0.0, delay_jitter)))

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


def product_from_search_item(item: dict[str, Any], code: str) -> dict[str, Any] | None:
    if item.get("isTradingCard") is False:
        return None
    pid = str(item.get("id") or "").strip()
    title = str(item.get("name") or "").strip()
    if not pid or not title or not exact_code_match(title, code):
        return None
    return {
        "snkr_product_id": pid,
        "snkr_product_name": title,
        "thumbnail_url": item.get("thumbnailUrl") or item.get("imageUrl") or item.get("image") or "",
        "min_price": item.get("minPrice"),
        "min_price_format": item.get("minPriceFormat"),
        "listing_count": item.get("listingCount"),
        "all_listing_count": item.get("allListingCount"),
        "offer_count": item.get("offerCount"),
    }


def fallback_search_keywords(code: str, cards: list[dict[str, Any]]) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for card in sorted(cards, key=lambda item: str(item.get("card_id") or "")):
        name = search_name(card.get("name") or card.get("name_en"))
        if not name or name in seen:
            continue
        seen.add(name)
        keywords.append(f"{name} [{code}]")
    return keywords


def fetch_keyword_products(
    session: Any,
    code: str,
    keyword: str,
    query_kind: str,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    products_by_id: dict[str, dict[str, Any]] = {}
    pages_seen = 0
    cache_hits = 0
    errors: list[dict[str, Any]] = []
    total_count = None
    for page in range(1, args.max_pages_per_code + 1):
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
            delay_jitter=args.delay_jitter,
        )
        pages_seen += 1
        cache_hits += int(bool(payload.get("from_cache")))
        if payload.get("error"):
            errors.append({"card_code": code, "page": page, "error": payload.get("error")})
            break
        items, page_total_count = page_items(payload.get("data") or {})
        if page_total_count is not None:
            total_count = page_total_count
        for item in items:
            product = product_from_search_item(item, code)
            if not product:
                continue
            products_by_id.setdefault(product["snkr_product_id"], product)
        if not items:
            break
        if total_count is not None:
            needed_pages = max(1, (total_count + args.per_page - 1) // args.per_page)
            if page >= needed_pages:
                break
        if len(items) < args.per_page:
            break
    meta = {
        "card_code": code,
        "keyword": keyword,
        "query_kind": query_kind,
        "pages_seen": pages_seen,
        "cache_hits": cache_hits,
        "total_count": total_count,
        "unique_products": len(products_by_id),
        "errors": errors,
    }
    return products_by_id, meta


def fetch_code_products(
    session: Any,
    code: str,
    cards: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    products_by_id: dict[str, dict[str, Any]] = {}
    query_metas: list[dict[str, Any]] = []
    keywords = [(code, "card_code")]
    keywords.extend((keyword, "name_code_fallback") for keyword in fallback_search_keywords(code, cards))

    for index, (keyword, query_kind) in enumerate(keywords):
        if index > 0 and products_by_id:
            break
        query_products, query_meta = fetch_keyword_products(session, code, keyword, query_kind, args)
        query_metas.append(query_meta)
        for product_id, product in query_products.items():
            products_by_id.setdefault(product_id, product)

    total_counts = [meta.get("total_count") for meta in query_metas if meta.get("total_count") is not None]
    errors = [error for meta in query_metas for error in meta.get("errors", [])]
    meta = {
        "card_code": code,
        "keyword": code,
        "queries": query_metas,
        "fallback_used": any(meta.get("query_kind") != "card_code" for meta in query_metas),
        "pages_seen": sum(int(meta.get("pages_seen") or 0) for meta in query_metas),
        "cache_hits": sum(int(meta.get("cache_hits") or 0) for meta in query_metas),
        "total_count": sum(int(count) for count in total_counts) if total_counts else None,
        "unique_products": len(products_by_id),
        "errors": errors,
    }
    return list(products_by_id.values()), meta


def rejected_candidate_record(product: dict[str, Any], parsed: dict[str, Any], reject_reasons: list[str]) -> dict[str, Any]:
    return {
        "snkr_product_id": product.get("snkr_product_id"),
        "snkr_product_name": product.get("snkr_product_name"),
        "snkr_url": f"https://snkrdunk.com/apparels/{product['snkr_product_id']}",
        "thumbnail_url": product.get("thumbnail_url"),
        "snkr_base_rarity": parsed.get("base_rarity"),
        "snkr_modifier": parsed.get("modifier"),
        "snkr_descriptor": parsed.get("descriptor"),
        "snkr_language": parsed.get("language") or thumbnail_language(product.get("thumbnail_url")),
        "thumbnail_suffix": thumbnail_suffix(product.get("thumbnail_url")),
        "reject_reasons": reject_reasons,
    }


def candidate_results_for_card(
    card: dict[str, Any],
    products: list[dict[str, Any]],
    base_rarity_by_code: dict[str, str | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code = str(card.get("card_code") or "")
    expected_rarity = expected_base_rarity(card, base_rarity_by_code)
    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for product in products:
        title = str(product.get("snkr_product_name") or "")
        parsed = parse_snkr_title(title, code)
        reasons: list[str] = []
        if not parsed["code_exact"]:
            rejected.append(rejected_candidate_record(product, parsed, [f"code_mismatch:{code}"]))
            continue
        reasons.append(f"code_exact:{code}")

        language_ok, language_reason = snkr_language_ok(card, product, parsed)
        if not language_ok:
            rejected.append(rejected_candidate_record(product, parsed, [language_reason]))
            continue
        reasons.append(language_reason)

        if not names_match(card, parsed):
            rejected.append(rejected_candidate_record(product, parsed, ["name_mismatch"]))
            continue
        reasons.append("name_exact_normalized")

        if not expected_rarity or parsed.get("base_rarity") != expected_rarity:
            got = parsed.get("base_rarity") or "unknown"
            rejected.append(rejected_candidate_record(product, parsed, [f"rarity_mismatch:{expected_rarity or 'unknown'}!={got}"]))
            continue
        reasons.append(f"rarity_base:{expected_rarity}")

        variant_ok, variant_reasons = variant_match(card, product, parsed)
        if not variant_ok:
            rejected.append(rejected_candidate_record(product, parsed, variant_reasons))
            continue
        reasons.extend(variant_reasons)

        candidate = dict(product)
        candidate.update(
            {
                "snkr_url": f"https://snkrdunk.com/apparels/{product['snkr_product_id']}",
                "snkr_base_rarity": parsed.get("base_rarity"),
                "snkr_modifier": parsed.get("modifier"),
                "snkr_descriptor": parsed.get("descriptor"),
                "snkr_language": parsed.get("language") or thumbnail_language(product.get("thumbnail_url")),
                "thumbnail_suffix": thumbnail_suffix(product.get("thumbnail_url")),
                "match_reasons": reasons,
            }
        )
        verified.append(candidate)
    verified.sort(key=lambda item: (str(item.get("snkr_product_name") or "").lower(), item["snkr_product_id"]))
    rejected.sort(key=lambda item: (str(item.get("snkr_product_name") or "").lower(), str(item.get("snkr_product_id") or "")))
    return verified, rejected


def reason_group(reason: str | None) -> str | None:
    if not reason:
        return None
    return str(reason).split(":", 1)[0]


def top_reason(counter: Counter[str]) -> str | None:
    return counter.most_common(1)[0][0] if counter else None


def mapping_record(
    card: dict[str, Any],
    verified: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    fetch_meta: dict[str, Any],
) -> dict[str, Any]:
    if len(verified) == 1:
        status = "matched"
        selected = verified[0]
    elif len(verified) > 1:
        status = "multiple_verified_matches"
        selected = None
    else:
        status = "no_verified_match"
        selected = None
    reject_reason_counts = Counter(
        reason
        for candidate in rejected
        for reason in candidate.get("reject_reasons", [])
    )
    reject_reason_group_counts = Counter(
        group
        for group in (reason_group(reason) for reason in reject_reason_counts.elements())
        if group
    )
    unmatched_reason = None
    unmatched_reason_detail = None
    if status == "multiple_verified_matches":
        unmatched_reason = "multiple_verified_candidates"
        unmatched_reason_detail = f"verified_candidate_count:{len(verified)}"
    elif status == "no_verified_match":
        if fetch_meta.get("errors"):
            unmatched_reason = "search_error"
            unmatched_reason_detail = "search_errors_present"
        elif not fetch_meta.get("unique_products"):
            unmatched_reason = "no_exact_code_products"
            unmatched_reason_detail = "search_returned_no_exact_code_products"
        else:
            unmatched_reason_detail = top_reason(reject_reason_counts)
            unmatched_reason = reason_group(unmatched_reason_detail) or top_reason(reject_reason_group_counts) or "no_verified_candidate"
    return {
        "game": "onepiece",
        "language": card.get("language"),
        "card_id": card.get("card_id"),
        "set_id": card.get("set_id"),
        "card_code": card.get("card_code"),
        "variant": card.get("variant"),
        "name": card.get("name"),
        "rarity": card.get("rarity"),
        "image_suffix": image_suffix(card.get("image_url")),
        "variant_kind": local_variant_kind(card),
        "match_status": status,
        "match_problem": None if status == "matched" else status,
        "unmatched_reason": unmatched_reason,
        "unmatched_reason_detail": unmatched_reason_detail,
        "snkr_product_id": selected.get("snkr_product_id") if selected else None,
        "snkr_product_name": selected.get("snkr_product_name") if selected else None,
        "snkr_url": selected.get("snkr_url") if selected else None,
        "snkr_thumbnail_url": selected.get("thumbnail_url") if selected else None,
        "snkr_min_price": selected.get("min_price") if selected else None,
        "snkr_min_price_format": selected.get("min_price_format") if selected else None,
        "verified_candidate_count": len(verified),
        "verified_candidates": verified,
        "rejected_candidate_count": len(rejected),
        "reject_reason_counts": dict(sorted(reject_reason_counts.items())),
        "reject_reason_group_counts": dict(sorted(reject_reason_group_counts.items())),
        "rejected_candidate_examples": rejected[:5],
        "search_total_count": fetch_meta.get("total_count"),
        "search_unique_products": fetch_meta.get("unique_products"),
        "search_pages_seen": fetch_meta.get("pages_seen"),
        "search_cache_hits": fetch_meta.get("cache_hits"),
        "search_query_count": len(fetch_meta.get("queries") or []),
        "search_queries": fetch_meta.get("queries") or [],
        "search_fallback_used": bool(fetch_meta.get("fallback_used")),
        "search_errors": fetch_meta.get("errors") or [],
        "matched_at": utc_now_iso(),
    }


def load_cards(path: Path) -> list[dict[str, Any]]:
    cards = []
    seen: set[tuple[str, str]] = set()
    for record in iter_jsonl(path):
        code = record.get("card_code")
        card_id = record.get("card_id")
        if not code or not card_id:
            continue
        key = (str(card_id), str(code))
        if key in seen:
            continue
        seen.add(key)
        cards.append(record)
    return cards


def summarize(records: list[dict[str, Any]], fetch_metas: list[dict[str, Any]], args: argparse.Namespace, started_at: str) -> dict[str, Any]:
    by_status = Counter(str(record.get("match_status") or "unknown") for record in records)
    by_variant_kind = Counter(str(record.get("variant_kind") or "unknown") for record in records)
    by_unmatched_reason = Counter(
        str(record.get("unmatched_reason") or "matched")
        for record in records
    )
    no_match_reject_reason_groups = Counter()
    no_match_reject_reasons = Counter()
    for record in records:
        if record.get("match_status") != "no_verified_match":
            continue
        no_match_reject_reason_groups.update(record.get("reject_reason_group_counts") or {})
        no_match_reject_reasons.update(record.get("reject_reason_counts") or {})
    failed_searches = [meta for meta in fetch_metas if meta.get("errors")]
    return {
        "source": "snkrdunk",
        "input_manifest": str(args.input),
        "records_written": len(records),
        "card_codes_seen": len({record.get("card_code") for record in records}),
        "searches_seen": len(fetch_metas),
        "search_queries_seen": sum(len(meta.get("queries") or []) for meta in fetch_metas),
        "fallback_searches_used": sum(1 for meta in fetch_metas if meta.get("fallback_used")),
        "by_status": dict(sorted(by_status.items())),
        "by_variant_kind": dict(sorted(by_variant_kind.items())),
        "by_unmatched_reason": dict(sorted(by_unmatched_reason.items())),
        "no_match_reject_reason_groups": dict(no_match_reject_reason_groups.most_common()),
        "no_match_reject_reasons_top": dict(no_match_reject_reasons.most_common(25)),
        "failed_searches": failed_searches,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "mapping": str(args.output),
            "summary": str(args.summary_output),
            "cache_dir": str(args.cache_dir),
        },
        "notes": [
            "A One Piece row is matched only when card code, language, normalized name, base rarity, and variant hint all match.",
            "If a card-code-only SNKRDUNK search returns no usable product items, the mapper retries with '<card name> [<card code>]'.",
            "Rows with zero or multiple strict SNKRDUNK candidates are not collapsed to a product id.",
            "Unmatched rows keep match_problem, unmatched_reason, reject_reason_counts, and candidate examples for later cleanup.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly map One Piece card variants to SNKRDUNK product IDs.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/manifests/onepiece_kaggle_catalog.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/snkr_onepiece_product_map.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/manifests/snkr_onepiece_product_map_summary.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/interim/snkr_onepiece_search_cache")
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--max-pages-per-code", type=int, default=2)
    parser.add_argument("--delay", type=float, default=4.0)
    parser.add_argument("--delay-jitter", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--rate-limit-sleep", type=float, default=120.0)
    parser.add_argument("--limit-codes", type=int, default=None)
    parser.add_argument("--codes", nargs="*", default=None)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now_iso()
    cards = load_cards(args.input)
    if not cards:
        raise SystemExit(f"No One Piece cards found in {args.input}")
    base_rarity_by_code = infer_base_rarity_by_code(cards)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        grouped.setdefault(str(card["card_code"]), []).append(card)
    codes = sorted(grouped)
    if args.codes:
        requested = {str(code) for code in args.codes}
        codes = [code for code in codes if code in requested]
    if args.limit_codes:
        codes = codes[: args.limit_codes]
    if not codes:
        raise SystemExit("No card codes selected.")

    session = build_session()
    records: list[dict[str, Any]] = []
    fetch_metas: list[dict[str, Any]] = []
    for index, code in enumerate(codes, start=1):
        print(f"[{index}/{len(codes)}] Fetching SNKRDUNK products for {code}", file=sys.stderr)
        products, meta = fetch_code_products(session, code, grouped[code], args)
        fetch_metas.append(meta)
        for card in sorted(grouped[code], key=lambda item: str(item.get("card_id") or "")):
            verified, rejected = candidate_results_for_card(card, products, base_rarity_by_code)
            records.append(mapping_record(card, verified, rejected, meta))
        if index % 25 == 0:
            write_jsonl(args.output, records)
            write_json(args.summary_output, summarize(records, fetch_metas, args, started_at))

    summary = summarize(records, fetch_metas, args, started_at)
    write_jsonl(args.output, records)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
