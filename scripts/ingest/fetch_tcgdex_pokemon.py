#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import write_json, write_jsonl
from scripts.lib.schema import (
    SOURCE_LICENSES,
    derive_set_id_from_card_id,
    empty_catalog_record,
    language_name_fields,
    truthy_variant_string,
    utc_now_iso,
)


DEFAULT_BASE_URL = "https://api.tcgdex.net/v2"


def fetch_json(url: str, timeout: int) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "card-scan-pipeline/0.1 (+https://tcgdex.dev/)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def card_to_record(card: dict[str, Any], language: str, detail_level: str) -> dict[str, Any]:
    card_id = card.get("id")
    name = card.get("name")
    set_info = card.get("set") if isinstance(card.get("set"), dict) else {}
    names = language_name_fields(language, name)

    return empty_catalog_record(
        game="pokemon",
        source="tcgdex",
        source_license=SOURCE_LICENSES["tcgdex"],
        card_id=card_id,
        card_code=str(card.get("localId")) if card.get("localId") is not None else None,
        set_id=set_info.get("id") or derive_set_id_from_card_id(card_id),
        language=language,
        name=name,
        name_en=names["name_en"],
        name_ja=names["name_ja"],
        rarity=card.get("rarity"),
        variant=truthy_variant_string(card.get("variants")),
        image_url=card.get("image"),
        updated_at=card.get("updated") or utc_now_iso(),
    ) | {"tcgdex_detail_level": detail_level}


def set_to_record(set_record: dict[str, Any], language: str) -> dict[str, Any]:
    count = set_record.get("cardCount") if isinstance(set_record.get("cardCount"), dict) else {}
    return {
        "game": "pokemon",
        "source": "tcgdex",
        "source_license": SOURCE_LICENSES["tcgdex"],
        "language": language,
        "set_id": set_record.get("id"),
        "name": set_record.get("name"),
        "logo_url": set_record.get("logo"),
        "symbol_url": set_record.get("symbol"),
        "card_count_official": count.get("official"),
        "card_count_total": count.get("total"),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }


def fetch_language(
    *,
    base_url: str,
    language: str,
    detail_level: str,
    limit: int | None,
    raw_dir: Path,
    timeout: int,
    sleep_seconds: float,
    continue_on_error: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    lang_base = f"{base_url.rstrip('/')}/{language}"
    cards = fetch_json(f"{lang_base}/cards", timeout)
    sets = fetch_json(f"{lang_base}/sets", timeout)

    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw_dir / f"{language}_cards_brief.json", cards)
    write_json(raw_dir / f"{language}_sets.json", sets)

    selected_cards = cards[:limit] if limit else cards
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for index, brief in enumerate(selected_cards, start=1):
        card_payload = brief
        if detail_level == "full":
            card_id = brief.get("id")
            try:
                card_payload = fetch_json(f"{lang_base}/cards/{card_id}", timeout)
                details.append(card_payload)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                error = {
                    "language": language,
                    "card_id": card_id,
                    "error": str(exc),
                    "created_at": utc_now_iso(),
                }
                errors.append(error)
                if not continue_on_error:
                    raise RuntimeError(f"Failed to fetch TCGdex card detail for {language}/{card_id}: {exc}") from exc
                continue
            if sleep_seconds:
                time.sleep(sleep_seconds)
            if index % 500 == 0:
                print(f"Fetched {index} {language} card details", file=sys.stderr)

        records.append(card_to_record(card_payload, language, detail_level))

    if detail_level == "full":
        write_jsonl(raw_dir / f"{language}_card_details.jsonl", details)

    set_records = [set_to_record(item, language) for item in sets]
    return records, set_records, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Pokemon card and set manifests from TCGdex.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--languages", nargs="+", default=["en", "ja"])
    parser.add_argument("--detail-level", choices=["brief", "full"], default="brief")
    parser.add_argument("--limit", type=int, default=None, help="Limit cards per language for smoke tests.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between detail requests.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/tcgdex")
    parser.add_argument("--cards-output", type=Path, default=ROOT / "data/manifests/tcgdex_pokemon_cards.jsonl")
    parser.add_argument("--sets-output", type=Path, default=ROOT / "data/manifests/tcgdex_pokemon_sets.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/manifests/tcgdex_pokemon_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_cards: list[dict[str, Any]] = []
    all_sets: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    started_at = utc_now_iso()

    for language in args.languages:
        cards, sets, errors = fetch_language(
            base_url=args.base_url,
            language=language,
            detail_level=args.detail_level,
            limit=args.limit,
            raw_dir=args.raw_dir,
            timeout=args.timeout,
            sleep_seconds=args.sleep,
            continue_on_error=args.continue_on_error,
        )
        all_cards.extend(cards)
        all_sets.extend(sets)
        all_errors.extend(errors)

    card_count = write_jsonl(args.cards_output, all_cards)
    set_count = write_jsonl(args.sets_output, all_sets)
    if all_errors:
        write_jsonl(ROOT / "data/manifests/tcgdex_pokemon_errors.jsonl", all_errors)

    summary = {
        "source": "tcgdex",
        "base_url": args.base_url,
        "languages": args.languages,
        "detail_level": args.detail_level,
        "limit_per_language": args.limit,
        "cards_written": card_count,
        "sets_written": set_count,
        "errors": len(all_errors),
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "cards": str(args.cards_output),
            "sets": str(args.sets_output),
            "raw_dir": str(args.raw_dir),
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
