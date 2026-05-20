#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import write_json, write_jsonl
from scripts.lib.schema import SOURCE_LICENSES, empty_catalog_record, utc_now_iso


DEFAULT_DATASET = "t22000t/optcg-en-cards"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def fetch_hf_rows(dataset: str, config: str, split: str, page_size: int, limit: int | None, sleep_seconds: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        length = page_size
        if limit is not None:
            remaining = limit - len(rows)
            if remaining <= 0:
                break
            length = min(length, remaining)

        params = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        request = urllib.request.Request(
            f"{ROWS_ENDPOINT}?{params}",
            headers={"User-Agent": "card-scan-pipeline/0.1 (+https://huggingface.co/)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)

        total = payload["num_rows_total"]
        page_rows = [item["row"] for item in payload["rows"]]
        rows.extend(page_rows)
        offset += len(page_rows)
        if not page_rows:
            break
        if sleep_seconds:
            time.sleep(sleep_seconds)

    return rows


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("Reading parquet requires pandas and pyarrow. Install them or omit --input-parquet.") from exc

    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise SystemExit(f"Could not read parquet {path}: {exc}") from exc
    return frame.to_dict(orient="records")


def parse_variant(card_id: str | None) -> str | None:
    if not card_id or "_" not in card_id:
        return None
    return card_id.split("_", 1)[1]


def row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    card_id = row.get("id")
    name = row.get("name")
    return empty_catalog_record(
        game="onepiece",
        source="optcg_hf_cards",
        source_license=SOURCE_LICENSES["optcg_hf_cards"],
        card_id=card_id,
        card_code=row.get("code"),
        set_id=row.get("set_code") or row.get("pack_id"),
        language=row.get("language") or "en",
        name=name,
        name_en=name if (row.get("language") or "en") == "en" else None,
        rarity=row.get("rarity"),
        variant=parse_variant(card_id),
    ) | {
        "card_type": row.get("card_type"),
        "colors": row.get("colors"),
        "effect_text": row.get("effect_text"),
        "trigger_text": row.get("trigger_text"),
        "keywords": row.get("keywords"),
        "set_name": row.get("set_name"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import t22000t/optcg-en-cards from Hugging Face.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for smoke tests.")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--input-parquet", type=Path, default=None, help="Read a local parquet instead of the HF rows API.")
    parser.add_argument("--raw-output", type=Path, default=ROOT / "data/raw/huggingface/optcg_en_cards_rows.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/optcg_hf_catalog.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/manifests/optcg_hf_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now_iso()
    if args.input_parquet:
        rows = read_parquet_rows(args.input_parquet)
        mode = "local_parquet"
        if args.limit:
            rows = rows[: args.limit]
    else:
        rows = fetch_hf_rows(args.dataset, args.config, args.split, args.page_size, args.limit, args.sleep)
        mode = "hf_rows_api"

    write_jsonl(args.raw_output, rows)
    records = [row_to_record(row) for row in rows]
    count = write_jsonl(args.output, records)

    summary = {
        "source": "optcg_hf_cards",
        "source_license": SOURCE_LICENSES["optcg_hf_cards"],
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "mode": mode,
        "rows_read": len(rows),
        "records_written": count,
        "limit": args.limit,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "raw_rows": str(args.raw_output),
            "catalog": str(args.output),
        },
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
