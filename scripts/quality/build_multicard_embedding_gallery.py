#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.schema import utc_now_iso
from scripts.server.recognition_api import (
    DEFAULT_ASPECT_TOLERANCE,
    DEFAULT_CONFIDENCE,
    DEFAULT_IMGSZ,
    DEFAULT_PADDING,
    DEFAULT_TARGET_ASPECT,
    cv2,
    detect_card_crops_from_image,
    existing_local_image_path,
    np,
    service,
)


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Detect multiple cards, search each crop, and build a debug gallery.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/reports" / f"multicard_tcgp_obb_embedding_probe_{timestamp}")
    parser.add_argument("--detector", default="tcgp_obb")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--max-cards", type=int, default=30)
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING)
    parser.add_argument("--target-aspect", type=float, default=DEFAULT_TARGET_ASPECT)
    parser.add_argument("--aspect-tolerance", type=float, default=DEFAULT_ASPECT_TOLERANCE)
    parser.add_argument("--sort", default="reading_order")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--per-index-top-k", type=int, default=5)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def h(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def result_name(result: dict[str, Any] | None) -> str:
    if not result:
        return "missing"
    return str(result.get("name_ja") or result.get("name_en") or result.get("name") or result.get("card_id") or "unknown")


def result_code(result: dict[str, Any] | None) -> str:
    if not result:
        return "-"
    return "-".join(str(value) for value in (result.get("set_id"), result.get("card_code")) if value)


def result_caption(result: dict[str, Any] | None) -> str:
    if not result:
        return "missing"
    bits = [
        f"#{result.get('rank')}",
        str(result.get("index") or ""),
        result_code(result),
        str(result.get("language") or ""),
        f"{float(result.get('score') or 0):.3f}",
        result_name(result),
    ]
    return " | ".join(bit for bit in bits if bit and bit != "-")


def save_bgr_jpeg(path: Path, image: Any, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def copy_reference_image(result: dict[str, Any], output_path: Path, max_side: int = 700) -> str | None:
    local_path = existing_local_image_path(result.get("local_image_path"))
    if local_path is None:
        return None
    image = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image is None:
        shutil.copy2(local_path, output_path)
        return output_path.name
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(1, max(height, width)))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    save_bgr_jpeg(output_path, image, quality=90)
    return output_path.name


def draw_annotated_image(image: Any, cards: list[dict[str, Any]]) -> Any:
    annotated = image.copy()
    for card in cards:
        points = np.asarray(card["polygon"], dtype="int32").reshape((-1, 1, 2))
        cv2.polylines(annotated, [points], isClosed=True, color=(0, 220, 80), thickness=3)
        center = card.get("center") or {}
        x = int(round(center.get("x") or points[:, 0, 0].mean()))
        y = int(round(center.get("y") or points[:, 0, 1].mean()))
        label = str(card.get("index"))
        cv2.circle(annotated, (x, y), 18, (0, 0, 0), -1)
        cv2.putText(annotated, label, (x - 10, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def image_tile(label: str, src: str | None, caption: str, css_class: str = "") -> str:
    if src:
        body = f'<img src="{h(src)}" alt="{h(label)}">'
    else:
        body = '<div class="missing">missing</div>'
    return f"""
      <figure class="tile {h(css_class)}">
        <div class="tile-label">{h(label)}</div>
        {body}
        <figcaption>{h(caption)}</figcaption>
      </figure>
    """


def build_html(report: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path) -> str:
    summary = report["summary"]
    metrics = [
        ("Cards", summary["cards_returned"]),
        ("Detections", summary["detections_total"]),
        ("Detector", summary["detector"]),
        ("Detect", f"{summary['timings'].get('detect_seconds', 0):.3f}s"),
        ("Embed", f"{summary['timings'].get('embedding_seconds', 0):.3f}s"),
        ("Search", f"{summary['timings'].get('search_seconds', 0):.3f}s"),
        ("Total", f"{summary['timings'].get('total_seconds', 0):.3f}s"),
    ]
    metric_html = "\n".join(
        f'<div class="metric"><span>{h(label)}</span><strong>{h(value)}</strong></div>' for label, value in metrics
    )
    top1_counts = Counter(row.get("top1_card_id") or "missing" for row in rows)
    top1_html = ", ".join(f"{h(card_id)} x{count}" for card_id, count in top1_counts.most_common(8))

    case_html: list[str] = []
    for row in rows:
        top_results = row.get("results") or []
        top1 = top_results[0] if top_results else None
        first_row_tiles = [
            image_tile("listing", row["annotated_image"], f"{row['input_name']} / boxes numbered"),
            image_tile("query crop", row["crop_image"], f"card #{row['card_index']} conf {row['confidence']:.3f}"),
            image_tile("embedding Top1", top1.get("report_image") if top1 else None, result_caption(top1), "top1"),
            image_tile("embedding Top2", top_results[1].get("report_image") if len(top_results) > 1 else None, result_caption(top_results[1] if len(top_results) > 1 else None)),
            image_tile("embedding Top3", top_results[2].get("report_image") if len(top_results) > 2 else None, result_caption(top_results[2] if len(top_results) > 2 else None)),
        ]
        strip = "\n".join(
            image_tile(f"Top{result.get('rank')}", result.get("report_image"), result_caption(result))
            for result in top_results
        )
        result_lines = "\n".join(result_caption(result) for result in top_results)
        case_html.append(
            f"""
            <section class="case neutral" id="card-{row['card_index']:02d}">
              <div class="case-head">
                <div>
                  <h2>#{row['card_index']:02d} {h(result_name(top1))}</h2>
                  <p>{h(result_caption(top1))}</p>
                </div>
                <a href="#top">top</a>
              </div>
              <div class="image-row">
                {''.join(first_row_tiles)}
              </div>
              <h3>Full Top5</h3>
              <div class="top5-strip">{strip}</div>
              <details>
                <summary>diagnostics</summary>
                <pre>{h(result_lines)}</pre>
              </details>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{h(report['title'])}</title>
  <style>
    :root {{ color-scheme: light; --border:#d8dedb; --muted:#5e6b66; --bg:#f7faf8; --ink:#17211d; --accent:#16784f; }}
    body {{ margin:0; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:24px 28px 18px; border-bottom:1px solid var(--border); background:#fff; position:sticky; top:0; z-index:5; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    h2 {{ margin:0 0 4px; font-size:18px; }}
    h3 {{ margin:16px 0 8px; font-size:13px; text-transform:uppercase; color:var(--muted); }}
    a {{ color:var(--accent); text-decoration:none; }}
    .sub {{ color:var(--muted); max-width:980px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:8px; margin:16px 0 12px; }}
    .metric {{ border:1px solid var(--border); border-radius:8px; padding:9px 10px; background:#fbfdfc; }}
    .metric span {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .metric strong {{ display:block; margin-top:2px; font-size:17px; }}
    main {{ padding:18px 28px 40px; }}
    .case {{ background:#fff; border:1px solid var(--border); border-left:6px solid #7f9d91; border-radius:10px; margin:0 0 18px; padding:14px; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
    .case-head {{ display:flex; justify-content:space-between; gap:16px; margin-bottom:10px; }}
    .case-head p {{ margin:0; color:var(--muted); }}
    .image-row {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; align-items:start; }}
    .top5-strip {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; }}
    .tile {{ margin:0; border:1px solid var(--border); border-radius:8px; overflow:hidden; background:#fdfefd; min-width:0; }}
    .tile-label {{ padding:6px 8px; font-size:11px; font-weight:700; color:#fff; background:#44524d; text-transform:uppercase; }}
    .tile.top1 .tile-label {{ background:var(--accent); }}
    .tile img {{ display:block; width:100%; height:220px; object-fit:contain; background:#eef2ef; }}
    .tile figcaption {{ padding:7px 8px; min-height:52px; color:#33413b; font-size:12px; word-break:break-word; }}
    .missing {{ display:grid; place-items:center; height:220px; color:#8b9691; background:#eef2ef; }}
    details {{ margin-top:8px; }}
    pre {{ white-space:pre-wrap; overflow:auto; background:#f3f6f4; border:1px solid var(--border); border-radius:8px; padding:10px; }}
    @media (max-width:1100px) {{ .image-row,.top5-strip {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} header {{ position:static; }} }}
  </style>
</head>
<body>
  <header id="top">
    <h1>{h(report['title'])}</h1>
    <div class="sub">Direct crop-to-embedding check. This run uses TCGP OBB server ONNX detection, then sends each detected crop straight to the existing embedding search. No OCR, no language rerank, no expected answer comparison.</div>
    <div class="metrics">{metric_html}</div>
    <div class="sub">Files: <a href="{h(report['csv'])}">CSV</a> · <a href="{h(report['jsonl'])}">JSONL</a> · <a href="{h(report['summary_json'])}">summary JSON</a></div>
    <div class="sub">Common Top1 card ids: {top1_html}</div>
  </header>
  <main>
    {''.join(case_html)}
  </main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"OpenCV could not read image: {args.input}")

    output_dir = args.output_dir.resolve()
    crops_dir = output_dir / "crops"
    refs_dir = output_dir / "refs"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_copy = output_dir / "input.jpg"
    save_bgr_jpeg(input_copy, image)

    detected = detect_card_crops_from_image(
        image=image,
        detector=args.detector,
        confidence=args.confidence,
        imgsz=args.imgsz,
        max_cards=args.max_cards,
        padding=args.padding,
        target_aspect=args.target_aspect,
        aspect_tolerance=args.aspect_tolerance,
        sort=args.sort,
    )
    annotated_path = output_dir / "input_annotated.jpg"
    save_bgr_jpeg(annotated_path, draw_annotated_image(image, detected["cards"]))

    indexes = service.load_indexes()
    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    embedding_seconds = 0.0
    search_seconds = 0.0
    search_top_k = max(args.top_k, args.per_index_top_k)

    for card in detected["cards"]:
        card_index = int(card["index"])
        crop_image = card["crop_image"]
        crop_path = crops_dir / f"card_{card_index:02d}.jpg"
        save_bgr_jpeg(crop_path, crop_image)

        encode_start = time.perf_counter()
        query = service.encode(crop_image)
        card_embedding_seconds = time.perf_counter() - encode_start
        search_start = time.perf_counter()
        combined, _per_index = service.search(
            query,
            per_index_top_k=max(args.per_index_top_k, search_top_k),
            combined_top_k=search_top_k,
        )
        card_search_seconds = time.perf_counter() - search_start
        embedding_seconds += card_embedding_seconds
        search_seconds += card_search_seconds
        for rank, result in enumerate(combined, start=1):
            result["rank"] = rank

        top_results: list[dict[str, Any]] = []
        for result in combined[: args.top_k]:
            item = dict(result)
            ref_name = copy_reference_image(item, refs_dir / f"card_{card_index:02d}_rank_{item['rank']:02d}.jpg")
            item["report_image"] = f"refs/{ref_name}" if ref_name else None
            top_results.append(item)

        top1 = top_results[0] if top_results else {}
        row = {
            "card_index": card_index,
            "input_name": args.input.name,
            "annotated_image": rel(annotated_path, output_dir),
            "crop_image": rel(crop_path, output_dir),
            "confidence": float(card.get("confidence") or 0.0),
            "confidence_rank": card.get("confidence_rank"),
            "box": card.get("box"),
            "center": card.get("center"),
            "area_ratio": card.get("area_ratio"),
            "crop": card.get("crop"),
            "results": top_results,
            "top1_card_id": top1.get("card_id"),
            "top1_code": result_code(top1),
            "top1_language": top1.get("language"),
            "top1_name": result_name(top1),
            "top1_score": top1.get("score"),
        }
        rows.append(row)
        csv_rows.append(
            {
                "card_index": card_index,
                "confidence": row["confidence"],
                "top1_index": top1.get("index"),
                "top1_score": top1.get("score"),
                "top1_card_id": top1.get("card_id"),
                "top1_code": row["top1_code"],
                "top1_language": top1.get("language"),
                "top1_name": row["top1_name"],
                "top5": " || ".join(result_caption(result) for result in top_results),
                "crop_image": row["crop_image"],
            }
        )

    timings = dict(detected["timings"])
    timings["embedding_seconds"] = embedding_seconds
    timings["search_seconds"] = search_seconds
    timings["total_seconds"] = time.perf_counter() - started
    summary = {
        "created_at": utc_now_iso(),
        "input_path": str(args.input.resolve()),
        "output_dir": str(output_dir),
        "detector": detected["detector"],
        "detections_total": detected["detections_total"],
        "cards_returned": len(rows),
        "params": {
            "confidence": args.confidence,
            "imgsz": detected["imgsz"],
            "max_cards": args.max_cards,
            "padding": args.padding,
            "target_aspect": args.target_aspect,
            "aspect_tolerance": args.aspect_tolerance,
            "sort": args.sort,
            "top_k": args.top_k,
            "per_index_top_k": args.per_index_top_k,
        },
        "indexes": [{"name": loaded.name, "records": len(loaded.records), "path": str(loaded.path)} for loaded in indexes],
        "timings": timings,
    }

    csv_path = output_dir / "multicard_embedding_results.csv"
    jsonl_path = output_dir / "multicard_embedding_results.jsonl"
    summary_path = output_dir / "summary.json"
    gallery_path = output_dir / "gallery_multicard_embedding.html"
    write_csv(
        csv_path,
        csv_rows,
        [
            "card_index",
            "confidence",
            "top1_index",
            "top1_score",
            "top1_card_id",
            "top1_code",
            "top1_language",
            "top1_name",
            "top5",
            "crop_image",
        ],
    )
    write_jsonl(jsonl_path, rows)
    write_json(summary_path, summary)
    report = {
        "title": "Multi-card TCGP OBB direct embedding probe",
        "summary": summary,
        "csv": rel(csv_path, output_dir),
        "jsonl": rel(jsonl_path, output_dir),
        "summary_json": rel(summary_path, output_dir),
    }
    gallery_path.write_text(build_html(report, rows, output_dir), encoding="utf-8")
    print(json.dumps({"gallery": str(gallery_path), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
