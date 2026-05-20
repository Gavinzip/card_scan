#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import write_json, write_jsonl
from scripts.lib.schema import utc_now_iso

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_REPO_ID = "AlecKarfonta/cardcaptor-v3"
DEFAULT_MODEL_FILE = "weights/cardcaptor_v3_best.pt"


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in path.stem)


def resolve_inputs(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise SystemExit(f"Input path does not exist: {input_path}")
    globber = input_path.rglob if recursive else input_path.glob
    return sorted(path for path in globber("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS)


def ensure_model(model_path: Path, repo_id: str, filename: str) -> Path:
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise SystemExit("Auto crop requires huggingface_hub. Install the cropping requirements.") from exc

    model_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    shutil.copy2(cached_path, model_path)
    return model_path


def require_deps() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Auto crop requires ultralytics and OpenCV. Install with: "
            "uv pip install --python /path/to/python ultralytics opencv-python-headless"
        ) from exc
    return cv2, np, YOLO


def order_points(points: Any, np: Any) -> Any:
    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(4)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def expand_points(points: Any, padding: float, image_width: int, image_height: int, np: Any) -> Any:
    if padding <= 0:
        return points
    center = points.mean(axis=0)
    vectors = points - center
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1.0)
    expanded = center + vectors * ((lengths + padding) / lengths)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, image_width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, image_height - 1)
    return expanded.astype("float32")


def warp_card(image: Any, polygon: Any, padding: float, cv2: Any, np: Any) -> Any:
    image_height, image_width = image.shape[:2]
    rect = order_points(polygon, np)
    rect = expand_points(rect, padding, image_width, image_height, np)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = max(1, int(round(max(width_a, width_b))))
    max_height = max(1, int(round(max(height_a, height_b))))
    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def trim_to_aspect(image: Any, target_aspect: float, tolerance: float) -> tuple[Any, bool]:
    if target_aspect <= 0:
        return image, False
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image, False
    aspect = width / height
    if abs(aspect - target_aspect) / target_aspect <= tolerance:
        return image, False
    if aspect > target_aspect:
        new_width = max(1, int(round(height * target_aspect)))
        left = max(0, (width - new_width) // 2)
        return image[:, left : left + new_width], True
    new_height = max(1, int(round(width / target_aspect)))
    top = max(0, (height - new_height) // 2)
    return image[top : top + new_height, :], True


def box_to_polygon(box: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def detections_from_result(result: Any, np: Any) -> list[dict[str, Any]]:
    names = getattr(result, "names", {}) or {}
    detections: list[dict[str, Any]] = []
    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        polygons = obb.xyxyxyxy.detach().cpu().numpy().reshape(-1, 4, 2)
        confidences = obb.conf.detach().cpu().numpy().tolist()
        classes = obb.cls.detach().cpu().numpy().astype(int).tolist()
        for polygon, confidence, class_id in zip(polygons, confidences, classes, strict=False):
            detections.append(
                {
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "class_name": names.get(int(class_id), str(class_id)),
                    "polygon": polygon.astype(float).tolist(),
                }
            )
        return detections

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
        confidences = boxes.conf.detach().cpu().numpy().tolist()
        classes = boxes.cls.detach().cpu().numpy().astype(int).tolist()
        for box, confidence, class_id in zip(xyxy, confidences, classes, strict=False):
            detections.append(
                {
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "class_name": names.get(int(class_id), str(class_id)),
                    "polygon": box_to_polygon([float(value) for value in box]),
                }
            )
    return detections


def save_debug_image(image: Any, polygon: list[list[float]], output_path: Path, cv2: Any, np: Any) -> None:
    annotated = image.copy()
    points = np.asarray(polygon, dtype="int32").reshape((-1, 1, 2))
    cv2.polylines(annotated, [points], isClosed=True, color=(0, 255, 0), thickness=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and perspective-crop trading cards from photos.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/crops")
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=ROOT / "data/models/cardcaptor_v3_best.pt")
    parser.add_argument("--model-repo", default=DEFAULT_REPO_ID)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--padding", type=float, default=0.0, help="Approximate extra pixels around the detected card.")
    parser.add_argument(
        "--target-aspect",
        type=float,
        default=63 / 88,
        help="Center-trim crops to this width/height ratio. Use 0 to disable.",
    )
    parser.add_argument("--aspect-tolerance", type=float, default=0.05)
    parser.add_argument("--max-crops", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cv2, np, YOLO = require_deps()
    started_at = utc_now_iso()
    model_path = ensure_model(args.model_path, args.model_repo, args.model_file)
    inputs = resolve_inputs(args.input, args.recursive)
    if not inputs:
        raise SystemExit(f"No supported image files found in: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_output = args.manifest_output or args.output_dir / "crop_manifest.jsonl"
    summary_output = args.summary_output or args.output_dir / "crop_summary.json"

    model = YOLO(str(model_path))
    records: list[dict[str, Any]] = []
    cropped_count = 0
    no_detection_count = 0
    error_count = 0

    for input_path in inputs:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            error_count += 1
            records.append(
                {
                    "status": "error",
                    "error": "OpenCV could not read image",
                    "input_path": str(input_path.resolve()),
                    "output_path": None,
                    "created_at": utc_now_iso(),
                }
            )
            continue

        try:
            predict_kwargs: dict[str, Any] = {
                "source": str(input_path),
                "conf": args.confidence,
                "imgsz": args.imgsz,
                "verbose": False,
            }
            if args.device:
                predict_kwargs["device"] = args.device
            result = model.predict(**predict_kwargs)[0]
            detections = sorted(detections_from_result(result, np), key=lambda item: item["confidence"], reverse=True)
        except Exception as exc:
            error_count += 1
            records.append(
                {
                    "status": "error",
                    "error": str(exc),
                    "input_path": str(input_path.resolve()),
                    "output_path": None,
                    "model_path": str(model_path.resolve()),
                    "created_at": utc_now_iso(),
                }
            )
            continue

        if not detections:
            no_detection_count += 1
            records.append(
                {
                    "status": "no_detection",
                    "input_path": str(input_path.resolve()),
                    "output_path": None,
                    "model_path": str(model_path.resolve()),
                    "confidence_threshold": args.confidence,
                    "created_at": utc_now_iso(),
                }
            )
            continue

        for crop_index, detection in enumerate(detections[: args.max_crops], start=1):
            polygon = np.asarray(detection["polygon"], dtype="float32")
            crop = warp_card(image, polygon, args.padding, cv2, np)
            crop, aspect_trimmed = trim_to_aspect(crop, args.target_aspect, args.aspect_tolerance)
            suffix = "_cardcrop" if args.max_crops == 1 else f"_cardcrop_{crop_index:02d}"
            output_path = args.output_dir / f"{safe_stem(input_path)}{suffix}.png"
            cv2.imwrite(str(output_path), crop)
            debug_path = None
            if args.save_debug:
                debug_path = args.output_dir / f"{safe_stem(input_path)}{suffix}_debug.png"
                save_debug_image(image, detection["polygon"], debug_path, cv2, np)
            crop_height, crop_width = crop.shape[:2]
            image_height, image_width = image.shape[:2]
            cropped_count += 1
            records.append(
                {
                    "status": "cropped",
                    "input_path": str(input_path.resolve()),
                    "output_path": str(output_path.resolve()),
                    "debug_path": str(debug_path.resolve()) if debug_path else None,
                    "model_path": str(model_path.resolve()),
                    "model_repo": args.model_repo,
                    "model_file": args.model_file,
                    "confidence_threshold": args.confidence,
                    "confidence": detection["confidence"],
                    "class_id": detection["class_id"],
                    "class_name": detection["class_name"],
                    "polygon": detection["polygon"],
                    "input_width": image_width,
                    "input_height": image_height,
                    "crop_width": crop_width,
                    "crop_height": crop_height,
                    "target_aspect": args.target_aspect,
                    "aspect_trimmed": aspect_trimmed,
                    "created_at": utc_now_iso(),
                }
            )

    count = write_jsonl(manifest_output, records)
    summary = {
        "model_path": str(model_path.resolve()),
        "model_repo": args.model_repo,
        "model_file": args.model_file,
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "images_seen": len(inputs),
        "records_written": count,
        "cropped": cropped_count,
        "no_detection": no_detection_count,
        "errors": error_count,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "manifest": str(manifest_output),
            "summary": str(summary_output),
        },
    }
    write_json(summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
