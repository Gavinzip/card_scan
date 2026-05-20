#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from scripts.cropping.auto_crop_cards import (
    DEFAULT_MODEL_FILE,
    DEFAULT_REPO_ID,
    detections_from_result,
    ensure_model,
    require_deps,
    safe_stem,
    trim_to_aspect,
    warp_card,
)
from scripts.lib.schema import utc_now_iso

MODEL_PATH = Path(os.environ.get("CARD_SCAN_CROP_MODEL_PATH", ROOT / "data/models/cardcaptor_v3_best.pt"))
MODEL_REPO = os.environ.get("CARD_SCAN_CROP_MODEL_REPO", DEFAULT_REPO_ID)
MODEL_FILE = os.environ.get("CARD_SCAN_CROP_MODEL_FILE", DEFAULT_MODEL_FILE)
OUTPUT_DIR = Path(os.environ.get("CARD_SCAN_CROP_OUTPUT_DIR", ROOT / "data/processed/crops/server"))
DEFAULT_CONFIDENCE = float(os.environ.get("CARD_SCAN_CROP_CONFIDENCE", "0.25"))
DEFAULT_IMGSZ = int(os.environ.get("CARD_SCAN_CROP_IMGSZ", "1024"))
DEFAULT_PADDING = float(os.environ.get("CARD_SCAN_CROP_PADDING", "0"))
DEFAULT_TARGET_ASPECT = float(os.environ.get("CARD_SCAN_CROP_TARGET_ASPECT", str(63 / 88)))
DEFAULT_ASPECT_TOLERANCE = float(os.environ.get("CARD_SCAN_CROP_ASPECT_TOLERANCE", "0.05"))
DEFAULT_DEVICE = os.environ.get("CARD_SCAN_CROP_DEVICE") or None

cv2, np, YOLO = require_deps()
_model: Any | None = None

app = FastAPI(title="Card Scan Crop API", version="0.1.0")


def get_model() -> Any:
    global _model
    if _model is None:
        model_path = ensure_model(MODEL_PATH, MODEL_REPO, MODEL_FILE)
        _model = YOLO(str(model_path))
    return _model


def append_manifest(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "crop_api_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


async def decode_upload(file: UploadFile) -> Any:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="OpenCV could not decode uploaded image.")
    return image


def crop_one(
    image: Any,
    original_name: str,
    confidence: float,
    imgsz: int,
    padding: float,
    target_aspect: float,
    aspect_tolerance: float,
    device: str | None,
) -> dict[str, Any]:
    predict_kwargs: dict[str, Any] = {
        "source": image,
        "conf": confidence,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device
    result = get_model().predict(**predict_kwargs)[0]
    detections = sorted(detections_from_result(result, np), key=lambda item: item["confidence"], reverse=True)
    image_height, image_width = image.shape[:2]
    base_record: dict[str, Any] = {
        "input_name": original_name,
        "model_path": str(MODEL_PATH.resolve()),
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILE,
        "confidence_threshold": confidence,
        "input_width": image_width,
        "input_height": image_height,
        "created_at": utc_now_iso(),
    }
    if not detections:
        return base_record | {"status": "no_detection", "output_path": None}

    detection = detections[0]
    polygon = np.asarray(detection["polygon"], dtype="float32")
    crop = warp_card(image, polygon, padding, cv2, np)
    crop, aspect_trimmed = trim_to_aspect(crop, target_aspect, aspect_tolerance)
    crop_height, crop_width = crop.shape[:2]
    output_name = f"{safe_stem(Path(original_name or 'upload'))}_{uuid.uuid4().hex[:10]}_cardcrop.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_name
    cv2.imwrite(str(output_path), crop)

    return base_record | {
        "status": "cropped",
        "output_path": str(output_path.resolve()),
        "confidence": detection["confidence"],
        "class_id": detection["class_id"],
        "class_name": detection["class_name"],
        "polygon": detection["polygon"],
        "crop_width": crop_width,
        "crop_height": crop_height,
        "target_aspect": target_aspect,
        "aspect_trimmed": aspect_trimmed,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": _model is not None,
        "model_path": str(MODEL_PATH),
        "output_dir": str(OUTPUT_DIR),
    }


@app.post("/crop", response_model=None)
async def crop(
    file: UploadFile = File(...),
    confidence: float = DEFAULT_CONFIDENCE,
    imgsz: int = DEFAULT_IMGSZ,
    padding: float = DEFAULT_PADDING,
    target_aspect: float = DEFAULT_TARGET_ASPECT,
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
    device: str | None = DEFAULT_DEVICE,
    return_image: bool = False,
) -> Any:
    image = await decode_upload(file)
    record = crop_one(
        image=image,
        original_name=file.filename or "upload",
        confidence=confidence,
        imgsz=imgsz,
        padding=padding,
        target_aspect=target_aspect,
        aspect_tolerance=aspect_tolerance,
        device=device,
    )
    append_manifest(record)
    if record["status"] != "cropped":
        return record
    if return_image:
        return FileResponse(record["output_path"], media_type="image/png", filename=Path(record["output_path"]).name)
    return record
