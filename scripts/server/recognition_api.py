#!/usr/bin/env python3
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Importing PyTorch, OpenCV/Ultralytics, and FAISS in one service can load
# multiple OpenMP runtimes on macOS. Linux Docker builds are usually fine, but
# this keeps local smoke tests from crashing.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[2]

from fastapi import FastAPI, File, HTTPException, UploadFile

from scripts.cropping.auto_crop_cards import (
    DEFAULT_MODEL_FILE,
    DEFAULT_REPO_ID,
    detections_from_result,
    ensure_model,
    require_deps,
    trim_to_aspect,
    warp_card,
)
from scripts.indexing.query_image_index import load_model
from scripts.lib.io_utils import iter_jsonl, read_json
from scripts.lib.schema import utc_now_iso

APP_VERSION = "0.1.0"

DEFAULT_INDEXES = (
    "pokemon_en=data/processed/image_index,"
    "pokemon_ja=data/processed/pokemon_ja_canonical_image_index"
)

INDEX_CONFIG = os.environ.get("CARD_SCAN_INDEXES", DEFAULT_INDEXES)
CROP_MODEL_PATH = Path(os.environ.get("CARD_SCAN_CROP_MODEL_PATH", ROOT / "data/models/cardcaptor_v3_best.pt"))
CROP_MODEL_REPO = os.environ.get("CARD_SCAN_CROP_MODEL_REPO", DEFAULT_REPO_ID)
CROP_MODEL_FILE = os.environ.get("CARD_SCAN_CROP_MODEL_FILE", DEFAULT_MODEL_FILE)
DEFAULT_CONFIDENCE = float(os.environ.get("CARD_SCAN_CROP_CONFIDENCE", "0.25"))
DEFAULT_IMGSZ = int(os.environ.get("CARD_SCAN_CROP_IMGSZ", "1024"))
DEFAULT_PADDING = float(os.environ.get("CARD_SCAN_CROP_PADDING", "0"))
DEFAULT_TARGET_ASPECT = float(os.environ.get("CARD_SCAN_CROP_TARGET_ASPECT", str(63 / 88)))
DEFAULT_ASPECT_TOLERANCE = float(os.environ.get("CARD_SCAN_CROP_ASPECT_TOLERANCE", "0.05"))
DEFAULT_DEVICE = os.environ.get("CARD_SCAN_DEVICE") or None
PRELOAD = os.environ.get("CARD_SCAN_PRELOAD", "false").lower() in {"1", "true", "yes"}

cv2, np, YOLO = require_deps()

app = FastAPI(title="TCG Card Recognition API", version=APP_VERSION)


@dataclass
class LoadedIndex:
    name: str
    path: Path
    records: list[dict[str, Any]]
    index: Any
    summary: dict[str, Any]


class RecognitionService:
    def __init__(self) -> None:
        self.crop_model: Any | None = None
        self.embedding_model: Any | None = None
        self.preprocess: Any | None = None
        self.device: str | None = None
        self.torch: Any | None = None
        self.image_module: Any | None = None
        self.backend: str | None = None
        self.indexes: list[LoadedIndex] | None = None

    def load_crop_model(self) -> Any:
        if self.crop_model is None:
            model_path = ensure_model(CROP_MODEL_PATH, CROP_MODEL_REPO, CROP_MODEL_FILE)
            self.crop_model = YOLO(str(model_path))
        return self.crop_model

    def load_indexes(self) -> list[LoadedIndex]:
        if self.indexes is not None:
            return self.indexes

        import faiss

        loaded: list[LoadedIndex] = []
        for name, path in configured_indexes().items():
            summary_path = path / "summary.json"
            index_path = path / "faiss.index"
            manifest_path = path / "image_embedding_manifest.jsonl"
            if not summary_path.exists() or not index_path.exists() or not manifest_path.exists():
                raise RuntimeError(f"Index directory is incomplete: {path}")
            summary = read_json(summary_path)
            loaded.append(
                LoadedIndex(
                    name=name,
                    path=path,
                    records=list(iter_jsonl(manifest_path)),
                    index=faiss.read_index(str(index_path)),
                    summary=summary,
                )
            )
        self.indexes = loaded
        return loaded

    def load_embedding_model(self) -> None:
        if self.embedding_model is not None:
            return
        indexes = self.load_indexes()
        if not indexes:
            raise RuntimeError("No indexes configured.")

        first_summary = indexes[0].summary
        for loaded in indexes[1:]:
            if loaded.summary.get("backend") != first_summary.get("backend"):
                raise RuntimeError("Configured indexes use different embedding backends.")
            if loaded.summary.get("model") != first_summary.get("model"):
                raise RuntimeError("Configured indexes use different embedding models.")

        np_mod, torch, Image, model_pack, device = load_model(first_summary, DEFAULT_DEVICE)
        self.embedding_model, self.preprocess = model_pack
        self.device = device
        self.torch = torch
        self.image_module = Image
        self.backend = first_summary.get("backend", "timm")

    def preload(self) -> None:
        self.load_indexes()
        self.load_embedding_model()
        self.load_crop_model()

    def crop(self, image: Any, confidence: float, imgsz: int, padding: float, target_aspect: float, aspect_tolerance: float) -> dict[str, Any]:
        result = self.load_crop_model().predict(
            source=image,
            conf=confidence,
            imgsz=imgsz,
            verbose=False,
        )[0]
        detections = sorted(detections_from_result(result, np), key=lambda item: item["confidence"], reverse=True)
        if not detections:
            return {
                "status": "no_detection",
                "detections": [],
                "crop_image": None,
                "fallback_used": False,
            }

        detection = detections[0]
        polygon = np.asarray(detection["polygon"], dtype="float32")
        crop_image = warp_card(image, polygon, padding, cv2, np)
        crop_image, aspect_trimmed = trim_to_aspect(crop_image, target_aspect, aspect_tolerance)
        crop_height, crop_width = crop_image.shape[:2]
        return {
            "status": "cropped",
            "detections": detections,
            "selected_detection": {
                "confidence": detection.get("confidence"),
                "class_id": detection.get("class_id"),
                "class_name": detection.get("class_name"),
                "polygon": detection.get("polygon"),
            },
            "crop_width": crop_width,
            "crop_height": crop_height,
            "aspect_trimmed": aspect_trimmed,
            "fallback_used": False,
            "crop_image": crop_image,
        }

    def encode(self, image_bgr: Any) -> Any:
        self.load_embedding_model()
        assert self.embedding_model is not None
        assert self.preprocess is not None
        assert self.torch is not None
        assert self.image_module is not None
        assert self.device is not None

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = self.image_module.fromarray(rgb)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            if self.backend == "open_clip":
                embedding = self.embedding_model.encode_image(tensor)
            else:
                embedding = self.embedding_model(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().astype("float32")

    def search(self, query: Any, per_index_top_k: int, combined_top_k: int) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        per_index: dict[str, list[dict[str, Any]]] = {}
        combined: list[dict[str, Any]] = []
        for loaded in self.load_indexes():
            scores, indices = loaded.index.search(query, per_index_top_k)
            results = []
            for rank, (score, index_id) in enumerate(zip(scores[0].tolist(), indices[0].tolist(), strict=False), start=1):
                if index_id < 0 or index_id >= len(loaded.records):
                    continue
                result = format_result(loaded.name, rank, float(score), loaded.records[index_id])
                results.append(result)
                combined.append(result)
            per_index[loaded.name] = results

        combined.sort(key=lambda item: item["score"], reverse=True)
        return combined[:combined_top_k], per_index


service = RecognitionService()


def configured_indexes() -> dict[str, Path]:
    indexes: dict[str, Path] = {}
    for item in INDEX_CONFIG.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(f"Invalid CARD_SCAN_INDEXES entry: {item!r}")
        name, path = item.split("=", 1)
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = ROOT / path_obj
        indexes[name.strip()] = path_obj
    return indexes


def format_result(index_name: str, rank: int, score: float, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index_name,
        "rank": rank,
        "score": score,
        "game": record.get("game"),
        "source": record.get("source"),
        "canonical_source": record.get("canonical_source"),
        "image_source": record.get("image_source"),
        "card_id": record.get("card_id"),
        "canonical_id": record.get("canonical_id"),
        "set_id": record.get("set_id"),
        "card_code": record.get("card_code"),
        "language": record.get("language"),
        "name": record.get("name"),
        "name_en": record.get("name_en"),
        "name_ja": record.get("name_ja"),
        "rarity": record.get("rarity"),
        "variant": record.get("variant"),
        "image_url": record.get("image_url"),
        "image_sha256": record.get("image_sha256"),
        "source_record_count": record.get("source_record_count"),
        "duplicate_source_count": record.get("duplicate_source_count"),
        "duplicate_image_count": record.get("duplicate_image_count"),
        "snkr": {
            "match_status": record.get("snkr_match_status"),
            "product_id": record.get("snkr_product_id"),
            "product_name": record.get("snkr_product_name"),
            "url": record.get("snkr_url"),
            "min_price": record.get("snkr_min_price"),
            "min_price_format": record.get("snkr_min_price_format"),
            "verified_candidate_count": record.get("snkr_verified_candidate_count"),
            "matched_at": record.get("snkr_matched_at"),
        },
    }


async def decode_upload(file: UploadFile) -> Any:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="OpenCV could not decode uploaded image.")
    return image


def encode_debug_crop(crop_image: Any) -> str:
    ok, buffer = cv2.imencode(".jpg", crop_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode debug crop image.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


@app.on_event("startup")
def startup() -> None:
    if PRELOAD:
        service.preload()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "TCG Card Recognition API",
        "version": APP_VERSION,
        "endpoints": ["/health", "/recognize"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    configured = configured_indexes()
    return {
        "ok": True,
        "version": APP_VERSION,
        "preload": PRELOAD,
        "crop_model_loaded": service.crop_model is not None,
        "embedding_model_loaded": service.embedding_model is not None,
        "indexes_loaded": service.indexes is not None,
        "configured_indexes": {name: str(path) for name, path in configured.items()},
        "available_indexes": {
            name: {
                "exists": path.exists(),
                "summary": (path / "summary.json").exists(),
                "faiss": (path / "faiss.index").exists(),
                "manifest": (path / "image_embedding_manifest.jsonl").exists(),
            }
            for name, path in configured.items()
        },
        "device": service.device or DEFAULT_DEVICE or "auto",
    }


@app.post("/warmup")
def warmup() -> dict[str, Any]:
    started = time.perf_counter()
    service.preload()
    return {
        "ok": True,
        "seconds": time.perf_counter() - started,
        "indexes": [
            {"name": loaded.name, "records": len(loaded.records), "path": str(loaded.path)}
            for loaded in (service.indexes or [])
        ],
        "device": service.device,
    }


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    crop: bool = True,
    fallback_to_original: bool = False,
    top_k: int = 5,
    per_index_top_k: int = 5,
    confidence: float = DEFAULT_CONFIDENCE,
    imgsz: int = DEFAULT_IMGSZ,
    padding: float = DEFAULT_PADDING,
    target_aspect: float = DEFAULT_TARGET_ASPECT,
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
    include_debug_crop_base64: bool = False,
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    started_at = utc_now_iso()
    total_start = time.perf_counter()
    image = await decode_upload(file)
    input_height, input_width = image.shape[:2]

    crop_payload: dict[str, Any] = {
        "status": "skipped",
        "fallback_used": False,
        "detections": [],
    }
    query_image = image
    if crop:
        crop_start = time.perf_counter()
        crop_payload = service.crop(
            image=image,
            confidence=confidence,
            imgsz=imgsz,
            padding=padding,
            target_aspect=target_aspect,
            aspect_tolerance=aspect_tolerance,
        )
        timings["crop_seconds"] = time.perf_counter() - crop_start
        if crop_payload["status"] == "cropped":
            query_image = crop_payload.pop("crop_image")
        elif fallback_to_original:
            crop_payload["fallback_used"] = True
            crop_payload.pop("crop_image", None)
            query_image = image
        else:
            crop_payload.pop("crop_image", None)
            timings["total_seconds"] = time.perf_counter() - total_start
            return {
                "status": "no_detection",
                "started_at": started_at,
                "input": {
                    "filename": file.filename,
                    "width": input_width,
                    "height": input_height,
                },
                "crop": crop_payload,
                "results": [],
                "results_by_index": {},
                "timings": timings,
            }

    encode_start = time.perf_counter()
    query = service.encode(query_image)
    timings["embedding_seconds"] = time.perf_counter() - encode_start

    search_start = time.perf_counter()
    combined, per_index = service.search(query, per_index_top_k=per_index_top_k, combined_top_k=top_k)
    timings["search_seconds"] = time.perf_counter() - search_start
    timings["total_seconds"] = time.perf_counter() - total_start

    if include_debug_crop_base64 and crop and crop_payload.get("status") == "cropped":
        crop_payload["debug_crop_jpeg_base64"] = encode_debug_crop(query_image)

    return {
        "status": "ok",
        "started_at": started_at,
        "input": {
            "filename": file.filename,
            "width": input_width,
            "height": input_height,
        },
        "crop": crop_payload,
        "results": combined,
        "results_by_index": per_index,
        "timings": timings,
    }
