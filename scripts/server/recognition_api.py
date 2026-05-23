#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import quote

# Importing PyTorch, OpenCV/Ultralytics, and FAISS in one service can load
# multiple OpenMP runtimes on macOS. Linux Docker builds are usually fine, but
# this keeps local smoke tests from crashing.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[2]

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from scripts.cropping.auto_crop_cards import (
    DEFAULT_MODEL_FILE,
    DEFAULT_REPO_ID,
    detections_from_result,
    ensure_model,
    expand_points,
    order_points,
    require_deps,
    trim_to_aspect,
    warp_card,
)
from scripts.indexing.query_image_index import load_model
from scripts.lib.io_utils import iter_jsonl, read_json
from scripts.lib.schema import utc_now_iso

APP_VERSION = "0.1.0"
FRONTEND_DIR = ROOT / "web"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"
FRONTEND_ASSETS = FRONTEND_DIR / "assets"

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
DEFAULT_RERANK_MODEL = os.environ.get("CARD_SCAN_RERANK_MODEL", "siglip").strip().lower()
DEFAULT_VISUAL_RERANK_CANDIDATES = int(os.environ.get("CARD_SCAN_VISUAL_RERANK_CANDIDATES", "100"))
DEFAULT_VISUAL_RERANK_WEIGHT = float(os.environ.get("CARD_SCAN_VISUAL_RERANK_WEIGHT", "0.40"))
DEFAULT_VISUAL_COLOR_WEIGHT = float(os.environ.get("CARD_SCAN_VISUAL_COLOR_WEIGHT", "0.50"))
DEFAULT_SIGLIP_MODEL = os.environ.get("CARD_SCAN_SIGLIP_MODEL", "vit_base_patch16_siglip_224.webli")
DEFAULT_SIGLIP_BATCH_SIZE = int(os.environ.get("CARD_SCAN_SIGLIP_BATCH_SIZE", "32"))
DEFAULT_CARD_CODE_OCR = os.environ.get("CARD_SCAN_CARD_CODE_OCR", "false").lower() in {"1", "true", "yes"}
DEFAULT_CARD_CODE_OCR_TIMEOUT = float(os.environ.get("CARD_SCAN_CARD_CODE_OCR_TIMEOUT", "8"))
DEFAULT_CARD_CODE_OCR_EXACT_BOOST = float(os.environ.get("CARD_SCAN_CARD_CODE_OCR_EXACT_BOOST", "0.09"))
PRELOAD = os.environ.get("CARD_SCAN_PRELOAD", "false").lower() in {"1", "true", "yes"}
PRELOAD_SIGLIP = os.environ.get("CARD_SCAN_PRELOAD_SIGLIP", "false").lower() in {"1", "true", "yes"}
PRELOAD_CROP_MODEL = os.environ.get("CARD_SCAN_PRELOAD_CROP_MODEL", "true").lower() in {"1", "true", "yes"}
REFERENCE_IMAGE_ROUTE = os.environ.get("CARD_SCAN_REFERENCE_IMAGE_ROUTE", "/reference-images").rstrip("/") or "/reference-images"
REFERENCE_IMAGE_ROOTS_CONFIG = os.environ.get("CARD_SCAN_IMAGE_ROOTS", "")
LOCAL_PATH_REWRITES_CONFIG = os.environ.get("CARD_SCAN_LOCAL_PATH_REWRITES", "")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CARD_SCAN_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

cv2, np, YOLO = require_deps()

app = FastAPI(title="TCG Card Recognition API", version=APP_VERSION)
VISUAL_SIGNATURE_CACHE: dict[str, tuple[int, int, dict[str, Any]]] = {}
SET_PATTERN = (
    r"(?:PROMO\s*[_-]?\s*SWSH|SWSH|WCS\s*\d{2,}|"
    r"SV\s*[-]?\s*P|S\s*[-]?\s*P|SM\s*[-]?\s*P|XY\s*[-]?\s*P|M\s*[-]?\s*P|"
    r"(?:SV|S|SM|XY|BW|DP|M|ME)\s*\d{1,2}[A-Z]?)"
)
CARD_CODE_PATTERN = re.compile(
    rf"\b(?P<set>{SET_PATTERN})\s+(?P<number>[A-Z]?\d{{1,3}}|[A-Z]{{2}}\d{{1,3}})\s*/\s*(?P<total>\d{{1,3}})\b",
    re.I,
)
REVERSE_PROMO_PATTERN = re.compile(
    rf"\b(?P<number>\d{{1,3}})\s*/\s*(?P<set>{SET_PATTERN})\b",
    re.I,
)
SWSH_PROMO_PATTERN = re.compile(r"\b(?P<set>SWSH)\s*[- ]?(?P<number>\d{1,3})\b", re.I)
LOOSE_SET_NUMBER_PATTERN = re.compile(
    rf"\b(?P<set>{SET_PATTERN})\s+(?P<number>[A-Z]?\d{{1,3}}|[A-Z]{{2}}\d{{1,3}})\b",
    re.I,
)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

if FRONTEND_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="assets")


def parse_named_paths(config: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for item in config.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(f"Invalid CARD_SCAN_IMAGE_ROOTS entry: {item!r}")
        name, path = item.split("=", 1)
        name = safe_url_part(name.strip())
        if not name:
            raise RuntimeError(f"Invalid CARD_SCAN_IMAGE_ROOTS name: {item!r}")
        paths[name] = Path(path.strip()).resolve()
    return paths


def parse_path_rewrites(config: str) -> list[tuple[Path, Path]]:
    rewrites: list[tuple[Path, Path]] = []
    for item in config.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(f"Invalid CARD_SCAN_LOCAL_PATH_REWRITES entry: {item!r}")
        source, target = item.split("=", 1)
        rewrites.append((Path(source.strip()).resolve(), Path(target.strip()).resolve()))
    return rewrites


def safe_url_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


REFERENCE_IMAGE_ROOTS = parse_named_paths(REFERENCE_IMAGE_ROOTS_CONFIG)
LOCAL_PATH_REWRITES = parse_path_rewrites(LOCAL_PATH_REWRITES_CONFIG)

for image_root_name, image_root_path in REFERENCE_IMAGE_ROOTS.items():
    if image_root_path.exists():
        app.mount(
            f"{REFERENCE_IMAGE_ROUTE}/{image_root_name}",
            StaticFiles(directory=image_root_path),
            name=f"reference_images_{image_root_name}",
        )


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
        self.siglip_model: Any | None = None
        self.siglip_preprocess: Any | None = None
        self.siglip_device: str | None = None
        self.siglip_torch: Any | None = None
        self.siglip_image_module: Any | None = None
        self.siglip_embedding_cache: dict[str, tuple[int, int, Any]] = {}

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
        if PRELOAD_SIGLIP:
            self.load_siglip_model()
        if PRELOAD_CROP_MODEL:
            self.load_crop_model()

    def load_siglip_model(self) -> None:
        if self.siglip_model is not None:
            return
        import torch
        import timm
        from PIL import Image
        from timm.data import create_transform, resolve_model_data_config

        if DEFAULT_DEVICE:
            device = DEFAULT_DEVICE
        elif torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        model = timm.create_model(DEFAULT_SIGLIP_MODEL, pretrained=True, num_classes=0)
        model = model.to(device)
        model.eval()
        self.siglip_model = model
        self.siglip_preprocess = create_transform(**resolve_model_data_config(model), is_training=False)
        self.siglip_device = device
        self.siglip_torch = torch
        self.siglip_image_module = Image

    def encode_siglip_bgr(self, image_bgr: Any) -> Any:
        self.load_siglip_model()
        assert self.siglip_model is not None
        assert self.siglip_preprocess is not None
        assert self.siglip_torch is not None
        assert self.siglip_image_module is not None
        assert self.siglip_device is not None

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = self.siglip_image_module.fromarray(rgb)
        tensor = self.siglip_preprocess(pil_image).unsqueeze(0).to(self.siglip_device)
        with self.siglip_torch.no_grad():
            embedding = self.siglip_model(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().astype("float32")[0]

    def encode_siglip_paths(self, paths: list[Path]) -> dict[str, Any]:
        self.load_siglip_model()
        assert self.siglip_model is not None
        assert self.siglip_preprocess is not None
        assert self.siglip_torch is not None
        assert self.siglip_image_module is not None
        assert self.siglip_device is not None

        output: dict[str, Any] = {}
        missing: list[Path] = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            cache_key = str(path.resolve())
            cached = self.siglip_embedding_cache.get(cache_key)
            if cached and cached[0] == stat.st_size and cached[1] == stat.st_mtime_ns:
                output[cache_key] = cached[2]
            else:
                missing.append(path)

        with self.siglip_torch.no_grad():
            for start in range(0, len(missing), DEFAULT_SIGLIP_BATCH_SIZE):
                batch_paths = missing[start : start + DEFAULT_SIGLIP_BATCH_SIZE]
                images = []
                kept_paths = []
                for path in batch_paths:
                    try:
                        stat = path.stat()
                        with self.siglip_image_module.open(path) as image:
                            images.append(self.siglip_preprocess(image.convert("RGB")))
                        kept_paths.append((path, stat.st_size, stat.st_mtime_ns))
                    except Exception:
                        continue
                if not images:
                    continue
                tensor = self.siglip_torch.stack(images).to(self.siglip_device)
                embeddings = self.siglip_model(tensor)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                for (path, size, mtime_ns), vector in zip(kept_paths, embeddings.cpu().numpy().astype("float32"), strict=False):
                    cache_key = str(path.resolve())
                    self.siglip_embedding_cache[cache_key] = (size, mtime_ns, vector)
                    output[cache_key] = vector
        return output

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
        crop_image = warp_card_to_target_aspect(image, polygon, padding, target_aspect)
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


def rewritten_local_paths(local_image_path: str | None) -> list[Path]:
    if not local_image_path:
        return []
    original = Path(local_image_path).resolve()
    candidates = [original]
    for source_prefix, target_prefix in LOCAL_PATH_REWRITES:
        try:
            relative = original.relative_to(source_prefix)
        except ValueError:
            continue
        candidates.append(target_prefix / relative)
    return candidates


def reference_image_url(local_image_path: str | None) -> str | None:
    for candidate in rewritten_local_paths(local_image_path):
        for root_name, root_path in REFERENCE_IMAGE_ROOTS.items():
            if not root_path.exists():
                continue
            try:
                relative = candidate.relative_to(root_path)
            except ValueError:
                continue
            if not candidate.exists():
                continue
            encoded = "/".join(quote(part) for part in relative.parts)
            return f"{REFERENCE_IMAGE_ROUTE}/{root_name}/{encoded}"
    return None


def existing_local_image_path(local_image_path: str | None) -> Path | None:
    for candidate in rewritten_local_paths(local_image_path):
        if candidate.exists():
            return candidate
    return None


def format_result(index_name: str, rank: int, score: float, record: dict[str, Any]) -> dict[str, Any]:
    remote_image_url = record.get("image_url")
    local_reference_image_url = reference_image_url(record.get("local_image_path"))
    return {
        "index": index_name,
        "rank": rank,
        "score": score,
        "game": record.get("game"),
        "source": record.get("source"),
        "source_license": record.get("source_license"),
        "canonical_source": record.get("canonical_source"),
        "canonical_source_license": record.get("canonical_source_license"),
        "merge_status": record.get("merge_status"),
        "supplemental_sources": record.get("supplemental_sources"),
        "image_source": record.get("image_source"),
        "image_source_license": record.get("image_source_license"),
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
        "image_url": remote_image_url,
        "reference_image_url": local_reference_image_url,
        "display_image_url": remote_image_url or local_reference_image_url,
        "local_image_path": record.get("local_image_path"),
        "image_sha256": record.get("image_sha256"),
        "source_record_count": record.get("source_record_count"),
        "duplicate_source_count": record.get("duplicate_source_count"),
        "duplicate_image_count": record.get("duplicate_image_count"),
        "training_allowed": record.get("training_allowed"),
        "reference_index_allowed_internal": record.get("reference_index_allowed_internal"),
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


def crop_ratio(image: Any, box: tuple[float, float, float, float]) -> Any:
    height, width = image.shape[:2]
    left, top, right, bottom = box
    x1 = max(0, min(width - 1, int(round(width * left))))
    y1 = max(0, min(height - 1, int(round(height * top))))
    x2 = max(x1 + 1, min(width, int(round(width * right))))
    y2 = max(y1 + 1, min(height, int(round(height * bottom))))
    return image[y1:y2, x1:x2]


def card_code_ocr_variants(roi: Any) -> list[tuple[str, Any]]:
    height, width = roi.shape[:2]
    scale = max(3, min(8, int(round(900 / max(1, height)))))
    enlarged = cv2.resize(roi, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sharpened = cv2.addWeighted(clahe, 1.8, cv2.GaussianBlur(clahe, (0, 0), 1.2), -0.8, 0)
    binary = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    return [
        ("color", enlarged),
        ("gray", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)),
        ("sharp", cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)),
        ("binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)),
    ]


def normalize_ocr_text(text: str) -> str:
    normalized = text.upper()
    substitutions = {
        "§": "S",
        "$": "S",
        "Ｏ": "0",
        "Ｉ": "1",
        "|": "1",
        "／": "/",
        "\\": "/",
        "—": "-",
        "–": "-",
    }
    for source, target in substitutions.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\b(SV|S|SM|XY|M)([ILO])([A-Z])\b", r"\g<1>1\3", normalized)
    normalized = re.sub(r"(?<=\d)[OQ](?=\d)", "0", normalized)
    normalized = re.sub(r"(?<=/)[OQ](?=\d)", "0", normalized)
    normalized = re.sub(r"(?<=\s)[OQ](?=\d{2,3})", "0", normalized)
    normalized = re.sub(r"(?<=\d)\s*[IL]\s*(?=\d)", "1", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def normalize_ocr_set(value: str | None) -> str | None:
    if not value:
        return None
    text = value.upper().replace(" ", "").replace("--", "-").replace("_", "-")
    text = re.sub(r"^(SV|S|SM|XY|M)-?P$", r"\1-P", text)
    if text == "SWSH":
        return "PROMO-SWSH"
    return text


def normalize_card_number(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).upper().strip()
    if "/" in text:
        text = text.split("/", 1)[0]
    text = re.sub(r"[^A-Z0-9]", "", text)
    if re.fullmatch(r"[A-Z]\d{2,3}", text) and not re.match(r"^(G|T|R|S)", text):
        text = text[1:]
    return text.lstrip("0") or "0"


def normalize_set_for_match(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).upper().replace("_", "-")
    text = re.sub(r"[^A-Z0-9-]", "", text)
    text = re.sub(r"^(SV|S|SM|XY|M)-?P$", r"\1-P", text)
    if text in {"SVP", "SV-P"}:
        return "SV-P"
    if text in {"SWSH", "PROMOSWSH", "PROMO-SWSH"}:
        return "PROMO-SWSH"
    return text


def extract_card_code(text: str) -> dict[str, Any] | None:
    normalized = normalize_ocr_text(text)
    patterns = [
        ("strict", CARD_CODE_PATTERN),
        ("reverse_promo", REVERSE_PROMO_PATTERN),
        ("swsh_promo", SWSH_PROMO_PATTERN),
        ("loose", LOOSE_SET_NUMBER_PATTERN),
    ]
    for pattern_name, pattern in patterns:
        match = pattern.search(normalized)
        if match:
            return {
                "quality": pattern_name,
                "set_id": normalize_ocr_set(match.group("set")),
                "card_number": normalize_card_number(match.group("number")),
                "total": normalize_card_number(match.groupdict().get("total")),
                "raw_match": match.group(0),
                "normalized_text": normalized,
            }
    return None


def better_card_code_attempt(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    quality_rank = {"strict": 4, "reverse_promo": 4, "swsh_promo": 3, "loose": 2}
    candidate_parsed = candidate.get("parsed") or {}
    best_parsed = best.get("parsed") or {}
    candidate_rank = quality_rank.get(candidate_parsed.get("quality"), 0)
    best_rank = quality_rank.get(best_parsed.get("quality"), 0)
    if candidate_rank != best_rank:
        return candidate_rank > best_rank
    return float(candidate.get("confidence") or 0.0) > float(best.get("confidence") or 0.0)


def recognize_card_bottom_code(card_image: Any) -> dict[str, Any]:
    swift_script = ROOT / "scripts/quality/ocr_vision_text.swift"
    if not swift_script.exists():
        return {"status": "unavailable", "reason": "ocr_vision_text.swift is missing"}

    rois = {
        "bottom_left_tight": (0.00, 0.875, 0.38, 0.985),
        "bottom_left_wide": (0.00, 0.835, 0.55, 0.995),
        "bottom_full": (0.00, 0.835, 1.00, 0.995),
    }
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        image_paths: list[Path] = []
        meta: list[dict[str, str]] = []
        for roi_name, box in rois.items():
            roi = crop_ratio(card_image, box)
            for variant_name, variant_image in card_code_ocr_variants(roi):
                path = temp_path / f"{roi_name}_{variant_name}.png"
                cv2.imwrite(str(path), variant_image)
                image_paths.append(path)
                meta.append({"roi": roi_name, "variant": variant_name})

        command = ["swift", str(swift_script), *[str(path) for path in image_paths]]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=DEFAULT_CARD_CODE_OCR_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "seconds": time.perf_counter() - started,
                "timeout_seconds": DEFAULT_CARD_CODE_OCR_TIMEOUT,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "seconds": time.perf_counter() - started,
                "error": str(exc),
            }

    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for item_meta, line in zip(meta, completed.stdout.splitlines(), strict=False):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = row.get("text") or ""
        observation_confidence = max(
            [item.get("confidence") or 0.0 for item in row.get("observations") or []],
            default=0.0,
        )
        parsed = extract_card_code(text)
        attempt = {
            **item_meta,
            "text": text,
            "confidence": observation_confidence,
            "parsed": parsed,
        }
        attempts.append(attempt)
        if parsed and better_card_code_attempt(attempt, best):
            best = attempt

    status = "ok" if best else "not_found"
    return {
        "status": status,
        "seconds": time.perf_counter() - started,
        "best": best,
        "attempt_count": len(attempts),
        "parsed_attempt_count": sum(1 for attempt in attempts if attempt.get("parsed")),
    }


def candidate_matches_ocr_code(candidate: dict[str, Any], parsed: dict[str, Any]) -> bool:
    ocr_set = normalize_set_for_match(parsed.get("set_id"))
    ocr_number = normalize_card_number(parsed.get("card_number"))
    candidate_set = normalize_set_for_match(candidate.get("set_id"))
    candidate_number = normalize_card_number(candidate.get("card_code"))
    if not ocr_set or not ocr_number or not candidate_set or not candidate_number:
        return False
    return ocr_set == candidate_set and ocr_number == candidate_number


def apply_card_code_ocr_boost(results: list[dict[str, Any]], ocr_payload: dict[str, Any], boost: float) -> list[dict[str, Any]]:
    best = ocr_payload.get("best") or {}
    parsed = best.get("parsed") or {}
    if ocr_payload.get("status") != "ok" or not parsed:
        return results

    boosted: list[dict[str, Any]] = []
    for result in results:
        item = dict(result)
        match = candidate_matches_ocr_code(item, parsed)
        item["ocr_card_code_match"] = match
        item["ocr_card_code_boost"] = boost if match else 0.0
        if match:
            item["pre_ocr_score"] = item.get("score")
            item["score"] = float(item.get("score") or 0.0) + boost
        boosted.append(item)

    boosted.sort(key=lambda item: item["score"], reverse=True)
    return boosted


def trim_box_to_aspect(box: tuple[int, int, int, int], width: int, height: int, target_aspect: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0 or target_aspect <= 0:
        return box
    aspect = box_width / box_height
    if aspect > target_aspect:
        new_width = max(1, int(round(box_height * target_aspect)))
        center_x = (x1 + x2) // 2
        x1 = center_x - (new_width // 2)
        x2 = x1 + new_width
    else:
        new_height = max(1, int(round(box_width / target_aspect)))
        center_y = (y1 + y2) // 2
        y1 = center_y - (new_height // 2)
        y2 = y1 + new_height
    x1 = max(0, min(width - 2, x1))
    y1 = max(0, min(height - 2, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def expand_box_ratio(box: tuple[int, int, int, int], width: int, height: int, ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x = int(round((x2 - x1) * ratio))
    pad_y = int(round((y2 - y1) * ratio))
    x1 = max(0, min(width - 2, x1 - pad_x))
    y1 = max(0, min(height - 2, y1 - pad_y))
    x2 = max(x1 + 1, min(width, x2 + pad_x))
    y2 = max(y1 + 1, min(height, y2 + pad_y))
    return x1, y1, x2, y2


def largest_component_stats(mask: Any) -> dict[str, float]:
    components = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    count, _labels, stats, _centroids = components
    if count <= 1:
        return {"width_ratio": 0.0, "height_ratio": 0.0, "area_ratio": 0.0}
    height, width = mask.shape[:2]
    best = max(range(1, count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    return {
        "width_ratio": float(stats[best, cv2.CC_STAT_WIDTH] / max(1, width)),
        "height_ratio": float(stats[best, cv2.CC_STAT_HEIGHT] / max(1, height)),
        "area_ratio": float(stats[best, cv2.CC_STAT_AREA] / max(1, width * height)),
    }


def horizontal_run_ratio(mask: Any, min_column_coverage: float = 0.04) -> float:
    if mask.size == 0:
        return 0.0
    columns = (mask.mean(axis=0) >= min_column_coverage).astype("uint8")
    best = 0
    current = 0
    for value in columns:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return float(best / max(1, columns.shape[0]))


def psa_label_score(image: Any) -> dict[str, float]:
    height, width = image.shape[:2]
    band = image[int(height * 0.03) : int(height * 0.23), int(width * 0.04) : int(width * 0.96)]
    if band.size == 0:
        return {
            "white_ratio": 0.0,
            "red_ratio": 0.0,
            "score": 0.0,
            "is_psa_label": False,
            "red_component_width_ratio": 0.0,
            "white_component_width_ratio": 0.0,
        }
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    white_mask = ((saturation < 55) & (value > 120)).astype("uint8")
    red_mask = (((hue < 10) | (hue > 170)) & (saturation > 70) & (value > 80)).astype("uint8")
    white_ratio = float(white_mask.mean())
    red_ratio = float(red_mask.mean())
    white_component = largest_component_stats(white_mask)
    red_component = largest_component_stats(red_mask)
    white_horizontal_run = horizontal_run_ratio(white_mask)
    red_horizontal_run = horizontal_run_ratio(red_mask)
    is_psa_label = (
        white_ratio >= 0.18
        and red_ratio >= 0.035
        and white_component["width_ratio"] >= 0.34
        and white_component["height_ratio"] >= 0.30
        and white_component["area_ratio"] >= 0.08
        and red_component["width_ratio"] >= 0.32
        and red_component["height_ratio"] >= 0.30
        and red_component["area_ratio"] >= 0.03
        and white_horizontal_run >= 0.34
        and red_horizontal_run >= 0.32
    )
    return {
        "white_ratio": white_ratio,
        "red_ratio": red_ratio,
        "score": min(1.0, white_ratio + (red_ratio * 1.5)),
        "is_psa_label": is_psa_label,
        "white_component_width_ratio": white_component["width_ratio"],
        "white_component_area_ratio": white_component["area_ratio"],
        "red_component_width_ratio": red_component["width_ratio"],
        "red_component_area_ratio": red_component["area_ratio"],
        "white_component_height_ratio": white_component["height_ratio"],
        "red_component_height_ratio": red_component["height_ratio"],
        "white_horizontal_run_ratio": white_horizontal_run,
        "red_horizontal_run_ratio": red_horizontal_run,
    }


def slab_ratio_fallback_box(image: Any) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    if height / max(1, width) > 1.18:
        return (
            int(round(width * 0.09)),
            int(round(height * 0.24)),
            int(round(width * 0.91)),
            int(round(height * 0.94)),
        )
    return (
        int(round(width * 0.305)),
        int(round(height * 0.2925)),
        int(round(width * 0.695)),
        int(round(height * 0.8375)),
    )


def contour_slab_inner_card_box(image: Any, target_aspect: float) -> tuple[tuple[int, int, int, int] | None, dict[str, Any]]:
    height, width = image.shape[:2]
    y_offset = int(round(height * 0.18))
    lower = image[y_offset : int(round(height * 0.97)), :]
    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (((saturation > 42) & (value > 35)) | ((value < 115) & (saturation > 12))).astype("uint8") * 255
    mask[: int(mask.shape[0] * 0.05), :] = 0
    kernel_size = max(9, (min(width, height) // 75) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, Any]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        y += y_offset
        area_ratio = (box_width * box_height) / float(width * height)
        aspect = box_width / max(1, box_height)
        if area_ratio < 0.08 or box_height < height * 0.33:
            continue
        if not 0.42 <= aspect <= 0.92:
            continue
        center_x = (x + (box_width / 2)) / width
        center_penalty = abs(center_x - 0.5)
        aspect_penalty = abs(aspect - target_aspect) / max(0.01, target_aspect)
        top_penalty = max(0.0, ((height * 0.20) - y) / height)
        score = area_ratio - (0.18 * center_penalty) - (0.08 * aspect_penalty) - top_penalty
        candidates.append(
            {
                "box": (x, y, x + box_width, y + box_height),
                "score": score,
                "area_ratio": area_ratio,
                "aspect": aspect,
                "center_x": center_x,
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    if not candidates:
        return None, {"candidate_count": 0}
    return candidates[0]["box"], {"candidate_count": len(candidates), "best": candidates[0]}


def slab_inner_card_crop(image: Any, target_aspect: float) -> dict[str, Any] | None:
    height, width = image.shape[:2]
    label = psa_label_score(image)
    if not label.get("is_psa_label"):
        return None

    box, details = contour_slab_inner_card_box(image, target_aspect)
    method = "slab_contour"
    if box is None:
        box = slab_ratio_fallback_box(image)
        method = "slab_ratio_fallback"

    box = expand_box_ratio(box, width, height, 0.012)
    box = trim_box_to_aspect(box, width, height, target_aspect)
    x1, y1, x2, y2 = box
    crop_image = image[y1:y2, x1:x2]
    crop_height, crop_width = crop_image.shape[:2]
    return {
        "status": "slab_inner_card",
        "fallback_used": False,
        "detections": [],
        "crop_width": crop_width,
        "crop_height": crop_height,
        "slab_crop_method": method,
        "slab_label": label,
        "slab_details": details,
        "slab_crop_box_ratio": {
            "left": x1 / width,
            "top": y1 / height,
            "right": x2 / width,
            "bottom": y2 / height,
        },
        "crop_image": crop_image,
    }


def warp_card_to_target_aspect(image: Any, polygon: Any, padding: float, target_aspect: float) -> Any:
    if target_aspect <= 0:
        return warp_card(image, polygon, padding, cv2, np)
    image_height, image_width = image.shape[:2]
    rect = order_points(np.asarray(polygon, dtype="float32"), np)
    rect = expand_points(rect, padding, image_width, image_height, np)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    measured_width = max(1.0, float(max(width_a, width_b)))
    measured_height = max(1.0, float(max(height_a, height_b)))
    output_height = max(measured_height, measured_width / target_aspect)
    output_width = output_height * target_aspect
    output_width_i = max(1, int(round(output_width)))
    output_height_i = max(1, int(round(output_height)))
    destination = np.array(
        [
            [0, 0],
            [output_width_i - 1, 0],
            [output_width_i - 1, output_height_i - 1],
            [0, output_height_i - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (output_width_i, output_height_i))


def generic_graded_slab_crop(image: Any) -> dict[str, Any] | None:
    height, width = image.shape[:2]
    image_aspect = height / max(1, width)
    if not 1.12 <= image_aspect <= 1.95:
        return None

    if image_aspect >= 1.65:
        left = 0.19
        top = 0.42
        right = 0.70
        bottom = 0.805
    else:
        left = 0.10
        top = 0.27
        right = 0.82
        bottom = 0.965
    x1 = max(0, min(width - 2, int(round(width * left))))
    y1 = max(0, min(height - 2, int(round(height * top))))
    x2 = max(x1 + 1, min(width, int(round(width * right))))
    y2 = max(y1 + 1, min(height, int(round(height * bottom))))
    crop_image = image[y1:y2, x1:x2]
    crop_height, crop_width = crop_image.shape[:2]
    return {
        "status": "graded_slab_ratio_card",
        "fallback_used": True,
        "detections": [],
        "crop_width": crop_width,
        "crop_height": crop_height,
        "graded_slab_crop_box_ratio": {
            "left": x1 / width,
            "top": y1 / height,
            "right": x2 / width,
            "bottom": y2 / height,
        },
        "crop_image": crop_image,
    }


def contour_crop_looks_like_graded_slab_miss(crop_payload: dict[str, Any] | None) -> bool:
    if crop_payload is None:
        return False
    details = crop_payload.get("contour_details") or {}
    best = details.get("best") or {}
    area_ratio = float(best.get("area_ratio") or 0.0)
    aspect = float(best.get("aspect") or 0.0)
    center_x = float(best.get("center_x") or 0.5)
    return area_ratio < 0.30 and aspect < 0.66 and center_x < 0.46


def yolo_crop_looks_like_label_detection(crop_payload: dict[str, Any] | None, image_height: int) -> bool:
    if crop_payload is None or crop_payload.get("status") != "cropped":
        return False
    detection = crop_payload.get("selected_detection") or {}
    polygon = np.asarray(detection.get("polygon") or [], dtype="float32")
    if polygon.shape != (4, 2):
        return False
    center_y = float(polygon[:, 1].mean())
    box_width = float(polygon[:, 0].max() - polygon[:, 0].min())
    box_height = float(polygon[:, 1].max() - polygon[:, 1].min())
    box_aspect = box_width / max(1.0, box_height)
    is_top_band = center_y < max(380.0, float(image_height) * 0.32)
    return is_top_band and box_aspect > 1.15


def polygon_from_contour(contour: Any) -> tuple[Any, str]:
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for epsilon_ratio in (0.012, 0.018, 0.025, 0.035, 0.05, 0.07):
        approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32"), "quad"
    return cv2.boxPoints(cv2.minAreaRect(contour)).astype("float32"), "min_area_rect"


def foreground_card_mask(image: Any) -> Any:
    height, width = image.shape[:2]
    border = max(4, int(round(min(width, height) * 0.04)))
    samples = np.concatenate(
        [
            image[:border, :, :].reshape(-1, 3),
            image[-border:, :, :].reshape(-1, 3),
            image[:, :border, :].reshape(-1, 3),
            image[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype("float32")
    background = np.median(samples, axis=0)
    diff = np.linalg.norm(image.astype("float32") - background, axis=2)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    background_value = float(np.median(value[:border, :]))
    mask = (
        (diff > 26.0)
        | (value.astype("float32") > background_value + 32.0)
        | ((saturation > 42) & (value > 38))
    ).astype("uint8") * 255
    close_size = max(17, (min(width, height) // 32) | 1)
    open_size = max(5, (min(width, height) // 140) | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    filled = mask.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype="uint8")
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(filled)
    return cv2.bitwise_or(mask, holes)


def contour_card_crop(image: Any, padding: float, target_aspect: float, aspect_tolerance: float) -> dict[str, Any] | None:
    height, width = image.shape[:2]
    candidates: list[dict[str, Any]] = []

    def add_contour_candidates(contours: list[Any], source: str) -> None:
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / max(1.0, float(width * height))
            if area_ratio < 0.08 or area_ratio > 0.88:
                continue
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (rect_width, rect_height), _angle = rect
            long_side = max(rect_width, rect_height)
            short_side = min(rect_width, rect_height)
            if short_side <= 1 or long_side <= 1:
                continue
            aspect = short_side / long_side
            if not 0.48 <= aspect <= 0.90:
                continue
            polygon, polygon_method = polygon_from_contour(contour)
            center_penalty = abs((center_x / max(1, width)) - 0.5) + abs((center_y / max(1, height)) - 0.5)
            aspect_penalty = abs(aspect - target_aspect) / max(0.01, target_aspect)
            source_bonus = 0.04 if source == "foreground" else 0.0
            method_bonus = 0.05 if polygon_method == "quad" else 0.0
            score = area_ratio + source_bonus + method_bonus - (0.08 * center_penalty) - (0.04 * aspect_penalty)
            candidates.append(
                {
                    "polygon": polygon,
                    "polygon_method": polygon_method,
                    "source": source,
                    "score": score,
                    "area_ratio": area_ratio,
                    "aspect": aspect,
                    "center_x": center_x / max(1, width),
                    "center_y": center_y / max(1, height),
                }
            )

    foreground = foreground_card_mask(image)
    foreground_contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_contour_candidates(foreground_contours, "foreground")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_contour_candidates(edge_contours, "edge")

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    effective_padding = max(float(padding), float(min(width, height)) * 0.018)
    crop_image = warp_card_to_target_aspect(image, best["polygon"], effective_padding, target_aspect)
    crop_image, aspect_trimmed = trim_to_aspect(crop_image, target_aspect, aspect_tolerance)
    crop_height, crop_width = crop_image.shape[:2]
    return {
        "status": "contour_card",
        "fallback_used": True,
        "detections": [],
        "crop_width": crop_width,
        "crop_height": crop_height,
        "aspect_trimmed": aspect_trimmed,
        "contour_details": {
            "candidate_count": len(candidates),
            "best": {
                "score": float(best["score"]),
                "area_ratio": float(best["area_ratio"]),
                "aspect": float(best["aspect"]),
                "center_x": float(best["center_x"]),
                "center_y": float(best["center_y"]),
                "source": best["source"],
                "polygon_method": best["polygon_method"],
            },
        },
        "crop_image": crop_image,
    }


def visual_region_rgb(image_rgb: Any) -> Any:
    height, width = image_rgb.shape[:2]
    x1 = max(0, min(width - 1, int(round(width * 0.05))))
    y1 = max(0, min(height - 1, int(round(height * 0.06))))
    x2 = max(x1 + 1, min(width, int(round(width * 0.95))))
    y2 = max(y1 + 1, min(height, int(round(height * 0.94))))
    region = image_rgb[y1:y2, x1:x2]
    return cv2.resize(region, (224, 312), interpolation=cv2.INTER_AREA)


def normalized_vector(vector: Any) -> Any:
    vector = vector.astype("float32").ravel()
    vector /= np.linalg.norm(vector) + 1e-9
    return vector


def hsv_histogram(image_rgb: Any) -> Any:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype("float32")
    hue = hsv[:, :, 0] / 180.0
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    mask = (saturation > 0.12) & (value > 0.12) & (value < 0.98)
    if int(mask.sum()) < 100:
        mask = np.ones(hue.shape, dtype=bool)
    samples = np.stack([hue[mask], saturation[mask], value[mask]], axis=1)
    hist, _ = np.histogramdd(
        samples,
        bins=(24, 6, 6),
        range=((0, 1), (0, 1), (0, 1)),
    )
    return normalized_vector(hist)


def rgb_histogram(image_rgb: Any) -> Any:
    image = image_rgb.astype("float32") / 255.0
    channels = []
    for channel_index in range(3):
        hist, _ = np.histogram(image[:, :, channel_index].ravel(), bins=32, range=(0, 1))
        channels.append(hist.astype("float32"))
    return normalized_vector(np.concatenate(channels))


def gray_vector(image_rgb: Any) -> Any:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (32, 44), interpolation=cv2.INTER_AREA).astype("float32")
    gray -= float(gray.mean())
    return normalized_vector(gray)


def edge_vector(image_rgb: Any) -> Any:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 160)
    edges = cv2.resize(edges, (32, 44), interpolation=cv2.INTER_AREA).astype("float32")
    return normalized_vector(edges)


def dhash_bits(image_rgb: Any) -> int:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten().tolist():
        value = (value << 1) | int(bool(bit))
    return value


def colorfulness(image_rgb: Any) -> float:
    image = image_rgb.astype("float32")
    red_green = np.abs(image[:, :, 0] - image[:, :, 1])
    yellow_blue = np.abs(0.5 * (image[:, :, 0] + image[:, :, 1]) - image[:, :, 2])
    return float(
        np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2)
        + 0.3 * np.sqrt(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
    )


def visual_signature_from_rgb(image_rgb: Any) -> dict[str, Any]:
    region = visual_region_rgb(image_rgb)
    height = region.shape[0]
    region_rows = [
        region[: height // 3],
        region[height // 3 : 2 * height // 3],
        region[2 * height // 3 :],
    ]
    return {
        "hsv_histogram": hsv_histogram(region),
        "rgb_histogram": rgb_histogram(region),
        "region_hsv_histograms": [hsv_histogram(row) for row in region_rows],
        "gray_vector": gray_vector(region),
        "edge_vector": edge_vector(region),
        "dhash": dhash_bits(region),
        "colorfulness": colorfulness(region),
    }


def visual_signature_from_bgr(image_bgr: Any) -> dict[str, Any]:
    return visual_signature_from_rgb(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def visual_signature_from_path(path: Path) -> dict[str, Any] | None:
    try:
        stat = path.stat()
        cache_key = str(path.resolve())
        cached = VISUAL_SIGNATURE_CACHE.get(cache_key)
        if cached and cached[0] == stat.st_size and cached[1] == stat.st_mtime_ns:
            return cached[2]

        from PIL import Image

        with Image.open(path) as image:
            signature = visual_signature_from_rgb(np.asarray(image.convert("RGB")))
        VISUAL_SIGNATURE_CACHE[cache_key] = (stat.st_size, stat.st_mtime_ns, signature)
        return signature
    except Exception:
        return None


def hamming_similarity(left: int, right: int) -> float:
    return 1.0 - ((left ^ right).bit_count() / 64.0)


def visual_similarity(
    query_signature: dict[str, Any],
    candidate_signature: dict[str, Any],
    color_weight: float = DEFAULT_VISUAL_COLOR_WEIGHT,
) -> dict[str, float]:
    hsv_score = float(np.dot(query_signature["hsv_histogram"], candidate_signature["hsv_histogram"]))
    rgb_score = float(np.dot(query_signature["rgb_histogram"], candidate_signature["rgb_histogram"]))
    region_scores = [
        float(np.dot(query_row, candidate_row))
        for query_row, candidate_row in zip(
            query_signature["region_hsv_histograms"],
            candidate_signature["region_hsv_histograms"],
            strict=False,
        )
    ]
    region_hsv_score = float(sum(region_scores) / max(1, len(region_scores)))
    gray_score = float(np.dot(query_signature["gray_vector"], candidate_signature["gray_vector"]))
    edge_score = float(np.dot(query_signature["edge_vector"], candidate_signature["edge_vector"]))
    dhash_score = hamming_similarity(query_signature["dhash"], candidate_signature["dhash"])
    colorfulness_score = max(0.0, 1.0 - abs(query_signature["colorfulness"] - candidate_signature["colorfulness"]) / 100.0)
    color_score = (
        (0.45 * hsv_score)
        + (0.25 * rgb_score)
        + (0.25 * region_hsv_score)
        + (0.05 * colorfulness_score)
    )
    structure_score = (0.45 * gray_score) + (0.30 * edge_score) + (0.25 * dhash_score)
    color_weight = max(0.0, min(1.0, color_weight))
    visual_score = (color_weight * color_score) + ((1.0 - color_weight) * structure_score)
    return {
        "visual_score": visual_score,
        "visual_color_score": color_score,
        "visual_structure_score": structure_score,
        "visual_hsv_score": hsv_score,
        "visual_rgb_score": rgb_score,
        "visual_region_hsv_score": region_hsv_score,
        "visual_gray_score": gray_score,
        "visual_edge_score": edge_score,
        "visual_dhash_score": dhash_score,
        "visual_colorfulness_score": colorfulness_score,
    }


def rerank_by_visual_similarity(
    query_image: Any,
    results: list[dict[str, Any]],
    top_k: int,
    weight: float,
) -> list[dict[str, Any]]:
    weight = max(0.0, min(1.0, weight))
    query_signature = visual_signature_from_bgr(query_image)
    reranked: list[dict[str, Any]] = []
    for result in results:
        item = dict(result)
        item["embedding_score"] = result.get("score")
        item["original_rank"] = result.get("rank")
        local_path = existing_local_image_path(result.get("local_image_path"))
        candidate_signature = visual_signature_from_path(local_path) if local_path else None
        if candidate_signature is not None:
            visual = visual_similarity(query_signature, candidate_signature)
            item.update(visual)
            item["score"] = ((1.0 - weight) * float(result.get("score") or 0.0)) + (weight * visual["visual_score"])
            item["visual_rerank_applied"] = True
        else:
            item["visual_rerank_applied"] = False
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(reranked[:top_k], start=1):
        item["rank"] = rank
    return reranked[:top_k]


def rerank_by_siglip_similarity(
    query_image: Any,
    results: list[dict[str, Any]],
    top_k: int,
    weight: float,
) -> list[dict[str, Any]]:
    weight = max(0.0, min(1.0, weight))
    query_vector = service.encode_siglip_bgr(query_image)
    path_by_result_index: dict[int, Path] = {}
    unique_paths: dict[str, Path] = {}
    for index, result in enumerate(results):
        local_path = existing_local_image_path(result.get("local_image_path"))
        if local_path:
            resolved = local_path.resolve()
            path_by_result_index[index] = resolved
            unique_paths[str(resolved)] = resolved

    reference_vectors = service.encode_siglip_paths(list(unique_paths.values()))
    raw_scores: list[float | None] = []
    for index, _result in enumerate(results):
        path = path_by_result_index.get(index)
        reference_vector = reference_vectors.get(str(path)) if path else None
        raw_scores.append(float(np.dot(query_vector, reference_vector)) if reference_vector is not None else None)

    available_scores = [score for score in raw_scores if score is not None]
    low = min(available_scores) if available_scores else 0.0
    high = max(available_scores) if available_scores else 0.0
    span = high - low

    reranked: list[dict[str, Any]] = []
    for result, siglip_score in zip(results, raw_scores, strict=False):
        item = dict(result)
        embedding_score = float(result.get("score") or 0.0)
        siglip_norm = ((siglip_score - low) / span) if siglip_score is not None and span > 1e-9 else 0.0
        item["embedding_score"] = embedding_score
        item["original_rank"] = result.get("rank")
        item["siglip_score"] = siglip_score
        item["siglip_norm"] = siglip_norm
        item["siglip_rerank_applied"] = siglip_score is not None
        item["rerank_model"] = "dinov2_siglip"
        item["rerank_weight"] = weight
        item["score"] = ((1.0 - weight) * embedding_score) + (weight * siglip_norm)
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(reranked[:top_k], start=1):
        item["rank"] = rank
    return reranked[:top_k]


def fixed_inner_card_crop(
    image: Any,
    left: float = 0.305,
    top: float = 0.2925,
    right: float = 0.695,
    bottom: float = 0.8375,
) -> Any:
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, int(round(width * left))))
    y1 = max(0, min(height - 1, int(round(height * top))))
    x2 = max(x1 + 1, min(width, int(round(width * right))))
    y2 = max(y1 + 1, min(height, int(round(height * bottom))))
    return image[y1:y2, x1:x2]


@app.on_event("startup")
def startup() -> None:
    if PRELOAD:
        service.preload()


def api_info() -> dict[str, Any]:
    return {
        "name": "TCG Card Recognition API",
        "version": APP_VERSION,
        "frontend": "/",
        "endpoints": ["/api", "/health", "/warmup", "/recognize"],
        "reference_image_route": REFERENCE_IMAGE_ROUTE,
    }


@app.get("/", include_in_schema=False)
def frontend() -> Any:
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return api_info()


@app.get("/api")
def api_root() -> dict[str, Any]:
    return api_info()


@app.get("/health")
def health() -> dict[str, Any]:
    configured = configured_indexes()
    return {
        "ok": True,
        "version": APP_VERSION,
        "preload": PRELOAD,
        "preload_siglip": PRELOAD_SIGLIP,
        "preload_crop_model": PRELOAD_CROP_MODEL,
        "crop_model_loaded": service.crop_model is not None,
        "embedding_model_loaded": service.embedding_model is not None,
        "siglip_model_loaded": service.siglip_model is not None,
        "siglip_model": DEFAULT_SIGLIP_MODEL,
        "siglip_cache_size": len(service.siglip_embedding_cache),
        "default_rerank_model": DEFAULT_RERANK_MODEL,
        "default_rerank_weight": DEFAULT_VISUAL_RERANK_WEIGHT,
        "default_rerank_candidates": DEFAULT_VISUAL_RERANK_CANDIDATES,
        "default_card_code_ocr": DEFAULT_CARD_CODE_OCR,
        "card_code_ocr_script_exists": (ROOT / "scripts/quality/ocr_vision_text.swift").exists(),
        "card_code_ocr_exact_boost": DEFAULT_CARD_CODE_OCR_EXACT_BOOST,
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
        "reference_images": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "route": f"{REFERENCE_IMAGE_ROUTE}/{name}",
            }
            for name, path in REFERENCE_IMAGE_ROOTS.items()
        },
        "local_path_rewrites": [
            {"source": str(source), "target": str(target)}
            for source, target in LOCAL_PATH_REWRITES
        ],
        "device": service.device or DEFAULT_DEVICE or "auto",
        "siglip_device": service.siglip_device or DEFAULT_DEVICE or "auto",
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
        "siglip_device": service.siglip_device,
    }


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    crop: bool = True,
    crop_mode: str = "auto",
    fallback_to_original: bool = False,
    top_k: int = 5,
    per_index_top_k: int = 5,
    visual_rerank: bool = False,
    visual_rerank_candidates: int = DEFAULT_VISUAL_RERANK_CANDIDATES,
    visual_rerank_weight: float = DEFAULT_VISUAL_RERANK_WEIGHT,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    card_code_ocr: bool = DEFAULT_CARD_CODE_OCR,
    card_code_ocr_boost: float = DEFAULT_CARD_CODE_OCR_EXACT_BOOST,
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
    debug_crop_image = None
    normalized_crop_mode = crop_mode.strip().lower().replace("-", "_")
    if not crop:
        normalized_crop_mode = "none"

    if normalized_crop_mode == "auto":
        crop_start = time.perf_counter()
        crop_payload = slab_inner_card_crop(image, target_aspect)
        if crop_payload is None:
            crop_payload = contour_card_crop(
                image=image,
                padding=padding,
                target_aspect=target_aspect,
                aspect_tolerance=aspect_tolerance,
            )
            if contour_crop_looks_like_graded_slab_miss(crop_payload):
                graded_slab_payload = generic_graded_slab_crop(image)
                if graded_slab_payload is not None:
                    graded_slab_payload["replaced_crop"] = crop_payload.get("status")
                    graded_slab_payload["replaced_crop_details"] = crop_payload.get("contour_details")
                    crop_payload = graded_slab_payload
            if crop_payload is None:
                crop_payload = service.crop(
                    image=image,
                    confidence=confidence,
                    imgsz=imgsz,
                    padding=padding,
                    target_aspect=target_aspect,
                    aspect_tolerance=aspect_tolerance,
                )
                if yolo_crop_looks_like_label_detection(crop_payload, input_height):
                    graded_slab_payload = generic_graded_slab_crop(image)
                    if graded_slab_payload is not None:
                        graded_slab_payload["replaced_crop"] = crop_payload.get("status")
                        graded_slab_payload["replaced_crop_details"] = crop_payload.get("selected_detection")
                        crop_payload = graded_slab_payload
        timings["crop_seconds"] = time.perf_counter() - crop_start
        debug_crop_image = crop_payload.get("crop_image")
        if crop_payload["status"] in {"cropped", "slab_inner_card", "contour_card", "graded_slab_ratio_card"}:
            query_image = debug_crop_image
        elif fallback_to_original:
            crop_payload["fallback_used"] = True
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
    elif normalized_crop_mode == "fixed_inner_card":
        crop_start = time.perf_counter()
        debug_crop_image = fixed_inner_card_crop(image)
        timings["crop_seconds"] = time.perf_counter() - crop_start
        crop_height, crop_width = debug_crop_image.shape[:2]
        query_image = debug_crop_image
        crop_payload = {
            "status": "fixed_inner_card",
            "fallback_used": False,
            "detections": [],
            "crop_width": crop_width,
            "crop_height": crop_height,
            "fixed_crop_box_ratio": {
                "left": 0.305,
                "top": 0.2925,
                "right": 0.695,
                "bottom": 0.8375,
            },
            "crop_image": debug_crop_image,
        }
    elif normalized_crop_mode not in {"none", "original", "skipped"}:
        raise HTTPException(status_code=400, detail=f"Unsupported crop_mode: {crop_mode}")

    encode_start = time.perf_counter()
    query = service.encode(query_image)
    timings["embedding_seconds"] = time.perf_counter() - encode_start

    search_start = time.perf_counter()
    search_top_k = max(top_k, visual_rerank_candidates) if (visual_rerank or card_code_ocr) else top_k
    search_per_index_top_k = max(per_index_top_k, search_top_k) if (visual_rerank or card_code_ocr) else per_index_top_k
    combined, per_index = service.search(query, per_index_top_k=search_per_index_top_k, combined_top_k=search_top_k)
    timings["search_seconds"] = time.perf_counter() - search_start

    if visual_rerank:
        rerank_start = time.perf_counter()
        normalized_rerank_model = rerank_model.strip().lower().replace("-", "_")
        if normalized_rerank_model in {"", "siglip", "dinov2_siglip"}:
            combined = rerank_by_siglip_similarity(
                query_image=query_image,
                results=combined,
                top_k=search_top_k,
                weight=visual_rerank_weight,
            )
        elif normalized_rerank_model in {"composite", "visual"}:
            combined = rerank_by_visual_similarity(
                query_image=query_image,
                results=combined,
                top_k=search_top_k,
                weight=visual_rerank_weight,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported rerank_model: {rerank_model}")
        timings["rerank_seconds"] = time.perf_counter() - rerank_start
        timings["visual_rerank_seconds"] = timings["rerank_seconds"]

    ocr_payload: dict[str, Any] = {
        "enabled": card_code_ocr,
        "status": "skipped",
        "boost": 0.0,
    }
    if card_code_ocr:
        ocr_start = time.perf_counter()
        crop_status = crop_payload.get("status")
        if crop_status in {"cropped", "fixed_inner_card", "slab_inner_card", "contour_card", "graded_slab_ratio_card"}:
            ocr_payload = recognize_card_bottom_code(query_image)
            ocr_payload["enabled"] = True
            ocr_payload["boost"] = card_code_ocr_boost
            combined = apply_card_code_ocr_boost(
                combined,
                ocr_payload,
                boost=max(0.0, card_code_ocr_boost),
            )
        else:
            ocr_payload = {
                "enabled": True,
                "status": "skipped",
                "reason": f"card crop status is {crop_status}",
                "boost": 0.0,
            }
        timings["ocr_seconds"] = time.perf_counter() - ocr_start

    combined = combined[:top_k]
    for rank, item in enumerate(combined, start=1):
        item["rank"] = rank

    timings["total_seconds"] = time.perf_counter() - total_start

    if include_debug_crop_base64 and debug_crop_image is not None:
        crop_payload["debug_crop_jpeg_base64"] = encode_debug_crop(debug_crop_image)
    crop_payload.pop("crop_image", None)

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
        "visual_rerank": {
            "enabled": visual_rerank,
            "model": rerank_model if visual_rerank else None,
            "candidates": search_top_k if visual_rerank else 0,
            "weight": visual_rerank_weight if visual_rerank else 0,
        },
        "card_code_ocr": ocr_payload,
        "timings": timings,
    }
