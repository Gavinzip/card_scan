#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

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
    "pokemon_en=data/processed/image_index_base,"
    "pokemon_ja=data/processed/pokemon_ja_canonical_image_index_base,"
    "onepiece=data/processed/onepiece_image_index_base"
)

INDEX_CONFIG = os.environ.get("CARD_SCAN_INDEXES", DEFAULT_INDEXES)
CROP_MODEL_PATH = Path(os.environ.get("CARD_SCAN_CROP_MODEL_PATH", ROOT / "data/models/production/current/cardcaptor_v3_best.pt"))
CROP_MODEL_REPO = os.environ.get("CARD_SCAN_CROP_MODEL_REPO", DEFAULT_REPO_ID)
CROP_MODEL_FILE = os.environ.get("CARD_SCAN_CROP_MODEL_FILE", DEFAULT_MODEL_FILE)
DEFAULT_CONFIDENCE = float(os.environ.get("CARD_SCAN_CROP_CONFIDENCE", "0.25"))
DEFAULT_IMGSZ = int(os.environ.get("CARD_SCAN_CROP_IMGSZ", "1024"))
DEFAULT_PADDING = float(os.environ.get("CARD_SCAN_CROP_PADDING", "0"))
DEFAULT_TARGET_ASPECT = float(os.environ.get("CARD_SCAN_CROP_TARGET_ASPECT", str(63 / 88)))
DEFAULT_ASPECT_TOLERANCE = float(os.environ.get("CARD_SCAN_CROP_ASPECT_TOLERANCE", "0.05"))
DEFAULT_DEVICE = os.environ.get("CARD_SCAN_DEVICE") or None
DEFAULT_RERANK_MODEL = os.environ.get("CARD_SCAN_RERANK_MODEL", "siglip").strip().lower()
DEFAULT_VISUAL_RERANK_CANDIDATES = int(os.environ.get("CARD_SCAN_VISUAL_RERANK_CANDIDATES", "5"))
DEFAULT_VISUAL_RERANK_WEIGHT = float(os.environ.get("CARD_SCAN_VISUAL_RERANK_WEIGHT", "0.40"))
DEFAULT_VISUAL_COLOR_WEIGHT = float(os.environ.get("CARD_SCAN_VISUAL_COLOR_WEIGHT", "0.50"))
DEFAULT_SIGLIP_MODEL = os.environ.get("CARD_SCAN_SIGLIP_MODEL", "vit_base_patch16_siglip_224.webli")
DEFAULT_SIGLIP_BATCH_SIZE = int(os.environ.get("CARD_SCAN_SIGLIP_BATCH_SIZE", "32"))
DEFAULT_CARD_CODE_OCR = os.environ.get("CARD_SCAN_CARD_CODE_OCR", "false").lower() in {"1", "true", "yes"}
DEFAULT_CARD_CODE_OCR_TIMEOUT = float(os.environ.get("CARD_SCAN_CARD_CODE_OCR_TIMEOUT", "8"))
DEFAULT_CARD_CODE_OCR_EXACT_BOOST = float(os.environ.get("CARD_SCAN_CARD_CODE_OCR_EXACT_BOOST", "0.09"))
DEFAULT_LANGUAGE_RERANK = os.environ.get("CARD_SCAN_LANGUAGE_RERANK", "true").lower() in {"1", "true", "yes"}
DEFAULT_LANGUAGE_RERANK_BOOST = float(os.environ.get("CARD_SCAN_LANGUAGE_RERANK_BOOST", "0.08"))
DEFAULT_LANGUAGE_RERANK_CANDIDATES = int(os.environ.get("CARD_SCAN_LANGUAGE_RERANK_CANDIDATES", "25"))
DEFAULT_LANGUAGE_OCR = os.environ.get("CARD_SCAN_LANGUAGE_OCR", "false").lower() in {"1", "true", "yes"}
DEFAULT_LANGUAGE_OCR_ENGINE = os.environ.get("CARD_SCAN_LANGUAGE_OCR_ENGINE", "rapidocr").strip().lower()
DEFAULT_LANGUAGE_OCR_TIMEOUT = float(os.environ.get("CARD_SCAN_LANGUAGE_OCR_TIMEOUT", "6"))
DEFAULT_HINT_CARD_CODE_BOOST = float(os.environ.get("CARD_SCAN_HINT_CARD_CODE_BOOST", "0.10"))
DEFAULT_SLAB_BARCODE_LOOKUP = os.environ.get("CARD_SCAN_SLAB_BARCODE_LOOKUP", "true").lower() in {"1", "true", "yes"}
DEFAULT_SLAB_BARCODE_SCAN_ALWAYS = os.environ.get("CARD_SCAN_SLAB_BARCODE_SCAN_ALWAYS", "false").lower() in {"1", "true", "yes"}
DEFAULT_SLAB_LABEL_OCR = os.environ.get("CARD_SCAN_SLAB_LABEL_OCR", "true").lower() in {"1", "true", "yes"}
DEFAULT_SLAB_LABEL_OCR_ENGINE = os.environ.get("CARD_SCAN_SLAB_LABEL_OCR_ENGINE", "rapidocr").strip().lower()
SLAB_CERT_LOOKUP_PATH = os.environ.get("CARD_SCAN_SLAB_CERT_LOOKUP_PATH", "").strip()
SLAB_CERT_API_TIMEOUT = float(os.environ.get("CARD_SCAN_SLAB_CERT_API_TIMEOUT", "4"))
SLAB_CERT_LOOKUP_PROVIDERS = [
    provider.strip().lower()
    for provider in os.environ.get("CARD_SCAN_SLAB_CERT_LOOKUP_PROVIDERS", "").split(",")
    if provider.strip()
]
SLAB_CERT_API_LOOKUP_ENABLED = os.environ.get("CARD_SCAN_SLAB_CERT_API_LOOKUP", "false").lower() in {
    "1",
    "true",
    "yes",
}
SLAB_LABEL_SET_ALIASES_PATH = Path(
    os.environ.get("CARD_SCAN_SLAB_LABEL_SET_ALIASES", ROOT / "data/config/slab_label_set_aliases.csv")
)
SEREBII_SET_NAME_MAP_PATH = ROOT / "data/config/serebii_pokemon_ja_set_name_map.csv"
SLAB_CATALOG_LOOKUP_PATHS = [
    Path(path.strip())
    for path in os.environ.get(
        "CARD_SCAN_SLAB_CATALOG_LOOKUP_PATHS",
        str(ROOT / "data/processed/pokemon_ja_canonical_catalog.jsonl"),
    ).split(",")
    if path.strip()
]
PSA_PUBLIC_API_TOKEN = os.environ.get("CARD_SCAN_PSA_API_TOKEN", "").strip()
CGC_DEALER_API_TOKEN = os.environ.get("CARD_SCAN_CGC_DEALER_API_TOKEN", "").strip()
DEFAULT_AMBIGUOUS_CANDIDATE_SCORE_MARGIN = float(os.environ.get("CARD_SCAN_AMBIGUOUS_CANDIDATE_SCORE_MARGIN", "0.04"))
DEFAULT_AMBIGUOUS_CANDIDATE_MIN_SCORE = float(os.environ.get("CARD_SCAN_AMBIGUOUS_CANDIDATE_MIN_SCORE", "0.70"))
DEFAULT_AMBIGUOUS_CANDIDATE_LIMIT = int(os.environ.get("CARD_SCAN_AMBIGUOUS_CANDIDATE_LIMIT", "6"))
PRELOAD = os.environ.get("CARD_SCAN_PRELOAD", "false").lower() in {"1", "true", "yes"}
PRELOAD_SIGLIP = os.environ.get("CARD_SCAN_PRELOAD_SIGLIP", "false").lower() in {"1", "true", "yes"}
PRELOAD_CROP_MODEL = os.environ.get("CARD_SCAN_PRELOAD_CROP_MODEL", "true").lower() in {"1", "true", "yes"}
REFERENCE_IMAGE_ROUTE = os.environ.get("CARD_SCAN_REFERENCE_IMAGE_ROUTE", "/reference-images").rstrip("/") or "/reference-images"
REFERENCE_IMAGE_ROOTS_CONFIG = os.environ.get("CARD_SCAN_IMAGE_ROOTS", "")
LOCAL_PATH_REWRITES_CONFIG = os.environ.get("CARD_SCAN_LOCAL_PATH_REWRITES", "")
SNKR_MAPPING_FLAGS_PATH = Path(
    os.environ.get("CARD_SCAN_SNKR_MAPPING_FLAGS", ROOT / "data/config/snkr_product_mapping_flags.csv")
)
SUPPRESS_FLAGGED_SNKR_MAPPINGS = os.environ.get(
    "CARD_SCAN_SUPPRESS_FLAGGED_SNKR_MAPPINGS",
    "true",
).lower() in {"1", "true", "yes"}
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CARD_SCAN_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

cv2, np, YOLO = require_deps()

app = FastAPI(title="TCG Card Recognition API", version=APP_VERSION)
VISUAL_SIGNATURE_CACHE: dict[str, tuple[int, int, dict[str, Any]]] = {}
OCR_ENGINE_CACHE: dict[str, Any] = {}
BARCODE_ENGINE_CACHE: dict[str, Any] = {}
SEGMENTATION_ENGINE_CACHE: dict[str, Any] = {}
SLAB_CERT_LOOKUP_CACHE: dict[str, Any] = {"path": None, "mtime_ns": None, "rows": {}}
SLAB_LABEL_SET_ALIAS_CACHE: dict[str, Any] = {"paths": None, "mtime_ns": None, "rows": []}
SLAB_CATALOG_LOOKUP_CACHE: dict[str, Any] = {"paths": None, "mtime_ns": None, "records": []}
SNKR_MAPPING_FLAGS: dict[str, dict[str, Any]] = {}
SET_PATTERN = (
    r"(?:PROMO\s*[_-]?\s*SWSH|SWSH|WCS\s*\d{2,}|"
    r"SV\s*[-]?\s*P|S\s*[-]?\s*P|SM\s*[-]?\s*P|XY\s*[-]?\s*P|M\s*[-]?\s*P|"
    r"(?:SV|S|SM|XY|BW|DP|M|ME)\s*\d{1,2}[A-Z]?)"
)
CARD_CODE_PATTERN = re.compile(
    rf"\b(?P<set>{SET_PATTERN})\s+(?P<number>[A-Z]?\d{{1,3}}|[A-Z]{{2}}\d{{1,3}})\s*/\s*(?P<total>\d{{1,3}})\b",
    re.I,
)
TRAILING_SET_CARD_CODE_PATTERN = re.compile(
    rf"\b(?P<number>[A-Z]?\d{{1,3}}|[A-Z]{{2}}\d{{1,3}})\s*/\s*(?P<total>\d{{1,3}})\s+(?P<set>{SET_PATTERN})\b",
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
JAPANESE_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
JAPANESE_SET_CODE_RE = re.compile(r"\b(?:SV|S|SM|XY|BW|DP|M|L)\d{1,2}[a-z]\b", re.I)
ENGLISH_SET_CODE_RE = re.compile(r"\b(?:SWSH\d{1,2}|SV\d{1,2}|SM\d{1,2}|XY\d{1,2}|BW\d{1,2}|DP\d{1,2})\b", re.I)
EXPLICIT_JAPANESE_RE = re.compile(r"POKEMON\s*(?:JAPANESE|JPN|JP)|POKEMONJPN|\b(?:JAPANESE|JPN|JP)\b|日版|日本語|日文", re.I)
EXPLICIT_ENGLISH_RE = re.compile(r"\b(?:ENGLISH|ENG)\b|英版|英文", re.I)
LATIN_TEXT_RE = re.compile(r"[A-Za-z]{3,}")

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


def load_snkr_mapping_flags(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    flags: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            product_id = str(row.get("snkr_product_id") or "").strip()
            if not product_id:
                continue
            flags[product_id] = {
                "severity": row.get("severity") or "suspicious",
                "card_id": row.get("card_id"),
                "set_id": row.get("set_id"),
                "card_code": row.get("card_code"),
                "catalog_name": row.get("catalog_name"),
                "snkr_product_name": row.get("snkr_product_name"),
                "match_reasons": row.get("match_reasons"),
                "audit_source": str(path),
            }
    return flags


def safe_url_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


REFERENCE_IMAGE_ROOTS = parse_named_paths(REFERENCE_IMAGE_ROOTS_CONFIG)
LOCAL_PATH_REWRITES = parse_path_rewrites(LOCAL_PATH_REWRITES_CONFIG)
SNKR_MAPPING_FLAGS = load_snkr_mapping_flags(SNKR_MAPPING_FLAGS_PATH)

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


def normalize_game_family(value: Any) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
    if not normalized:
        return None
    if "onepiece" in normalized or "optcg" in normalized or "opcg" in normalized:
        return "onepiece"
    if "pokemon" in normalized:
        return "pokemon"
    return None


def infer_index_family(index_name: Any) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "", str(index_name or "").casefold())
    if not normalized:
        return None
    if "onepiece" in normalized:
        return "onepiece"
    if normalized in {"pokemonen", "pokemonja"} or normalized.startswith("pokemon"):
        return "pokemon"
    return None


def candidate_family(candidate: dict[str, Any]) -> str | None:
    for key in ("game_family", "candidate_family", "game"):
        family = normalize_game_family(candidate.get(key))
        if family:
            return family
    family = infer_index_family(candidate.get("index"))
    if family:
        return family
    for key in ("source", "canonical_source", "image_source", "metadata_source"):
        family = normalize_game_family(candidate.get(key))
        if family:
            return family
    return None


def is_pokemon_candidate(candidate: dict[str, Any]) -> bool:
    return candidate_family(candidate) == "pokemon"


def format_result(index_name: str, rank: int, score: float, record: dict[str, Any]) -> dict[str, Any]:
    remote_image_url = record.get("image_url")
    local_reference_image_url = reference_image_url(record.get("local_image_path"))
    raw_snkr_product_id = record.get("snkr_product_id")
    snkr_mapping_flag = SNKR_MAPPING_FLAGS.get(str(raw_snkr_product_id)) if raw_snkr_product_id else None
    snkr_mapping_suppressed = bool(snkr_mapping_flag and SUPPRESS_FLAGGED_SNKR_MAPPINGS)
    snkr_product_id = None if snkr_mapping_suppressed else raw_snkr_product_id
    game_family = candidate_family({**record, "index": index_name})
    return {
        "index": index_name,
        "rank": rank,
        "score": score,
        "game_family": game_family,
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
        "canonical_key": record.get("canonical_key"),
        "set_id": record.get("set_id"),
        "card_code": record.get("card_code"),
        "language": record.get("language"),
        "name": record.get("name"),
        "name_en": record.get("name_en"),
        "name_ja": record.get("name_ja"),
        "rarity": record.get("rarity"),
        "variant": record.get("variant"),
        "edition": record.get("edition"),
        "variant_source": record.get("variant_source"),
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
            "match_status": "suppressed_suspicious_mapping" if snkr_mapping_suppressed else record.get("snkr_match_status"),
            "product_id": snkr_product_id,
            "has_product_id": bool(snkr_product_id),
            "product_name": None if snkr_mapping_suppressed else record.get("snkr_product_name"),
            "url": None if snkr_mapping_suppressed else record.get("snkr_url"),
            "min_price": None if snkr_mapping_suppressed else record.get("snkr_min_price"),
            "min_price_format": None if snkr_mapping_suppressed else record.get("snkr_min_price_format"),
            "verified_candidate_count": None if snkr_mapping_suppressed else record.get("snkr_verified_candidate_count"),
            "matched_at": record.get("snkr_matched_at"),
            "mapping_flag": snkr_mapping_flag,
            "mapping_status": snkr_mapping_flag.get("severity") if snkr_mapping_flag else None,
            "mapping_suppressed": snkr_mapping_suppressed,
            "suppressed_product_id": raw_snkr_product_id if snkr_mapping_suppressed else None,
            "suppressed_product_name": record.get("snkr_product_name") if snkr_mapping_suppressed else None,
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
    promo_aliases = {
        "BWP": "BWP",
        "BW-P": "BWP",
        "DPP": "DPP",
        "DP-P": "DPP",
        "XYP": "XYP",
        "XY-P": "XYP",
        "SMP": "SMP",
        "SM-P": "SMP",
    }
    if text in promo_aliases:
        return promo_aliases[text]
    if text in {"SVP", "SV-P"}:
        return "SV-P"
    if text in {"SWSH", "PROMOSWSH", "PROMO-SWSH"}:
        return "PROMO-SWSH"
    return text


def extract_card_code(text: str) -> dict[str, Any] | None:
    normalized = normalize_ocr_text(text)
    patterns = [
        ("strict", CARD_CODE_PATTERN),
        ("trailing_set", TRAILING_SET_CARD_CODE_PATTERN),
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


def normalize_language(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower().replace("_", "-")
    aliases = {
        "ja": "ja",
        "jp": "ja",
        "jpn": "ja",
        "japanese": "ja",
        "日本語": "ja",
        "日文": "ja",
        "日版": "ja",
        "en": "en",
        "eng": "en",
        "english": "en",
        "英文": "en",
        "英版": "en",
    }
    return aliases.get(text)


def infer_language_from_text(text: str | None, source: str) -> dict[str, Any]:
    raw_text = (text or "").strip()
    payload: dict[str, Any] = {
        "source": source,
        "text": raw_text,
        "language": None,
        "confidence": 0.0,
        "signals": [],
    }
    if not raw_text:
        payload["status"] = "empty"
        return payload

    direct = normalize_language(raw_text)
    if direct:
        payload.update(
            {
                "status": "ok",
                "language": direct,
                "confidence": 1.0,
                "signals": [f"explicit:{direct}"],
            }
        )
        return payload

    signals: list[tuple[str, str, float]] = []
    if JAPANESE_SCRIPT_RE.search(raw_text):
        signals.append(("ja", "japanese_script", 0.92))
    if EXPLICIT_JAPANESE_RE.search(raw_text):
        signals.append(("ja", "explicit_japanese", 0.98))
    if JAPANESE_SET_CODE_RE.search(raw_text):
        signals.append(("ja", "japanese_set_code", 0.94))
    if EXPLICIT_ENGLISH_RE.search(raw_text):
        signals.append(("en", "explicit_english", 0.96))
    if ENGLISH_SET_CODE_RE.search(raw_text) and not JAPANESE_SET_CODE_RE.search(raw_text):
        signals.append(("en", "english_set_code", 0.80))

    if not signals:
        payload["status"] = "not_found"
        return payload

    scores: dict[str, float] = {}
    for language, signal, confidence in signals:
        scores[language] = max(scores.get(language, 0.0), confidence)
        payload["signals"].append(signal)

    # Japanese-specific set codes and script should win over generic English words in listing titles.
    if scores.get("ja", 0.0) >= scores.get("en", 0.0):
        language = "ja"
    else:
        language = "en"
    payload.update(
        {
            "status": "ok",
            "language": language,
            "confidence": scores[language],
        }
    )
    return payload


def candidate_language(candidate: dict[str, Any]) -> str | None:
    language = normalize_language(candidate.get("language"))
    if language:
        return language
    index_name = str(candidate.get("index") or "").lower()
    if index_name.endswith("_ja") or "pokemon_ja" in index_name:
        return "ja"
    if index_name.endswith("_en") or "pokemon_en" in index_name:
        return "en"
    return None


def apply_language_rerank(results: list[dict[str, Any]], language_payload: dict[str, Any], boost: float) -> list[dict[str, Any]]:
    target = normalize_language(language_payload.get("language"))
    if language_payload.get("status") != "ok" or target not in {"ja", "en"} or boost <= 0:
        return results

    primary_family = candidate_family(results[0]) if results else None
    language_payload["primary_candidate_family"] = primary_family
    if primary_family and primary_family != "pokemon":
        language_payload["boost_skipped"] = f"primary_candidate_family:{primary_family}"
        skipped: list[dict[str, Any]] = []
        for result in results:
            item = dict(result)
            item["language_hint"] = target
            item["language_hint_match"] = False
            item["language_hint_boost"] = 0.0
            item["language_hint_skipped"] = f"primary_candidate_family:{primary_family}"
            skipped.append(item)
        return skipped

    reranked: list[dict[str, Any]] = []
    for result in results:
        item = dict(result)
        if not is_pokemon_candidate(item):
            item["language_hint"] = target
            item["language_hint_match"] = False
            item["language_hint_boost"] = 0.0
            item["language_hint_skipped"] = "non_pokemon_candidate"
            reranked.append(item)
            continue
        language = candidate_language(item)
        match = language == target
        item["language_hint"] = target
        item["language_hint_match"] = match
        item["language_hint_boost"] = boost if match else 0.0
        if match:
            item["pre_language_score"] = item.get("score")
            item["score"] = float(item.get("score") or 0.0) + boost
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked


def build_hint_card_code_payload(text: str | None) -> dict[str, Any]:
    parsed = extract_card_code(text or "")
    if not parsed:
        return {
            "enabled": bool((text or "").strip()),
            "status": "not_found" if (text or "").strip() else "skipped",
            "boost": 0.0,
        }
    return {
        "enabled": True,
        "status": "ok",
        "source": "hint_text",
        "boost": DEFAULT_HINT_CARD_CODE_BOOST,
        "best": {
            "text": text,
            "confidence": 1.0,
            "parsed": parsed,
        },
    }


def ocr_image_variants_for_language(card_image: Any) -> list[tuple[str, Any]]:
    height, width = card_image.shape[:2]
    scale = min(1.0, 1200 / max(1, max(height, width)))
    if scale < 1.0:
        resized = cv2.resize(card_image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    else:
        resized = card_image
    return [
        ("full", resized),
        ("top", crop_ratio(resized, (0.00, 0.00, 1.00, 0.42))),
        ("middle", crop_ratio(resized, (0.00, 0.18, 1.00, 0.76))),
    ]


def rapidocr_available() -> bool:
    return bool(importlib.util.find_spec("rapidocr_onnxruntime") or importlib.util.find_spec("rapidocr"))


def get_rapidocr_engine() -> Any:
    if "rapidocr" not in OCR_ENGINE_CACHE:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception:
            from rapidocr import RapidOCR  # type: ignore
        OCR_ENGINE_CACHE["rapidocr"] = RapidOCR()
    return OCR_ENGINE_CACHE["rapidocr"]


def flatten_rapidocr_result(result: Any) -> tuple[str, float]:
    if result is None:
        return "", 0.0
    if isinstance(result, tuple):
        result = result[0]

    texts: list[str] = []
    confidence = 0.0
    for item in result or []:
        text = None
        item_confidence = None
        if isinstance(item, dict):
            text = item.get("text") or item.get("rec_text")
            item_confidence = item.get("confidence") or item.get("rec_score")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            text = item[1]
            if len(item) >= 3:
                item_confidence = item[2]
        if text:
            texts.append(str(text))
        try:
            confidence = max(confidence, float(item_confidence))
        except (TypeError, ValueError):
            pass
    return "\n".join(texts), confidence


def recognize_language_with_rapidocr(card_image: Any) -> dict[str, Any]:
    if not rapidocr_available():
        return {"source": "rapidocr", "status": "unavailable", "reason": "rapidocr_onnxruntime is missing"}

    started = time.perf_counter()
    try:
        engine = get_rapidocr_engine()
    except Exception as exc:  # noqa: BLE001
        return {"source": "rapidocr", "status": "unavailable", "reason": str(exc)}

    texts: list[str] = []
    max_confidence = 0.0
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        for variant_name, variant_image in ocr_image_variants_for_language(card_image):
            path = temp_path / f"language_{variant_name}.png"
            cv2.imwrite(str(path), variant_image)
            try:
                result = engine(str(path))
            except Exception as exc:  # noqa: BLE001
                return {
                    "source": "rapidocr",
                    "status": "error",
                    "seconds": time.perf_counter() - started,
                    "error": f"{variant_name}: {exc}",
                }
            text, confidence = flatten_rapidocr_result(result)
            if text:
                texts.append(text)
            max_confidence = max(max_confidence, confidence)

    text = "\n".join(texts)
    payload = infer_language_from_text(text, source="rapidocr")
    if payload.get("status") == "not_found" and LATIN_TEXT_RE.search(text):
        payload.update({"status": "ok", "language": "en", "confidence": 0.55, "signals": ["latin_text"]})
    payload["seconds"] = time.perf_counter() - started
    payload["ocr_confidence"] = max_confidence
    return payload


def recognize_language_with_vision_ocr(card_image: Any) -> dict[str, Any]:
    swift_script = ROOT / "scripts/quality/ocr_vision_text.swift"
    if not swift_script.exists():
        return {"source": "vision", "status": "unavailable", "reason": "ocr_vision_text.swift is missing"}

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        image_paths: list[Path] = []
        for variant_name, variant_image in ocr_image_variants_for_language(card_image):
            path = temp_path / f"language_{variant_name}.png"
            cv2.imwrite(str(path), variant_image)
            image_paths.append(path)

        command = ["swift", str(swift_script), *[str(path) for path in image_paths]]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=DEFAULT_LANGUAGE_OCR_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {
                "source": "vision",
                "status": "timeout",
                "seconds": time.perf_counter() - started,
                "timeout_seconds": DEFAULT_LANGUAGE_OCR_TIMEOUT,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "source": "vision",
                "status": "error",
                "seconds": time.perf_counter() - started,
                "error": str(exc),
            }

    texts: list[str] = []
    max_confidence = 0.0
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = row.get("text") or ""
        if text:
            texts.append(text)
        max_confidence = max(
            max_confidence,
            max([item.get("confidence") or 0.0 for item in row.get("observations") or []], default=0.0),
        )

    payload = infer_language_from_text("\n".join(texts), source="vision")
    payload["seconds"] = time.perf_counter() - started
    payload["ocr_confidence"] = max_confidence
    return payload


def recognize_language_with_ocr(card_image: Any, engine: str | None = None) -> dict[str, Any]:
    normalized_engine = (engine or DEFAULT_LANGUAGE_OCR_ENGINE or "rapidocr").strip().lower().replace("-", "_")
    if normalized_engine in {"rapidocr", "rapid"}:
        return recognize_language_with_rapidocr(card_image)
    if normalized_engine in {"vision", "macos_vision", "macos"}:
        return recognize_language_with_vision_ocr(card_image)
    if normalized_engine == "auto":
        rapid_payload = recognize_language_with_rapidocr(card_image)
        if rapid_payload.get("status") in {"ok", "not_found"}:
            return rapid_payload
        vision_payload = recognize_language_with_vision_ocr(card_image)
        vision_payload["fallback_from"] = rapid_payload
        return vision_payload
    return {
        "source": normalized_engine,
        "status": "error",
        "reason": f"Unsupported language OCR engine: {engine}",
    }


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
    if shutil.which("swift") is None:
        return {"status": "unavailable", "reason": "swift is not installed"}

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


def apply_card_code_ocr_boost(
    results: list[dict[str, Any]],
    ocr_payload: dict[str, Any],
    boost: float,
    field_prefix: str = "ocr_card_code",
) -> list[dict[str, Any]]:
    best = ocr_payload.get("best") or {}
    parsed = best.get("parsed") or {}
    if ocr_payload.get("status") != "ok" or not parsed:
        return results

    boosted: list[dict[str, Any]] = []
    for result in results:
        item = dict(result)
        if not is_pokemon_candidate(item):
            item[f"{field_prefix}_match"] = False
            item[f"{field_prefix}_boost"] = 0.0
            item[f"{field_prefix}_skipped"] = "non_pokemon_candidate"
            boosted.append(item)
            continue
        match = candidate_matches_ocr_code(item, parsed)
        item[f"{field_prefix}_match"] = match
        item[f"{field_prefix}_boost"] = boost if match else 0.0
        if match:
            item[f"pre_{field_prefix}_score"] = item.get("score")
            item["score"] = float(item.get("score") or 0.0) + boost
        boosted.append(item)

    boosted.sort(key=lambda item: item["score"], reverse=True)
    return boosted


def normalized_candidate_subject(candidate: dict[str, Any]) -> str | None:
    for key in ("name_ja", "name_en", "name"):
        value = candidate.get(key)
        if value:
            text = unicodedata.normalize("NFKC", str(value)).casefold()
            text = re.sub(r"[\s\-_・:：'\"`´’‘“”()\[\]{}<>/\\|.,]+", "", text)
            return text or None
    return None


def candidate_similarity_score(candidate: dict[str, Any]) -> float:
    for key in (
        "pre_ocr_card_code_score",
        "pre_hint_card_code_score",
        "embedding_score",
        "pre_language_score",
        "score",
    ):
        value = candidate.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def candidate_has_exact_code_signal(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("ocr_card_code_match") or candidate.get("hint_card_code_match"))


def build_candidate_selection(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(results) < 2:
        return None

    top = results[0]
    top_subject = normalized_candidate_subject(top)
    top_language = candidate_language(top)
    top_family = candidate_family(top)
    top_score = float(top.get("score") or 0.0)
    top_similarity = candidate_similarity_score(top)
    if top_score < DEFAULT_AMBIGUOUS_CANDIDATE_MIN_SCORE and top_similarity < DEFAULT_AMBIGUOUS_CANDIDATE_MIN_SCORE:
        return None

    selected: list[dict[str, Any]] = [top]
    reasons: set[str] = set()
    for candidate in results[1:]:
        if len(selected) >= DEFAULT_AMBIGUOUS_CANDIDATE_LIMIT:
            break
        if candidate.get("card_id") == top.get("card_id"):
            continue

        subject = normalized_candidate_subject(candidate)
        same_subject = bool(top_subject and subject and top_subject == subject)
        candidate_family_value = candidate_family(candidate)
        same_family = not top_family or not candidate_family_value or top_family == candidate_family_value
        candidate_language_value = candidate_language(candidate)
        same_language = not top_language or not candidate_language_value or top_language == candidate_language_value
        if not same_subject or not same_family or not same_language:
            continue

        score = float(candidate.get("score") or 0.0)
        similarity = candidate_similarity_score(candidate)
        close_score = abs(top_score - score) <= DEFAULT_AMBIGUOUS_CANDIDATE_SCORE_MARGIN
        close_similarity = abs(top_similarity - similarity) <= DEFAULT_AMBIGUOUS_CANDIDATE_SCORE_MARGIN
        exact_code_signal = candidate_has_exact_code_signal(candidate) or candidate_has_exact_code_signal(top)
        if close_score or close_similarity or exact_code_signal:
            item = dict(candidate)
            item["ambiguity_score_delta"] = abs(top_score - score)
            item["ambiguity_similarity_delta"] = abs(top_similarity - similarity)
            selected.append(item)
            if close_score:
                reasons.add("close_final_score")
            if close_similarity:
                reasons.add("close_visual_score")
            if exact_code_signal:
                reasons.add("card_code_signal")

    if len(selected) < 2:
        return None

    exact_matches = [candidate for candidate in selected if candidate_has_exact_code_signal(candidate)]
    recommended = exact_matches[0] if len(exact_matches) == 1 else selected[0]
    status = "resolved_by_card_code" if len(exact_matches) == 1 else "needs_user_choice"
    return {
        "status": status,
        "reason": "same_subject_close_versions",
        "reasons": sorted(reasons) or ["same_subject_close_versions"],
        "recommended_card_id": recommended.get("card_id"),
        "recommended_rank": recommended.get("rank"),
        "needs_user_choice": status != "resolved_by_card_code",
        "score_margin": DEFAULT_AMBIGUOUS_CANDIDATE_SCORE_MARGIN,
        "candidates": selected,
    }


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


def mask_runs(values: Any) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        if (not value or index == len(values) - 1) and start is not None:
            end = index if not value else index + 1
            runs.append((start, end))
            start = None
    return runs


def slab_label_panel_body_signal(image: Any, box: tuple[int, int, int, int]) -> dict[str, Any]:
    height, width = image.shape[:2]
    x1, _y1, x2, y2 = box
    lower_top = max(y2 + int(round(height * 0.025)), int(round(height * 0.20)))
    lower_bottom = int(round(height * 0.96))
    lower_left = max(0, x1 - int(round(width * 0.12)))
    lower_right = min(width, x2 + int(round(width * 0.12)))
    if lower_top >= lower_bottom or lower_left >= lower_right:
        return {"detected": False, "candidate_count": 0}

    lower = image[lower_top:lower_bottom, lower_left:lower_right]
    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (((saturation > 42) & (value > 35)) | ((value < 115) & (saturation > 12))).astype("uint8") * 255
    kernel_size = max(7, (min(width, height) // 85) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict[str, Any]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width <= 0 or box_height <= 0:
            continue
        full_x = x + lower_left
        full_y = y + lower_top
        area_ratio = (box_width * box_height) / max(1, width * height)
        aspect = box_width / max(1, box_height)
        center_x = (full_x + (box_width / 2)) / max(1, width)
        if area_ratio < 0.045 or box_height < height * 0.18:
            continue
        if not 0.38 <= aspect <= 0.95:
            continue
        candidates.append(
            {
                "box_ratio": {
                    "left": full_x / width,
                    "top": full_y / height,
                    "right": (full_x + box_width) / width,
                    "bottom": (full_y + box_height) / height,
                },
                "area_ratio": area_ratio,
                "aspect": aspect,
                "center_x": center_x,
                "score": area_ratio - (abs(center_x - 0.5) * 0.08),
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return {
        "detected": bool(candidates),
        "candidate_count": len(candidates),
        "best": candidates[0] if candidates else None,
    }


def detect_slab_label_panel(image: Any) -> dict[str, Any]:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return {"detected": False, "candidate_count": 0}

    search_bottom = max(1, int(round(height * 0.34)))
    margin_x = int(round(width * 0.025))
    band = image[0:search_bottom, margin_x : width - margin_x]
    if band.size == 0:
        return {"detected": False, "candidate_count": 0}

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    light_neutral = (((saturation < 72) & (value > 118)) | ((saturation < 95) & (value > 165))).astype("uint8") * 255
    candidates: list[dict[str, Any]] = []

    def add_candidate(
        x: int,
        y: int,
        box_width: int,
        box_height: int,
        source: str,
        source_score: float = 0.0,
    ) -> None:
        if box_width <= 0 or box_height <= 0:
            return
        full_x = x + margin_x
        full_y = y
        aspect = box_width / max(1, box_height)
        width_ratio = box_width / max(1, width)
        height_ratio = box_height / max(1, height)
        area_ratio = (box_width * box_height) / max(1, width * height)
        center_y = (full_y + (box_height / 2)) / max(1, height)
        center_x = (full_x + (box_width / 2)) / max(1, width)
        if not 3.0 <= aspect <= 12.0:
            return
        if not 0.025 <= area_ratio <= 0.22:
            return
        if not 0.36 <= width_ratio <= 0.96:
            return
        if not 0.025 <= height_ratio <= 0.20:
            return
        if center_y >= 0.28 or full_y / max(1, height) >= 0.22:
            return
        mask_crop = light_neutral[y : y + box_height, x : x + box_width]
        fill_ratio = float(mask_crop.mean() / 255.0) if mask_crop.size else 0.0
        if fill_ratio < 0.22:
            return
        full_box = (full_x, full_y, full_x + box_width, full_y + box_height)
        body_signal = slab_label_panel_body_signal(image, full_box)
        center_penalty = abs(center_x - 0.5)
        aspect_bonus = 1.0 - min(1.0, abs(aspect - 6.0) / 6.0)
        body_bonus = 0.12 if body_signal.get("detected") else 0.0
        score = (
            (width_ratio * 0.40)
            + (fill_ratio * 0.24)
            + (aspect_bonus * 0.18)
            + body_bonus
            + source_score
            - (center_penalty * 0.18)
        )
        candidates.append(
            {
                "source": source,
                "box_ratio": {
                    "left": full_x / width,
                    "top": full_y / height,
                    "right": (full_x + box_width) / width,
                    "bottom": (full_y + box_height) / height,
                },
                "aspect": aspect,
                "width_ratio": width_ratio,
                "height_ratio": height_ratio,
                "area_ratio": area_ratio,
                "center_x": center_x,
                "center_y": center_y,
                "fill_ratio": fill_ratio,
                "body_below": body_signal,
                "score": score,
            }
        )

    def add_row_band_candidates(mask: Any) -> None:
        row_coverage = (mask > 0).mean(axis=1)
        for row_threshold in (0.18, 0.26, 0.34, 0.42):
            for y1, y2 in mask_runs(row_coverage >= row_threshold):
                box_height = y2 - y1
                if not int(round(height * 0.025)) <= box_height <= int(round(height * 0.20)):
                    continue
                if ((y1 + y2) / 2) / max(1, height) >= 0.30:
                    continue
                slice_mask = mask[y1:y2] > 0
                column_coverage = slice_mask.mean(axis=0)
                column_runs = mask_runs(column_coverage >= 0.12)
                if not column_runs:
                    continue
                x1, x2 = max(column_runs, key=lambda item: item[1] - item[0])
                add_candidate(
                    x1,
                    y1,
                    x2 - x1,
                    box_height,
                    "light_neutral_row_band",
                    source_score=min(0.08, float(row_coverage[y1:y2].mean()) * 0.08),
                )

    kernel_w = max(9, (width // 95) | 1)
    kernel_h = max(3, (height // 500) | 1)
    row_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    row_mask = cv2.morphologyEx(light_neutral, cv2.MORPH_CLOSE, row_kernel, iterations=1)
    add_row_band_candidates(row_mask)

    hue = hsv[:, :, 0]
    red_mask = (((hue < 12) | (hue > 168)) & (saturation > 45) & (value > 50)).astype("uint8") * 255
    red_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(11, (width // 90) | 1), max(3, (height // 500) | 1)),
    )
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, red_kernel, iterations=1)
    red_row_coverage = (red_mask > 0).mean(axis=1)
    for row_threshold in (0.05, 0.08, 0.12):
        for y1, y2 in mask_runs(red_row_coverage >= row_threshold):
            red_height = y2 - y1
            if red_height <= 0 or red_height > height * 0.08:
                continue
            if ((y1 + y2) / 2) / max(1, height) >= 0.28:
                continue
            slice_mask = red_mask[y1:y2] > 0
            column_coverage = slice_mask.mean(axis=0)
            column_runs = mask_runs(column_coverage >= 0.018)
            if not column_runs:
                continue
            x1, x2 = max(column_runs, key=lambda item: item[1] - item[0])
            box_width = x2 - x1
            if box_width / max(1, width) < 0.36:
                continue
            estimated_height = int(
                round(
                    max(
                        red_height * 2.2,
                        box_width / 5.2,
                        height * 0.045,
                    )
                )
            )
            estimated_height = min(estimated_height, int(round(height * 0.18)))
            estimated_y = max(0, int(round(y1 - (estimated_height * 0.10))))
            add_candidate(
                x1,
                estimated_y,
                box_width,
                estimated_height,
                "red_label_border_estimate",
                source_score=min(0.10, float(red_row_coverage[y1:y2].mean()) * 0.16),
            )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(13, (width // 55) | 1), max(5, (height // 260) | 1)))
    contour_mask = cv2.morphologyEx(light_neutral, cv2.MORPH_CLOSE, kernel, iterations=2)
    contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        add_candidate(x, y, box_width, box_height, "light_neutral_contour")

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0] if candidates else None
    return {
        "detected": bool(best),
        "candidate_count": len(candidates),
        "best": best,
    }


def psa_label_score(image: Any) -> dict[str, Any]:
    height, width = image.shape[:2]
    band = image[int(height * 0.03) : int(height * 0.23), int(width * 0.04) : int(width * 0.96)]
    if band.size == 0:
        return {
            "white_ratio": 0.0,
            "red_ratio": 0.0,
            "score": 0.0,
            "is_psa_label": False,
            "is_slab_label_region": False,
            "red_background_flood": False,
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
    panel = detect_slab_label_panel(image)
    red_background_flood = (
        red_ratio >= 0.14
        and red_component["width_ratio"] >= 0.94
        and red_component["height_ratio"] >= 0.88
        and red_horizontal_run >= 0.72
    )
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
        and not red_background_flood
    )
    panel_best = panel.get("best") or {}
    panel_body = panel_best.get("body_below") or {}
    is_slab_label_region = bool(panel.get("detected") and panel_body.get("detected") and not red_background_flood)
    return {
        "white_ratio": white_ratio,
        "red_ratio": red_ratio,
        "score": min(1.0, white_ratio + (red_ratio * 1.5)),
        "is_psa_label": is_psa_label,
        "is_slab_label_region": is_slab_label_region,
        "label_panel": panel,
        "red_background_flood": red_background_flood,
        "white_component_width_ratio": white_component["width_ratio"],
        "white_component_area_ratio": white_component["area_ratio"],
        "red_component_width_ratio": red_component["width_ratio"],
        "red_component_area_ratio": red_component["area_ratio"],
        "white_component_height_ratio": white_component["height_ratio"],
        "red_component_height_ratio": red_component["height_ratio"],
        "white_horizontal_run_ratio": white_horizontal_run,
        "red_horizontal_run_ratio": red_horizontal_run,
    }


def zxing_available() -> bool:
    return bool(importlib.util.find_spec("zxingcpp"))


def get_zxingcpp() -> Any:
    if "zxingcpp" not in BARCODE_ENGINE_CACHE:
        import zxingcpp  # type: ignore

        BARCODE_ENGINE_CACHE["zxingcpp"] = zxingcpp
    return BARCODE_ENGINE_CACHE["zxingcpp"]


def normalize_cert_key(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    return digits or text.upper()


def slab_barcode_text_type(text: str) -> str:
    value = text.strip()
    if re.fullmatch(r"\d{7,10}", value):
        return "numeric_cert_like"
    if re.fullmatch(r"\d+", value):
        return "numeric_other_length"
    if value.lower().startswith(("http://", "https://")):
        return "url"
    return "other_text"


def slab_barcode_candidates(image: Any) -> list[dict[str, Any]]:
    if not zxing_available():
        return []
    zxingcpp = get_zxingcpp()
    height, width = image.shape[:2]
    rois = {
        "full": image,
        "top45": image[: int(height * 0.45), :],
        "top30": image[: int(height * 0.30), :],
        "top_center": image[: int(height * 0.35), int(width * 0.05) : int(width * 0.95)],
    }
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for roi_name, roi in rois.items():
        if roi.size == 0:
            continue
        for scale in (1, 2):
            scan = roi if scale == 1 else cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            try:
                decoded = zxingcpp.read_barcodes(scan)
            except Exception:  # noqa: BLE001
                decoded = []
            for item in decoded:
                text = str(getattr(item, "text", "") or "").strip()
                barcode_format = str(getattr(item, "format", "") or "")
                if not text:
                    continue
                key = (barcode_format, text)
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    {
                        "format": barcode_format,
                        "text": text,
                        "text_type": slab_barcode_text_type(text),
                        "cert_key": normalize_cert_key(text),
                        "roi": roi_name,
                        "scale": scale,
                    }
                )
    return found


def load_slab_cert_lookup() -> dict[str, dict[str, Any]]:
    if not SLAB_CERT_LOOKUP_PATH:
        return {}
    path = Path(SLAB_CERT_LOOKUP_PATH)
    if not path.exists():
        return {}
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return {}
    if SLAB_CERT_LOOKUP_CACHE.get("path") == str(path) and SLAB_CERT_LOOKUP_CACHE.get("mtime_ns") == mtime_ns:
        return dict(SLAB_CERT_LOOKUP_CACHE.get("rows") or {})

    rows: dict[str, dict[str, Any]] = {}
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = list(iter_jsonl(path))
    else:
        payload = read_json(path)
        if isinstance(payload, dict):
            records = []
            for key, value in payload.items():
                record = dict(value) if isinstance(value, dict) else {"value": value}
                record.setdefault("cert", key)
                records.append(record)
        elif isinstance(payload, list):
            records = [item for item in payload if isinstance(item, dict)]
        else:
            records = []

    for record in records:
        cert = (
            record.get("cert")
            or record.get("psa_cert")
            or record.get("certificate")
            or record.get("certificate_number")
            or record.get("barcode")
            or record.get("barcode_text")
        )
        key = normalize_cert_key(str(cert) if cert is not None else None)
        if key:
            rows[key] = dict(record)
    SLAB_CERT_LOOKUP_CACHE.update({"path": str(path), "mtime_ns": mtime_ns, "rows": rows})
    return rows


SLAB_TEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "card",
    "cards",
    "japan",
    "japanese",
    "pokemon",
    "tcg",
    "the",
}


def normalize_lookup_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", text.lower())


def deep_find_scalar(payload: Any, candidate_keys: list[str]) -> Any:
    normalized_keys = {normalize_lookup_key(key) for key in candidate_keys}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if normalize_lookup_key(key) in normalized_keys and not isinstance(value, (dict, list)):
                return value
        for value in payload.values():
            found = deep_find_scalar(value, candidate_keys)
            if found not in (None, ""):
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = deep_find_scalar(value, candidate_keys)
            if found not in (None, ""):
                return found
    return None


def text_tokens(*values: Any) -> set[str]:
    joined = " ".join(str(value) for value in values if value not in (None, ""))
    text = unicodedata.normalize("NFKD", joined).encode("ascii", "ignore").decode("ascii").lower()
    raw_tokens = set(re.findall(r"[a-z0-9]+", text))
    tokens: set[str] = set()
    embedded_terms = (
        "anniversary",
        "champion",
        "collection",
        "concept",
        "dark",
        "expansion",
        "legendary",
        "premium",
        "pack",
        "phantasma",
        "promo",
        "shine",
        "shiny",
    )
    for token in raw_tokens:
        if token in SLAB_TEXT_STOPWORDS:
            continue
        tokens.add(token)
        compact = token.replace("0", "o") if any(char.isalpha() for char in token) else token
        if token.startswith("exp"):
            tokens.add("expansion")
        if "anniv" in token:
            tokens.add("anniversary")
            number = re.match(r"(\d{1,2})(?:st|nd|rd|th)?anniv", token)
            if number:
                tokens.add(f"{number.group(1)}th")
        if token.endswith("sted") and token[:-2]:
            tokens.add(token[:-2])
        if token.endswith("coll"):
            tokens.add("collection")
        if "shine" in token or "shiny" in token:
            tokens.add("shine")
            tokens.add("shiny")
        if "gojpn" in token or token == "go":
            tokens.add("go")
        if compact != token and "anniv" in compact:
            tokens.add("anniversary")
        for term in embedded_terms:
            if term in token and term != token:
                tokens.add(term)
    return {token for token in tokens if token not in SLAB_TEXT_STOPWORDS}


def title_similarity(lookup: dict[str, Any], record: dict[str, Any]) -> float:
    lookup_tokens = text_tokens(
        lookup.get("brand_title"),
        lookup.get("set_title"),
        lookup.get("variety"),
        lookup.get("pedigree"),
        lookup.get("description"),
        lookup.get("title"),
    )
    record_tokens = text_tokens(
        record.get("set_id"),
        record.get("serebii_set_name"),
        record.get("serebii_set_label"),
        record.get("snkr_product_name"),
        record.get("card_id"),
    )
    if not lookup_tokens or not record_tokens:
        return 0.0
    overlap = lookup_tokens & record_tokens
    return len(overlap) / max(1, min(len(lookup_tokens), len(record_tokens)))


def subject_similarity(lookup: dict[str, Any], record: dict[str, Any]) -> float:
    lookup_compacts = [
        normalize_lookup_key(lookup.get(key))
        for key in ("subject", "name", "card_name", "description")
        if normalize_lookup_key(lookup.get(key))
    ]
    record_compacts = [
        normalize_lookup_key(record.get(key))
        for key in ("name", "name_en", "name_ja", "pokemon_species")
        if normalize_lookup_key(record.get(key))
    ]
    for lookup_compact in lookup_compacts:
        for record_compact in record_compacts:
            if len(lookup_compact) >= 3 and len(record_compact) >= 3 and (
                lookup_compact in record_compact or record_compact in lookup_compact
            ):
                return 1.0
    lookup_tokens = text_tokens(lookup.get("subject"), lookup.get("name"), lookup.get("card_name"))
    record_tokens = text_tokens(
        record.get("name"),
        record.get("name_en"),
        record.get("name_ja"),
        record.get("pokemon_species"),
        record.get("snkr_product_name"),
    )
    if not lookup_tokens or not record_tokens:
        return 0.0
    if lookup_tokens <= record_tokens:
        return 1.0
    return len(lookup_tokens & record_tokens) / max(1, len(lookup_tokens))


def slab_record_lookup_score(record: dict[str, Any], lookup: dict[str, Any]) -> float:
    snkr_product_id = lookup.get("snkr_product_id") or lookup.get("product_id")
    if snkr_product_id and str(record.get("snkr_product_id") or "") == str(snkr_product_id):
        return 1.0
    for key in ("canonical_id", "card_id"):
        if lookup.get(key) and str(record.get(key) or "") == str(lookup.get(key)):
            return 0.99

    lookup_set = normalize_set_for_match(lookup.get("set_id") or lookup.get("set"))
    lookup_number = normalize_card_number(
        lookup.get("card_code")
        or lookup.get("card_number")
        or lookup.get("number")
        or lookup.get("cardNo")
    )
    record_number = normalize_card_number(record.get("card_code"))
    if lookup_set and lookup_number:
        if normalize_set_for_match(record.get("set_id")) == lookup_set and record_number == lookup_number:
            return 0.96
        return 0.0

    if not lookup_number or lookup_number != record_number:
        return 0.0

    name_score = subject_similarity(lookup, record)
    set_title_score = title_similarity(lookup, record)
    is_label_ocr_lookup = lookup.get("provider") == "slab_label_ocr" or lookup.get("lookup_source") == "slab_label_ocr"
    title_threshold = 0.12 if is_label_ocr_lookup else 0.35
    if name_score >= 0.99 and set_title_score >= title_threshold:
        return min(0.95, 0.82 + (set_title_score * 0.10))
    if name_score >= 0.99 and not any(lookup.get(key) for key in ("brand_title", "set_title", "variety", "pedigree", "description", "title")):
        return 0.72
    if name_score >= 0.65 and set_title_score >= 0.55:
        return min(0.90, 0.70 + (set_title_score * 0.15))
    return 0.0


def record_matches_slab_lookup(record: dict[str, Any], lookup: dict[str, Any]) -> bool:
    return slab_record_lookup_score(record, lookup) > 0.0


def slab_lookup_results(lookup: dict[str, Any], barcode: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for loaded in service.load_indexes():
        for record in loaded.records:
            match_score = slab_record_lookup_score(record, lookup)
            if match_score <= 0.0:
                continue
            key = (str(record.get("canonical_id") or record.get("card_id") or ""), str(record.get("snkr_product_id") or ""))
            seen_keys.add(key)
            result = format_result(loaded.name, len(matches) + 1, match_score, record)
            result["slab_barcode_match"] = True
            result["slab_barcode_text"] = barcode.get("text")
            result["slab_cert_key"] = barcode.get("cert_key")
            result["slab_lookup_source"] = lookup.get("lookup_source") or str(SLAB_CERT_LOOKUP_PATH)
            result["slab_lookup_provider"] = lookup.get("provider")
            result["slab_lookup_match_score"] = round(match_score, 4)
            if lookup.get("language"):
                result["slab_lookup_language"] = lookup.get("language")
            matches.append(result)
    for source_name, record in load_slab_catalog_lookup_records():
        match_score = slab_record_lookup_score(record, lookup)
        if match_score <= 0.0:
            continue
        key = (str(record.get("canonical_id") or record.get("card_id") or ""), str(record.get("snkr_product_id") or ""))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        result = format_result(source_name, len(matches) + 1, match_score, record)
        result["slab_barcode_match"] = True
        result["slab_barcode_text"] = barcode.get("text")
        result["slab_cert_key"] = barcode.get("cert_key")
        result["slab_lookup_source"] = lookup.get("lookup_source") or str(SLAB_CERT_LOOKUP_PATH)
        result["slab_lookup_provider"] = lookup.get("provider")
        result["slab_lookup_match_score"] = round(match_score, 4)
        if lookup.get("language"):
            result["slab_lookup_language"] = lookup.get("language")
        result["slab_catalog_match"] = True
        matches.append(result)
    target_language = normalize_language(lookup.get("language"))

    def sort_key(item: dict[str, Any]) -> tuple[float, int, int, int]:
        language_match = 1 if target_language and candidate_language(item) == target_language else 0
        snkr = item.get("snkr") or {}
        has_snkr_product = 1 if snkr.get("product_id") or item.get("snkr_product_id") else 0
        is_japanese_index = 1 if str(item.get("index") or "").endswith("_ja") else 0
        return (
            float(item.get("slab_lookup_match_score") or 0.0),
            language_match,
            has_snkr_product,
            is_japanese_index,
        )

    matches.sort(key=sort_key, reverse=True)
    for rank, item in enumerate(matches[:top_k], start=1):
        item["rank"] = rank
    return matches[:top_k]


def load_slab_catalog_lookup_records() -> list[tuple[str, dict[str, Any]]]:
    paths = [path if path.is_absolute() else ROOT / path for path in SLAB_CATALOG_LOOKUP_PATHS]
    existing_paths = [path for path in paths if path.exists()]
    try:
        mtime_ns = tuple(path.stat().st_mtime_ns for path in existing_paths)
    except OSError:
        mtime_ns = ()
    cache_paths = tuple(str(path) for path in existing_paths)
    if SLAB_CATALOG_LOOKUP_CACHE.get("paths") == cache_paths and SLAB_CATALOG_LOOKUP_CACHE.get("mtime_ns") == mtime_ns:
        return list(SLAB_CATALOG_LOOKUP_CACHE.get("records") or [])

    records: list[tuple[str, dict[str, Any]]] = []
    for path in existing_paths:
        source_name = f"catalog:{path.stem}"
        try:
            for record in iter_jsonl(path):
                if isinstance(record, dict):
                    records.append((source_name, record))
        except Exception:  # noqa: BLE001
            continue
    SLAB_CATALOG_LOOKUP_CACHE.update({"paths": cache_paths, "mtime_ns": mtime_ns, "records": records})
    return records


def http_get_json(url: str, headers: dict[str, str], timeout: float = SLAB_CERT_API_TIMEOUT) -> dict[str, Any]:
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            text = raw.decode("utf-8", errors="replace")
            try:
                payload = json.loads(text) if text else None
            except json.JSONDecodeError:
                payload = None
            return {
                "status": "ok",
                "http_status": int(getattr(response, "status", 200)),
                "payload": payload,
                "body_preview": text[:500] if payload is None else None,
            }
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        return {"status": "http_error", "http_status": exc.code, "body_preview": body}
    except (TimeoutError, URLError, OSError) as exc:
        return {"status": "network_error", "error": str(exc)}


def normalize_provider_cert_payload(provider: str, cert_key: str, barcode: dict[str, Any], payload: Any) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    cert_number = deep_find_scalar(source, ["certNumber", "cert_number", "cert", "certificateNumber"]) or cert_key
    card_number = deep_find_scalar(source, ["cardNumber", "card_number", "cardNo", "number"])
    subject = deep_find_scalar(source, ["subject", "cardName", "card_name", "name"])
    brand_title = deep_find_scalar(source, ["brandTitle", "brand/title", "brand", "title", "setName", "set_name"])
    variety = deep_find_scalar(source, ["variety", "pedigree", "varietyPedigree", "variety/pedigree"])
    description = deep_find_scalar(source, ["description", "itemDescription", "labelDescription"])
    year = deep_find_scalar(source, ["year"])
    grade = deep_find_scalar(source, ["gradeDescription", "displayGrade", "grade"])
    category = deep_find_scalar(source, ["category"])
    lookup: dict[str, Any] = {
        "cert": str(cert_number),
        "certificate_number": str(cert_number),
        "provider": provider,
        "lookup_source": f"{provider}_api",
        "barcode_text": barcode.get("text"),
        "barcode_format": barcode.get("format"),
        "card_number": str(card_number) if card_number not in (None, "") else None,
        "card_code": str(card_number) if card_number not in (None, "") else None,
        "subject": str(subject) if subject not in (None, "") else None,
        "name": str(subject) if subject not in (None, "") else None,
        "brand_title": str(brand_title) if brand_title not in (None, "") else None,
        "set_title": str(brand_title) if brand_title not in (None, "") else None,
        "variety": str(variety) if variety not in (None, "") else None,
        "description": str(description) if description not in (None, "") else None,
        "year": str(year) if year not in (None, "") else None,
        "grade": str(grade) if grade not in (None, "") else None,
        "category": str(category) if category not in (None, "") else None,
        "provider_payload": source,
    }
    if provider == "psa":
        lookup["cert_url"] = f"https://www.psacard.com/cert/{cert_key}/psa"
    return {key: value for key, value in lookup.items() if value not in (None, "")}


def psa_lookup_cert(cert_key: str, barcode: dict[str, Any]) -> dict[str, Any]:
    attempt: dict[str, Any] = {"provider": "psa", "cert_key": cert_key}
    if not PSA_PUBLIC_API_TOKEN:
        attempt["status"] = "not_configured"
        attempt["reason"] = "CARD_SCAN_PSA_API_TOKEN is not set"
        return attempt
    if not re.fullmatch(r"\d{7,10}", cert_key or ""):
        attempt["status"] = "skipped_cert_format"
        return attempt
    url = f"https://api.psacard.com/publicapi/cert/GetByCertNumber/{quote(cert_key)}"
    response = http_get_json(
        url,
        {
            "Accept": "application/json",
            "Authorization": f"bearer {PSA_PUBLIC_API_TOKEN}",
        },
    )
    attempt.update({key: value for key, value in response.items() if key != "payload"})
    payload = response.get("payload")
    if response.get("status") != "ok":
        return attempt
    if isinstance(payload, dict):
        server_message = str(payload.get("ServerMessage") or payload.get("serverMessage") or "")
        is_valid = payload.get("IsValidRequest", payload.get("isValidRequest", True))
        if is_valid is False or "no data" in server_message.lower():
            attempt["status"] = "not_found"
            attempt["server_message"] = server_message
            return attempt
    attempt["status"] = "matched"
    attempt["entry"] = normalize_provider_cert_payload("psa", cert_key, barcode, payload)
    return attempt


def cgc_lookup_cert(cert_key: str, barcode: dict[str, Any]) -> dict[str, Any]:
    attempt: dict[str, Any] = {"provider": "cgc", "cert_key": cert_key}
    if not CGC_DEALER_API_TOKEN:
        attempt["status"] = "not_configured"
        attempt["reason"] = "CARD_SCAN_CGC_DEALER_API_TOKEN is not set"
        return attempt
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {CGC_DEALER_API_TOKEN}",
    }
    urls = [
        ("cert_v3", f"https://dealer-api.collectiblesgroup.com/cards/certifications/v3/lookup/{quote(cert_key)}"),
        ("cert_v2", f"https://dealer-api.collectiblesgroup.com/cards/certifications/v2/lookup/{quote(cert_key)}"),
    ]
    barcode_text = str(barcode.get("text") or "").strip()
    if barcode_text and barcode_text != cert_key:
        encoded_barcode = quote(barcode_text, safe="")
        urls.extend(
            [
                (
                    "barcode_v3",
                    f"https://dealer-api.collectiblesgroup.com/cards/certifications/v3/barcode/{encoded_barcode}",
                ),
                (
                    "barcode_v2",
                    f"https://dealer-api.collectiblesgroup.com/cards/certifications/v2/barcode/{encoded_barcode}",
                ),
            ]
        )
    responses: list[dict[str, Any]] = []
    for lookup_kind, url in urls:
        response = http_get_json(url, headers)
        responses.append({key: value for key, value in {"lookup_kind": lookup_kind, **response}.items() if key != "payload"})
        if response.get("status") == "ok" and response.get("payload"):
            attempt["status"] = "matched"
            attempt["lookup_kind"] = lookup_kind
            attempt["responses"] = responses
            attempt["entry"] = normalize_provider_cert_payload("cgc", cert_key, barcode, response.get("payload"))
            return attempt
    attempt["status"] = "not_found"
    attempt["responses"] = responses
    return attempt


def slab_barcode_provider_hint(barcode: dict[str, Any]) -> str | None:
    text = str(barcode.get("text") or "").lower()
    if "psacard.com" in text:
        return "psa"
    if "cgccards.com" in text or "collectiblesgroup.com" in text:
        return "cgc"
    return None


SLAB_LABEL_GRADE_TOKENS = {
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "AUTHENTIC",
    "EX",
    "EX-MT",
    "FAIR",
    "GEM",
    "GEMMT",
    "GEM MT",
    "GOOD",
    "MINT",
    "MT",
    "NM",
    "NM-MT",
    "PR",
    "PSA",
    "VG",
}


def slab_label_ocr_variants(image: Any) -> list[tuple[str, Any]]:
    height, width = image.shape[:2]
    top_label_bottom = 0.34 if height / max(1, width) > 1.18 else 0.24
    boxes = [
        ("top", (0.00, 0.00, 1.00, top_label_bottom)),
        ("left_label", (0.015, 0.015, 0.50, 0.215)),
    ]
    variants: list[tuple[str, Any]] = []
    for name, box in boxes:
        crop = crop_ratio(image, box)
        crop_height, crop_width = crop.shape[:2]
        if crop_height < 24 or crop_width < 80:
            continue
        scale = max(1, min(5, int(round(1200 / max(1, crop_width)))))
        if scale > 1:
            crop = cv2.resize(crop, (crop_width * scale, crop_height * scale), interpolation=cv2.INTER_CUBIC)
        variants.append((name, crop))
    return variants


def recognize_slab_label_text_with_rapidocr(image: Any) -> dict[str, Any]:
    if not rapidocr_available():
        return {"source": "rapidocr", "status": "unavailable", "reason": "rapidocr_onnxruntime is missing"}
    started = time.perf_counter()
    try:
        engine = get_rapidocr_engine()
    except Exception as exc:  # noqa: BLE001
        return {"source": "rapidocr", "status": "unavailable", "reason": str(exc)}

    variant_rows: list[dict[str, Any]] = []
    texts: list[str] = []
    max_confidence = 0.0
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        for variant_name, variant_image in slab_label_ocr_variants(image):
            path = temp_path / f"slab_label_{variant_name}.jpg"
            cv2.imwrite(str(path), variant_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            try:
                result = engine(str(path))
            except Exception as exc:  # noqa: BLE001
                variant_rows.append({"variant": variant_name, "status": "error", "error": str(exc)})
                continue
            text, confidence = flatten_rapidocr_result(result)
            max_confidence = max(max_confidence, confidence)
            if text:
                texts.append(text)
            variant_rows.append(
                {
                    "variant": variant_name,
                    "status": "ok" if text else "empty",
                    "text": text,
                    "confidence": confidence,
                    "width": int(variant_image.shape[1]),
                    "height": int(variant_image.shape[0]),
                }
            )
    text = "\n".join(texts)
    return {
        "source": "rapidocr",
        "status": "ok" if text else "not_found",
        "text": text,
        "ocr_confidence": max_confidence,
        "variants": variant_rows,
        "seconds": time.perf_counter() - started,
    }


def normalize_slab_label_ocr_text(text: str) -> str:
    normalized = normalize_ocr_text(text or "")
    normalized = normalized.replace("POKEMONJAPANESE", "POKEMON JAPANESE")
    normalized = normalized.replace("POKEMONENGLISH", "POKEMON ENGLISH")
    normalized = normalized.replace("GEMMT", "GEM MT")
    normalized = re.sub(r"(?<=\d)[OQ](?=\d)", "0", normalized)
    normalized = re.sub(r"\b2OTH\b", "20TH", normalized)
    normalized = re.sub(r"\b2OTHANNIV\b", "20THANNIV", normalized)
    return normalized


def normalize_slab_set_alias_key(value: str | None) -> str:
    text = normalize_slab_label_ocr_text(value or "")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.upper().replace("&", "AND")
    text = re.sub(r"[^A-Z0-9]+", "", text)
    text = text.replace("2OTH", "20TH")
    text = text.replace("CHAMPI0N", "CHAMPION")
    return text


def add_slab_label_set_alias(
    aliases: dict[tuple[str, str], dict[str, Any]],
    alias: str | None,
    set_id: str | None,
    source: str,
    confidence: str,
    notes: str = "",
) -> None:
    alias_key = normalize_slab_set_alias_key(alias)
    normalized_set = normalize_set_for_match(set_id)
    if not alias_key or len(alias_key) < 3 or not normalized_set:
        return
    key = (alias_key, normalized_set)
    current = aliases.get(key)
    row = {
        "alias": alias,
        "alias_key": alias_key,
        "set_id": set_id,
        "normalized_set_id": normalized_set,
        "source": source,
        "confidence": confidence,
        "notes": notes,
    }
    if current is None or len(alias_key) > len(str(current.get("alias_key") or "")):
        aliases[key] = row


def load_slab_label_set_aliases() -> list[dict[str, Any]]:
    paths = [SLAB_LABEL_SET_ALIASES_PATH, SEREBII_SET_NAME_MAP_PATH]
    existing_paths = [path for path in paths if path.exists()]
    cache_paths = tuple(str(path) for path in existing_paths)
    try:
        mtime_ns = tuple(path.stat().st_mtime_ns for path in existing_paths)
    except OSError:
        mtime_ns = ()
    if (
        SLAB_LABEL_SET_ALIAS_CACHE.get("paths") == cache_paths
        and SLAB_LABEL_SET_ALIAS_CACHE.get("mtime_ns") == mtime_ns
    ):
        return list(SLAB_LABEL_SET_ALIAS_CACHE.get("rows") or [])

    aliases: dict[tuple[str, str], dict[str, Any]] = {}
    if SLAB_LABEL_SET_ALIASES_PATH.exists():
        with SLAB_LABEL_SET_ALIASES_PATH.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("enabled") or "true").strip().lower() in {"0", "false", "no"}:
                    continue
                add_slab_label_set_alias(
                    aliases,
                    row.get("alias"),
                    row.get("set_id"),
                    row.get("source") or "slab_label_set_aliases",
                    row.get("confidence") or "manual",
                    row.get("notes") or "",
                )

    if SEREBII_SET_NAME_MAP_PATH.exists():
        with SEREBII_SET_NAME_MAP_PATH.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                set_id = row.get("mapped_set_id") or row.get("set_id")
                confidence = row.get("confidence") or "serebii_map"
                for alias in (
                    row.get("serebii_set_name"),
                    row.get("canonical_snkr_evidence_name"),
                    row.get("evidence_source"),
                ):
                    add_slab_label_set_alias(aliases, alias, set_id, "serebii_set_name_map", confidence)

    rows = sorted(
        aliases.values(),
        key=lambda item: (
            len(str(item.get("alias_key") or "")),
            1 if str(item.get("confidence") or "").lower() == "high" else 0,
        ),
        reverse=True,
    )
    SLAB_LABEL_SET_ALIAS_CACHE.update({"paths": cache_paths, "mtime_ns": mtime_ns, "rows": rows})
    return list(rows)


def match_slab_label_set_alias(normalized_text: str) -> dict[str, Any] | None:
    text_key = normalize_slab_set_alias_key(normalized_text)
    if not text_key:
        return None
    for row in load_slab_label_set_aliases():
        alias_key = str(row.get("alias_key") or "")
        if not alias_key or len(alias_key) < 3:
            continue
        if alias_key in text_key:
            return {
                "set_id": row.get("set_id"),
                "normalized_set_id": row.get("normalized_set_id"),
                "matched_alias": row.get("alias"),
                "matched_alias_key": alias_key,
                "source": row.get("source"),
                "confidence": row.get("confidence"),
                "notes": row.get("notes"),
            }
    return None


def extract_slab_label_set_id(normalized_text: str) -> dict[str, Any] | None:
    direct_patterns = [
        r"\b(?:JPN|JP|JAPANESE)\.?\s*(?P<set>(?:SV|S|SM|XY|BW|DP|PCG|ADV)-?P)\b",
        r"\b(?P<set>(?:SV|S|SM|XY|BW|DP|PCG|ADV)-?P)\s*(?:PROMO|PROMOS)\b",
    ]
    for pattern in direct_patterns:
        match = re.search(pattern, normalized_text, flags=re.I)
        if not match:
            continue
        raw_set = match.group("set")
        normalized_set = normalize_set_for_match(raw_set)
        if normalized_set:
            return {
                "set_id": normalized_set,
                "normalized_set_id": normalized_set,
                "matched_alias": raw_set,
                "matched_alias_key": normalize_slab_set_alias_key(raw_set),
                "source": "slab_label_direct_set_code",
                "confidence": "high",
                "notes": "Direct set code parsed from slab label OCR.",
            }
    return None


def parse_slab_label_ocr_lookup(text: str, cert_key: str | None = None) -> dict[str, Any]:
    normalized = normalize_slab_label_ocr_text(text)
    set_alias_match = match_slab_label_set_alias(normalized) or extract_slab_label_set_id(normalized)
    language_payload = infer_language_from_text(normalized, source="slab_label_ocr")
    number_match = re.search(r"#\s*([A-Z]?\d{1,4})\b", normalized) or re.search(
        r"\bNO\.?\s*([A-Z]?\d{1,4})\b",
        normalized,
    )
    card_number = number_match.group(1) if number_match else None
    stop_compact = {token.replace(" ", "") for token in SLAB_LABEL_GRADE_TOKENS}

    lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = re.sub(r"[^A-Z0-9# /.-]+", " ", raw_line).strip()
        line = re.sub(r"\s+", " ", line)
        compact = line.replace(" ", "")
        if not line or compact in stop_compact:
            continue
        if re.fullmatch(r"\d{7,10}", compact):
            continue
        if re.fullmatch(r"#?[A-Z]?\d{1,4}", compact):
            continue
        lines.append(line)

    subject: str | None = None
    title_parts: list[str] = []
    for line in lines:
        clean = re.sub(r"^\d{4}\s*", "", line).strip()
        clean = clean.replace("POKEMON JAPANESE", "").replace("POKEMON ENGLISH", "")
        clean = clean.replace("POKEMON", "").replace("JAPANESE", "").replace("ENGLISH", "").strip()
        if not clean:
            title_parts.append(line)
            continue
        if subject is None and len(clean.split()) <= 4 and not re.search(
            r"\b(?:ANNIV|CHAMPION|COLLECTION|DECK|EXPANSION|PACK|SERIES)\b",
            clean,
        ):
            subject = clean
        else:
            title_parts.append(clean)

    lookup: dict[str, Any] = {
        "provider": "slab_label_ocr",
        "lookup_source": "slab_label_ocr",
        "set_id": set_alias_match.get("set_id") if set_alias_match else None,
        "card_number": card_number,
        "card_code": card_number,
        "subject": subject,
        "name": subject,
        "brand_title": " ".join(title_parts),
        "set_title": " ".join(title_parts),
        "description": normalized,
    }
    if language_payload.get("status") == "ok":
        lookup["language"] = language_payload.get("language")
        lookup["language_source"] = language_payload.get("source")
        lookup["language_signals"] = language_payload.get("signals")
    if set_alias_match:
        lookup["set_alias_match"] = set_alias_match
    if cert_key:
        lookup["cert"] = cert_key
        lookup["certificate_number"] = cert_key
    return {key: value for key, value in lookup.items() if value not in (None, "")}


def recognize_slab_label_ocr(image: Any, barcode: dict[str, Any] | None, top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = {
        "enabled": DEFAULT_SLAB_LABEL_OCR,
        "engine": DEFAULT_SLAB_LABEL_OCR_ENGINE,
        "status": "skipped" if not DEFAULT_SLAB_LABEL_OCR else "not_found",
    }
    if not DEFAULT_SLAB_LABEL_OCR:
        return payload, []
    if DEFAULT_SLAB_LABEL_OCR_ENGINE not in {"rapidocr", "rapid"}:
        payload["status"] = "unsupported"
        payload["reason"] = f"Unsupported slab label OCR engine: {DEFAULT_SLAB_LABEL_OCR_ENGINE}"
        return payload, []

    text_payload = recognize_slab_label_text_with_rapidocr(image)
    payload.update(text_payload)
    if text_payload.get("status") != "ok":
        return payload, []
    cert_key = str((barcode or {}).get("cert_key") or "") or None
    lookup = parse_slab_label_ocr_lookup(str(text_payload.get("text") or ""), cert_key=cert_key)
    payload["lookup_entry"] = lookup
    required_keys = {
        "set_id": bool(lookup.get("set_id")),
        "card_number": bool(lookup.get("card_number") or lookup.get("card_code")),
    }
    payload["direct_lookup_required_keys"] = required_keys
    if not all(required_keys.values()):
        payload["matched_result_count"] = 0
        payload["status"] = "insufficient_lookup_keys"
        payload["reason"] = "slab label OCR direct lookup requires both set_id and card_number"
        return payload, []

    results = slab_lookup_results(lookup, barcode or {}, top_k)
    payload["matched_result_count"] = len(results)
    payload["status"] = "matched" if results else "lookup_without_index_match"
    return payload, results


def external_slab_lookup(barcode: dict[str, Any], top_k: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    cert_key = str(barcode.get("cert_key") or "")
    if not cert_key:
        return None, [], []
    if not SLAB_CERT_API_LOOKUP_ENABLED:
        return None, [], [
            {
                "status": "disabled",
                "reason": "CARD_SCAN_SLAB_CERT_API_LOOKUP is disabled; slab label OCR/catalog lookup is preferred",
                "cert_key": cert_key,
            }
        ]
    providers: list[str] = []
    hint = slab_barcode_provider_hint(barcode)
    if hint:
        providers.append(hint)
    providers.extend(SLAB_CERT_LOOKUP_PROVIDERS)
    ordered_providers = list(dict.fromkeys(providers))
    attempts: list[dict[str, Any]] = []
    for provider in ordered_providers:
        if provider == "psa":
            attempt = psa_lookup_cert(cert_key, barcode)
        elif provider in {"cgc", "ccg"}:
            attempt = cgc_lookup_cert(cert_key, barcode)
        else:
            attempt = {"provider": provider, "status": "unsupported"}
        attempts.append(attempt)
        entry = attempt.get("entry")
        if isinstance(entry, dict):
            results = slab_lookup_results(entry, barcode, top_k)
            return entry, results, attempts
    return None, [], attempts


def recognize_slab_barcode(image: Any, top_k: int, scan_always: bool = DEFAULT_SLAB_BARCODE_SCAN_ALWAYS) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label = psa_label_score(image)
    payload: dict[str, Any] = {
        "enabled": True,
        "zxing_available": zxing_available(),
        "lookup_path": SLAB_CERT_LOOKUP_PATH or None,
        "lookup_providers": SLAB_CERT_LOOKUP_PROVIDERS,
        "provider_tokens_configured": {
            "psa": bool(PSA_PUBLIC_API_TOKEN),
            "cgc": bool(CGC_DEALER_API_TOKEN),
        },
        "barcode_lookup_enabled": False,
        "label_ocr_enabled": DEFAULT_SLAB_LABEL_OCR,
        "label_ocr_engine": DEFAULT_SLAB_LABEL_OCR_ENGINE if DEFAULT_SLAB_LABEL_OCR else None,
        "scan_always": scan_always,
        "psa_label": label,
        "barcodes": [],
        "cert_candidates": [],
        "lookup": {"status": "label_ocr_not_found"},
    }
    if not zxing_available():
        payload["zxing_available"] = False
    if not scan_always and not label.get("is_slab_label_region"):
        payload["status"] = "skipped_non_slab"
        return payload, []

    label_ocr_payload: dict[str, Any] = {"enabled": DEFAULT_SLAB_LABEL_OCR, "status": "skipped"}
    label_ocr_results: list[dict[str, Any]] = []
    if DEFAULT_SLAB_LABEL_OCR:
        label_ocr_payload, label_ocr_results = recognize_slab_label_ocr(image, {}, top_k)
        payload["lookup"]["label_ocr"] = label_ocr_payload
        if label_ocr_results:
            payload["status"] = "lookup_match"
            payload["lookup"] = {
                "status": "label_ocr_matched",
                "source": "label_ocr",
                "entry": label_ocr_payload.get("lookup_entry"),
                "matched_result_count": len(label_ocr_results),
                "label_ocr": label_ocr_payload,
            }
            return payload, label_ocr_results

    payload["status"] = "not_found"
    payload["reason"] = "slab label OCR did not resolve a catalog match; barcode/cert lookup is disabled"
    payload["lookup"] = {
        "status": "label_ocr_not_found",
        "source": "label_ocr",
        "label_ocr": label_ocr_payload,
    }
    return payload, []


def slab_label_ocr_text_from_payload(payload: dict[str, Any]) -> str:
    lookup = payload.get("lookup") or {}
    label_ocr = lookup.get("label_ocr") or {}
    text = str(label_ocr.get("text") or "").strip()
    if text:
        return text
    entry = label_ocr.get("lookup_entry") or lookup.get("entry") or {}
    return str(entry.get("description") or entry.get("brand_title") or "").strip()


def infer_language_from_request_or_slab(
    language_hint: str,
    hint_text: str,
    slab_text: str,
    *,
    enabled: bool,
    boost: float,
    ocr_enabled: bool,
    ocr_engine: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": enabled,
        "status": "skipped",
        "boost": 0.0,
        "ocr_enabled": ocr_enabled,
        "ocr_engine": ocr_engine if ocr_enabled else None,
    }
    if not enabled:
        return payload

    explicit_language = normalize_language(language_hint)
    if explicit_language:
        return {
            "enabled": True,
            "status": "ok",
            "source": "explicit",
            "language": explicit_language,
            "confidence": 1.0,
            "signals": [f"explicit:{explicit_language}"],
            "boost": boost,
            "ocr_enabled": ocr_enabled,
            "ocr_engine": ocr_engine if ocr_enabled else None,
        }

    source_text = hint_text.strip()
    source = "hint_text"
    if not source_text and slab_text.strip():
        source_text = slab_text.strip()
        source = "slab_label_ocr"

    payload = infer_language_from_text(source_text, source=source)
    payload["enabled"] = True
    payload["boost"] = boost
    payload["ocr_enabled"] = ocr_enabled
    payload["ocr_engine"] = ocr_engine if ocr_enabled else None
    return payload


def rembg_available() -> bool:
    return bool(importlib.util.find_spec("rembg") and importlib.util.find_spec("PIL"))


def get_rembg_session(model_name: str = "u2netp") -> Any:
    cache_key = f"rembg:{model_name}"
    if cache_key not in SEGMENTATION_ENGINE_CACHE:
        from rembg import new_session

        SEGMENTATION_ENGINE_CACHE[cache_key] = new_session(model_name)
    return SEGMENTATION_ENGINE_CACHE[cache_key]


def u2netp_foreground_crop(image: Any, target_aspect: float) -> dict[str, Any]:
    if not rembg_available():
        return {
            "status": "unavailable",
            "reason": "rembg is not installed",
            "fallback_used": False,
            "detections": [],
        }
    from PIL import Image
    from rembg import remove

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    mask = remove(pil_image, session=get_rembg_session("u2netp"), only_mask=True)
    mask_array = np.asarray(mask)
    ys, xs = np.where(mask_array > 24)
    if len(xs) == 0 or len(ys) == 0:
        return {
            "status": "no_detection",
            "fallback_used": False,
            "detections": [],
            "segmentation_model": "u2netp",
        }

    mask_box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    box = expand_box_ratio(mask_box, width, height, 0.018)
    box = trim_box_to_aspect(box, width, height, target_aspect)
    x1, y1, x2, y2 = box
    crop_image = image[y1:y2, x1:x2]
    crop_height, crop_width = crop_image.shape[:2]
    return {
        "status": "u2netp_foreground",
        "fallback_used": False,
        "detections": [],
        "segmentation_model": "u2netp",
        "mask_threshold": 24,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "mask_box_ratio": {
            "left": mask_box[0] / width,
            "top": mask_box[1] / height,
            "right": mask_box[2] / width,
            "bottom": mask_box[3] / height,
        },
        "crop_box_ratio": {
            "left": x1 / width,
            "top": y1 / height,
            "right": x2 / width,
            "bottom": y2 / height,
        },
        "crop_area_ratio": ((x2 - x1) * (y2 - y1)) / max(1, width * height),
        "crop_image": crop_image,
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
    if not (label.get("is_psa_label") or label.get("is_slab_label_region")):
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
        "card_code_ocr_available": (
            (ROOT / "scripts/quality/ocr_vision_text.swift").exists()
            and shutil.which("swift") is not None
        ),
        "card_code_ocr_exact_boost": DEFAULT_CARD_CODE_OCR_EXACT_BOOST,
        "default_language_rerank": DEFAULT_LANGUAGE_RERANK,
        "default_language_rerank_boost": DEFAULT_LANGUAGE_RERANK_BOOST,
        "default_language_rerank_candidates": DEFAULT_LANGUAGE_RERANK_CANDIDATES,
        "default_language_ocr": DEFAULT_LANGUAGE_OCR,
        "default_language_ocr_engine": DEFAULT_LANGUAGE_OCR_ENGINE,
        "rapidocr_available": rapidocr_available(),
        "default_slab_barcode_lookup": DEFAULT_SLAB_BARCODE_LOOKUP,
        "default_slab_barcode_scan_always": DEFAULT_SLAB_BARCODE_SCAN_ALWAYS,
        "default_slab_label_ocr": DEFAULT_SLAB_LABEL_OCR,
        "default_slab_label_ocr_engine": DEFAULT_SLAB_LABEL_OCR_ENGINE,
        "zxing_available": zxing_available(),
        "rembg_available": rembg_available(),
        "slab_cert_lookup_path": SLAB_CERT_LOOKUP_PATH or None,
        "slab_cert_lookup_configured": bool(SLAB_CERT_LOOKUP_PATH),
        "slab_cert_lookup_providers": SLAB_CERT_LOOKUP_PROVIDERS,
        "slab_label_set_aliases_path": str(SLAB_LABEL_SET_ALIASES_PATH),
        "slab_label_set_alias_count": len(load_slab_label_set_aliases()),
        "slab_catalog_lookup_paths": [str(path) for path in SLAB_CATALOG_LOOKUP_PATHS],
        "slab_cert_api_timeout": SLAB_CERT_API_TIMEOUT,
        "slab_cert_api_lookup_enabled": SLAB_CERT_API_LOOKUP_ENABLED,
        "slab_cert_provider_tokens_configured": {
            "psa": bool(PSA_PUBLIC_API_TOKEN),
            "cgc": bool(CGC_DEALER_API_TOKEN),
        },
        "default_hint_card_code_boost": DEFAULT_HINT_CARD_CODE_BOOST,
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
        "snkr_mapping_flags": {
            "path": str(SNKR_MAPPING_FLAGS_PATH),
            "exists": SNKR_MAPPING_FLAGS_PATH.exists(),
            "records": len(SNKR_MAPPING_FLAGS),
            "suppress_flagged_mappings": SUPPRESS_FLAGGED_SNKR_MAPPINGS,
        },
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
    language_rerank: bool = DEFAULT_LANGUAGE_RERANK,
    language_rerank_boost: float = DEFAULT_LANGUAGE_RERANK_BOOST,
    language_rerank_candidates: int = DEFAULT_LANGUAGE_RERANK_CANDIDATES,
    language_hint: str = "",
    language_hint_text: str = "",
    language_ocr: bool = DEFAULT_LANGUAGE_OCR,
    language_ocr_engine: str = DEFAULT_LANGUAGE_OCR_ENGINE,
    slab_barcode_lookup: bool = DEFAULT_SLAB_BARCODE_LOOKUP,
    hint_card_code: bool = True,
    hint_card_code_boost: float = DEFAULT_HINT_CARD_CODE_BOOST,
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
    slab_barcode_payload: dict[str, Any] = {
        "enabled": slab_barcode_lookup,
        "status": "skipped",
    }
    query_image = image
    debug_crop_image = None
    normalized_crop_mode = crop_mode.strip().lower().replace("-", "_")
    if not crop:
        normalized_crop_mode = "none"

    if slab_barcode_lookup:
        slab_barcode_start = time.perf_counter()
        slab_barcode_payload, slab_barcode_results = recognize_slab_barcode(image, top_k=top_k)
        timings["slab_barcode_seconds"] = time.perf_counter() - slab_barcode_start
        if slab_barcode_results:
            for rank, item in enumerate(slab_barcode_results, start=1):
                item["rank"] = rank
            slab_language_payload = infer_language_from_request_or_slab(
                language_hint,
                language_hint_text,
                slab_label_ocr_text_from_payload(slab_barcode_payload),
                enabled=language_rerank,
                boost=max(0.0, language_rerank_boost),
                ocr_enabled=False,
                ocr_engine=language_ocr_engine,
            )
            lookup_source = str((slab_barcode_payload.get("lookup") or {}).get("source") or "")
            recognition_route = "slab_label_ocr_lookup" if lookup_source == "label_ocr" else "slab_barcode_lookup"
            timings["total_seconds"] = time.perf_counter() - total_start
            return {
                "status": "ok",
                "recognition_route": recognition_route,
                "started_at": started_at,
                "input": {
                    "filename": file.filename,
                    "width": input_width,
                    "height": input_height,
                },
                "crop": crop_payload,
                "slab_barcode": slab_barcode_payload,
                "results": slab_barcode_results[:top_k],
                "results_by_index": {},
                "candidate_selection": build_candidate_selection(slab_barcode_results),
                "visual_rerank": {
                    "enabled": False,
                    "model": None,
                    "weight": 0.0,
                    "candidates": 0,
                },
                "card_code_ocr": {
                    "enabled": False,
                    "status": "skipped",
                    "boost": 0.0,
                },
                "language_rerank": {
                    **slab_language_payload,
                    "direct_lookup_only": True,
                },
                "timings": timings,
            }

    if normalized_crop_mode == "auto":
        crop_start = time.perf_counter()
        crop_payload = slab_inner_card_crop(image, target_aspect)
        if crop_payload is None:
            slab_barcode_status = slab_barcode_payload.get("status")
            slab_label = slab_barcode_payload.get("psa_label") or {}
            slab_signal = slab_barcode_status == "decoded" or bool(
                slab_label.get("is_psa_label") or slab_label.get("is_slab_label_region")
            )
            if slab_signal:
                crop_payload = generic_graded_slab_crop(image)
                if crop_payload is not None:
                    crop_payload["source_signal"] = "slab_barcode_or_label"
            else:
                crop_payload = u2netp_foreground_crop(image=image, target_aspect=target_aspect)

            if crop_payload is None or crop_payload.get("status") not in {
                "graded_slab_ratio_card",
                "u2netp_foreground",
            }:
                previous_crop_payload = crop_payload
                crop_payload = contour_card_crop(
                    image=image,
                    padding=padding,
                    target_aspect=target_aspect,
                    aspect_tolerance=aspect_tolerance,
                )
                if previous_crop_payload is not None and crop_payload is not None:
                    crop_payload["previous_crop_attempt"] = previous_crop_payload.get("status")
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
        if crop_payload["status"] in {"cropped", "slab_inner_card", "contour_card", "graded_slab_ratio_card", "u2netp_foreground"}:
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
                "slab_barcode": slab_barcode_payload,
                "results": [],
                "results_by_index": {},
                "timings": timings,
            }
    elif normalized_crop_mode in {"contour", "contour_card"}:
        crop_start = time.perf_counter()
        crop_payload = contour_card_crop(
            image=image,
            padding=padding,
            target_aspect=target_aspect,
            aspect_tolerance=aspect_tolerance,
        ) or {
            "status": "no_detection",
            "detections": [],
            "fallback_used": False,
        }
        timings["crop_seconds"] = time.perf_counter() - crop_start
        debug_crop_image = crop_payload.get("crop_image")
        if crop_payload["status"] == "contour_card":
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
                "slab_barcode": slab_barcode_payload,
                "results": [],
                "results_by_index": {},
                "timings": timings,
            }
    elif normalized_crop_mode in {"yolo", "model", "detector"}:
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
        debug_crop_image = crop_payload.get("crop_image")
        if crop_payload["status"] == "cropped":
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
                "slab_barcode": slab_barcode_payload,
                "results": [],
                "results_by_index": {},
                "timings": timings,
            }
    elif normalized_crop_mode in {"u2netp", "u2netp_foreground", "segmentation"}:
        crop_start = time.perf_counter()
        crop_payload = u2netp_foreground_crop(image=image, target_aspect=target_aspect)
        timings["crop_seconds"] = time.perf_counter() - crop_start
        debug_crop_image = crop_payload.get("crop_image")
        if crop_payload["status"] == "u2netp_foreground":
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
                "slab_barcode": slab_barcode_payload,
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
    candidate_windows = [top_k]
    if visual_rerank or card_code_ocr:
        candidate_windows.append(visual_rerank_candidates)
    if language_rerank or hint_card_code:
        candidate_windows.append(language_rerank_candidates)
    search_top_k = max(candidate_windows)
    search_per_index_top_k = max(per_index_top_k, search_top_k)
    combined, per_index = service.search(query, per_index_top_k=search_per_index_top_k, combined_top_k=search_top_k)
    timings["search_seconds"] = time.perf_counter() - search_start

    slab_hint_text = slab_label_ocr_text_from_payload(slab_barcode_payload)
    hint_text = language_hint_text.strip() or slab_hint_text
    hint_text_source = "request" if language_hint_text.strip() else ("slab_label_ocr" if slab_hint_text else "empty")
    hint_card_code_payload: dict[str, Any] = {
        "enabled": hint_card_code,
        "status": "skipped",
        "boost": 0.0,
    }
    if hint_card_code and hint_text:
        hint_card_code_payload = build_hint_card_code_payload(hint_text)
        hint_card_code_payload["boost"] = max(0.0, hint_card_code_boost)
        hint_card_code_payload["source"] = hint_text_source
        combined = apply_card_code_ocr_boost(
            combined,
            hint_card_code_payload,
            boost=max(0.0, hint_card_code_boost),
            field_prefix="hint_card_code",
        )

    language_payload: dict[str, Any] = {
        "enabled": language_rerank,
        "status": "skipped",
        "boost": 0.0,
        "ocr_enabled": language_ocr,
        "ocr_engine": language_ocr_engine if language_ocr else None,
    }
    if language_rerank:
        language_start = time.perf_counter()
        language_payload = infer_language_from_request_or_slab(
            language_hint,
            language_hint_text,
            slab_hint_text,
            enabled=True,
            boost=max(0.0, language_rerank_boost),
            ocr_enabled=language_ocr,
            ocr_engine=language_ocr_engine,
        )
        if language_payload.get("status") != "ok" and language_ocr:
            ocr_language_payload = recognize_language_with_ocr(query_image, language_ocr_engine)
            ocr_language_payload["enabled"] = True
            ocr_language_payload["boost"] = max(0.0, language_rerank_boost)
            language_payload = ocr_language_payload
        combined = apply_language_rerank(
            combined,
            language_payload,
            boost=max(0.0, language_rerank_boost),
        )
        language_payload.setdefault("ocr_enabled", language_ocr)
        language_payload.setdefault("ocr_engine", language_ocr_engine if language_ocr else None)
        timings["language_rerank_seconds"] = time.perf_counter() - language_start

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
        if crop_status in {"cropped", "fixed_inner_card", "slab_inner_card", "contour_card", "graded_slab_ratio_card", "u2netp_foreground"}:
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

    for rank, item in enumerate(combined, start=1):
        item["rank"] = rank
    candidate_selection = build_candidate_selection(combined)
    combined = combined[:top_k]

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
        "slab_barcode": slab_barcode_payload,
        "results": combined,
        "results_by_index": per_index,
        "candidate_selection": candidate_selection,
        "visual_rerank": {
            "enabled": visual_rerank,
            "model": rerank_model if visual_rerank else None,
            "candidates": search_top_k if visual_rerank else 0,
            "weight": visual_rerank_weight if visual_rerank else 0,
        },
        "language_rerank": language_payload,
        "hint_card_code": hint_card_code_payload,
        "card_code_ocr": ocr_payload,
        "timings": timings,
    }
