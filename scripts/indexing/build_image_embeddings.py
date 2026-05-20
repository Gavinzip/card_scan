#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import iter_jsonl, write_json, write_jsonl
from scripts.lib.schema import utc_now_iso


def require_common_embedding_deps() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Embedding build requires optional deps. Install with: "
            "python3 -m pip install -r requirements-embedding.txt"
        ) from exc
    return np, torch, Image


def require_faiss() -> Any:
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise SystemExit("FAISS index writing requires faiss-cpu. Install requirements-embedding.txt.") from exc
    return faiss


def build_timm_model(model_name: str, torch: Any, device: str) -> tuple[Any, Any, str]:
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ModuleNotFoundError as exc:
        raise SystemExit("DINOv2/timm backend requires timm. Install requirements-embedding.txt.") from exc

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()
    data_config = resolve_model_data_config(model)
    preprocess = create_transform(**data_config, is_training=False)
    return model, preprocess, "timm"


def build_open_clip_model(model_name: str, pretrained: str, torch: Any, device: str) -> tuple[Any, Any, str]:
    try:
        import open_clip
    except ModuleNotFoundError as exc:
        raise SystemExit("OpenCLIP backend requires open-clip-torch. Install requirements-embedding.txt.") from exc

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    return model, preprocess, "open_clip"


def eligible_records(
    manifest_path: Path,
    allow_watermarked: bool,
    allow_sample: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = []
    excluded = {
        "missing_local_image_path": 0,
        "sample": 0,
        "watermarked": 0,
    }
    for record in iter_jsonl(manifest_path):
        path = record.get("local_image_path")
        if not path:
            excluded["missing_local_image_path"] += 1
            continue
        if record.get("is_sample") and not allow_sample:
            excluded["sample"] += 1
            continue
        if record.get("is_watermarked") and not allow_watermarked:
            excluded["watermarked"] += 1
            continue
        records.append(record)
    return records, excluded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP image embeddings and a FAISS index from clean image manifests.")
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/image_index")
    parser.add_argument("--backend", choices=["timm", "open_clip"], default="timm")
    parser.add_argument("--model", default="vit_small_patch14_dinov2")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow-watermarked",
        action="store_true",
        help="Include records flagged as watermarked, for explicit local reference indexes.",
    )
    parser.add_argument(
        "--allow-sample",
        action="store_true",
        help="Include records flagged as sample assets, for explicit local reference indexes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    np, torch, Image = require_common_embedding_deps()
    started_at = utc_now_iso()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if args.backend == "timm":
        model, preprocess, backend_name = build_timm_model(args.model, torch, device)
    else:
        model, preprocess, backend_name = build_open_clip_model(args.model, args.pretrained, torch, device)

    records, excluded = eligible_records(
        args.image_manifest,
        allow_watermarked=args.allow_watermarked,
        allow_sample=args.allow_sample,
    )
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit(
            "No eligible images found. Check local_image_path and sample/watermark flags. "
            "For explicit local reference indexes, pass --allow-watermarked or --allow-sample."
        )

    vectors = []
    kept_records = []
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        tqdm = None
    batch_starts = range(0, len(records), args.batch_size)
    if tqdm is not None:
        batch_starts = tqdm(batch_starts, desc="Embedding image batches")

    with torch.no_grad():
        for start in batch_starts:
            batch_records = records[start : start + args.batch_size]
            images = []
            batch_kept = []
            for record in batch_records:
                try:
                    image = Image.open(record["local_image_path"]).convert("RGB")
                    images.append(preprocess(image))
                    batch_kept.append(record)
                except OSError as exc:
                    print(f"Skipping unreadable image {record['local_image_path']}: {exc}", file=sys.stderr)
            if not images:
                continue
            tensor = torch.stack(images).to(device)
            if args.backend == "open_clip":
                embedding = model.encode_image(tensor)
            else:
                embedding = model(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            vectors.append(embedding.cpu().numpy())
            kept_records.extend(batch_kept)

    matrix = np.concatenate(vectors).astype("float32")
    # Import FAISS only after Torch has finished model setup and inference. On
    # some macOS builds, importing FAISS first can crash Torch weight init.
    faiss = require_faiss()
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    embeddings_path = args.output_dir / "image_embeddings.npy"
    manifest_path = args.output_dir / "image_embedding_manifest.jsonl"
    index_path = args.output_dir / "faiss.index"
    summary_path = args.output_dir / "summary.json"

    np.save(embeddings_path, matrix)
    faiss.write_index(index, str(index_path))
    write_jsonl(manifest_path, kept_records)
    summary = {
        "image_manifest": str(args.image_manifest),
        "records_indexed": len(kept_records),
        "embedding_dim": int(matrix.shape[1]),
        "model": args.model,
        "backend": backend_name,
        "pretrained": args.pretrained,
        "device": device,
        "allow_watermarked": args.allow_watermarked,
        "allow_sample": args.allow_sample,
        "excluded_before_limit": excluded,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "outputs": {
            "embeddings": str(embeddings_path),
            "manifest": str(manifest_path),
            "faiss_index": str(index_path),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
