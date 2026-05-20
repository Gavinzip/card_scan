#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import iter_jsonl, read_json


def load_model(summary: dict[str, Any], device_arg: str | None) -> tuple[Any, Any, Any, Any, str]:
    import numpy as np
    import torch
    from PIL import Image

    if device_arg:
        device = device_arg
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    backend = summary.get("backend", "timm")
    model_name = summary["model"]
    if backend == "timm":
        import timm
        from timm.data import create_transform, resolve_model_data_config

        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model = model.to(device)
        model.eval()
        preprocess = create_transform(**resolve_model_data_config(model), is_training=False)
    elif backend == "open_clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=summary.get("pretrained"),
        )
        model = model.to(device)
        model.eval()
    else:
        raise SystemExit(f"Unsupported backend: {backend}")

    return np, torch, Image, (model, preprocess), device


def encode_query(image_path: Path, summary: dict[str, Any], device_arg: str | None) -> Any:
    np, torch, Image, model_pack, device = load_model(summary, device_arg)
    model, preprocess = model_pack
    backend = summary.get("backend", "timm")
    with Image.open(image_path) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        if backend == "open_clip":
            embedding = model.encode_image(tensor)
        else:
            embedding = model(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype("float32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a FAISS image index with a card/photo image.")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data/processed/image_index")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--query-vector", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def search_faiss(args: argparse.Namespace, query: Any) -> dict[str, Any]:
    index_path = args.index_dir / "faiss.index"
    manifest_path = args.index_dir / "image_embedding_manifest.jsonl"
    if not index_path.exists() or not manifest_path.exists():
        raise SystemExit(f"Index directory is incomplete: {args.index_dir}")

    import faiss

    index = faiss.read_index(str(index_path))
    records = list(iter_jsonl(manifest_path))
    scores, indices = index.search(query, args.top_k)

    results = []
    for score, index_id in zip(scores[0].tolist(), indices[0].tolist(), strict=False):
        if index_id < 0 or index_id >= len(records):
            continue
        record = records[index_id]
        results.append(
            {
                "rank": len(results) + 1,
                "score": score,
                "source": record.get("source"),
                "card_id": record.get("card_id"),
                "card_code": record.get("card_code"),
                "set_id": record.get("set_id"),
                "language": record.get("language"),
                "name": record.get("name"),
                "name_en": record.get("name_en"),
                "name_ja": record.get("name_ja"),
                "variant": record.get("variant"),
                "local_image_path": record.get("local_image_path"),
                "image_sha256": record.get("image_sha256"),
            }
        )

    payload = {
        "query_image": str(args.image) if args.image else None,
        "index_dir": str(args.index_dir),
        "top_k": args.top_k,
        "results": results,
    }
    return payload


def main() -> int:
    args = parse_args()
    summary_path = args.index_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Index directory is incomplete: {args.index_dir}")

    if os.environ.get("CARD_SCAN_FAISS_WORKER") == "1":
        if not args.query_vector:
            raise SystemExit("FAISS worker requires --query-vector.")
        import numpy as np

        payload = search_faiss(args, np.load(args.query_vector).astype("float32"))
    else:
        if not args.image:
            raise SystemExit("Pass --image.")
        if not args.image.exists():
            raise SystemExit(f"Query image does not exist: {args.image}")
        summary = read_json(summary_path)
        query = encode_query(args.image, summary, args.device)
        with tempfile.TemporaryDirectory() as tmpdir:
            query_path = Path(tmpdir) / "query.npy"
            import numpy as np

            np.save(query_path, query)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--index-dir",
                str(args.index_dir),
                "--query-vector",
                str(query_path),
                "--top-k",
                str(args.top_k),
            ]
            if args.image:
                command.extend(["--image", str(args.image)])
            env = os.environ.copy()
            env["CARD_SCAN_FAISS_WORKER"] = "1"
            completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
            payload = json.loads(completed.stdout)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
