# Zeabur Deployment

This service deploys as a Docker app because the runtime includes PyTorch, FAISS, OpenCV, and Ultralytics.

## Included Runtime Assets

The Docker image serves the frontend from:

```text
web/
```

The Docker image copies these local indexes:

```text
data/processed/image_index/
data/processed/pokemon_ja_canonical_image_index/
data/processed/pokemon_ja_canonical_catalog.jsonl
data/processed/pokemon_ja_canonical_summary.json
data/processed/pokemon_ja_canonical_image_manifest.jsonl
```

The crop model is downloaded lazily from Hugging Face when `/warmup` or the
first cropped `/recognize` request runs:

```text
AlecKarfonta/cardcaptor-v3 / weights/cardcaptor_v3_best.pt
```

## API

Frontend:

```text
$APP_URL/
```

API metadata:

```bash
curl "$APP_URL/api"
```

Health:

```bash
curl "$APP_URL/health"
```

Warm loaded models manually:

```bash
curl -X POST "$APP_URL/warmup"
```

Recognize one card photo:

```bash
curl -F "file=@/path/to/photo.jpg" \
  "$APP_URL/recognize?top_k=5&per_index_top_k=5"
```

Generate recognition, Raw/A and PSA 10 prices, the text report, and TCGPro
poster PNGs in one request:

```bash
curl -F "file=@/path/to/photo.jpg" \
  "$APP_URL/market-report?crop=true&crop_mode=tcgp_obb&top_k=5&include_posters=true"
```

The endpoint writes generated files below `CARD_SCAN_REPORT_OUTPUT_DIR` and
serves them from `/reports/{report_id}/{filename}`. It uses SNKRDUNK trading
histories, applies a per-bucket IQR outlier filter, and returns the filtered
Raw/A and PSA 10 summaries in JSON.

No silent crop fallback is used by default. If the card detector does not find a card, the API returns `status=no_detection`. To explicitly search the original image anyway:

```bash
curl -F "file=@/path/to/photo.jpg" \
  "$APP_URL/recognize?fallback_to_original=true"
```

## Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8080` | HTTP port. Zeabur injects this in Docker deployments. |
| `WEB_CONCURRENCY` | `1` | Keep at `1` unless the instance has enough RAM for duplicate model copies. |
| `CARD_SCAN_PRELOAD` | `false` | Load models lazily. Set to `true` only when the instance has enough startup time and memory. |
| `CARD_SCAN_DEVICE` | `cpu` | Use `cpu` on Zeabur CPU instances. |
| `CARD_SCAN_INDEXES` | `pokemon_en=data/processed/image_index,pokemon_ja=data/processed/pokemon_ja_canonical_image_index` | Comma-separated index name/path pairs. |
| `CARD_SCAN_IMAGE_ROOTS` | empty | Optional comma-separated `name=/path` roots for local reference images served through `/reference-images/{name}/...`. |
| `CARD_SCAN_LOCAL_PATH_REWRITES` | empty | Optional comma-separated `old_prefix=new_prefix` rules for mapping manifest `local_image_path` values to mounted server paths. |
| `CARD_SCAN_REFERENCE_IMAGE_ROUTE` | `/reference-images` | URL route prefix for mounted reference images. |
| `CARD_SCAN_CORS_ORIGINS` | empty | Optional comma-separated origins if a separate frontend calls this API. Leave empty for same-origin Zeabur deployment. |
| `CARD_SCAN_CROP_CONFIDENCE` | `0.25` | YOLO detection confidence threshold. |
| `CARD_SCAN_CROP_IMGSZ` | `1024` | YOLO inference image size. |
| `CARD_SCAN_REPORT_OUTPUT_DIR` | `/tmp/card_scan_reports` | Directory for `/market-report` markdown and poster files served under `/reports`. |
| `CARD_SCAN_SNKR_HISTORY_TIMEOUT` | `12` | Timeout in seconds for SNKRDUNK trading-history requests. |
| `CARD_SCAN_MARKET_JPY_RATE` | `150` | JPY/USD conversion rate used for poster labels when SNKRDUNK records are rendered as JPY with USD hints. |

## Expected Performance

Local MPS hot-path timing was about 0.5 seconds per image. A CPU-only Zeabur instance will likely be slower because both YOLO and DINOv2 run on CPU. Start with at least 2 CPU cores and enough memory for PyTorch, Ultralytics, FAISS, and the loaded indexes.

## Deployment Notes

- Do not push raw card images; the Docker image only needs the FAISS indexes and embedding manifests.
- Ensure `data/processed/image_index/` and `data/processed/pokemon_ja_canonical_image_index/` are present in the deployed repository or build context.
- `zbpack.json` pins Zeabur to the root `Dockerfile`.
- The Dockerfile avoids heredoc Python blocks because Zeabur can inject build-time `ARG`/`ENV` lines into multi-line build commands.
- The crop model is not downloaded during image build; use `/warmup` after deploy if you want the first scan to be faster.
- The Docker image installs the X11/OpenGL runtime libraries that OpenCV wheels import even when the service uses headless image processing.
- The returned `local_image_path` values are provenance paths from the build machine and should not be treated as public URLs. Use `image_url` for remote display when available.
- To show local Kaggle English images in the frontend, mount the image folder and configure `CARD_SCAN_IMAGE_ROOTS` plus `CARD_SCAN_LOCAL_PATH_REWRITES`; see `docs/reference_images.md`.
- Official Japanese and TCGdex images are local reference/search assets, not training or redistribution assets.
