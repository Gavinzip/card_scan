from __future__ import annotations

import re
from urllib.parse import unquote, urlparse


SAMPLE_RE = re.compile(r"(^|[_./?&=-])sample([_./?&=-]|$)|\bsample\b", re.IGNORECASE)

# The official Bandai card-list image path commonly renders preview images with
# visible sample markings. Treat it as a training-data risk until a pixel-level
# watermark check says otherwise.
WATERMARK_HINT_RE = re.compile(
    r"(sample|watermark|cardlist/card|onepiece-cardgame\.com)", re.IGNORECASE
)


def flag_onepiece_image_url(url: str | None) -> dict[str, object]:
    if not url:
        return {"is_sample": False, "is_watermarked": False, "reasons": []}

    decoded = unquote(url)
    parsed = urlparse(decoded)
    haystack = " ".join([decoded, parsed.netloc, parsed.path, parsed.query])

    reasons: list[str] = []
    is_sample = bool(SAMPLE_RE.search(haystack))
    if is_sample:
        reasons.append("url_contains_sample")

    is_watermarked = bool(WATERMARK_HINT_RE.search(haystack))
    if is_watermarked:
        reasons.append("url_watermark_hint")

    return {
        "is_sample": is_sample,
        "is_watermarked": is_watermarked,
        "reasons": reasons,
    }
