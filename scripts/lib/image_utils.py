from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def iter_image_files(root: str | Path) -> Iterator[Path]:
    base = Path(root)
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: str | Path) -> dict[str, object]:
    target = Path(path)
    try:
        with Image.open(target) as image:
            image.load()
            width, height = image.size
            return {
                "is_valid_image": True,
                "width": width,
                "height": height,
                "format": image.format,
                "image_error": None,
            }
    except (OSError, UnidentifiedImageError) as exc:
        return {
            "is_valid_image": False,
            "width": None,
            "height": None,
            "format": None,
            "image_error": str(exc),
        }


def infer_card_fields_from_image_path(path: str | Path, image_root: str | Path) -> dict[str, str | None]:
    target = Path(path)
    root = Path(image_root)
    try:
        relative = target.relative_to(root)
    except ValueError:
        relative = target

    parts = relative.parts
    set_id = parts[-2] if len(parts) >= 2 else None
    return {
        "set_id": set_id,
        "card_code": target.stem,
        "variant": None,
    }


def infer_pokemon_kaggle_fields(path: str | Path, image_root: str | Path) -> dict[str, str | bool | None]:
    target = Path(path)
    root = Path(image_root)
    try:
        relative = target.relative_to(root)
    except ValueError:
        relative = target

    folder_slug = relative.parts[-2] if len(relative.parts) >= 2 else None
    stem = target.stem
    result: dict[str, str | bool | None] = {
        "card_id": None,
        "card_code": stem,
        "set_id": folder_slug,
        "language": None,
        "variant": None,
        "kaggle_set_slug": folder_slug,
        "kaggle_filename": target.name,
        "pokemon_filename_parse_rule": "unparsed",
        "pokemon_filename_parsed": False,
    }

    modern = re.match(
        r"^(?P<set_id>[^_]+)_(?P<language>[a-z]{2})_(?P<card_code>[^_]+)(?:_(?P<variant>.+))?$",
        stem,
    )
    if modern:
        groups = modern.groupdict()
        result.update(groups)
        result["card_id"] = f"{groups['set_id']}-{groups['card_code']}"
        result["pokemon_filename_parse_rule"] = "modern_underscore"
        result["pokemon_filename_parsed"] = True
        return result

    regional = re.match(
        r"^(?P<language>[a-z]{2})_[A-Z]{2}-(?P<set_id>.+)-(?P<card_code>[A-Za-z]*\d+[A-Za-z]?)-(?P<name_slug>.+)$",
        stem,
    )
    if regional:
        groups = regional.groupdict()
        result.update({key: groups[key] for key in ("language", "set_id", "card_code")})
        result["card_id"] = f"{groups['set_id']}-{groups['card_code']}"
        result["pokemon_name_slug"] = groups["name_slug"]
        result["pokemon_filename_parse_rule"] = "regional_hyphen"
        result["pokemon_filename_parsed"] = True
        return result

    slugged = re.match(r"^(?P<name_and_set>.+)-(?P<card_code>[A-Za-z]*\d+[A-Za-z]?)$", stem)
    if slugged and folder_slug:
        result.update(
            {
                "card_code": slugged.group("card_code"),
                "card_id": f"{folder_slug}-{slugged.group('card_code')}",
                "pokemon_name_slug": slugged.group("name_and_set"),
                "pokemon_filename_parse_rule": "legacy_slugged",
                "pokemon_filename_parsed": True,
            }
        )
        return result

    compact = re.match(r"^(?P<name_slug>.+?)(?P<card_code>\d+)$", stem)
    if compact and folder_slug:
        result.update(
            {
                "card_code": compact.group("card_code"),
                "card_id": f"{folder_slug}-{compact.group('card_code')}",
                "pokemon_name_slug": compact.group("name_slug"),
                "pokemon_filename_parse_rule": "legacy_compact_suffix",
                "pokemon_filename_parsed": True,
            }
        )
        return result

    return result
