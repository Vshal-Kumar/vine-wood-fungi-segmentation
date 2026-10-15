"""Shared paths, image discovery and annotation helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset.yaml"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the central dataset configuration and resolve paths at project root."""
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["_config_path"] = config_path.resolve()
    config["_project_root"] = config_path.resolve().parents[1]
    for key in ("archive", "extracted_root", "processed_root", "splits_file"):
        config[key] = (config["_project_root"] / config[key]).resolve()
    config["image_extensions"] = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in config.get("image_extensions", SUPPORTED_EXTENSIONS)
    }
    return config


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for exact duplicate detection."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sample_id(relative_image: str) -> str:
    """Create a portable unique ID from a source-relative image path."""
    path = Path(relative_image)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")
    parent = re.sub(r"[^A-Za-z0-9_-]+", "_", path.parent.parent.as_posix()).strip("_")
    short_hash = hashlib.sha1(relative_image.encode("utf-8")).hexdigest()[:8]
    return f"{parent}_{stem}_{short_hash}"


def find_image_mask_pairs(
    dataset_root: Path, extensions: set[str] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Discover sibling images/ and masks/ directories; match by exact filename."""
    extensions = extensions or SUPPORTED_EXTENSIONS
    records: list[dict[str, Any]] = []
    unmatched_annotations: list[str] = []
    for image_dir in sorted(path for path in dataset_root.rglob("images") if path.is_dir()):
        mask_dir = image_dir.parent / "masks"
        if not mask_dir.is_dir():
            continue
        masks = {path.name: path for path in mask_dir.iterdir() if path.is_file()}
        for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
            if image_path.suffix.lower() not in extensions:
                continue
            mask_path = masks.pop(image_path.name, None)
            group = image_dir.parent.relative_to(dataset_root).as_posix()
            relative = image_path.relative_to(dataset_root).as_posix()
            records.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "image_relative": relative,
                    "mask_relative": mask_path.relative_to(dataset_root).as_posix()
                    if mask_path
                    else None,
                    "group": group,
                    "sample_id": stable_sample_id(relative),
                }
            )
        unmatched_annotations.extend(
            path.relative_to(dataset_root).as_posix() for path in sorted(masks.values())
        )
    return records, unmatched_annotations


def mask_to_class_ids(
    mask: Image.Image,
    *,
    synthetic: bool,
    synthetic_threshold: int = 128,
    ignore_index: int = 255,
) -> tuple[Image.Image, dict[str, int]]:
    """Convert documented binary masks into IDs 0/1 and explicit ignore pixels.

    Synthetic masks have antialiased grayscale values; the documentation calls
    the target binary, so the configured midpoint threshold is explicit. Real
    masks use exact black/white colors. Unexpected colors become ignore_index
    rather than being silently assigned to fungus or background.
    """
    rgb = mask.convert("RGB")
    class_pixels: list[int] = []
    counts = {"background": 0, "fungus": 0, "ignored_unrecognized": 0}
    for red, green, blue in rgb.get_flattened_data():
        if synthetic and red == green == blue:
            label = 1 if red >= synthetic_threshold else 0
        elif (red, green, blue) == (0, 0, 0):
            label = 0
        elif (red, green, blue) == (255, 255, 255):
            label = 1
        else:
            label = ignore_index
        class_pixels.append(label)
        if label == 0:
            counts["background"] += 1
        elif label == 1:
            counts["fungus"] += 1
        else:
            counts["ignored_unrecognized"] += 1
    converted = Image.new("L", rgb.size)
    converted.putdata(class_pixels)
    return converted, counts


def is_synthetic_group(group: str) -> bool:
    return group.startswith("Thumbnail images/Synthetic images")