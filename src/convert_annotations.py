"""Convert the dataset's binary semantic masks to auditable class-ID masks."""

from __future__ import annotations

from PIL import Image

from src.utils import is_synthetic_group, load_config, mask_to_class_ids


def convert_mask(
    mask: Image.Image,
    group: str,
    synthetic_threshold: int = 128,
    ignore_index: int = 255,
) -> tuple[Image.Image, dict[str, int]]:
    """Convert one official binary mask while preserving its size and unknowns."""
    return mask_to_class_ids(
        mask,
        synthetic=is_synthetic_group(group),
        synthetic_threshold=synthetic_threshold,
        ignore_index=ignore_index,
    )


def get_conversion_settings() -> dict:
    """Expose the centralized label values and documented synthetic threshold."""
    config = load_config()
    return {
        "classes": config["classes"],
        "ignore_index": config["ignore_index"],
        "synthetic_mask_threshold": config["synthetic_mask_threshold"],
    }