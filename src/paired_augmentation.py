"""Conservative geometry transforms applied identically to image and mask."""

from __future__ import annotations

from PIL import Image, ImageOps


GEOMETRIC_TRANSFORMS = ("identity", "horizontal_flip", "vertical_flip", "rotate_90", "rotate_180", "rotate_270")


def apply_paired_geometry(
    image: Image.Image, mask: Image.Image, transform: str
) -> tuple[Image.Image, Image.Image]:
    """Apply an exact, interpolation-free transform to image and target mask.

    This helper deliberately does not add color jitter, resize, blur or resample
    the masks. It is opt-in at training time; preparation outputs are untouched.
    """
    if image.size != mask.size:
        raise ValueError(f"Image and mask must align before augmentation: {image.size} != {mask.size}")
    operations = {
        "identity": lambda item: item.copy(),
        "horizontal_flip": ImageOps.mirror,
        "vertical_flip": ImageOps.flip,
        "rotate_90": lambda item: item.transpose(Image.Transpose.ROTATE_90),
        "rotate_180": lambda item: item.transpose(Image.Transpose.ROTATE_180),
        "rotate_270": lambda item: item.transpose(Image.Transpose.ROTATE_270),
    }
    if transform not in operations:
        raise ValueError(f"Unsupported transform {transform!r}; choose from {GEOMETRIC_TRANSFORMS}")
    operation = operations[transform]
    transformed_image, transformed_mask = operation(image), operation(mask)
    if transformed_image.size != transformed_mask.size:
        raise AssertionError("Paired transform broke image/mask alignment")
    return transformed_image, transformed_mask