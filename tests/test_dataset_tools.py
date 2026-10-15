"""Fast unit tests for annotation semantics and dataset discovery helpers."""

from pathlib import Path

from PIL import Image

from src.convert_annotations import convert_mask
from src.analyze_mask_thresholds import otsu_threshold_from_histogram
from src.paired_augmentation import apply_paired_geometry
from src.utils import find_image_mask_pairs, stable_sample_id
from src.visualize_annotations import display_mask


def test_real_binary_mask_conversion_and_unknown_color_ignore():
    mask = Image.new("RGB", (3, 1))
    mask.putdata([(0, 0, 0), (255, 255, 255), (0, 255, 0)])
    converted, counts = convert_mask(mask, "Thumbnail images/Real images/With fungi")
    assert list(converted.get_flattened_data()) == [0, 1, 255]
    assert counts == {"background": 1, "fungus": 1, "ignored_unrecognized": 1}


def test_synthetic_threshold_is_explicit():
    mask = Image.new("RGB", (4, 1))
    mask.putdata([(0, 0, 0), (127, 127, 127), (128, 128, 128), (255, 255, 255)])
    converted, _ = convert_mask(mask, "Thumbnail images/Synthetic images", synthetic_threshold=128)
    assert list(converted.get_flattened_data()) == [0, 0, 1, 1]


def test_display_mask_scales_class_ids_for_human_review_only():
    mask = Image.new("L", (3, 1))
    mask.putdata([0, 1, 255])
    shown = display_mask(mask)
    assert list(shown.get_flattened_data()) == [
        (0, 0, 0), (255, 255, 255), (255, 200, 0)
    ]
    assert list(mask.get_flattened_data()) == [0, 1, 255]


def test_otsu_threshold_separates_two_intensity_modes():
    histogram = [0] * 256
    histogram[20] = 10
    histogram[220] = 10
    assert otsu_threshold_from_histogram(histogram) == 20


def test_paired_geometric_transform_keeps_image_and_mask_pixels_aligned():
    image = Image.new("L", (3, 2))
    mask = Image.new("L", (3, 2))
    image.putdata([10, 20, 30, 40, 50, 60])
    mask.putdata([0, 0, 1, 0, 1, 1])
    for transform in ("horizontal_flip", "vertical_flip", "rotate_90", "rotate_180", "rotate_270"):
        transformed_image, transformed_mask = apply_paired_geometry(image, mask, transform)
        assert transformed_image.size == transformed_mask.size
        expected = [1 if value in {30, 50, 60} else 0 for value in transformed_image.get_flattened_data()]
        assert list(transformed_mask.get_flattened_data()) == expected


def test_paired_transform_rejects_unaligned_inputs_and_unknown_names():
    image = Image.new("RGB", (3, 2))
    mask = Image.new("L", (2, 3))
    import pytest

    with pytest.raises(ValueError, match="must align"):
        apply_paired_geometry(image, mask, "identity")
    with pytest.raises(ValueError, match="Unsupported transform"):
        apply_paired_geometry(image, Image.new("L", image.size), "arbitrary_crop")


def test_discovery_matches_exact_names_and_reports_unmatched_masks(tmp_path: Path):
    images = tmp_path / "sample" / "images"
    masks = tmp_path / "sample" / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    Image.new("RGB", (2, 2)).save(images / "one.tif")
    Image.new("L", (2, 2)).save(masks / "one.tif")
    Image.new("L", (2, 2)).save(masks / "orphan.tif")
    pairs, unmatched = find_image_mask_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0]["mask_path"] == masks / "one.tif"
    assert unmatched == ["sample/masks/orphan.tif"]


def test_sample_ids_are_stable_and_unique_for_source_paths():
    first = stable_sample_id("Synthetic/images/0.tif")
    assert first == stable_sample_id("Synthetic/images/0.tif")
    assert first != stable_sample_id("Real/images/0.tif")