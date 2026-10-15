"""Compare fixed and Otsu thresholds on synthetic masks without changing labels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from src.utils import load_config


def otsu_threshold_from_histogram(histogram: list[int]) -> int:
    """Return the Otsu threshold maximizing between-class variance."""
    total = sum(histogram)
    if total == 0:
        raise ValueError("Cannot calculate Otsu threshold from an empty histogram.")
    weighted_total = sum(level * count for level, count in enumerate(histogram))
    background_weight = 0
    background_sum = 0
    best_variance = -1.0
    best_threshold = 0
    for threshold, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_total - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def analyze_thresholds() -> dict:
    """Compare the configured fixed threshold with pooled/per-mask Otsu."""
    config = load_config()
    threshold_fixed = int(config["synthetic_mask_threshold"])
    root: Path = config["extracted_root"] / "Thumbnail images" / "Synthetic images"
    image_dir = root / "images"
    mask_dir = root / "masks"
    pooled_histogram = [0] * 256
    per_mask_thresholds: list[int] = []
    total_pixels = 0
    fixed_positive = 0
    pooled_otsu_relabelled = 0
    per_mask_otsu_relabelled = 0
    per_mask_change_fractions: list[float] = []
    cached: list[tuple[Path, Path, list[int], int]] = []
    for mask_path in sorted(mask_dir.glob("*.tif")):
        image_path = image_dir / mask_path.name
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing paired synthetic image: {image_path}")
        with Image.open(mask_path) as source_mask:
            gray = source_mask.convert("L")
            histogram = gray.histogram()
        threshold = otsu_threshold_from_histogram(histogram)
        per_mask_thresholds.append(threshold)
        pixels = sum(histogram)
        total_pixels += pixels
        fixed_positive += sum(histogram[threshold_fixed:])
        per_mask_otsu_relabelled += sum(histogram[threshold + 1 : threshold_fixed])
        per_mask_change_fractions.append(
            sum(histogram[threshold + 1 : threshold_fixed]) / pixels
        )
        pooled_histogram = [left + right for left, right in zip(pooled_histogram, histogram)]
        cached.append((image_path, mask_path, histogram, threshold))
    if not cached:
        raise FileNotFoundError(f"No synthetic TIFF masks found in {mask_dir}")
    pooled_threshold = otsu_threshold_from_histogram(pooled_histogram)
    pooled_otsu_positive = sum(pooled_histogram[pooled_threshold + 1 :])
    fixed_positive_pooled = sum(pooled_histogram[threshold_fixed:])
    pooled_otsu_relabelled = max(0, pooled_otsu_positive - fixed_positive_pooled)
    sorted_thresholds = sorted(per_mask_thresholds)
    sorted_changes = sorted(per_mask_change_fractions)
    summary = {
        "synthetic_mask_count": len(cached),
        "current_method": f"fixed grayscale >= {threshold_fixed}",
        "per_mask_otsu_threshold_min_median_max": [
            sorted_thresholds[0], sorted_thresholds[len(sorted_thresholds) // 2], sorted_thresholds[-1]
        ],
        "per_mask_otsu_threshold_distribution": dict(sorted(Counter(per_mask_thresholds).items())),
        "pooled_otsu_threshold": pooled_threshold,
        "pixels_changed_by_pooled_otsu_vs_current": pooled_otsu_relabelled,
        "fraction_all_mask_pixels_changed_by_pooled_otsu": pooled_otsu_relabelled / total_pixels,
        "pixels_changed_by_per_mask_otsu_vs_current": per_mask_otsu_relabelled,
        "fraction_all_mask_pixels_changed_by_per_mask_otsu": per_mask_otsu_relabelled / total_pixels,
        "median_fraction_pixels_changed_per_mask": sorted_changes[len(sorted_changes) // 2],
        "fixed_foreground_pixels": fixed_positive,
        "pooled_otsu_foreground_pixels": pooled_otsu_positive,
        "per_mask_otsu_foreground_pixels": fixed_positive + per_mask_otsu_relabelled,
        "threshold_otsu_convention": "classify intensity > threshold as fungus",
        "training_masks_modified": False,
    }
    report_dir = config["_project_root"] / "reports"
    visual_dir = report_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    # Visualize the median per-mask Otsu case; retain the image, original grayscale
    # mask, and both threshold outputs at the original thumbnail resolution.
    selected_threshold = sorted_thresholds[len(sorted_thresholds) // 2]
    selected = min(cached, key=lambda item: abs(item[3] - selected_threshold))
    image_path, mask_path, _, selected_otsu = selected
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
    with Image.open(mask_path) as mask_file:
        gray_mask = mask_file.convert("L")
    fixed_mask = gray_mask.point(lambda value: 255 if value >= threshold_fixed else 0)
    otsu_mask = gray_mask.point(lambda value: 255 if value > selected_otsu else 0)
    panel = Image.new("RGB", (1024, 290), "white")
    draw = ImageDraw.Draw(panel)
    panels = [
        (image, "source image"),
        (gray_mask.convert("RGB"), "original grayscale mask"),
        (fixed_mask.convert("RGB"), f"fixed >= {threshold_fixed}"),
        (otsu_mask.convert("RGB"), f"per-mask Otsu > {selected_otsu}"),
    ]
    for index, (tile, title) in enumerate(panels):
        x = index * 256
        panel.paste(tile.resize((256, 256), Image.Resampling.NEAREST), (x, 28))
        draw.text((x + 5, 8), title, fill="black")
    visual_path = visual_dir / "otsu_mask_threshold_comparison.png"
    panel.save(visual_path)
    summary["comparison_visualization"] = visual_path.relative_to(config["_project_root"]).as_posix()
    (report_dir / "mask_threshold_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Synthetic mask threshold comparison", "",
        "This is a **diagnostic experiment only**. It did not modify the processed masks or the configured training threshold.", "",
        f"- Synthetic masks inspected: {summary['synthetic_mask_count']}",
        f"- Current preprocessing rule: `{summary['current_method']}`",
        f"- Per-mask Otsu thresholds (min/median/max): {summary['per_mask_otsu_threshold_min_median_max']}",
        f"- Pooled-dataset Otsu threshold: {pooled_threshold}",
        f"- Pixels changed using pooled Otsu vs current rule: {pooled_otsu_relabelled:,} / {total_pixels:,} ({100 * summary['fraction_all_mask_pixels_changed_by_pooled_otsu']:.3f}%)",
        f"- Pixels changed using per-mask Otsu vs current rule: {per_mask_otsu_relabelled:,} / {total_pixels:,} ({100 * summary['fraction_all_mask_pixels_changed_by_per_mask_otsu']:.3f}%)",
        f"- Foreground pixels: current={fixed_positive:,}; pooled Otsu={pooled_otsu_positive:,}; per-mask Otsu={summary['per_mask_otsu_foreground_pixels']:,}.",
        f"- Visual comparison: `{summary['comparison_visualization']}`",
        "", "## Interpretation", "",
        "Otsu is a histogram-based binarizer, not an edge detector and not a measure of annotation correctness. It chooses a threshold that separates grayscale intensities; the dataset README does not define what each synthetic gray level means. Here Otsu thresholds are consistently below the current midpoint and would label more faint boundary pixels as fungus. That may thicken/expand boundaries rather than improve ground truth.",
        "",
        "Keep the current midpoint labels as the baseline for now. If comparing Otsu, make it a separate training-data variant, train otherwise identical models, select using validation only, and inspect thin structures/boundaries. Do not replace or tune labels using the held-out test set.",
        "",
        "Edge methods such as Sobel/Canny should likewise not replace masks. A future experiment can add a derived edge input channel or boundary-aware training loss, but that requires a model-specific implementation and validation comparison. No edge filtering was applied to images or masks in this preparation run.",
        "",
    ]
    (report_dir / "mask_threshold_comparison.md").write_text("\n".join(report), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze_thresholds(), indent=2))