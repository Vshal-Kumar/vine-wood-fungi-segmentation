"""Validate processed semantic masks, pair alignment and split integrity."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.utils import load_config, sha256_file


def validate_dataset() -> dict:
    config = load_config()
    project = config["_project_root"]
    split_data = json.loads(config["splits_file"].read_text(encoding="utf-8"))
    problems: list[str] = []
    split_hashes: dict[str, set[str]] = defaultdict(set)
    split_groups: dict[str, Counter] = defaultdict(Counter)
    split_foreground: dict[str, int] = Counter()
    split_empty: dict[str, int] = Counter()
    split_counts: dict[str, int] = Counter()
    seen_ids: set[str] = set()
    for split, rows in split_data["splits"].items():
        for row in rows:
            split_counts[split] += 1
            split_groups[split][row["group"]] += 1
            if row["sha256"] in split_hashes[split]:
                pass  # Exact duplicate records are allowed but remain within one split.
            split_hashes[split].add(row["sha256"])
            if row["sample_id"] in seen_ids:
                problems.append(f"Duplicate sample ID across manifest: {row['sample_id']}")
            seen_ids.add(row["sample_id"])
            image_path = project / row["image"]
            mask_path = project / row["mask"]
            # Split tree is the trainer-facing dataset; also validate source files.
            split_image = config["processed_root"] / "images" / split / (
                row["sample_id"] + image_path.suffix.lower()
            )
            split_mask = config["processed_root"] / "masks" / split / f"{row['sample_id']}.png"
            for path in (image_path, mask_path, split_image, split_mask):
                if not path.is_file():
                    problems.append(f"Missing dataset file: {path}")
            if not all(path.is_file() for path in (image_path, mask_path, split_image, split_mask)):
                continue
            try:
                if sha256_file(image_path) != row["sha256"]:
                    problems.append(f"Source image hash changed: {row['sample_id']}")
                if sha256_file(split_image) != row["sha256"]:
                    problems.append(f"Processed image bytes differ from source: {row['sample_id']}")
                with Image.open(image_path) as image, Image.open(mask_path) as mask:
                    image.load()
                    mask.load()
                    if image.size != mask.size:
                        problems.append(f"Source dimension mismatch: {row['sample_id']} {image.size} vs {mask.size}")
                    if image.size != (256, 256):
                        problems.append(f"Unexpected thumbnail dimensions: {row['sample_id']} {image.size}")
                with Image.open(split_image) as image, Image.open(split_mask) as mask:
                    image.load()
                    mask.load()
                    if image.size != mask.size:
                        problems.append(f"Split dimension mismatch: {row['sample_id']} {image.size} vs {mask.size}")
                    values = set(mask.get_flattened_data())
                    invalid = values - {0, 1, config["ignore_index"]}
                    if invalid:
                        problems.append(f"Invalid mask IDs {sorted(invalid)} in {row['sample_id']}")
                    positive = mask.histogram()[1]
                    if positive:
                        split_foreground[split] += positive
                    else:
                        split_empty[split] += 1
            except Exception as error:
                problems.append(f"Unreadable processed pair {row['sample_id']}: {error}")
    overlap = {
        f"{left}/{right}": sorted(split_hashes[left] & split_hashes[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    for pair, hashes in overlap.items():
        if hashes:
            problems.append(f"Exact duplicate image hashes leak across {pair}: {len(hashes)}")
    expected = split_data["record_count_including_duplicates"]
    actual = sum(split_counts.values())
    if expected != actual:
        problems.append(f"Manifest record count {expected} != split count {actual}")
    result = {
        "status": "PASS" if not problems else "FAIL",
        "task": "binary semantic segmentation",
        "split_record_counts": dict(split_counts),
        "split_unique_image_hash_counts": {name: len(values) for name, values in split_hashes.items()},
        "split_source_groups": {name: dict(groups) for name, groups in split_groups.items()},
        "split_fungus_pixel_counts": dict(split_foreground),
        "split_empty_annotation_counts": dict(split_empty),
        "exact_hash_overlap": {pair: len(values) for pair, values in overlap.items()},
        "problems": problems,
    }
    report = [
        "# Dataset validation report", "", f"**Status: {result['status']}**", "",
        "Checks cover image and mask file existence/readability, source and split alignment, expected thumbnail dimensions, permitted mask values {0, 1, 255}, manifest consistency, and exact-image-hash leakage across splits.",
        "", "## Split counts and coverage", "",
        "| Split | Records | Unique image hashes | Empty masks | Fungus pixels | Source groups |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for split in ("train", "val", "test"):
        report.append(
            f"| {split} | {split_counts[split]} | {len(split_hashes[split])} | {split_empty[split]} | {split_foreground[split]} | {dict(split_groups[split])} |"
        )
    report.extend(["", "## Exact duplicate leakage", "", *[f"- `{pair}`: {len(hashes)} overlapping hashes" for pair, hashes in overlap.items()], "", "## Errors", ""])
    report.extend([f"- {problem}" for problem in problems] if problems else ["- None."])
    report.extend([
        "", "## Interpretation", "",
        "A successful file-level validation does not resolve the synthetic-mask grayscale convention; the configured midpoint threshold remains a documented modeling assumption. Review the generated overlays and annotation analysis before training.",
        "", "Training is not started by this project preparation workflow. This dataset is semantic segmentation, not standard YOLO instance segmentation.", "",
    ])
    (project / "reports" / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    (project / "reports" / "validation_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = validate_dataset()
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)