"""Inventory the extracted dataset and its image/mask relationships."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.utils import find_image_mask_pairs, load_config, sha256_file


def inspect_dataset(config_path: Path | None = None) -> dict:
    config = load_config(config_path) if config_path else load_config()
    root: Path = config["extracted_root"]
    records, unmatched_masks = find_image_mask_pairs(root, config["image_extensions"])
    image_formats: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    annotation_extensions: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    hashes_to_rows: dict[str, list[dict]] = defaultdict(list)
    corrupt: list[dict[str, str]] = []
    trainable_groups = set(config["include_groups"])
    for record in records:
        image_path: Path = record["image_path"]
        relative = record["image_relative"]
        record["image_sha256"] = sha256_file(image_path)
        hashes[record["image_sha256"]].append(relative)
        record["train_candidate"] = record["group"] in trainable_groups
        try:
            with Image.open(image_path) as image:
                image.load()
                record.update(
                    width=image.width,
                    height=image.height,
                    image_format=image.format,
                    image_mode=image.mode,
                    channels=len(image.getbands()),
                    bit_depth=8 if image.mode in {"1", "L", "P", "RGB", "RGBA"} else None,
                    file_size_bytes=image_path.stat().st_size,
                    decodable=True,
                )
                image_formats[str(image.format)] += 1
                image_modes[image.mode] += 1
                dimensions[f"{image.width}x{image.height}"] += 1
        except Exception as error:  # pragma: no cover - depends on damaged inputs
            record.update(decodable=False, decode_error=str(error))
            corrupt.append({"path": relative, "error": str(error)})
        if record["mask_path"]:
            annotation_extensions[Path(record["mask_path"]).suffix.lower()] += 1
            record["mask_sha256"] = sha256_file(Path(record["mask_path"]))
        hashes_to_rows[record["image_sha256"]].append(record)
        record["image_path"] = relative
        record["mask_path"] = record["mask_relative"]
        del record["mask_relative"]
    docs = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".pdf", ".json", ".xml", ".csv"}:
            docs.append({"path": path.relative_to(root).as_posix(), "extension": path.suffix.lower()})
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    duplicate_groups_with_mask_disagreement = [
        [row["image_relative"] for row in rows]
        for rows in hashes_to_rows.values()
        if len(rows) > 1 and len({row.get("mask_sha256") for row in rows}) > 1
    ]
    inventory = {
        "dataset_root": root.as_posix(),
        "archive": config["archive"].as_posix(),
        "image_count": len(records),
        "annotation_count": sum(record["mask_path"] is not None for record in records)
        + len(unmatched_masks),
        "matched_pairs": sum(record["mask_path"] is not None for record in records),
        "unmatched_images": sum(record["mask_path"] is None for record in records),
        "unmatched_annotations": unmatched_masks,
        "image_formats": dict(image_formats),
        "image_modes": dict(image_modes),
        "dimensions": dict(dimensions),
        "annotation_extensions": dict(annotation_extensions),
        "exact_duplicate_image_group_count": len(duplicate_groups),
        "exact_duplicate_extra_image_count": sum(len(group) - 1 for group in duplicate_groups),
        "exact_duplicate_groups": duplicate_groups,
        "duplicate_image_groups_with_different_masks": duplicate_groups_with_mask_disagreement,
        "corrupt_images": corrupt,
        "documentation_files": docs,
        "groups": dict(Counter(record["group"] for record in records)),
        "records": records,
    }
    reports = config["_project_root"] / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "dataset_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )
    quality = [
        "# Dataset quality report", "",
        f"- Decoded image records: {len(records)}",
        f"- Matched image/mask pairs: {inventory['matched_pairs']}",
        f"- Unmatched images: {inventory['unmatched_images']}",
        f"- Unmatched masks: {len(unmatched_masks)}",
        f"- Corrupt images: {len(corrupt)}",
        f"- Image dimensions: {dict(dimensions)}",
        f"- Image formats: {dict(image_formats)}",
        f"- Image modes: {dict(image_modes)}",
        f"- Exact duplicate image groups: {len(duplicate_groups)}; duplicate extra records: {inventory['exact_duplicate_extra_image_count']}",
        f"- Exact duplicate image groups with different mask content: {len(duplicate_groups_with_mask_disagreement)}",
        "- Near-duplicate detection was not performed; exact duplicate groups are kept together during splitting.",
        "- Original full-resolution image pairs are retained as reference data and excluded from thumbnail training splits, as directed by the dataset README.",
        "", "## Files and grouping", "",
        *[f"- `{name}`: {count}" for name, count in sorted(inventory["groups"].items())],
        "", "## Documentation found", "",
        *[f"- `{doc['path']}`" for doc in docs], "",
    ]
    (reports / "dataset_quality_report.md").write_text("\n".join(quality), encoding="utf-8")
    print(f"Inventory saved: {reports / 'dataset_inventory.json'}")
    print(f"Images: {len(records)}; matched pairs: {inventory['matched_pairs']}; corrupt: {len(corrupt)}")
    return inventory


if __name__ == "__main__":
    inspect_dataset()