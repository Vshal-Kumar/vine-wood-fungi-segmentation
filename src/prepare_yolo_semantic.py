"""Materialize duplicate-free splits for Ultralytics semantic segmentation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml
from PIL import Image

from src.utils import load_config, sha256_file


def prepare_yolo_semantic_dataset() -> Path:
    """Build a trainer-ready semantic dataset using Ultralytics' mask-folder YAML.

    Input images are hard-linked when possible (otherwise copied) without image
    decoding/re-encoding. Label masks are lossless single-channel class-ID PNGs.
    The unique-hash manifests keep duplicates out of train, validation and test.
    """
    config = load_config()
    project: Path = config["_project_root"]
    dataset_root = project / "data" / "processed" / "yolo_semantic_unique"
    splits_root = project / "data" / "splits"
    for split in ("train", "val", "test"):
        manifest_path = splits_root / f"{split}_unique.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing {manifest_path}; run: python -m src.experiment_preparation"
            )
        records = json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
        expected_images = {
            f"{row['sample_id']}{Path(row['image']).suffix.lower()}" for row in records
        }
        expected_masks = {f"{row['sample_id']}.png" for row in records}
        for kind, expected in (("images", expected_images), ("masks", expected_masks)):
            split_dir = dataset_root / kind / split
            if split_dir.exists():
                existing = {path.name for path in split_dir.iterdir() if path.is_file()}
                extra = existing - expected
                if extra:
                    raise RuntimeError(
                        f"Stale files in {split_dir}: {sorted(extra)[:5]}. "
                        "Move the old generated yolo_semantic_unique directory aside before rebuilding; "
                        "the preparation script will not silently delete samples."
                    )
        for row in records:
            source_image = project / row["image"]
            source_mask = project / row["mask"]
            if not source_image.is_file() or not source_mask.is_file():
                raise FileNotFoundError(f"Missing pair for sample {row['sample_id']}")
            image_target = dataset_root / "images" / split / f"{row['sample_id']}{source_image.suffix.lower()}"
            mask_target = dataset_root / "masks" / split / f"{row['sample_id']}.png"
            image_target.parent.mkdir(parents=True, exist_ok=True)
            mask_target.parent.mkdir(parents=True, exist_ok=True)
            if not image_target.exists():
                try:
                    image_target.hardlink_to(source_image)
                except OSError:
                    shutil.copy2(source_image, image_target)
            if not mask_target.exists():
                try:
                    mask_target.hardlink_to(source_mask)
                except OSError:
                    shutil.copy2(source_mask, mask_target)
            if sha256_file(image_target) != row["sha256"]:
                raise RuntimeError(f"TIFF image copy/hash mismatch for {row['sample_id']}")
            if sha256_file(mask_target) != sha256_file(source_mask):
                raise RuntimeError(f"PNG mask copy/hash mismatch for {row['sample_id']}")
            with Image.open(image_target) as image, Image.open(mask_target) as mask:
                image.load()
                mask.load()
                if image.size != mask.size:
                    raise ValueError(f"Image/mask size mismatch for {row['sample_id']}: {image.size} vs {mask.size}")
                invalid = set(mask.convert("L").get_flattened_data()) - {0, 1, config["ignore_index"]}
                if invalid:
                    raise ValueError(f"Unexpected class IDs for {row['sample_id']}: {sorted(invalid)}")

    dataset_yaml = dataset_root / "dataset.yaml"
    dataset_config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "masks_dir": "masks",
        "names": {0: "background", 1: "fungus"},
    }
    dataset_yaml.write_text(
        "# Ultralytics YOLO semantic-segmentation dataset (PNG class-ID masks).\n"
        + yaml.safe_dump(dataset_config, sort_keys=False),
        encoding="utf-8",
    )
    return dataset_yaml


if __name__ == "__main__":
    print(prepare_yolo_semantic_dataset())