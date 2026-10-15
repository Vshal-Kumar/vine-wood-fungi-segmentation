"""Prepare lossless native-resolution images and semantic masks for training."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from src.utils import is_synthetic_group, load_config, mask_to_class_ids


def preprocess_images() -> dict:
    config = load_config()
    inventory = json.loads(
        (config["_project_root"] / "reports" / "dataset_inventory.json").read_text(encoding="utf-8")
    )
    processed_root: Path = config["processed_root"]
    processed_root.mkdir(parents=True, exist_ok=True)
    report_groups: dict[str, dict] = {}
    ignored_total = 0
    candidate_count = 0
    for record in inventory["records"]:
        if not record["train_candidate"]:
            continue
        candidate_count += 1
        if not record["decodable"] or not record["mask_path"]:
            continue
        image_path = config["extracted_root"] / record["image_path"]
        source_mask = config["extracted_root"] / record["mask_path"]
        if record["width"] != 256 or record["height"] != 256:
            raise ValueError(
                f"Training source is not a thumbnail (expected 256x256): {record['image_path']}"
            )
        image_out = processed_root / "images" / f"{record['sample_id']}{image_path.suffix.lower()}"
        mask_out = processed_root / "masks" / f"{record['sample_id']}.png"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_out)
        with Image.open(source_mask) as mask:
            converted, counts = mask_to_class_ids(
                mask,
                synthetic=is_synthetic_group(record["group"]),
                synthetic_threshold=config["synthetic_mask_threshold"],
                ignore_index=config["ignore_index"],
            )
        converted.save(mask_out, format="PNG", optimize=False)
        ignored_total += counts["ignored_unrecognized"]
        group = report_groups.setdefault(record["group"], {"count": 0, "background_pixels": 0,
                                                              "fungus_pixels": 0, "ignored_pixels": 0})
        group["count"] += 1
        group["background_pixels"] += counts["background"]
        group["fungus_pixels"] += counts["fungus"]
        group["ignored_pixels"] += counts["ignored_unrecognized"]
        record["processed_image"] = image_out.relative_to(config["_project_root"]).as_posix()
        record["processed_mask"] = mask_out.relative_to(config["_project_root"]).as_posix()
    inventory["processed_groups"] = report_groups
    (config["_project_root"] / "reports" / "dataset_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )
    summary = {
        "candidate_images": candidate_count,
        "processed_images": sum(group["count"] for group in report_groups.values()),
        "native_image_sizes_preserved": True,
        "image_files_copied_without_pixel_changes": True,
        "processed_mask_encoding": "single-channel uint8 PNG; 0=background, 1=fungus, 255=ignore",
        "synthetic_grayscale_threshold": config["synthetic_mask_threshold"],
        "ignored_unrecognized_pixels": ignored_total,
        "groups": report_groups,
    }
    (config["_project_root"] / "data" / "splits").mkdir(parents=True, exist_ok=True)
    (config["_project_root"] / "data" / "splits" / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "# Preprocessing report", "",
        "- Raw archive and extracted source images/masks were not edited.",
        "- Training candidates are only the three documented thumbnail collections; 16 full-resolution microscope pairs remain reference-only.",
        "- Thumbnail images are copied byte-for-byte to the processed image folder (no resize, enhancement, color conversion or lossy recompression).",
        "- Paired masks are converted to single-channel class-ID PNG without spatial resizing. Real masks use exact black/background and white/fungus; unexpected colors become ignore ID 255.",
        f"- Synthetic grayscale masks use the configurable midpoint threshold of {config['synthetic_mask_threshold']}; this resolves antialiased mask values into the documented binary task and can be changed centrally.",
        f"- Images processed: {summary['processed_images']} / {candidate_count}; ignored unexpected-color pixels: {ignored_total}.",
        "- Native thumbnail size is 256x256, so no tiling or resizing was performed.",
        "- Class-ID masks can be read by common semantic segmentation data loaders; standard Ultralytics YOLO instance segmentation is not the training format.",
        "", "## Counts by source group", "",
        *[f"- `{group}`: {counts['count']} images; background={counts['background_pixels']}, fungus={counts['fungus_pixels']}, ignored={counts['ignored_pixels']} pixels"
          for group, counts in sorted(report_groups.items())],
        "",
    ]
    (config["_project_root"] / "reports" / "preprocessing_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(preprocess_images(), indent=2))