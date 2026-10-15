"""Create deterministic leakage-safe train/validation/test manifests."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from src.utils import load_config


def build_splits(config_path: Path | None = None) -> dict:
    config = load_config(config_path) if config_path else load_config()
    inventory = json.loads(
        (config["_project_root"] / "reports" / "dataset_inventory.json").read_text(encoding="utf-8")
    )
    summaries_path = config["_project_root"] / "data" / "splits" / "preprocessing_summary.json"
    if not summaries_path.exists():
        raise FileNotFoundError("Run preprocessing before splitting the dataset.")
    groups_by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in inventory["records"]:
        if row.get("processed_image") and row.get("processed_mask"):
            groups_by_hash[row["image_sha256"]].append(row)
    # Exact duplicate imagery is kept as one split unit, but every original record
    # and every distinct mask remains represented in that split manifest.
    units = list(groups_by_hash.values())
    random.Random(config["seed"]).shuffle(units)
    ratios = config["split_ratios"]
    targets = {name: len(units) * ratios[name] for name in ("train", "val", "test")}
    assignments: dict[str, list[dict]] = {name: [] for name in targets}
    counts = {name: 0 for name in targets}
    # Greedy stratification by original source category, with image hashes as
    # indivisible units. This prevents exact duplicates crossing split boundaries.
    by_group: dict[str, list[list[dict]]] = defaultdict(list)
    for unit in units:
        category = unit[0]["group"]
        by_group[category].append(unit)
    for category, category_units in sorted(by_group.items()):
        random.Random(config["seed"] + len(category)).shuffle(category_units)
        for unit in category_units:
            split = min(targets, key=lambda name: counts[name] / max(targets[name], 1))
            assignments[split].extend(unit)
            counts[split] += 1
    # Include full provenance and duplicate members; downstream loader uses these
    # paths, with unique sample IDs so duplicated images are not overwritten.
    result = {
        "seed": config["seed"],
        "ratios": ratios,
        "unit_count_exact_unique_images": len(units),
        "record_count_including_duplicates": sum(len(rows) for rows in assignments.values()),
        "duplicate_policy": "Exact image hashes are indivisible split units; all associated records/masks stay in the same split.",
        "split_unit_counts": counts,
        "splits": {
            name: [
                {
                    "sample_id": row["sample_id"],
                    "group": row["group"],
                    "source_image": row["image_path"],
                    "source_mask": row["mask_path"],
                    "image": row["processed_image"],
                    "mask": row["processed_mask"],
                    "sha256": row["image_sha256"],
                }
                for row in sorted(rows, key=lambda item: item["sample_id"])
            ]
            for name, rows in assignments.items()
        },
    }
    config["splits_file"].parent.mkdir(parents=True, exist_ok=True)
    config["splits_file"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Materialize split directories using hardlinks where possible, copying only
    # if filesystem boundaries prevent links; source processed images remain intact.
    root = config["processed_root"]
    for split_name, rows in result["splits"].items():
        for kind in ("images", "masks"):
            (root / kind / split_name).mkdir(parents=True, exist_ok=True)
        for row in rows:
            for kind, key in (("images", "image"), ("masks", "mask")):
                source = config["_project_root"] / row[key]
                destination = root / kind / split_name / f"{row['sample_id']}{'.png' if kind == 'masks' else source.suffix}"
                if destination.exists():
                    continue
                try:
                    destination.hardlink_to(source)
                except OSError:
                    import shutil

                    shutil.copy2(source, destination)
    config_path = root / "dataset.yaml"
    config_path.write_text(
        "# Semantic segmentation dataset; this is not a YOLO instance-polygon YAML.\n"
        "path: .\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "masks_dir: masks\n"
        "names:\n  0: background\n  1: fungus\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    result = build_splits()
    print(json.dumps({"unit_counts": result["split_unit_counts"],
                      "records": result["record_count_including_duplicates"]}, indent=2))