"""Audit annotations and prepare deduplicated training-only sampling metadata."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import quantiles

from PIL import Image, ImageDraw

from src.utils import load_config


def prepare_experiment_audit() -> dict:
    """Write an audit and a train-only unique-hash manifest without deleting data."""
    config = load_config()
    project = config["_project_root"]
    inventory = json.loads((project / "reports" / "dataset_inventory.json").read_text(encoding="utf-8"))
    split_data = json.loads(config["splits_file"].read_text(encoding="utf-8"))
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    empty_positive_folder: list[str] = []
    class_id_counts: Counter[str] = Counter()
    for row in inventory["records"]:
        if not row.get("train_candidate") or not row.get("processed_mask"):
            continue
        groups[row["group"]][row["image_sha256"]].append(row)
        with Image.open(project / row["processed_mask"]) as mask:
            histogram = mask.histogram()
            foreground = histogram[1]
            class_id_counts["background"] += histogram[0]
            class_id_counts["fungus"] += foreground
            class_id_counts["ignore"] += histogram[config["ignore_index"]]
        if row["group"] == "Thumbnail images/Real images/With fungi" and foreground == 0:
            empty_positive_folder.append(row["image_path"])

    duplicate_rows_by_group = {}
    unique_rows_by_group = {}
    conflicts = []
    for group, hash_groups in sorted(groups.items()):
        duplicate_rows_by_group[group] = sum(len(rows) - 1 for rows in hash_groups.values())
        unique_rows_by_group[group] = len(hash_groups)
        for digest, rows in hash_groups.items():
            mask_hashes = {row.get("mask_sha256") for row in rows}
            if len(mask_hashes) > 1:
                conflicts.append({"group": group, "sha256": digest, "paths": [row["image_path"] for row in rows]})

    split_results = {}
    unique_manifests = {
        split: {"seed": config["seed"], "sampling": "one record per exact-image SHA-256",
                "records": []}
        for split in ("train", "val", "test")
    }
    unique_train_pixel_counts: Counter[str] = Counter()
    unique_train_positive_areas: list[int] = []
    unique_train_empty_mask_count = 0
    for split, rows in split_data["splits"].items():
        unique: dict[str, dict] = {}
        for row in rows:
            unique.setdefault(row["sha256"], row)
        split_results[split] = {
            "records_in_existing_split": len(rows),
            "unique_image_hashes": len(unique),
            "duplicate_exposures_if_all_records_sampled_once_per_epoch": len(rows) - len(unique),
            "multiplicity_histogram": dict(sorted(Counter(Counter(r["sha256"] for r in rows).values()).items())),
        }
        if split == "train":
            unique_manifests[split]["records"] = sorted(unique.values(), key=lambda item: item["sample_id"])
            for row in unique_manifests[split]["records"]:
                with Image.open(project / row["mask"]) as mask:
                    histogram = mask.histogram()
                unique_train_pixel_counts["background"] += histogram[0]
                unique_train_pixel_counts["fungus"] += histogram[1]
                unique_train_pixel_counts["ignore"] += histogram[config["ignore_index"]]
                if histogram[1]:
                    unique_train_positive_areas.append(histogram[1])
                else:
                    unique_train_empty_mask_count += 1
        else:
            unique_manifests[split]["records"] = sorted(unique.values(), key=lambda item: item["sample_id"])

    # Identify the empty masks that also occur as empty-mask duplicate records
    # elsewhere in the officially paired thumbnail collections.
    empty_duplicate_details = []
    empty_unique_hash_groups: dict[str, list[dict]] = {}
    rows_by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in inventory["records"]:
        if row.get("train_candidate"):
            rows_by_hash[row["image_sha256"]].append(row)
    for path in empty_positive_folder:
        source_row = next(row for row in inventory["records"] if row["image_path"] == path)
        empty_unique_hash_groups[source_row["image_sha256"]] = rows_by_hash[source_row["image_sha256"]]
        empty_duplicate_details.append({
            "image": path,
            "identical_image_records": [
                {"group": duplicate["group"], "image": duplicate["image_path"], "mask": duplicate["mask_path"]}
                for duplicate in rows_by_hash[source_row["image_sha256"]]
            ],
            "mask_sha256_matches_exact_duplicates": len({
                duplicate.get("mask_sha256") for duplicate in rows_by_hash[source_row["image_sha256"]]
            }) == 1,
        })
    empty_hashes_also_in_negative_group = sum(
        any(row["group"] == "Thumbnail images/Real images/Without fungi" for row in rows)
        for rows in empty_unique_hash_groups.values()
    )
    empty_hashes_without_negative_alias = len(empty_unique_hash_groups) - empty_hashes_also_in_negative_group

    output_dir = project / "data" / "splits"
    unique_manifest_paths = {}
    for split, manifest in unique_manifests.items():
        manifest_path = output_dir / f"{split}_unique.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        unique_manifest_paths[split] = manifest_path.relative_to(project).as_posix()
    visualization_dir = project / "reports" / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    columns = 3
    cell_width, cell_height = 256, 300
    rows_count = (len(empty_positive_folder) + columns - 1) // columns
    contact_sheet = Image.new("RGB", (columns * cell_width, rows_count * cell_height), "white")
    draw = ImageDraw.Draw(contact_sheet)
    for index, relative_image in enumerate(sorted(empty_positive_folder)):
        row = next(item for item in inventory["records"] if item["image_path"] == relative_image)
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        with Image.open(config["extracted_root"] / row["image_path"]) as source:
            source_image = source.convert("RGB")
        with Image.open(project / row["processed_mask"]) as source_mask:
            class_ids = source_mask.convert("L").get_flattened_data()
            display_mask = Image.new("RGB", source_mask.size)
            display_mask.putdata([
                (255, 200, 0) if value == config["ignore_index"]
                else (255, 255, 255) if value == 1 else (0, 0, 0)
                for value in class_ids
            ])
        draw.text((x + 4, y + 4), Path(relative_image).name + " — image then mask", fill="black")
        contact_sheet.paste(source_image, (x, y + 24))
        contact_sheet.paste(display_mask, (x, y + 24 + 256))
    empty_mask_visual = visualization_dir / "empty_with_fungi_mask_audit.png"
    contact_sheet.save(empty_mask_visual)
    known_pixels = unique_train_pixel_counts["background"] + unique_train_pixel_counts["fungus"]
    train_pos_weight = (
        unique_train_pixel_counts["background"] / unique_train_pixel_counts["fungus"]
        if unique_train_pixel_counts["fungus"]
        else None
    )
    area_quartiles = quantiles(unique_train_positive_areas, n=4, method="inclusive") if len(unique_train_positive_areas) >= 2 else []
    train_area_stats = {
        "positive_mask_count": len(unique_train_positive_areas),
        "empty_mask_count": unique_train_empty_mask_count,
        "min_fungus_pixels": min(unique_train_positive_areas, default=0),
        "q1_fungus_pixels": area_quartiles[0] if area_quartiles else 0,
        "median_fungus_pixels": area_quartiles[1] if area_quartiles else 0,
        "q3_fungus_pixels": area_quartiles[2] if area_quartiles else 0,
        "max_fungus_pixels": max(unique_train_positive_areas, default=0),
        "positive_masks_at_most_1_percent_area": sum(area <= 655 for area in unique_train_positive_areas),
        "positive_masks_at_most_5_percent_area": sum(area <= 3276 for area in unique_train_positive_areas),
    }
    result = {
        "empty_masks_in_with_fungi_folder": empty_duplicate_details,
        "empty_unique_image_hash_count": len(empty_unique_hash_groups),
        "empty_unique_hashes_also_listed_in_without_fungi": empty_hashes_also_in_negative_group,
        "empty_unique_hashes_still_requiring_expert_review": empty_hashes_without_negative_alias,
        "duplicate_extra_records_by_source_group": duplicate_rows_by_group,
        "unique_images_by_source_group": unique_rows_by_group,
        "duplicate_image_annotation_conflicts": conflicts,
        "split_sampling_effect": split_results,
        "training_pixels_by_class_id": dict(class_id_counts),
        "unique_train_pixels_by_class_id": dict(unique_train_pixel_counts),
        "unique_train_fungus_pixel_fraction_excluding_ignore": unique_train_pixel_counts["fungus"] / known_pixels if known_pixels else 0.0,
        "unique_train_bce_pos_weight_background_over_fungus_candidate": train_pos_weight,
        "unique_train_foreground_area_statistics": train_area_stats,
        "unique_manifests": unique_manifest_paths,
        "unique_sample_counts": {split: len(manifest["records"]) for split, manifest in unique_manifests.items()},
        "empty_with_fungi_contact_sheet": empty_mask_visual.relative_to(project).as_posix(),
        "all_original_and_processed_records_preserved": True,
    }
    report = [
        "# Experiment preparation audit", "",
        "This report audits data properties and prepares a unique-hash train manifest. It does **not** train a model or claim accuracy gains.",
        "", "## Empty masks in the 'With fungi' folder", "",
        f"Found {len(empty_duplicate_details)} such masks. The folder is a coarse grouping; the pixel masks are the training target, so these records remain intact. Review them manually before deciding whether to retain them.",
        f"These represent {len(empty_unique_hash_groups)} unique images; {empty_hashes_also_in_negative_group} are also listed in the 'Without fungi' group with exactly matching empty masks, while {empty_hashes_without_negative_alias} unique images have no negative-group alias and merit expert review.",
        f"Contact sheet: `{result['empty_with_fungi_contact_sheet']}` (image on top, class mask below; white is fungus, yellow is ignore).",
        "", "| Image | Exact duplicate records and groups | Duplicate mask content agrees |", "|---|---|---|",
    ]
    for item in empty_duplicate_details:
        aliases = "; ".join(f"{alias['group']}: {alias['image']}" for alias in item["identical_image_records"])
        report.append(f"| `{item['image']}` | {aliases} | {item['mask_sha256_matches_exact_duplicates']} |")
    report.extend(["", "## Exact duplicate sampling impact", "",
                   "All source/processed records remain available. Unique-hash manifests select one deterministic representative per image hash for train, validation, and test. Use them for fair sample-weighted metrics; they do not delete duplicate files.",
                   "", "| Split | Existing records | Unique image hashes | Extra repeated exposures if all records sampled | Multiplicity histogram (records per hash: groups) |",
                   "|---|---:|---:|---:|---|"])
    for split, values in split_results.items():
        report.append(f"| {split} | {values['records_in_existing_split']} | {values['unique_image_hashes']} | {values['duplicate_exposures_if_all_records_sampled_once_per_epoch']} | {values['multiplicity_histogram']} |")
    report.extend([
        "", f"Duplicate image groups with different mask checksums: {len(conflicts)}.",
        f"Unique-hash examples prepared by split: { {split: len(manifest['records']) for split, manifest in unique_manifests.items()} }.",
        f"Processed target pixel counts: {dict(class_id_counts)} (ignore pixels should be excluded from loss).",
        f"Unique-hash training known-pixel counts: background={unique_train_pixel_counts['background']:,}; fungus={unique_train_pixel_counts['fungus']:,}; ignore={unique_train_pixel_counts['ignore']:,}.",
        f"Fungus pixels are {100 * result['unique_train_fungus_pixel_fraction_excluding_ignore']:.2f}% of known training pixels; background/fungus ratio is {train_pos_weight:.2f}:1. This is a candidate starting `pos_weight` for BCE, not a validated optimum.",
        f"Unique training masks contain {train_area_stats['positive_mask_count']} nonempty and {train_area_stats['empty_mask_count']} empty labels. Positive-mask fungus area min/Q1/median/Q3/max: {train_area_stats['min_fungus_pixels']}/{train_area_stats['q1_fungus_pixels']:.1f}/{train_area_stats['median_fungus_pixels']:.1f}/{train_area_stats['q3_fungus_pixels']:.1f}/{train_area_stats['max_fungus_pixels']} pixels; {train_area_stats['positive_masks_at_most_1_percent_area']} positive masks are <=1% foreground.",
        "", "## Training-only experiments not run here", "",
        "Dice + weighted BCE, Tversky/Focal loss, boundary-aware loss, Sobel input channels, and augmentation-versus-baseline all require a semantic-segmentation model, PyTorch, and a training GPU. This workspace currently has no PyTorch and no visible NVIDIA device (`nvidia-smi` unavailable), and the friend’s laptop environment has not been inspected. No training comparison or accuracy claim is possible yet.",
        "",
        "Data-level test status: unit tests verify the class-ID converters, Otsu threshold helper, duplicate-aware manifest helpers, and paired geometry transforms. The geometric helper is not wired into a trainer and no augmented files were written.",
        "", "Recommended controlled order on the training laptop: (1) baseline BCE+Dice, unique-hash sampling, no augmentation; (2) change only loss to Tversky/Focal; (3) revert to best loss and add paired right-angle geometric augmentation; (4) separately test an edge-aware loss or Sobel input; select on validation; test the chosen configuration once on held-out test.",
        "", "The 11 empty positive-folder masks and synthetic threshold are annotation questions, not image-enhancement problems. Do not automatically repair them, apply morphology, or tune thresholds against the test split.", "",
    ])
    (project / "reports" / "experiment_preparation_report.md").write_text("\n".join(report), encoding="utf-8")
    (project / "reports" / "experiment_preparation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(prepare_experiment_audit(), indent=2))