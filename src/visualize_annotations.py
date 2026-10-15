"""Create representative image/mask/overlay visualizations for review."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from src.utils import load_config


def display_mask(mask: Image.Image) -> Image.Image:
    """Render class IDs visibly without changing the stored training mask."""
    class_ids = mask.convert("L").get_flattened_data()
    display = Image.new("RGB", mask.size)
    display.putdata(
        [
            (255, 255, 255) if value == 1 else (255, 200, 0) if value == 255 else (0, 0, 0)
            for value in class_ids
        ]
    )
    return display


def create_visualizations() -> list[str]:
    config = load_config()
    project = config["_project_root"]
    inventory = json.loads((project / "reports" / "dataset_inventory.json").read_text(encoding="utf-8"))
    output = project / "reports" / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    candidates = [row for row in inventory["records"] if row.get("processed_image") and row.get("processed_mask")]
    # Pick examples by mask area, retaining source metadata for reproducibility.
    ranked = []
    for row in candidates:
        path = project / row["processed_mask"]
        with Image.open(path) as mask:
            histogram = mask.histogram()
            area = histogram[1]
        ranked.append((area, row))
    picks: list[tuple[str, dict]] = []
    nonempty = sorted((pair for pair in ranked if pair[0] > 0), key=lambda pair: pair[0])
    if nonempty:
        picks.extend((name, row) for name, row in (
            ("small_positive", nonempty[0][1]),
            ("median_positive", nonempty[len(nonempty) // 2][1]),
            ("large_positive", nonempty[-1][1]),
        ))
    for group, label in (("Thumbnail images/Real images/Without fungi", "real_negative"),
                         ("Thumbnail images/Synthetic images", "synthetic"),
                         ("Thumbnail images/Real images/With fungi", "real_fungi_example")):
        matches = [row for row in candidates if row["group"] == group]
        if matches:
            if label == "real_fungi_example":
                matches = [row for row in matches if next(
                    area for area, candidate in ranked if candidate["sample_id"] == row["sample_id"]
                ) > 0]
                matches.sort(
                    key=lambda row: next(
                        area for area, candidate in ranked if candidate["sample_id"] == row["sample_id"]
                    )
                )
            if matches:
                picks.append((label, matches[len(matches) // 2] if label == "real_fungi_example" else matches[0]))
    paths: list[str] = []
    for label, row in picks:
        with Image.open(project / row["processed_image"]) as source:
            image = source.convert("RGB")
        with Image.open(project / row["processed_mask"]) as mask_image:
            mask = mask_image.convert("L")
        overlay = image.copy()
        tint = Image.new("RGB", image.size, (255, 30, 30))
        fungus_alpha = mask.point(lambda value: 110 if value == 1 else 0)
        ignore_tint = Image.new("RGB", image.size, (255, 200, 0))
        ignore_alpha = mask.point(lambda value: 100 if value == 255 else 0)
        alpha = fungus_alpha
        overlay = Image.composite(tint, overlay, alpha)
        overlay = Image.composite(ignore_tint, overlay, ignore_alpha)
        visible_mask = display_mask(mask)
        canvas = Image.new("RGB", (768, 300), "white")
        draw = ImageDraw.Draw(canvas)
        canvas.paste(image.resize((256, 256)), (0, 32))
        canvas.paste(visible_mask.resize((256, 256), Image.Resampling.NEAREST), (256, 32))
        canvas.paste(overlay.resize((256, 256)), (512, 32))
        draw.text((6, 10), f"{label} — image", fill="black")
        draw.text((262, 10), "mask: fungus=white, ignore=yellow", fill="black")
        draw.text((518, 10), "overlay: fungus=red, ignore=yellow", fill="black")
        destination = output / f"{label}_{row['sample_id']}.png"
        canvas.save(destination)
        paths.append(destination.relative_to(project).as_posix())
    return paths


if __name__ == "__main__":
    print("\n".join(create_visualizations()))