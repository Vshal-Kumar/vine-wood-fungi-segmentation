"""Inspect actual mask encodings and document the selected pixel task."""

from __future__ import annotations

import json
from collections import Counter

from PIL import Image

from src.utils import is_synthetic_group, load_config


def analyze_annotations() -> str:
    config = load_config()
    inventory_path = config["_project_root"] / "reports" / "dataset_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    summaries: dict[str, dict] = {}
    for record in inventory["records"]:
        if not record["train_candidate"] or not record["mask_path"]:
            continue
        mask_path = config["extracted_root"] / record["mask_path"]
        with Image.open(mask_path) as mask:
            mask.load()
            colors = Counter(mask.convert("RGB").get_flattened_data())
            group = record["group"]
            summary = summaries.setdefault(
                group,
                {"count": 0, "modes": Counter(), "sizes": Counter(), "unique_colors": set(),
                 "empty": 0, "positive_at_threshold": 0, "positive_any": 0,
                 "gray_intermediate_pixels": 0, "non_gray_pixels": 0},
            )
            summary["count"] += 1
            summary["modes"][mask.mode] += 1
            summary["sizes"][f"{mask.width}x{mask.height}"] += 1
            summary["unique_colors"].update(colors)
            if is_synthetic_group(group):
                positive = sum(count for color, count in colors.items()
                               if color[0] == color[1] == color[2]
                               and color[0] >= config["synthetic_mask_threshold"])
                any_positive = sum(count for color, count in colors.items()
                                   if color[0] == color[1] == color[2] and color[0] > 0)
                summary["gray_intermediate_pixels"] += sum(
                    count for color, count in colors.items()
                    if color[0] == color[1] == color[2] and color[0] not in (0, 255)
                )
                summary["non_gray_pixels"] += sum(
                    count for color, count in colors.items()
                    if not (color[0] == color[1] == color[2])
                )
            else:
                positive = colors.get((255, 255, 255), 0)
                any_positive = positive
                summary["non_gray_pixels"] += sum(
                    count for color, count in colors.items()
                    if color not in {(0, 0, 0), (255, 255, 255)}
                )
            summary["positive_at_threshold"] += positive
            summary["positive_any"] += any_positive
            if positive == 0:
                summary["empty"] += 1
    lines = [
        "# Annotation analysis", "", "## Finding and task", "",
        "The bundled `Readme.md` explicitly describes paired images and manually obtained **binary segmentation masks**. The supplied labels contain one foreground concept (fungal structures) and background; they do not assign identities to individual fungal objects.",
        "", "**Selected task: binary semantic segmentation.** Each pixel is background (class 0) or fungus (class 1). The masks do not support instance segmentation without inventing instance identities or extracting potentially misleading connected components.",
        "", "Standard Ultralytics YOLO-seg is an instance-polygon task and is not a direct fit for these pixel-class masks. The prepared dataset therefore keeps semantic class-ID masks and is ready for a semantic-segmentation trainer. A compatible model/framework must be selected on the training laptop; no model weights were downloaded and no training was run.",
        "", "## Encodings found and conversion decision", "",
        "- Real thumbnail masks are RGB black/white (black background, white fungus); 11 images in the folder named `With fungi` have fully black masks. They are retained as empty ground truth because the mask, not the folder name, is the pixel-level label.",
        "- All 312 masks in `Without fungi` are entirely black and are retained as valid negative samples.",
        f"- Synthetic masks are RGB grayscale with intermediate values despite the documentation describing binary masks. The prepared binary labels use the configured midpoint threshold `gray >= {config['synthetic_mask_threshold']}` as fungus and lower values as background. Raw masks remain unchanged under `data/extracted`; the decision and counts are reported.",
        "- A few real thumbnail masks contain rare colors other than exact black/white. Those pixels are mapped to ignore index 255 in derived masks, not guessed as either class.",
        "- No per-object IDs are present. There are no polygons, COCO annotations or object boxes in the archive.",
        "", "## Per-group inspection", "",
        "| Source group | Masks | Modes | Dimensions | Empty at selected encoding | Intermediate grayscale pixels | Other-color pixels |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for group, summary in sorted(summaries.items()):
        lines.append(
            f"| `{group}` | {summary['count']} | {dict(summary['modes'])} | {dict(summary['sizes'])} | {summary['empty']} | {summary['gray_intermediate_pixels']} | {summary['non_gray_pixels']} |"
        )
    lines.extend([
        "", "## Sufficiency and remaining ambiguity", "",
        "The thumbnail annotations are sufficient for supervised **semantic** segmentation after the documented synthetic-mask thresholding and explicit ignore-label handling. They are not sufficient for instance segmentation. The grayscale interpretation of synthetic masks should be revisited if the authors' generation code or a more detailed mask specification becomes available; the included overlay samples make the thresholding effect reviewable.",
        "", "The 16 full-resolution microscope image/mask pairs are present and decoded, but the dataset README says these are included for reference. They are not mixed into thumbnail training splits, avoiding an unsupported assumption that they are independent examples. They remain intact in the extracted source tree.", "",
    ])
    output = config["_project_root"] / "reports" / "annotation_analysis.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output.as_posix()


if __name__ == "__main__":
    print(f"Annotation analysis saved: {analyze_annotations()}")