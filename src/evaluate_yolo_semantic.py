"""Evaluate semantic predictions on each unique held-out test image once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from src.utils import PROJECT_ROOT


def evaluate(weights: Path, device: str = "0", imgsz: int = 256) -> dict:
    try:
        import numpy as np
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Evaluation requires NumPy, PyTorch and Ultralytics on the training machine.") from error
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu to run explicitly on CPU.")
    model = YOLO(str(weights))
    if getattr(model, "task", None) != "semantic":
        raise ValueError("This evaluator requires a semantic-segmentation checkpoint (-sem), not an instance-segmentation model.")
    manifest = json.loads((PROJECT_ROOT / "data" / "splits" / "test_unique.json").read_text(encoding="utf-8"))
    confusion = np.zeros((2, 2), dtype=np.int64)
    outputs = PROJECT_ROOT / "predictions" / "test_evaluation"
    mask_dir, overlay_dir = outputs / "masks", outputs / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest["records"]:
        image_path = PROJECT_ROOT / row["image"]
        target_path = PROJECT_ROOT / row["mask"]
        result = model.predict(str(image_path), imgsz=imgsz, device=device, verbose=False)[0]
        if result.semantic_mask is None:
            raise RuntimeError(f"Model returned no semantic class map for {image_path}")
        prediction = result.semantic_mask.data.detach().cpu().numpy().astype("uint8")
        with Image.open(target_path) as target_image:
            target = np.asarray(target_image.convert("L"), dtype=np.uint8)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch for {row['sample_id']}: {prediction.shape} vs {target.shape}")
        valid = target != 255
        target_binary = target[valid] == 1
        pred_binary = prediction[valid] == 1
        tn = int(np.count_nonzero(~target_binary & ~pred_binary))
        fp = int(np.count_nonzero(~target_binary & pred_binary))
        fn = int(np.count_nonzero(target_binary & ~pred_binary))
        tp = int(np.count_nonzero(target_binary & pred_binary))
        confusion += np.array([[tn, fp], [fn, tp]], dtype=np.int64)
        Image.fromarray(prediction, mode="L").save(mask_dir / f"{row['sample_id']}.png")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        red = Image.new("RGB", image.size, (255, 20, 20))
        alpha = Image.fromarray((prediction == 1).astype("uint8") * 100, mode="L")
        Image.composite(red, image, alpha).save(overlay_dir / f"{row['sample_id']}.png")

    metrics = {}
    for class_id, name in ((0, "background"), (1, "fungus")):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        tn = int(confusion.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        metrics[name] = {"iou": iou, "dice": dice, "precision": precision, "recall": recall}
    total_valid = int(confusion.sum())
    output = {
        "weights": str(weights),
        "task": "binary semantic segmentation",
        "test_samples_unique": len(manifest["records"]),
        "ignore_index": 255,
        "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
        "pixel_accuracy": float(confusion.trace() / total_valid) if total_valid else 0.0,
        "per_class": metrics,
        "mean_iou": sum(item["iou"] for item in metrics.values()) / 2,
        "prediction_masks": str(mask_dir.relative_to(PROJECT_ROOT)),
        "overlays": str(overlay_dir.relative_to(PROJECT_ROOT)),
    }
    report_path = PROJECT_ROOT / "reports" / "evaluation_report.md"
    report_path.write_text(
        "# Evaluation report\n\n"
        "Metrics below are computed once on the unique held-out test images; ignore pixels (255) are excluded.\n\n"
        f"- Checkpoint: `{weights}`\n- Unique test images: {output['test_samples_unique']}\n"
        f"- Pixel accuracy: {output['pixel_accuracy']:.6f}\n- Mean IoU: {output['mean_iou']:.6f}\n\n"
        "| Class | IoU | Dice | Precision | Recall |\n|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {name} | {values['iou']:.6f} | {values['dice']:.6f} | {values['precision']:.6f} | {values['recall']:.6f} |"
            for name, values in metrics.items()
        ) + "\n",
        encoding="utf-8",
    )
    (PROJECT_ROOT / "reports" / "evaluation_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Trained best.pt checkpoint")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.weights, args.device, args.imgsz), indent=2))


if __name__ == "__main__":
    main()