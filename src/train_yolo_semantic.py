"""Train an Ultralytics semantic-segmentation model on the prepared dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import PROJECT_ROOT


def train(
    weights: str = "yolo26n-sem.pt",
    data: Path | None = None,
    device: str = "0",
    imgsz: int = 256,
    batch: int = 4,
    epochs: int = 100,
    patience: int = 15,
    name: str = "yolo26n_sem_baseline",
) -> Path:
    """Start training only after checking Ultralytics, model task and CUDA device."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Training dependencies missing. Install PyTorch for the laptop's CUDA setup, then install Ultralytics."
        ) from error

    dataset_yaml = data or (PROJECT_ROOT / "data" / "processed" / "yolo_semantic_unique" / "dataset.yaml")
    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"Semantic YOLO dataset YAML not found: {dataset_yaml}. Run python -m src.prepare_yolo_semantic first."
        )
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch reports no CUDA device. Check the training laptop's NVIDIA driver and CUDA-enabled PyTorch install, or explicitly pass --device cpu."
        )

    model = YOLO(weights)
    if getattr(model, "task", None) != "semantic":
        raise ValueError(
            f"Checkpoint task is {getattr(model, 'task', None)!r}; this dataset requires semantic segmentation. Use a -sem checkpoint, not a -seg instance model."
        )
    run_dir = PROJECT_ROOT / "experiments" / name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment directory: {run_dir}")
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        seed=42,
        deterministic=True,
        device=device,
        amp=True,
        workers=4,
        project=str(PROJECT_ROOT / "experiments"),
        name=name,
        exist_ok=False,
        save=True,
        plots=True,
        cache=False,
        # Conservative microscopy baseline: no color, mosaic, mixup, or geometric
        # transforms. Compare paired flips in a separately named run later.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26n-sem.pt")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--device", default="0", help="CUDA index (default 0) or explicitly cpu")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--name", default="yolo26n_sem_baseline")
    args = parser.parse_args()
    print(train(args.weights, args.data, args.device, args.imgsz, args.batch,
                args.epochs, args.patience, args.name))


if __name__ == "__main__":
    main()