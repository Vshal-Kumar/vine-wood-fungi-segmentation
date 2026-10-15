"""Predict semantic fungal masks on a TIFF or image directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.utils import PROJECT_ROOT, SUPPORTED_EXTENSIONS


def predict(weights: Path, source: Path, output: Path, device: str = "0", imgsz: int = 256) -> int:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Prediction requires PyTorch and Ultralytics on the inference machine.") from error
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Prediction source does not exist: {source}")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu to use CPU inference.")
    model = YOLO(str(weights))
    if getattr(model, "task", None) != "semantic":
        raise ValueError("Expected a semantic-segmentation checkpoint (-sem), not an instance-segmentation model.")
    files = [source] if source.is_file() else sorted(
        path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported images found in {source}")
    mask_dir, overlay_dir = output / "masks", output / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(files):
        result = model.predict(str(image_path), imgsz=imgsz, device=device, verbose=False)[0]
        if result.semantic_mask is None:
            raise RuntimeError(f"Model returned no semantic class map for {image_path}")
        class_map = result.semantic_mask.data.detach().cpu().numpy().astype("uint8")
        with Image.open(image_path) as original:
            image = original.convert("RGB")
        if class_map.shape != (image.height, image.width):
            raise ValueError(f"Model mask shape {class_map.shape} differs from original {image.size}; refusing silent resize.")
        stem = f"{index:05d}_{image_path.stem}"
        Image.fromarray(class_map, mode="L").save(mask_dir / f"{stem}.png")
        red = Image.new("RGB", image.size, (255, 20, 20))
        alpha = Image.fromarray((class_map == 1).astype("uint8") * 100, mode="L")
        Image.composite(red, image, alpha).save(overlay_dir / f"{stem}.png")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained best.pt checkpoint")
    parser.add_argument("--source", type=Path, required=True, help="One image (.tif/.tiff supported) or directory")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "predictions" / "new_images")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=256)
    args = parser.parse_args()
    count = predict(args.weights, args.source, args.output, args.device, args.imgsz)
    print(f"Processed {count} image(s); class-ID masks and overlays saved under {args.output}")


if __name__ == "__main__":
    main()