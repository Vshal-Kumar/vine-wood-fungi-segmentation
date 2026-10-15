"""Download/load an Ultralytics pretrained model on the user's training machine."""

from __future__ import annotations

import argparse


def download_model(weights: str = "yolo26n-sem.pt") -> None:
    """Trigger Ultralytics' official automatic checkpoint download and verify task."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Ultralytics is not installed. Install the training requirements on the training laptop first."
        ) from error
    model = YOLO(weights)
    task = getattr(model, "task", None)
    if task != "semantic":
        raise SystemExit(
            f"Loaded task={task!r}, not semantic. Use a semantic checkpoint such as yolo26n-sem.pt; do not use -seg instance weights."
        )
    print(f"Semantic model loaded successfully: {weights} (task={task}).")
    print("Ultralytics manages the downloaded checkpoint in its model cache.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26n-sem.pt", help="Official semantic checkpoint name or local .pt path")
    args = parser.parse_args()
    download_model(args.weights)


if __name__ == "__main__":
    main()