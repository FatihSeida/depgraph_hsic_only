"""CLI helper for executing the default YOLOv8 pruner."""

import argparse

from .yolov8_pruner import DefaultYolov8SegPruner


def main() -> None:
    """Entry point for running a pruner from the command line."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 segmentation model pruners"
    )
    parser.add_argument(
        "--pretrained",
        default="yolov8n-seg.pt",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--cfg",
        default="biotech_model_train.yaml",
        help="Training config for DefaultYolov8SegPruner",
    )
    args = parser.parse_args()

    pruner = DefaultYolov8SegPruner(
        pretrained_path=args.pretrained,
        cfg=args.cfg,
    )

    pruner.run()


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()
