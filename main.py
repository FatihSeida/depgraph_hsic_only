"""Command-line entry point for the pruning pipeline."""

from __future__ import annotations

import argparse
import logging

if __package__:
    from .pipeline.yolo_pipeline import YoloPipeline
else:  # pragma: no cover - direct script execution
    from pipeline.yolo_pipeline import YoloPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Run the YOLO pruning pipeline")
    parser.add_argument(
        "--pretrained",
        default="yolov8n-seg.pt",
        help="Path to the pretrained YOLO model",
    )
    parser.add_argument(
        "--config",
        default="default.yaml",
        help="Path to the training data configuration file",
    )
    return parser.parse_args()


def main() -> None:
    """Instantiate the pipeline and execute it."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    pipeline = YoloPipeline(model_path=args.pretrained, data=args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
