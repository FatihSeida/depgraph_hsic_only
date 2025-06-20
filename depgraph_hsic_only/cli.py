import argparse

from .yolov8_pruner import DefaultYolov8SegPruner
from .hsic_pruner import HSICYolov8SegPruner


def main() -> None:
    """Entry point for running a pruner from the command line."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 segmentation model pruners"
    )
    parser.add_argument(
        "--pruner",
        choices=["default", "hsic"],
        default="default",
        help="Pruner implementation to use",
    )
    parser.add_argument(
        "--pretrained",
        default="yolov8n-seg.pt",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--cfg",
        default="default.yaml",
        help="Training config for DefaultYolov8SegPruner",
    )
    parser.add_argument(
        "--ratio-target", type=float, default=0.5, help="HSIC prune ratio target"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="HSIC batch size"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="HSIC input image size"
    )
    parser.add_argument(
        "--lambd-steps", type=int, default=10, help="HSIC lambda steps"
    )

    args = parser.parse_args()

    if args.pruner == "default":
        pruner = DefaultYolov8SegPruner(
            pretrained_path=args.pretrained, cfg=args.cfg
        )
    else:
        pruner = HSICYolov8SegPruner(
            pretrained_path=args.pretrained,
            ratio_target=args.ratio_target,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lambd_steps=args.lambd_steps,
        )

    pruner.run()


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()
