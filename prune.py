"""Command-line interface for running the default pruner."""

import argparse
from depgraph_hsic_only.yolov8_pruner import DefaultYolov8SegPruner


def main() -> None:
    """Run DefaultYolov8SegPruner using command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DefaultYolov8SegPruner")
    parser.add_argument(
        "--pretrained",
        default="yolov8n-seg.pt",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--cfg",
        default="default.yaml",
        help="Training config for the pruner",
    )
    parser.add_argument(
        "--iterative-steps",
        type=int,
        default=16,
        help="Number of pruning iterations",
    )
    parser.add_argument(
        "--target-prune-rate",
        type=float,
        default=0.5,
        help="Target overall prune ratio",
    )
    parser.add_argument(
        "--max-map-drop",
        type=float,
        default=0.2,
        help="Early stop if mAP drops below this value",
    )

    args = parser.parse_args()

    pruner = DefaultYolov8SegPruner(
        pretrained_path=args.pretrained,
        cfg=args.cfg,
        iterative_steps=args.iterative_steps,
        target_prune_rate=args.target_prune_rate,
        max_map_drop=args.max_map_drop,
    )
    pruner.run()


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
