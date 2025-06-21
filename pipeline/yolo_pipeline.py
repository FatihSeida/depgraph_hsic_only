"""YOLO pruning pipeline using HSIC-Lasso."""

from __future__ import annotations

from pathlib import Path

import logging
import torch
from ultralytics import YOLO

from .base import BasePruningPipeline

logger = logging.getLogger(__name__)

if __package__ and __package__.startswith("depgraph_hsic_only"):
    from ..prune_methods.hsic_lasso import HsicLassoPruner
    from ..metric_collector import YoloTrainingMetrics, YoloPruningMetrics
else:  # pragma: no cover - direct script execution
    from prune_methods.hsic_lasso import HsicLassoPruner
    from metric_collector import YoloTrainingMetrics, YoloPruningMetrics




class YoloPipeline(BasePruningPipeline):
    """Concrete pruning pipeline for YOLOv8 segmentation."""

    def __init__(self, model_path: str = "yolov8n-seg.pt", data: str = "default.yaml", epochs: int = 10, device: str | None = None):
        self.model_path = Path(model_path)
        self.data = data
        self.epochs = epochs
        self.device = device
        self.yolo: YOLO | None = None
        self._baseline_params: int | None = None
        self.training_metrics: YoloTrainingMetrics | None = None
        self.pruning_metrics: YoloPruningMetrics | None = None

    # ---------------------------------------------------------
    # Pipeline stages
    # ---------------------------------------------------------
    def train_model(self) -> None:  # type: ignore[override]
        logger.info("Loading model from %s", self.model_path)
        self.yolo = YOLO(str(self.model_path))
        logger.info("Starting training for %d epochs", self.epochs)
        self.yolo.train(data=self.data, epochs=self.epochs, device=self.device)
        self._baseline_params = sum(p.numel() for p in self.yolo.model.parameters())
        logger.info("Training complete")

    def apply_pruning(self) -> None:  # type: ignore[override]
        assert self.yolo is not None
        logger.info("Applying HSIC-Lasso pruning")
        pruner = HsicLassoPruner()
        pruner.prune(self.yolo.model)
        logger.info("Pruning complete")

    def fine_tune(self) -> None:  # type: ignore[override]
        assert self.yolo is not None
        logger.info("Starting fine-tuning")
        metrics = self.yolo.train(data=self.data, epochs=max(1, self.epochs // 2), device=self.device)
        self.training_metrics = YoloTrainingMetrics.from_results(metrics)
        logger.info("Fine-tuning complete")

    def collect_metrics(self) -> None:  # type: ignore[override]
        assert self.yolo is not None
        logger.info("Collecting pruning metrics")
        self.pruning_metrics = YoloPruningMetrics.from_model(self.yolo.model, self._baseline_params)
        logger.info("Metric collection complete")

    def visualize_results(self) -> None:  # type: ignore[override]
        if self.training_metrics or self.pruning_metrics:
            lines = ["Metrics summary:"]
            if self.training_metrics:
                tm = self.training_metrics
                lines.extend([
                    "Training metrics:",
                    f"  {'mAP50':<15}{tm.map:.4f}",
                    f"  {'mAP50-95':<15}{tm.map50_95:.4f}",
                    f"  {'Precision':<15}{tm.precision:.4f}",
                    f"  {'Recall':<15}{tm.recall:.4f}",
                ])
            if self.pruning_metrics:
                pm = self.pruning_metrics
                lines.extend([
                    "Pruning metrics:",
                    f"  {'Params':<15}{pm.parameters}",
                    f"  {'Size (MB)':<15}{pm.size:.2f}",
                    f"  {'Filter reduction':<15}{pm.filter_reduction:.2%}",
                ])
            logger.info("\n".join(lines))

    def run(self) -> None:
        """Execute the full pruning workflow."""
        super().run()

