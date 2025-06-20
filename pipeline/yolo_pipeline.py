"""YOLO pruning pipeline using HSIC-Lasso."""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

from .base import BasePruningPipeline
from prune_methods.hsic_lasso import HsicLassoPruner
from metric_collector import YoloTrainingMetrics, YoloPruningMetrics




class YoloPipeline(BasePruningPipeline):
    """Concrete pruning pipeline for YOLOv8 segmentation."""

    def __init__(self, model_path: str = "yolov8n-seg.pt", data: str = "biotech_model_train.yaml", epochs: int = 10, device: str | None = None):
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
        self.yolo = YOLO(str(self.model_path))
        self.yolo.train(data=self.data, epochs=self.epochs, device=self.device)
        self._baseline_params = sum(p.numel() for p in self.yolo.model.parameters())

    def apply_pruning(self) -> None:  # type: ignore[override]
        assert self.yolo is not None
        pruner = HsicLassoPruner()
        pruner.prune(self.yolo.model)

    def fine_tune(self) -> None:  # type: ignore[override]
        assert self.yolo is not None
        metrics = self.yolo.train(data=self.data, epochs=max(1, self.epochs // 2), device=self.device)
        self.training_metrics = YoloTrainingMetrics.from_results(metrics)

    def collect_metrics(self) -> None:  # type: ignore[override]
        assert self.yolo is not None
        self.pruning_metrics = YoloPruningMetrics.from_model(self.yolo.model, self._baseline_params)

    def visualize_results(self) -> None:  # type: ignore[override]
        if self.training_metrics:
            print("Training metrics:")
            print(f"  mAP50: {self.training_metrics.map:.4f}")
            print(f"  mAP50-95: {self.training_metrics.map50_95:.4f}")
            print(f"  Precision: {self.training_metrics.precision:.4f}")
            print(f"  Recall: {self.training_metrics.recall:.4f}")
        if self.pruning_metrics:
            print("Pruning metrics:")
            print(f"  Params: {self.pruning_metrics.parameters}")
            print(f"  Size (MB): {self.pruning_metrics.size:.2f}")
            print(f"  Filter reduction: {self.pruning_metrics.filter_reduction:.2%}")

    def run(self) -> None:
        """Execute the full pruning workflow."""
        super().run()

