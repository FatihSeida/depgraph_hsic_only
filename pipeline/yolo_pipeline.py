"""YOLO pruning pipeline using HSIC-Lasso."""

from __future__ import annotations

from pathlib import Path
import tempfile
import os

import torch
from ultralytics import YOLO

from .base import BasePruningPipeline
from prune_methods.hsic_lasso import HsicLassoPruner
from metric_collector import TrainingMetricCollector, PruningMetricCollector


class _TrainingMetrics(TrainingMetricCollector):
    """Simple training metrics wrapper for YOLO results."""

    def __init__(self, metrics: dict | None = None):
        self._m = metrics or {}

    @property
    def map(self) -> float:  # type: ignore[override]
        return float(self._m.get("metrics/mAP50", 0.0))

    @property
    def recall(self) -> float:  # type: ignore[override]
        return float(self._m.get("metrics/recall", 0.0))

    @property
    def precision(self) -> float:  # type: ignore[override]
        return float(self._m.get("metrics/precision", 0.0))

    @property
    def map50_95(self) -> float:  # type: ignore[override]
        return float(self._m.get("metrics/mAP50-95", 0.0))


class _PruningMetrics(PruningMetricCollector):
    """Collect simple pruning statistics."""

    def __init__(self, model: torch.nn.Module, baseline_params: int | None = None):
        self._params = sum(p.numel() for p in model.parameters())
        self._size = self._calc_size(model)
        self._flops = 0.0
        if baseline_params:
            self._filter_reduction = 1.0 - self._params / baseline_params
        else:
            self._filter_reduction = 0.0

    @staticmethod
    def _calc_size(model: torch.nn.Module) -> float:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            size = os.path.getsize(f.name) / 1e6
        os.unlink(f.name)
        return size

    @property
    def flops(self) -> float:  # type: ignore[override]
        return self._flops

    @property
    def parameters(self) -> int:  # type: ignore[override]
        return self._params

    @property
    def size(self) -> float:  # type: ignore[override]
        return self._size

    @property
    def filter_reduction(self) -> float:  # type: ignore[override]
        return self._filter_reduction


class YoloPipeline(BasePruningPipeline):
    """Concrete pruning pipeline for YOLOv8 segmentation."""

    def __init__(self, model_path: str = "yolov8n-seg.pt", data: str = "biotech_model_train.yaml", epochs: int = 10, device: str | None = None):
        self.model_path = Path(model_path)
        self.data = data
        self.epochs = epochs
        self.device = device
        self.yolo: YOLO | None = None
        self._baseline_params: int | None = None
        self.training_metrics: _TrainingMetrics | None = None
        self.pruning_metrics: _PruningMetrics | None = None

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
        self.training_metrics = _TrainingMetrics(metrics or {})

    def collect_metrics(self) -> None:  # type: ignore[override]
        assert self.yolo is not None
        self.pruning_metrics = _PruningMetrics(self.yolo.model, self._baseline_params)

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

