from __future__ import annotations

"""Concrete metric collectors for YOLO models."""

from dataclasses import dataclass, asdict
import os
import tempfile

import psutil
import torch
from ultralytics.utils.torch_utils import get_flops

from .training_metrics import TrainingMetricCollector
from .pruning_metrics import PruningMetricCollector
from .compute_metrics import ComputeMetricCollector


@dataclass
class YoloTrainingMetrics(TrainingMetricCollector):
    """Training metrics parsed from Ultralytics results."""

    _map: float = 0.0
    _recall: float = 0.0
    _precision: float = 0.0
    _map50_95: float = 0.0

    @classmethod
    def from_results(cls, metrics: dict | None) -> "YoloTrainingMetrics":
        metrics = metrics or {}
        return cls(
            _map=float(metrics.get("metrics/mAP50", 0.0)),
            _recall=float(metrics.get("metrics/recall", 0.0)),
            _precision=float(metrics.get("metrics/precision", 0.0)),
            _map50_95=float(metrics.get("metrics/mAP50-95", 0.0)),
        )

    # Properties implementing TrainingMetricCollector -----------------------
    @property
    def map(self) -> float:  # type: ignore[override]
        return self._map

    @property
    def recall(self) -> float:  # type: ignore[override]
        return self._recall

    @property
    def precision(self) -> float:  # type: ignore[override]
        return self._precision

    @property
    def map50_95(self) -> float:  # type: ignore[override]
        return self._map50_95

    # Utility ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class YoloPruningMetrics(PruningMetricCollector):
    """Pruning metrics for a YOLO model."""

    _flops: float
    _parameters: int
    _size: float
    _filter_reduction: float

    @classmethod
    def from_model(
        cls, model: torch.nn.Module, baseline_params: int | None = None, imgsz: int = 640
    ) -> "YoloPruningMetrics":
        params = sum(p.numel() for p in model.parameters())
        flops = get_flops(model, imgsz)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            size = os.path.getsize(f.name) / 1e6
        os.unlink(f.name)
        if baseline_params:
            fr = 1.0 - params / baseline_params
        else:
            fr = 0.0
        return cls(_flops=flops, _parameters=params, _size=size, _filter_reduction=fr)

    # Properties implementing PruningMetricCollector -----------------------
    @property
    def flops(self) -> float:  # type: ignore[override]
        return self._flops

    @property
    def parameters(self) -> int:  # type: ignore[override]
        return self._parameters

    @property
    def size(self) -> float:  # type: ignore[override]
        return self._size

    @property
    def filter_reduction(self) -> float:  # type: ignore[override]
        return self._filter_reduction

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SystemMetrics(ComputeMetricCollector):
    """System resource metrics collected via psutil and torch."""

    _peak_memory: float
    _gpu_usage: float
    _time: float
    _power: float

    @classmethod
    def capture(cls, start_time: float, end_time: float) -> "SystemMetrics":
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1e6
        if torch.cuda.is_available():
            gpu = torch.cuda.max_memory_allocated() / 1e6
        else:
            gpu = 0.0
        elapsed = end_time - start_time
        return cls(_peak_memory=mem, _gpu_usage=gpu, _time=elapsed, _power=0.0)

    # Properties implementing ComputeMetricCollector -----------------------
    @property
    def peak_memory(self) -> float:  # type: ignore[override]
        return self._peak_memory

    @property
    def gpu_usage(self) -> float:  # type: ignore[override]
        return self._gpu_usage

    @property
    def time(self) -> float:  # type: ignore[override]
        return self._time

    @property
    def power(self) -> float:  # type: ignore[override]
        return self._power

    def to_dict(self) -> dict:
        return asdict(self)
