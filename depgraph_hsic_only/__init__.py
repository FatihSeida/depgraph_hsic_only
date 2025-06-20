from .pruner_base import Yolov8SegPruner
from .yolov8_pruner import DefaultYolov8SegPruner

try:
    from .hsic_pruner import HSICYolov8SegPruner
except ImportError:  # pragma: no cover
    HSICYolov8SegPruner = None

__all__ = [
    "Yolov8SegPruner",
    "DefaultYolov8SegPruner",
]
if HSICYolov8SegPruner is not None:
    __all__.append("HSICYolov8SegPruner")
