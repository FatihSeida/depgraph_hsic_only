"""Abstract collector for training metrics."""

from __future__ import annotations

import abc


class TrainingMetricCollector(abc.ABC):
    """Collect metrics related to model training."""

    @property
    @abc.abstractmethod
    def map(self) -> float:
        """Return the mean average precision."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def recall(self) -> float:
        """Return the recall value."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def precision(self) -> float:
        """Return the precision value."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def map50_95(self) -> float:
        """Return the mAP@50-95 score."""
        raise NotImplementedError
