"""Abstract collector for pruning metrics."""

from __future__ import annotations

import abc


class PruningMetricCollector(abc.ABC):
    """Collect metrics related to model pruning."""

    @property
    @abc.abstractmethod
    def flops(self) -> float:
        """Return the number of floating point operations."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parameters(self) -> int:
        """Return the number of model parameters."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def size(self) -> float:
        """Return the serialized model size."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def filter_reduction(self) -> float:
        """Return the fraction of filters removed."""
        raise NotImplementedError
