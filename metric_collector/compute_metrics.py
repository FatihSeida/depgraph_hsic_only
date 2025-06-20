"""Abstract collector for compute metrics."""

from __future__ import annotations

import abc


class ComputeMetricCollector(abc.ABC):
    """Collect metrics related to compute resources."""

    @property
    @abc.abstractmethod
    def peak_memory(self) -> float:
        """Return the peak memory usage."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gpu_usage(self) -> float:
        """Return the GPU utilisation."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def time(self) -> float:
        """Return the elapsed time."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def power(self) -> float:
        """Return the consumed power."""
        raise NotImplementedError
