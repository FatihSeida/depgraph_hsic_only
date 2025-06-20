"""Base visualization class definitions."""

from __future__ import annotations

import abc


class BaseVisualization(abc.ABC):
    """Abstract base class for visualization utilities."""

    @abc.abstractmethod
    def plot_metrics(self, metrics):
        """Plot the provided metrics.

        Parameters
        ----------
        metrics : Any
            Metrics data to visualize.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path):
        """Save the visualization to ``path``.

        Parameters
        ----------
        path : str or PathLike
            Destination file path.
        """
        raise NotImplementedError
