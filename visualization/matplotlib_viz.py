"""Matplotlib based visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .base import BaseVisualization


class MatplotlibVisualization(BaseVisualization):
    """Visualize training and pruning metrics using Matplotlib."""

    def __init__(self) -> None:
        self.fig: plt.Figure | None = None
        self.axes: list[plt.Axes] | None = None

    # ---------------------------------------------------------
    # BaseVisualization interface
    # ---------------------------------------------------------
    def plot_metrics(self, metrics: Any) -> None:  # type: ignore[override]
        """Plot the provided metrics.

        Parameters
        ----------
        metrics : Any
            Metrics data collected from a pipeline. Expected to be a mapping with
            optional ``"training"`` and ``"pruning"`` entries. Each entry may
            contain scalar values or sequences representing metric curves.
        """
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dictionary of values")

        training = metrics.get("training")
        pruning = metrics.get("pruning")
        sections = int(bool(training)) + int(bool(pruning))
        self.fig, axarr = plt.subplots(1, max(1, sections), figsize=(6 * sections, 4))
        if not isinstance(axarr, (list, tuple)):
            axarr = [axarr]
        self.axes = list(axarr)

        idx = 0
        if training is not None:
            ax = self.axes[idx]
            self._plot_section(ax, training, title="Training Metrics")
            idx += 1
        if pruning is not None:
            ax = self.axes[idx]
            self._plot_section(ax, pruning, title="Pruning Metrics", bar_chart=True)

        self.fig.tight_layout()

    def save(self, path: str | Path) -> None:  # type: ignore[override]
        """Save the created figure to ``path``.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        if self.fig is None:
            raise RuntimeError("No figure has been created. Call plot_metrics() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path)

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    def _plot_section(
        self,
        ax: plt.Axes,
        data: Any,
        *,
        title: str,
        bar_chart: bool = False,
    ) -> None:
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        if not isinstance(data, dict):
            raise TypeError("section data must be a dictionary")

        ax.set_title(title)
        for name, values in data.items():
            if isinstance(values, (list, tuple)):
                ax.plot(range(1, len(values) + 1), list(values), label=str(name))
            else:
                if bar_chart:
                    ax.bar(str(name), values)
                else:
                    ax.plot([1], [values], marker="o", label=str(name))
        ax.legend()
