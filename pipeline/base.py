"""Base pruning pipeline definitions."""

from __future__ import annotations

import abc


class BasePruningPipeline(abc.ABC):
    """Abstract pipeline coordinating pruning workflow."""

    @abc.abstractmethod
    def train_model(self) -> None:
        """Train the initial model."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_pruning(self) -> None:
        """Apply pruning to the trained model."""
        raise NotImplementedError

    @abc.abstractmethod
    def fine_tune(self) -> None:
        """Fine-tune the pruned model."""
        raise NotImplementedError

    @abc.abstractmethod
    def collect_metrics(self) -> None:
        """Collect metrics after fine-tuning."""
        raise NotImplementedError

    @abc.abstractmethod
    def visualize_results(self) -> None:
        """Visualize collected metrics."""
        raise NotImplementedError

    def run(self) -> None:
        """Execute the complete pruning pipeline."""
        self.train_model()
        self.apply_pruning()
        self.fine_tune()
        self.collect_metrics()
        self.visualize_results()

