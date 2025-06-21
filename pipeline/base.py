"""Base pruning pipeline definitions."""

from __future__ import annotations

import abc
import logging


logger = logging.getLogger(__name__)


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
        logger.info("Starting train_model")
        self.train_model()
        logger.info("Completed train_model")

        logger.info("Starting apply_pruning")
        self.apply_pruning()
        logger.info("Completed apply_pruning")

        logger.info("Starting fine_tune")
        self.fine_tune()
        logger.info("Completed fine_tune")

        logger.info("Starting collect_metrics")
        self.collect_metrics()
        logger.info("Completed collect_metrics")

        logger.info("Starting visualize_results")
        self.visualize_results()
        logger.info("Completed visualize_results")

