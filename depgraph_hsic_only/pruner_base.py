"""Base classes defining the pruning workflow."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Yolov8SegPruner(ABC):
    """Abstract base class for pruning YOLOv8n-seg backbone layers."""

    def __init__(self, pretrained_path: str = "yolov8n-seg.pt") -> None:
        """Store path to pretrained weights."""
        self.pretrained_path = pretrained_path

    def run(self) -> None:
        """Complete workflow for training, pruning, and fine-tuning."""
        model = self.load_pretrained_model(self.pretrained_path)
        self.train(model)
        self.prune_backbone(model)
        self.fine_tune(model)
        self.save_model(model)

    @abstractmethod
    def load_pretrained_model(self, path: str):
        """Load the pretrained YOLOv8n-seg model."""
        raise NotImplementedError

    @abstractmethod
    def train(self, model) -> None:
        """Perform initial training using the pretrained model."""
        raise NotImplementedError

    @abstractmethod
    def prune_backbone(self, model) -> None:
        """Prune backbone layers 0-9 of the model."""
        raise NotImplementedError

    @abstractmethod
    def fine_tune(self, model) -> None:
        """Fine-tune the pruned model."""
        raise NotImplementedError

    @abstractmethod
    def save_model(self, model) -> None:
        """Persist the trained model to disk."""
        raise NotImplementedError
