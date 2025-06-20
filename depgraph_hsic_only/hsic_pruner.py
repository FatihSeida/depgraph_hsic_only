from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

import torch_pruning as tp
from pyHSICLasso import HSICLasso

from .pruner_base import Yolov8SegPruner


class HSICYolov8SegPruner(Yolov8SegPruner):
    """Prune YOLOv8 segmentation models using HSIC-Lasso."""

    def __init__(self, pretrained_path: str = "yolov8n-seg.pt", *, ratio_target: float = 0.5,
                 batch_size: int = 4, img_size: int = 640, lambd_steps: int = 10) -> None:
        super().__init__(pretrained_path)
        self.ratio_target = ratio_target
        self.batch_size = batch_size
        self.img_size = img_size
        self.lambd_steps = lambd_steps

    # ------------------------------------------------------------------
    # Yolov8SegPruner interface
    # ------------------------------------------------------------------
    def load_pretrained_model(self, path: str) -> YOLO:
        """Load the pretrained YOLOv8 model."""
        return YOLO(path)

    def train(self, model: YOLO) -> None:  # pragma: no cover - example
        """Placeholder training routine."""
        pass

    def fine_tune(self, model: YOLO) -> None:  # pragma: no cover - example
        """Placeholder fine‑tuning routine."""
        pass

    def save_model(self, model: YOLO) -> None:
        """Save the pruned model."""
        model.export(format="onnx")

    # ------------------------------------------------------------------
    # Pruning implementation
    # ------------------------------------------------------------------
    def prune_backbone(self, model: YOLO) -> None:
        """Apply HSIC-Lasso based pruning to backbone layers."""
        device = model.device
        example_inputs = torch.randn(self.batch_size, 3, self.img_size, self.img_size, device=device)

        # Build the dependency graph of the backbone
        dg = tp.DependencyGraph().build_dependency(model.model, example_inputs=example_inputs)

        # Collect all convolution layers within backbone layers 0-9
        backbone = model.model.model[:10]
        conv_layers = [m for m in backbone.modules() if isinstance(m, nn.Conv2d)]

        # Register hooks to capture feature maps
        feature_maps: dict[nn.Module, torch.Tensor] = {}

        def _hook(module: nn.Module, _inp: tuple[torch.Tensor], out: torch.Tensor) -> None:
            feature_maps[module] = out.detach()

        handles = [layer.register_forward_hook(_hook) for layer in conv_layers]
        with torch.no_grad():
            y = model.model(example_inputs)
        for h in handles:
            h.remove()

        if isinstance(y, (list, tuple)):
            y = y[0]
        target = y.mean(dim=[2, 3]).cpu().numpy()

        for layer in conv_layers:
            fmap = feature_maps.get(layer)
            if fmap is None:
                continue
            X = fmap.mean(dim=[2, 3]).cpu().numpy()  # samples x channels
            Y = target

            hsic = HSICLasso()
            hsic.input(X, Y)
            hsic.regression(num_feat=X.shape[1], B=max(1, min(self.batch_size, X.shape[0])))
            coeffs = np.array(hsic.beta)
            lam_path = np.array(hsic.lam).reshape(-1)

            selected = coeffs[:, -1]
            for i in range(coeffs.shape[1]):
                keep_ratio = np.count_nonzero(coeffs[:, i]) / coeffs.shape[0]
                if keep_ratio <= self.ratio_target:
                    selected = coeffs[:, i]
                    break

            prune_idx = [i for i, w in enumerate(selected) if w == 0]
            if not prune_idx:
                continue

            group = dg.get_pruning_group(layer, tp.prune_conv_out_channels, prune_idx)
            group.prune()

        dg.update_index_mapping()
