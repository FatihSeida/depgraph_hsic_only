from __future__ import annotations

"""Pruning method using HSIC-Lasso scoring."""

import yaml
import torch
import torch.nn as nn
import torch_pruning as tp
from sklearn.linear_model import Lasso

from .base import BasePruningMethod


class HsicLassoPruner(BasePruningMethod):
    """Prune convolutional filters based on HSIC-Lasso scores."""

    def __init__(self, threshold: float = 0.02, batch_size: int = 4, input_size: tuple[int, int] = (640, 640)):
        self.threshold = threshold
        self.batch_size = batch_size
        self.input_size = input_size

    # ---------------------------------------------------------
    # Config loading
    # ---------------------------------------------------------
    def load_config(self, cfg_path) -> None:
        """Load YAML configuration from ``cfg_path``."""
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self.threshold = float(cfg.get("threshold", self.threshold))
        self.batch_size = int(cfg.get("batch_size", self.batch_size))
        inp = cfg.get("input_size", self.input_size)
        self.input_size = tuple(inp)

    # ---------------------------------------------------------
    # Main pruning entry point
    # ---------------------------------------------------------
    def prune(self, model) -> None:
        """Prune ``model`` in-place using HSIC-Lasso scoring."""
        device = next(model.parameters()).device
        example_inputs = torch.randn(self.batch_size, 3, *self.input_size, device=device)

        dependency_graph = tp.DependencyGraph()
        dependency_graph.build_dependency(model, example_inputs=example_inputs)

        feat_in: dict[str, torch.Tensor] = {}
        feat_out: dict[str, torch.Tensor] = {}

        def save_in(module, x, y, name):
            feat_in[name] = x[0].detach().cpu()
            feat_out[name] = y.detach().cpu()

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(lambda m, x, y, n=name: save_in(m, x, y, n))

        with torch.no_grad():
            model(example_inputs)

        group_scores: dict[tp.Group, float] = {}
        for group in dependency_graph.get_all_groups():
            names = [node.name for node in group.nodes]
            kernels = []
            for nm in names:
                X = feat_in[nm]
                for k in range(X.size(1)):
                    kernels.append(self._gram_centered(X[:, k : k + 1]).flatten())
            Y = feat_out[names[-1]]
            target = self._gram_centered(Y).flatten()

            A = torch.stack(kernels, dim=1).numpy()
            lasso = Lasso(alpha=1e-3, positive=True, fit_intercept=False, max_iter=1000)
            lasso.fit(A, target.numpy())
            alphas = lasso.coef_
            group_scores[group] = float(alphas.mean())

        to_prune = [g for g, s in group_scores.items() if s < self.threshold]
        for group in to_prune:
            dependency_graph.prune_group(group)

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------
    @staticmethod
    def _gram_centered(z: torch.Tensor) -> torch.Tensor:
        n = z.shape[0]
        z = z.flatten(1)
        K = z @ z.T
        H = torch.eye(n) - torch.ones(n, n) / n
        return (H @ K @ H) / n
