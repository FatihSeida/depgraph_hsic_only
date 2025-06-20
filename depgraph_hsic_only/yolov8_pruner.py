"""Utilities for pruning YOLOv8 segmentation models.

The module defines :class:`DefaultYolov8SegPruner` which implements the
workflow described in :class:`~depgraph_hsic_only.pruner_base.Yolov8SegPruner`:

1. ``load_pretrained_model`` obtains the pretrained weights.
2. ``train`` performs an initial training run.
3. ``prune_backbone`` iteratively prunes backbone layers using
   Torch‑Pruning and records performance.
4. ``fine_tune`` trains the pruned model.
5. ``save_model`` exports the final result.

Helper utilities for plotting metrics, model conversion and custom training
hooks live in :mod:`depgraph_hsic_only.utils`.
"""

import math
from copy import deepcopy
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import initialize_weights

import torch_pruning as tp

from .pruner_base import Yolov8SegPruner
from .utils import (
    save_pruning_performance_graph,
    replace_c2f_with_c2f_v2,
    train_v2,
)


class DefaultYolov8SegPruner(Yolov8SegPruner):
    """Concrete implementation of ``Yolov8SegPruner`` using Torch-Pruning."""

    def __init__(
        self,
        pretrained_path: str = "yolov8n-seg.pt",
        cfg: str = "default.yaml",
        *,
        iterative_steps: int = 16,
        target_prune_rate: float = 0.5,
        max_map_drop: float = 0.2,
    ) -> None:
        """Initialize parameters controlling the pruning process."""
        super().__init__(pretrained_path)
        self.cfg = cfg
        self.iterative_steps = iterative_steps
        self.target_prune_rate = target_prune_rate
        self.max_map_drop = max_map_drop
        self._batch_size = None

    def load_pretrained_model(self, path: str) -> YOLO:
        """Load a YOLO model and bind custom training hooks."""
        model = YOLO(path)
        model.__setattr__("train_v2", train_v2.__get__(model))
        return model

    def train(self, model: YOLO) -> None:
        """Train the model using the provided config."""
        cfg = YAML.load(check_yaml(self.cfg))
        model.train_v2(data=self.cfg, **cfg)

    def prune_backbone(self, model: YOLO) -> None:
        """Prune backbone layers and record metrics."""
        pruning_cfg = YAML.load(check_yaml(self.cfg))
        self._batch_size = pruning_cfg['batch']
        pruning_cfg['data'] = "coco128.yaml"
        pruning_cfg['epochs'] = 10
        model.model.train()
        replace_c2f_with_c2f_v2(model.model)
        initialize_weights(model.model)
        for _, param in model.model.named_parameters():
            param.requires_grad = True
        example_inputs = torch.randn(
            1,
            3,
            pruning_cfg["imgsz"],
            pruning_cfg["imgsz"],
        ).to(model.device)
        macs_list, nparams_list, map_list, pruned_map_list = [], [], [], []
        base_macs, base_nparams = tp.utils.count_ops_and_params(
            model.model,
            example_inputs,
        )
        pruning_cfg['name'] = "baseline_val"
        pruning_cfg['batch'] = 1
        validation_model = deepcopy(model)
        metric = validation_model.val(**pruning_cfg)
        init_map = metric.box.map
        macs_list.append(base_macs)
        nparams_list.append(100)
        map_list.append(init_map)
        pruned_map_list.append(init_map)
        pruning_ratio = 1 - math.pow(
            (1 - self.target_prune_rate),
            1 / self.iterative_steps,
        )
        # Only prune layers 0-9 of the internal ``model.model.model``
        # list.  ``YOLO.model`` contains a ``ModuleList`` where the
        # first 10 entries form the backbone.  All later layers are
        # detection heads or other blocks that should remain intact
        # during pruning.  We mark them as ignored so ``GroupNormPruner``
        # will operate solely on the backbone modules.
        backbone_limit = 10
        head_modules = list(model.model.model[backbone_limit:])

        for i in range(self.iterative_steps):
            model.model.train()
            for _, param in model.model.named_parameters():
                param.requires_grad = True
            ignored_layers = []
            unwrapped_parameters = []
            ignored_layers.extend(head_modules)
            # Also ignore explicit Detect layers inside the model
            for m in model.model.modules():
                if isinstance(m, (Detect,)):
                    ignored_layers.append(m)
            example_inputs = example_inputs.to(model.device)
            pruner = tp.pruner.GroupNormPruner(
                model.model,
                example_inputs,
                importance=tp.importance.GroupMagnitudeImportance(),
                iterative_steps=1,
                pruning_ratio=pruning_ratio,
                ignored_layers=ignored_layers,
                unwrapped_parameters=unwrapped_parameters
            )
            pruner.step()
            pruning_cfg['name'] = f"step_{i}_pre_val"
            pruning_cfg['batch'] = 1
            validation_model.model = deepcopy(model.model)
            metric = validation_model.val(**pruning_cfg)
            pruned_map = metric.box.map
            pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(
                pruner.model,
                example_inputs,
            )
            current_speed_up = float(macs_list[0]) / pruned_macs
            print(
                f"After pruning iter {i + 1}: MACs={pruned_macs / 1e9} G, "
                f"#Params={pruned_nparams / 1e6} M, mAP={pruned_map}, "
                f"speed up={current_speed_up}"
            )
            for _, param in model.model.named_parameters():
                param.requires_grad = True
            pruning_cfg['name'] = f"step_{i}_finetune"
            pruning_cfg['batch'] = self._batch_size
            model.train_v2(pruning=True, **pruning_cfg)
            pruning_cfg['name'] = f"step_{i}_post_val"
            pruning_cfg['batch'] = 1
            validation_model = YOLO(model.trainer.best)
            metric = validation_model.val(**pruning_cfg)
            current_map = metric.box.map
            print(f"After fine tuning mAP={current_map}")
            macs_list.append(pruned_macs)
            nparams_list.append(pruned_nparams / base_nparams * 100)
            pruned_map_list.append(pruned_map)
            map_list.append(current_map)
            del pruner
            save_pruning_performance_graph(
                nparams_list,
                map_list,
                macs_list,
                pruned_map_list,
            )
            if init_map - current_map > self.max_map_drop:
                print("Pruning early stop")
                break
        self._final_macs = macs_list[-1]

    def fine_tune(self, model: YOLO) -> None:
        """Fine-tune the pruned model."""
        if self._batch_size is None:
            raise RuntimeError("Model must be pruned before fine tuning")
        model.train_v2(pruning=True, cfg=self.cfg, batch=self._batch_size)

    def save_model(self, model: YOLO) -> None:
        """Export the model to ONNX format."""
        model.export(format='onnx')
