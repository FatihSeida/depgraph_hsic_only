# depgraph_hsic_only

This repository aims to provide tooling for pruning the YOLOv8n segmentation model's backbone layers.

The initial implementation contains an abstract base class, `Yolov8SegPruner`, which defines the workflow:

1. Load a pretrained model (`yolov8n-seg.pt`).
2. Train the model before pruning.
3. Prune backbone layers 0–9.
4. Fine-tune the pruned model.

The repository now includes two implementations:

* `DefaultYolov8SegPruner` — adapts Torch-Pruning for iterative magnitude
  pruning.
* `HSICYolov8SegPruner` — builds a dependency graph and ranks filters using
  HSIC‑Lasso.  Filters with zero coefficients are removed and the graph is
  updated before fine‑tuning.

Both classes load a pretrained model, run training, prune the backbone layers
and export the result to ONNX format.

Developers can subclass `Yolov8SegPruner` to customize the process or reuse the
provided implementation as a starting point.

```
from depgraph_hsic_only import HSICYolov8SegPruner

pruner = HSICYolov8SegPruner(ratio_target=0.6)
pruner.run()
```
