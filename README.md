# depgraph_hsic_only

This repository aims to provide tooling for pruning the YOLOv8n segmentation model's backbone layers.

The initial implementation contains an abstract base class, `Yolov8SegPruner`, which defines the workflow:

1. Load a pretrained model (`yolov8n-seg.pt`).
2. Train the model before pruning.
3. Prune backbone layers 0–9.
4. Fine-tune the pruned model.

The repository now includes an example implementation,
`DefaultYolov8SegPruner`, which adapts code from the Torch-Pruning project to
perform iterative magnitude pruning and fine‑tuning.  It loads a pretrained
model, runs training, prunes the backbone layers, fine‑tunes, and finally exports the model to ONNX format. Utility helpers for model conversion, training hooks and metric plotting are provided in ``depgraph_hsic_only.utils``.

Developers can subclass `Yolov8SegPruner` to customize the process or reuse the
provided implementation as a starting point.

```
from depgraph_hsic_only import DefaultYolov8SegPruner

pruner = DefaultYolov8SegPruner(pretrained_path="yolov8n-seg.pt", cfg="default.yaml")
pruner.run()
```

You can also run the pruner from the command line::

    python -m depgraph_hsic_only --pretrained model.pt

Alternatively run the default implementation directly with ``prune.py``::

    python prune.py --pretrained yolov8n-seg.pt --cfg default.yaml

The repository provides a sample training configuration in ``default.yaml``.
This file contains the dataset path and basic parameters used by
``DefaultYolov8SegPruner``.  Pass a different file via ``--cfg`` to customize
the training behaviour.
