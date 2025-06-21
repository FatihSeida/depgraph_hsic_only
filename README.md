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

In addition, ``HsicLassoPruner`` implements pruning using HSIC‑Lasso scores.
This method measures the dependency between input and output feature maps and
removes filters with scores below a configurable threshold.

Developers can subclass `Yolov8SegPruner` to customize the process or reuse the
provided implementation as a starting point.

```
from depgraph_hsic_only import DefaultYolov8SegPruner

pruner = DefaultYolov8SegPruner(pretrained_path="yolov8n-seg.pt", cfg="default.yaml")
pruner.run()
```

The HSIC based approach can be invoked similarly::

    from depgraph_hsic_only import HsicLassoPruner

    pruner = HsicLassoPruner()
    pruner.prune(model)

You can also run the pruner from the command line::

    python -m depgraph_hsic_only --pretrained model.pt

Alternatively run the default implementation directly with ``prune.py``::

    python prune.py --pretrained yolov8n-seg.pt --cfg default.yaml

``HsicLassoPruner`` operates purely on an in-memory model and does not ship a
separate command-line interface.

The repository provides a sample training configuration in ``default.yaml``.
This file contains the dataset path and basic parameters used by
``DefaultYolov8SegPruner``.  Pass a different file via ``--cfg`` to customize
the training behaviour.

## Installation

Install the required Python packages using ``pip``.  The repository includes
a ``requirements.txt`` file::

    pip install -r requirements.txt

The ``default.yaml`` file defines dataset paths and training parameters that are
compatible with the Ultralytics YOLO API.  Update these paths to point to your
own dataset if necessary.

Running the pruner requires a recent PyTorch installation and, ideally, access
to a CUDA-capable GPU for training and pruning.

## Usage

After installing the dependencies, you can invoke the package module or the
stand‑alone script.  To run via the module entry point::

    python -m depgraph_hsic_only --pretrained yolov8n-seg.pt

Or run ``prune.py`` directly::

    python prune.py --pretrained yolov8n-seg.pt --cfg default.yaml

