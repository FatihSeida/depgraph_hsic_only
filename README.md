# depgraph_hsic_only

This repository aims to provide tooling for pruning the YOLOv8n segmentation model's backbone layers.

The initial implementation contains an abstract base class, `Yolov8SegPruner`, which defines the workflow:

1. Load a pretrained model (`yolov8n-seg.pt`).
2. Train the model before pruning.
3. Prune backbone layers 0–9.
4. Fine-tune the pruned model.

The repository now includes an example implementation, `DefaultYolov8SegPruner`,
which adapts code from the Torch-Pruning project to perform iterative pruning
and fine‑tuning.  It loads a pretrained model, runs training, prunes the
backbone layers, fine‑tunes, and finally exports the model to ONNX format.

Developers can subclass `Yolov8SegPruner` to customize the process or reuse the
provided implementation as a starting point.
