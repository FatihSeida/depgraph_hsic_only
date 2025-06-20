# depgraph_hsic_only

This repository aims to provide tooling for pruning the YOLOv8n segmentation model's backbone layers.

The initial implementation contains an abstract base class, `Yolov8SegPruner`, which defines the workflow:

1. Load a pretrained model (`yolov8n-seg.pt`).
2. Train the model before pruning.
3. Prune backbone layers 0–9.
4. Fine-tune the pruned model.

Concrete subclasses should implement the actual logic for each step. All methods raise `NotImplementedError` by default.
