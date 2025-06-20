from .test_yolov8_pruner import load_pruner_module


def test_default_pruner_can_be_instantiated():
    pruner_module = load_pruner_module()
    DefaultYolov8SegPruner = pruner_module.DefaultYolov8SegPruner
    pruner = DefaultYolov8SegPruner(pretrained_path="model.pt", cfg="cfg.yaml")
    assert pruner.pretrained_path == "model.pt"
    assert pruner.cfg == "cfg.yaml"
