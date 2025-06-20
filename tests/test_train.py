from .test_yolov8_pruner import load_pruner_module


def test_train_passes_dataset_path():
    pruner_module = load_pruner_module()
    DefaultYolov8SegPruner = pruner_module.DefaultYolov8SegPruner

    recorded = {}

    class DummyModel:
        def train_v2(self, **kwargs):
            recorded.update(kwargs)

    pruner = DefaultYolov8SegPruner(
        pretrained_path="model.pt",
        cfg="biotech_model_train.yaml",
    )
    pruner.train(DummyModel())

    assert recorded.get("data") == "biotech_model_train.yaml"
