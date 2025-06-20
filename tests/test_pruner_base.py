import importlib
import sys
import types
import pytest

# Avoid importing heavy dependencies when loading the package
dummy_mod1 = types.ModuleType('depgraph_hsic_only.yolov8_pruner')
dummy_mod1.DefaultYolov8SegPruner = object
sys.modules.setdefault('depgraph_hsic_only.yolov8_pruner', dummy_mod1)

from depgraph_hsic_only.pruner_base import Yolov8SegPruner


class DummyPruner(Yolov8SegPruner):
    """Concrete class that simply calls the base implementations."""

    def load_pretrained_model(self, path):
        return super().load_pretrained_model(path)

    def train(self, model):
        return super().train(model)

    def prune_backbone(self, model):
        return super().prune_backbone(model)

    def fine_tune(self, model):
        return super().fine_tune(model)

    def save_model(self, model):
        return super().save_model(model)


def test_abstract_methods_raise_not_implemented():
    pruner = DummyPruner()
    with pytest.raises(NotImplementedError):
        pruner.load_pretrained_model("model.pt")
    with pytest.raises(NotImplementedError):
        pruner.train(None)
    with pytest.raises(NotImplementedError):
        pruner.prune_backbone(None)
    with pytest.raises(NotImplementedError):
        pruner.fine_tune(None)
    with pytest.raises(NotImplementedError):
        pruner.save_model(None)
