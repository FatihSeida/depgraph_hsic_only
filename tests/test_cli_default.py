from importlib import import_module, reload
import sys
from unittest import mock

from .test_yolov8_pruner import load_pruner_module


def test_cli_runs_with_default_arguments(monkeypatch):
    pruner_module = load_pruner_module()
    with mock.patch.dict(sys.modules, {"depgraph_hsic_only.yolov8_pruner": pruner_module}):
        cli = import_module("depgraph_hsic_only.cli")
        reload(cli)
        created = {}

        class DummyPruner:
            def __init__(self, pretrained_path, cfg):
                created["pretrained"] = pretrained_path
                created["cfg"] = cfg

            def run(self):
                created["ran"] = True

        monkeypatch.setattr(cli, "DefaultYolov8SegPruner", DummyPruner)
        monkeypatch.setattr(sys, "argv", ["prog"])
        cli.main()

    assert created == {
        "pretrained": "yolov8n-seg.pt",
        "cfg": "biotech_model_train.yaml",
        "ran": True,
    }
