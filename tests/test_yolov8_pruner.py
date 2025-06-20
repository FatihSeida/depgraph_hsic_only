import importlib
import sys
import types
from unittest import mock


def load_pruner_module():
    """Import depgraph_hsic_only.yolov8_pruner with heavy dependencies stubbed."""
    dummy = lambda *args, **kwargs: None
    modules = {}

    # torch stub
    torch_mod = types.ModuleType("torch")
    torch_mod.save = dummy
    torch_mod.load = dummy
    torch_mod.randn = dummy
    torch_mod.device = lambda *a, **k: None
    torch_mod.Tensor = object
    torch_nn = types.ModuleType("torch.nn")
    class Module:
        def __init__(self, *a, **k):
            pass
    torch_nn.Module = Module
    torch_mod.nn = torch_nn
    modules["torch"] = torch_mod
    modules["torch.nn"] = torch_nn

    # torch_pruning stub
    modules["torch_pruning"] = types.ModuleType("torch_pruning")

    # matplotlib stub
    plt_mod = types.ModuleType("matplotlib.pyplot")
    plt_mod.style = types.SimpleNamespace(use=dummy)
    plt_mod.subplots = lambda *a, **k: (None, types.SimpleNamespace())
    plt_mod.savefig = dummy
    matplotlib_mod = types.ModuleType("matplotlib")
    matplotlib_mod.pyplot = plt_mod
    modules["matplotlib"] = matplotlib_mod
    modules["matplotlib.pyplot"] = plt_mod

    # numpy stub
    np_mod = types.ModuleType("numpy")
    np_mod.array = lambda x: x
    np_mod.argmax = lambda x: 0
    np_mod.argmin = lambda x: 0
    modules["numpy"] = np_mod

    # ultralytics stubs
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = type("YOLO", (), {})
    ultralytics.__version__ = "0"

    nn_modules = types.ModuleType("ultralytics.nn.modules")
    class Dummy:  # simple placeholder
        pass
    nn_modules.Detect = Dummy
    nn_modules.C2f = Dummy
    nn_modules.Conv = Dummy
    nn_modules.Bottleneck = Dummy

    nn_tasks = types.ModuleType("ultralytics.nn.tasks")
    nn_tasks.attempt_load_one_weight = dummy

    ultralytics_nn = types.ModuleType("ultralytics.nn")
    ultralytics_nn.modules = nn_modules
    ultralytics_nn.tasks = nn_tasks

    engine_model = types.ModuleType("ultralytics.yolo.engine.model")
    engine_model.TASK_MAP = {"segment": (None, object)}

    engine_trainer = types.ModuleType("ultralytics.yolo.engine.trainer")
    class BaseTrainer:
        pass
    engine_trainer.BaseTrainer = BaseTrainer

    yolo_engine = types.ModuleType("ultralytics.yolo.engine")
    yolo_engine.model = engine_model
    yolo_engine.trainer = engine_trainer

    yolo_utils = types.ModuleType("ultralytics.yolo.utils")
    yolo_utils.yaml_load = lambda *a, **k: {}
    yolo_utils.LOGGER = types.SimpleNamespace(info=dummy)
    yolo_utils.RANK = -1
    yolo_utils.DEFAULT_CFG_DICT = {}
    yolo_utils.DEFAULT_CFG_KEYS = []

    checks = types.ModuleType("ultralytics.yolo.utils.checks")
    checks.check_yaml = lambda x: x

    torch_utils = types.ModuleType("ultralytics.yolo.utils.torch_utils")
    torch_utils.initialize_weights = dummy
    torch_utils.de_parallel = lambda x: x

    yolo_utils.checks = checks
    yolo_utils.torch_utils = torch_utils

    yolo_mod = types.ModuleType("ultralytics.yolo")
    yolo_mod.engine = yolo_engine
    yolo_mod.utils = yolo_utils

    ultralytics.yolo = yolo_mod
    ultralytics.nn = ultralytics_nn

    modules.update({
        "ultralytics": ultralytics,
        "ultralytics.nn": ultralytics_nn,
        "ultralytics.nn.modules": nn_modules,
        "ultralytics.nn.tasks": nn_tasks,
        "ultralytics.yolo": yolo_mod,
        "ultralytics.yolo.engine": yolo_engine,
        "ultralytics.yolo.engine.model": engine_model,
        "ultralytics.yolo.engine.trainer": engine_trainer,
        "ultralytics.yolo.utils": yolo_utils,
        "ultralytics.yolo.utils.checks": checks,
        "ultralytics.yolo.utils.torch_utils": torch_utils,
    })

    with mock.patch.dict(sys.modules, modules):
        sys.modules.pop("depgraph_hsic_only.yolov8_pruner", None)
        return importlib.import_module("depgraph_hsic_only.yolov8_pruner")


def test_default_pruner_run_sequence():
    pruner_module = load_pruner_module()
    DefaultYolov8SegPruner = pruner_module.DefaultYolov8SegPruner
    pruner = DefaultYolov8SegPruner(pretrained_path="model.pt", cfg="cfg.yaml")

    model = object()
    calls = []

    with mock.patch.object(pruner, "load_pretrained_model", return_value=model) as load_mock, \
         mock.patch.object(pruner, "train", side_effect=lambda m: calls.append("train")) as train_mock, \
         mock.patch.object(pruner, "prune_backbone", side_effect=lambda m: calls.append("prune")) as prune_mock, \
         mock.patch.object(pruner, "fine_tune", side_effect=lambda m: calls.append("fine_tune")) as fine_mock, \
         mock.patch.object(pruner, "save_model", side_effect=lambda m: calls.append("save")) as save_mock:
        pruner.run()

    assert calls == ["train", "prune", "fine_tune", "save"]
    load_mock.assert_called_once_with("model.pt")
    train_mock.assert_called_once_with(model)
    prune_mock.assert_called_once_with(model)
    fine_mock.assert_called_once_with(model)
    save_mock.assert_called_once_with(model)
