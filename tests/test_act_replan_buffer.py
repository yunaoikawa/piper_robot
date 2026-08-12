import importlib.util
from pathlib import Path
import sys
import types


def load_server_module():
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    sys.modules.setdefault("torch", torch)
    act = types.ModuleType("act_inference")
    act.ACTInferencePolicy = type("ACTInferencePolicy", (), {})
    sys.modules.setdefault("act_inference", act)
    path = Path(__file__).parents[1] / "cloud_inference_clean-main" / "hpc_inference_act.py"
    spec = importlib.util.spec_from_file_location("hpc_inference_act_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replan_threshold_and_clear_generation():
    module = load_server_module()
    buffer = module.ActionBuffer(40)
    buffer.overwrite([{"chunk_index": index} for index in range(40)])
    assert not buffer.should_replan(12)
    for _ in range(28):
        buffer.pop_action()
    assert buffer.should_replan(12)
    buffer.clear()
    assert buffer.should_replan(12)


def test_right_only_wire_shape_is_documented():
    module = load_server_module()
    assert "quat_wxyz" in module.__doc__
