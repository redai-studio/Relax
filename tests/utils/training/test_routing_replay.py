"""Replay compact host-side indices as gather-compatible micro-batches."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def replay_type():
    source = Path(__file__).resolve().parents[3] / "relax/utils/training/routing_replay.py"
    tree = ast.parse(source.read_text())
    replay_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RoutingReplay")
    namespace = {
        "torch": torch,
        "device_utils": SimpleNamespace(make_current_torch_device=lambda: torch.device("cpu")),
    }
    exec(compile(ast.Module(body=[replay_class], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["RoutingReplay"], namespace


def test_routing_replay_upcasts_only_requested_microbatch_and_preserves_replay_order(replay_type) -> None:
    replay_class, _ = replay_type
    replay = replay_class()
    recorded = [
        torch.tensor([[7, 2], [4, 0]], dtype=torch.int32),
        torch.tensor([[1, 8]], dtype=torch.int32),
    ]
    # No CUDA pinning allocator is required to exercise the real pop methods.
    replay.top_indices_list = recorded
    scores = [torch.arange(20).reshape(2, 10).float(), torch.arange(10).reshape(1, 10).float()]

    for i, expected in enumerate(recorded):
        actual = replay.pop_forward()
        assert actual.dtype == torch.int64
        assert torch.equal(scores[i].gather(1, actual), scores[i].gather(1, expected.long()))
        assert replay.forward_index == i + 1
        assert replay.backward_index == 0
    replay.clear_forward()
    assert torch.equal(replay.pop_forward(), recorded[0].long())
    for i, expected in enumerate(recorded):
        assert torch.equal(replay.pop_backward(), expected.long())
        assert replay.backward_index == i + 1
    assert all(value.dtype == torch.int32 and value.device.type == "cpu" for value in recorded)
    replay.clear()
    assert replay.forward_index == replay.backward_index == 0
    assert replay.top_indices_list == []


@pytest.mark.skipif(
    os.environ.get("RELAX_ROUTING_REPLAY_GPU_TEST") != "1",
    reason="set RELAX_ROUTING_REPLAY_GPU_TEST=1 on an available GPU",
)
def test_routing_replay_record_pins_int32_and_replays_long_on_current_gpu(replay_type) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the pinned-memory allocator")
    replay_class, namespace = replay_type
    namespace["device_utils"] = SimpleNamespace(make_current_torch_device=lambda: torch.device("cuda", 0))
    replay = replay_class()
    recorded = torch.tensor([[7, 2], [4, 0]], dtype=torch.int32)

    replay.record(recorded)

    assert replay.top_indices_list[0].dtype == torch.int32
    assert replay.top_indices_list[0].device.type == "cpu"
    assert replay.top_indices_list[0].is_pinned()
    for actual in (replay.pop_forward(), replay.pop_backward()):
        assert actual.dtype == torch.int64 and actual.is_cuda
        assert torch.equal(actual.cpu(), recorded.long())
